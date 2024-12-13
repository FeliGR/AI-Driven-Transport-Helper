const CLASSES = ["crying_baby", "clock_alarm", "water_drops", "toilet_flush"];

const YAMNET_MODEL_URL = "https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1";
const MODEL_SAMPLE_RATE = 16000;
const NUM_SECONDS = 5;

let model;
let yamnet;
let audioContext;
let stream;

let isProcessing = false;
let isRecording = false; // Flag to track recording state
const alertQueue = [];
const alertHistory = [];
const MAX_HISTORY = 5;

const timeDataQueue = [];

let recorder = null; // Global reference to the recorder

// Elementos de la UI
const btnMicStart = document.querySelector("#btnMicStart");
const btnMicStop = document.querySelector("#btnMicStop");
const predictionClass = document.querySelector("#predictionClass");
const outputMessageEl = document.getElementById("outputMessage");
const outputMessageEl1 = document.getElementById("outputMessage1");
const outputMessageEl2 = document.getElementById("outputMessage2");
const alertHistoryEl = document.getElementById("alertHistory");

// Cargar los modelos de TensorFlow.js
async function loadYamnetModel() {
  try {
    console.log("Cargando modelo YamNet...");
    return await tf.loadGraphModel(YAMNET_MODEL_URL, { fromTFHub: true });
  } catch (error) {
    console.error("Error al cargar el modelo YamNet:", error);
    throw new Error("No se pudo cargar el modelo YamNet.");
  }
}

async function loadCustomAudioClassificationModel() {
  try {
    console.log("Cargando modelo personalizado...");
    return await tf.loadLayersModel("./model/model.json");
  } catch (error) {
    console.error("Error al cargar el modelo personalizado:", error);
    throw new Error("No se pudo cargar el modelo personalizado.");
  }
}

async function predict(yamnet, model, audioData) {
  try {
    const embeddings = await getEmbeddingsFromTimeDomainData(yamnet, audioData);
    const results = await model.predict(embeddings);
    const meanTensor = results.mean(0);

    // Get prediction probabilities and class index
    const predictionArray = meanTensor.dataSync();
    const predictedIndex = meanTensor.argMax(0).dataSync()[0];
    const confidence = predictionArray[predictedIndex];
    embeddings.dispose();
    results.dispose();
    meanTensor.dispose();

    return { predictedIndex, confidence };
  } catch (error) {
    console.error("Error during prediction:", error);
    throw new Error("Prediction failed.");
  }
}

async function getEmbeddingsFromTimeDomainData(yamnet, audioData) {
  try {
    // Crear un tensor 1-D a partir de audioData
    const waveformTensor = tf.tensor(audioData); // Forma: [80000]
    console.log("Forma del tensor 'waveform':", waveformTensor.shape); // Verificación

    // Crear un objeto con la clave 'waveform'
    const inputs = { waveform: waveformTensor };

    // Realizar la predicción
    const [scores, embeddings] = yamnet.predict(inputs);
    console.log("Forma de 'scores':", scores.shape);
    console.log("Forma de 'embeddings':", embeddings.shape);

    // Liberar memoria
    waveformTensor.dispose();

    return embeddings;
  } catch (error) {
    console.error("Error al extraer embeddings del audio:", error);
    throw new Error("No se pudieron extraer los embeddings del audio.");
  }
}

async function getAudioStream(audioTrackConstraints) {
  const options = audioTrackConstraints || {};
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: false,
      audio: {
        sampleRate: options.sampleRate || MODEL_SAMPLE_RATE,
        sampleSize: options.sampleSize || 16,
        channelCount: options.channelCount || 1,
      },
    });
  } catch (error) {
    console.error("Error al obtener el stream de audio:", error);
    throw new Error("No se pudo acceder al micrófono.");
  }
}

// Función para generar el prompt basado en la predicción
function generatePrompt(prediction, location = "Car 4", urgency = "Moderate") {
  const eventDescriptions = {
    crying_baby: "A baby crying sound was detected",
    clock_alarm: "An alarm sound was detected",
    toilet_flush: "A toilet flush sound was detected",
    water_drops: "A water dripping sound was detected",
    Equipaje: "A luggage moving sound was detected",
  };

  const description =
    eventDescriptions[prediction] || "An unidentified sound was detected";

  return [
    {
      role: "user",
      content: `Create a brief, clear announcement for train passengers (maximum 30 words).

Event: ${description}
Location: ${location}
Urgency: ${urgency}

Requirements:
- Be concise and informative
- Focus on essential information
- Avoid causing panic
- No bullet points or lists

Generate only the message, and just give the text. Do not include any additional information or instructions.`,
    },
  ];
}

// Inicializar y cargar el Web Worker para Transformers.js
const worker = new Worker("worker.js", { type: "module" }); // Asegúrate de que worker.js está en la misma carpeta

// Manejar mensajes del Worker
worker.onmessage = async (e) => {
  switch (e.data.type) {
    case "token":
      // Original English message
      globalMessage += e.data.token;
      outputMessageEl.textContent = globalMessage;
      break;
    case "ready":
      console.log("Worker está listo");
      break;
    case "done":
      if (alertQueue.length > 0) {
        const alert = alertQueue[0];

        // Get translations
        const [spanishTranslation, portugueseTranslation] = await Promise.all([
          translateMessage(globalMessage, "es"),
          translateMessage(globalMessage, "pt"),
        ]);

        // Create alert with all languages
        const alertWithTranslations = {
          ...alert,
          messages: {
            en: globalMessage,
            es: spanishTranslation,
            pt: portugueseTranslation,
          },
          timestamp: new Date(),
        };

        // Add to history
        alertHistory.unshift(alertWithTranslations);
        if (alertHistory.length > MAX_HISTORY) {
          alertHistory.pop();
        }

        updateAlertHistory();

        // Speak all languages
        await speakTextSequentially([
          { text: globalMessage, lang: "en" },
          { text: spanishTranslation, lang: "es" },
          { text: portugueseTranslation, lang: "pt" },
        ]);

        alertQueue.shift();
      }

      // Reset states
      isProcessing = false;
      predictionClass.textContent = "Waiting for input...";
      outputMessageEl.textContent = "";
      globalMessage = "";

      processNextAlert();
      break;
    case "error":
      console.error("Error en el Worker:", e.data.message);
      isProcessing = false;
      processNextAlert();
      break;

    default:
      console.warn("Tipo de mensaje desconocido del Worker:", e.data.type);
  }
};

// Actualizar el historial de alertas en la UI
function updateAlertHistory() {
  alertHistoryEl.innerHTML = `
    <h3>Recent Alerts</h3>
    ${alertHistory
      .map(
        (alert) => `
      <div class="alert-item">
        <div class="alert-header">
          <span class="alert-time">${alert.timestamp.toLocaleTimeString()}</span>
          <span class="alert-type">${alert.prediction}</span>
        </div>
        <div class="alert-messages">
          <div class="message">
            <div class="message-label">🇬🇧 English</div>
            <div class="message-text">${alert.messages.en}</div>
          </div>
          <div class="message">
            <div class="message-label">🇪🇸 Español</div>
            <div class="message-text">${alert.messages.es}</div>
          </div>
          <div class="message">
            <div class="message-label">🇵🇹 Português</div>
            <div class="message-text">${alert.messages.pt}</div>
          </div>
        </div>
      </div>
    `
      )
      .join("")}
  `;
}

// Función para procesar la siguiente alerta en la cola
async function processNextAlert() {
  if (isProcessing || alertQueue.length === 0) return;

  isProcessing = true;
  const alert = alertQueue[0];

  predictionClass.textContent = `Processing: ${alert.prediction}`;
  outputMessageEl.textContent = "";

  // Generate and process the message
  const prompt = generatePrompt(alert.prediction, "Train car 4", "Urgent");
  worker.postMessage({ type: "generate", prompt: prompt });
}

// Función para verificar si el audio es silencioso
function isSilent(audioData, threshold = 0.01) {
  const rms = Math.sqrt(
    audioData.reduce((sum, val) => sum + val * val, 0) / audioData.length
  );
  return rms < threshold;
}

// Función para detener la grabación
function stopRecording() {
  try {
    if (audioContext && stream) {
      audioContext.close();
      stream.getTracks().forEach((track) => track.stop());
    }

    if (recorder) {
      recorder.disconnect();
      recorder.port.onmessage = null; // Remove the message handler
      recorder = null;
    }

    timeDataQueue.splice(0);
    isRecording = false; // Update recording flag

    btnMicStart.disabled = false;
    btnMicStop.disabled = true;

    predictionClass.textContent = "Stopped";
    outputMessageEl.textContent = "";

    console.log("Grabación detenida.");
  } catch (error) {
    console.error("Error al detener la grabación:", error);
  }
}

// Función para manejar los mensajes de audio del Worklet
function handleAudioMessage(e) {
  try {
    const inputBuffer = e.data;
    if (!inputBuffer || inputBuffer.length === 0) return;

    // Check if recording is active
    if (!isRecording) return;

    timeDataQueue.push(...inputBuffer);

    if (timeDataQueue.length >= MODEL_SAMPLE_RATE * NUM_SECONDS) {
      const audioData = new Float32Array(
        timeDataQueue.splice(0, MODEL_SAMPLE_RATE * NUM_SECONDS)
      );

      // Optional: Check for silence
      if (isSilent(audioData)) {
        console.log("Audio is silent. Skipping prediction.");
        return;
      }

      predict(yamnet, model, audioData)
        .then(({ predictedIndex, confidence }) => {
          const predictedClass = CLASSES[predictedIndex];
          console.log(
            `Predicted Class: ${predictedClass}, Confidence: ${confidence.toFixed(
              2
            )}`
          );

          // Set confidence threshold
          const CONFIDENCE_THRESHOLD = 0.7;

          if (confidence >= CONFIDENCE_THRESHOLD) {
            // Add to alert queue
            alertQueue.push({
              prediction: predictedClass,
              timestamp: new Date(),
            });

            // Start processing if not already
            processNextAlert();
          } else {
            console.log("Confidence below threshold. Alert not added.");
          }
        })
        .catch((error) => {
          console.error("Error during prediction:", error);
        });
    }
  } catch (error) {
    console.error("Error processing audio:", error);
  }
}

// Iniciar la grabación y procesamiento de audio
btnMicStart.onclick = async () => {
  try {
    // Iniciar el micrófono y procesar audio
    stream = await getAudioStream();
    audioContext = new AudioContext({ sampleRate: MODEL_SAMPLE_RATE });
    const source = audioContext.createMediaStreamSource(stream);

    await audioContext.audioWorklet.addModule("recorder.worklet.js");
    recorder = new AudioWorkletNode(audioContext, "recorder.worklet");
    source.connect(recorder);

    recorder.port.onmessage = handleAudioMessage;

    isRecording = true; // Update recording flag

    btnMicStart.disabled = true;
    btnMicStop.disabled = false;

    predictionClass.textContent = "Listening...";
    outputMessageEl.textContent = "";

    console.log("Grabación iniciada.");
  } catch (error) {
    console.error("Error al iniciar el micrófono:", error);
  }
};

btnMicStop.onclick = () => {
  stopRecording();
};

// Función principal para inicializar la aplicación
async function main() {
  try {
    yamnet = await loadYamnetModel();
    console.log("Modelo YamNet cargado");

    model = await loadCustomAudioClassificationModel();
    console.log("Modelo de clasificación cargado");

    // Cargar el Worker
    worker.postMessage({ type: "load" });
  } catch (error) {
    console.error("Error en la inicialización del flujo principal:", error);
  }
}

// Ejecutar la función principal
(async function initApp() {
  try {
    await main();
  } catch (error) {
    console.error("Error en la inicialización de la aplicación:", error);
  }
})();

// ----------------------------------------------------------- GENERACIÓN DE TEXTO, TRADUCCIÓN Y SÍNTESIS DE VOZ -----------------------------------------------------------

let globalMessage = "";

function speakTextSequentially(messages) {
  return messages.reduce((promiseChain, messageObj) => {
    return promiseChain.then(() => speakText(messageObj.text, messageObj.lang));
  }, Promise.resolve());
}

function speakText(message, lang = "es") {
  return new Promise((resolve, reject) => {
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.lang = lang;

    utterance.onend = () => {
      resolve();
    };

    utterance.onerror = (e) => {
      console.error("Error en SpeechSynthesis:", e);
      resolve(); // Resolver para continuar con el siguiente mensaje incluso si hay un error
    };

    window.speechSynthesis.speak(utterance);
  });
}

async function translateMessage(message, targetLanguage) {
  // Codificar correctamente el mensaje para evitar problemas con caracteres especiales
  const encodedMessage = encodeURIComponent(message);
  // Construir la URL correctamente
  const url = `https://api.mymemory.translated.net/get?q=${encodedMessage}&langpair=en|${targetLanguage}`;

  // Realizar la solicitud GET a la API
  const response = await fetch(url);
  const data = await response.json();

  // Verificar si la traducción se obtuvo correctamente
  if (data.responseData && data.responseData.translatedText) {
    return data.responseData.translatedText;
  } else {
    console.error("Error en la traducción:", data);
    return message; // Retorna el mensaje original si ocurre un error
  }
}
