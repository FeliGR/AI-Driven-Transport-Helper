const CLASSES = ["crying_baby", "clock_alarm", "toilet_flush", "water_drops"];

const YAMNET_MODEL_URL = "https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1";
const MODEL_SAMPLE_RATE = 16000;
const NUM_SECONDS = 5;

let model;
let yamnet;
let audioContext;
let stream;

let isProcessing = false;
const alertQueue = [];
const alertHistory = [];
const MAX_HISTORY = 5;

const timeDataQueue = [];

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

    console.log(predictionArray)
    console.log(predictedIndex)
    console.log(confidence)

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

// Elementos de la UI
const btnMicStart = document.querySelector("#btnMicStart");
const btnMicStop = document.querySelector("#btnMicStop");
const predictionClass = document.querySelector("#predictionClass");
const outputMessageEl = document.getElementById("outputMessage");
const alertHistoryEl = document.getElementById("alertHistory");

// Manejar mensajes del Worker
worker.onmessage = async (e) => {
  switch (e.data.type) {
    case "token":
      outputMessageEl.textContent += e.data.token;
      break;

    case "ready":
      console.log("Worker está listo");
      break;

    case "done":
      console.log("Generación de texto completada");
      // Agregar el alert actual al historial
      if (alertQueue.length > 0) {
        const alert = alertQueue.shift();
        alert.message = outputMessageEl.textContent;
        alertHistory.unshift(alert);
        if (alertHistory.length > MAX_HISTORY) {
          alertHistory.pop();
        }
        updateAlertHistory();
      }
      // Resetear estado
      isProcessing = false;
      predictionClass.textContent = "Waiting for input...";
      outputMessageEl.textContent = "";

      // Procesar siguiente alerta si hay
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

function updateAlertHistory() {
  alertHistoryEl.innerHTML = `
    <h3>Recent Alerts</h3>
    ${alertHistory
      .map(
        (alert) => `
    <div class="alert-item">
      <div class="alert-time">${alert.timestamp.toLocaleTimeString()}</div>
      <div class="alert-type">${alert.prediction}</div>
      <div class="alert-message">${alert.message}</div>
    </div>
  `
      )
      .join("")}
  `;
}

function processNextAlert() {
  if (isProcessing || alertQueue.length === 0) return;

  isProcessing = true;
  const alert = alertQueue[0];

  predictionClass.textContent = `Processing: ${alert.prediction}`;
  outputMessageEl.textContent = "";

  // Generar prompt y enviar al Worker para generar el mensaje
  const prompt = generatePrompt(alert.prediction, "Train car 4", "Urgent");

  worker.postMessage({ type: "generate", prompt: prompt });
}

// Función para detener la grabación
function stopRecording() {
  try {
    if (audioContext && stream) {
      audioContext.close();
      stream.getTracks().forEach((track) => track.stop());
    }
    timeDataQueue.splice(0);

    btnMicStart.disabled = false;
    btnMicStop.disabled = true;

    predictionClass.textContent = "Stopped";
    outputMessageEl.textContent = "";

    console.log("Grabación detenida.");
  } catch (error) {
    console.error("Error al detener la grabación:", error);
  }
}

btnMicStart.onclick = async () => {
  try {
    // Iniciar el micrófono y procesar audio
    stream = await getAudioStream();
    audioContext = new AudioContext({ sampleRate: MODEL_SAMPLE_RATE });
    const source = audioContext.createMediaStreamSource(stream);

    await audioContext.audioWorklet.addModule("recorder.worklet.js");
    const recorder = new AudioWorkletNode(audioContext, "recorder.worklet");
    source.connect(recorder);

    recorder.port.onmessage = async (e) => {
      try {
        const inputBuffer = e.data;
        if (!inputBuffer || inputBuffer.length === 0) return;

        timeDataQueue.push(...inputBuffer);

        if (timeDataQueue.length >= MODEL_SAMPLE_RATE * NUM_SECONDS) {
          const audioData = new Float32Array(
            timeDataQueue.splice(0, MODEL_SAMPLE_RATE * NUM_SECONDS)
          );
          const { predictedIndex, confidence } = await predict(
            yamnet,
            model,
            audioData
          );
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
        }
      } catch (error) {
        console.error("Error processing audio:", error);
      }
    };

    btnMicStart.disabled = true;
    btnMicStop.disabled = false;
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
