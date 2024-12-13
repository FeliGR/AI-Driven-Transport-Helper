const CLASSES = ["crying_baby", "clock_alarm", "toilet_flush", "water_drops"];

const YAMNET_MODEL_URL = "https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1";
const MODEL_SAMPLE_RATE = 16000;
const NUM_SECONDS = 5;

let model;
let yamnet;

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
    const argMaxTensor = meanTensor.argMax(0);

    embeddings.dispose();
    results.dispose();
    meanTensor.dispose();

    return argMaxTensor.dataSync()[0];
  } catch (error) {
    console.error("Error durante la predicción:", error);
    throw new Error("No se pudo realizar la predicción.");
  }
}

async function getEmbeddingsFromTimeDomainData(yamnet, audioData) {
  try {
    const waveformTensor = tf.tensor(audioData);
    const [, embeddings] = yamnet.predict(waveformTensor);
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

async function main() {
  try {
    yamnet = await loadYamnetModel();
    console.log("Modelo YamNet cargado");

    model = await loadCustomAudioClassificationModel();
    console.log("Modelo de clasificación cargado");

    let audioContext;
    let stream;

    const btnMicStart = document.querySelector("#btnMicStart");
    const btnMicStop = document.querySelector("#btnMicStop");
    const predictionClass = document.querySelector("#predictionClass");

    const timeDataQueue = [];

    btnMicStart.onclick = async () => {
      try {
        stream = await getAudioStream();
        audioContext = new AudioContext({ sampleRate: MODEL_SAMPLE_RATE });
        const source = audioContext.createMediaStreamSource(stream);

        await audioContext.audioWorklet.addModule("recorder.worklet.js");
        const recorder = new AudioWorkletNode(audioContext, "recorder.worklet");
        source.connect(recorder);

        recorder.port.onmessage = async (e) => {
          try {
            const inputBuffer = Array.from(e.data);
            if (inputBuffer[0] === 0) return;

            timeDataQueue.push(...inputBuffer);

            if (timeDataQueue.length >= MODEL_SAMPLE_RATE * NUM_SECONDS) {
              const audioData = new Float32Array(
                timeDataQueue.splice(0, MODEL_SAMPLE_RATE * NUM_SECONDS)
              );
              const classIndex = await predict(yamnet, model, audioData);
              predictionClass.textContent = CLASSES[classIndex];
            }
          } catch (error) {
            console.error("Error al procesar el audio:", error);
          }
        };

        btnMicStart.disabled = true;
        btnMicStop.disabled = false;
      } catch (error) {
        console.error("Error al iniciar el micrófono:", error);
      }
    };

    btnMicStop.onclick = () => {
      try {
        if (audioContext && stream) {
          audioContext.close();
          stream.getTracks().forEach((track) => track.stop());
        }
        timeDataQueue.splice(0);

        btnMicStart.disabled = false;
        btnMicStop.disabled = true;
      } catch (error) {
        console.error("Error al detener el micrófono:", error);
      }
    };
  } catch (error) {
    console.error("Error en la inicialización del flujo principal:", error);
  }
}

(async function initApp() {
  try {
    await main();
  } catch (error) {
    console.error("Error en la inicialización de la aplicación:", error);
  }
})();
