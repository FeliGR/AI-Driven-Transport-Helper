import {
  TextStreamer,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.2";

const TASK_NAME = "text-generation";
const MODEL_NAME = "onnx-community/Qwen2.5-0.5B-Instruct";

let generator = null;
let streamer = null;
let stopGeneration = false;

self.onmessage = async (e) => {
  switch (e.data.type) {
    case "load":
      await load();
      break;
    case "generate":
      stopGeneration = false;
      await generate(e.data.prompt);
      break;
    default:
      console.warn("Tipo de mensaje desconocido:", e.data.type);
  }
};

async function load() {
  try {
    console.log("Cargando modelo de generación de texto...");
    generator = await pipeline(TASK_NAME, MODEL_NAME, {
      dtype: "fp16",
      device: "wasm",
    });

    streamer = new TextStreamer(generator.tokenizer, {
      skip_prompt: true,
      callback_function,
    });

    self.postMessage({ type: "ready" });
  } catch (error) {
    console.error("Error al cargar el modelo en el Worker:", error);
    self.postMessage({ type: "error", message: error.message });
  }
}

async function generate(prompt) {
  try {
    await generator(prompt, {
      max_new_tokens: 500,
      temperature: 0,
      top_p: 0,
      do_sample: false,
      early_stopping: true,
      streamer,
    });

    if (!stopGeneration) {
      self.postMessage({ type: "done" });
    }
  } catch (error) {
    console.error("Error al generar texto en el Worker:", error);
    self.postMessage({ type: "error", message: error.message });
  }
}

function callback_function(token) {
  if (token.trim() === "" || token.trim() === "<|im_end|>") {
    if (token.trim() === "<|im_end|>") {
      stopGeneration = true;
      self.postMessage({ type: "done" });
    }
    return;
  }
  self.postMessage({ type: "token", token });
}
