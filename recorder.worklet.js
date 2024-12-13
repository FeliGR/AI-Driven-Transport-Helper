// recorder.worklet.js

class RecorderWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input.length > 0) {
      const channelData = input[0];
      // Copiar los datos del canal para evitar referencias compartidas
      const buffer = new Float32Array(channelData);
      // Enviar los datos al hilo principal
      this.port.postMessage(buffer);
    }
    return true; // Mantener el worklet activo
  }
}

registerProcessor('recorder.worklet', RecorderWorklet);
