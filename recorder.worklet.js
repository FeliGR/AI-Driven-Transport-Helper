class RecorderWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input.length > 0) {
      const channelData = input[0];
      const buffer = new Float32Array(channelData);
      this.port.postMessage(buffer);
    }
    return true;
  }
}

registerProcessor("recorder.worklet", RecorderWorklet);
