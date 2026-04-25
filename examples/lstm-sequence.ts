import { Architect, Network, loadWasmKernel } from "../src/index.js";

const kernel = await loadWasmKernel();
const model = Architect.LSTM(2, 8, 1, { seed: 1 });
const fast = Network.fromJSON(model.toJSON()).useWasm(kernel);
const sequence = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]);

console.log("ts", model.runSequence(sequence).map((output) => Array.from(output)));
console.log("wasm", fast.runSequence(sequence).map((output) => Array.from(output)));
