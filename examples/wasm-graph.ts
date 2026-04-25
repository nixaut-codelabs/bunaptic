import { Architect, Network, loadWasmKernel } from "../src/index.js";

const kernel = await loadWasmKernel();
const tsModel = Architect.Perceptron(2, 8, 1);
const wasmModel = Network.fromJSON(tsModel.toJSON()).useWasm(kernel);
const input = new Float32Array([0.25, 0.75]);

console.log("ts", Array.from(tsModel.activate(input)));
console.log("wasm", Array.from(wasmModel.activate(input)));

const batch = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]);
console.log("wasm batch", Array.from(kernel.graphForwardBatch(tsModel.compiledGraph(), batch)));
