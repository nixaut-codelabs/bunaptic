import { Architect, Network, loadWasmKernel, packSequenceDataset } from "../src/index.js";

const set = [
  { input: new Float32Array([0, 0, 1, 1]), length: 2, output: [1] },
  { input: new Float32Array([1, 0, 0, 0]), length: 2, output: [0] },
  { input: new Float32Array([1, 1, 1, 0]), length: 2, output: [1] },
  { input: new Float32Array([0, 0, 0, 0]), length: 2, output: [0] },
];

const model = Architect.GRU(2, 5, 1, { seed: 14 });
const before = model.testSequences(set).error;
const result = model.trainSequences(set, { iterations: 100, rate: 0.05, clip: 1, truncatedSteps: 2 });
const kernel = await loadWasmKernel({ simd: "auto" });
const packed = packSequenceDataset(set, 2, 1);
const wasmModel = Network.fromJSON(model.toJSON()).useWasm(kernel);
const wasmBefore = wasmModel.testSequences(packed).error;
const wasmResult = wasmModel.trainSequences(packed, { optimizer: "sgd", iterations: 20, rate: 0.02, clip: 1 });

console.log({ before, after: result.error, iterations: result.iterations, wasmBefore, wasmAfter: wasmResult.error, wasmIterations: wasmResult.iterations });
