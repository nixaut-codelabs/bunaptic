import { Architect, Network, loadWasmKernel } from "../src/index.js";

const kernel = await loadWasmKernel();
const model = Architect.GRU(2, 8, 1, { seed: 1 });
const fast = Network.fromJSON(model.toJSON()).useWasm(kernel);
const batch = {
  values: new Float32Array([0, 0, 0, 1, 1, 0, 1, 1, 0, 1]),
  offsets: new Uint32Array([0, 4, 10]),
  featureSize: 2,
};

console.log("ts ragged", model.runBatchRagged(batch).map((output) => Array.from(output)));
console.log("wasm ragged", fast.runBatchRagged(batch).map((output) => Array.from(output)));
