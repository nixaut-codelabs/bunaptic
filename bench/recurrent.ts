import { Architect, Network, loadWasmKernel, packSequenceDataset } from "../src/index.js";
import { benchmarkProfile, measure, outputMetadata, printBenchmark, sumNestedOutputs } from "./harness.js";

const profileConfig = benchmarkProfile();
const kernel = await loadWasmKernel({ simd: "auto" });
console.log(JSON.stringify(outputMetadata({ simd: kernel.simdEnabled })));
const profiles = [
  { name: "short", steps: Math.max(8, Math.floor(32 * profileConfig.scale)) },
  { name: "medium", steps: Math.max(64, Math.floor(512 * profileConfig.scale)) },
  { name: "long", steps: Math.max(256, Math.floor(4096 * profileConfig.scale)) },
];

for (const profile of profiles) {
  const values = sequence(profile.steps, 8);
  console.log(`\n${profile.name.toUpperCase()} ${profile.steps} steps`);
  for (const kind of ["LSTM", "GRU"] as const) {
    const tsModel = kind === "LSTM" ? Architect.LSTM(8, 32, 4, { seed: 42 }) : Architect.GRU(8, 32, 4, { seed: 42 });
    const wasmModel = Network.fromJSON(tsModel.toJSON()).useWasm(kernel);
    const handle = kernel.createRecurrentHandle(tsModel.recurrentSpec!);
    printBenchmark(kind, [
      await measure("TS reference", () => sumNestedOutputs(tsModel.runSequence(values)), profileConfig),
      await measure("WASM fused", () => sumNestedOutputs(wasmModel.runSequence(values)), profileConfig),
      await measure("WASM recurrent handle", () => sumNestedOutputs(kernel.recurrentForwardSequenceHandle(handle, values)), profileConfig),
    ]);
    kernel.disposeRecurrentHandle(handle);
  }
}

const ragged = {
  values: sequence(8 + 32 + 128 + 512 + 2048, 8),
  offsets: offsets([8, 32, 128, 512, 2048], 8),
  featureSize: 8,
};

console.log("\nRAGGED");
for (const kind of ["LSTM", "GRU"] as const) {
  const tsModel = kind === "LSTM" ? Architect.LSTM(8, 32, 4, { seed: 7 }) : Architect.GRU(8, 32, 4, { seed: 7 });
  const wasmModel = Network.fromJSON(tsModel.toJSON()).useWasm(kernel);
  const handle = kernel.createRecurrentHandle(tsModel.recurrentSpec!);
  printBenchmark(kind, [
    await measure("TS ragged", () => sumNestedOutputs(tsModel.runBatchRagged(ragged)), profileConfig),
    await measure("WASM ragged", () => sumNestedOutputs(wasmModel.runBatchRagged(ragged)), profileConfig),
    await measure("WASM ragged handle", () => sumNestedOutputs(kernel.recurrentForwardRaggedHandle(handle, ragged)), profileConfig),
  ]);
  kernel.disposeRecurrentHandle(handle);
}

const trainSet = packSequenceDataset([
  { input: sequence(4, 2), length: 4, output: [1] },
  { input: sequence(4, 2).map((value) => 1 - value), length: 4, output: [0] },
], 2, 1);
printBenchmark("RECURRENT TRAIN", [
  await measure("GRU TS train", () => {
    const model = Architect.GRU(2, 8, 1, { seed: 9 });
    return model.trainSequences(trainSet, { optimizer: "sgd", iterations: 4, rate: 0.01 }).error;
  }, profileConfig),
  await measure("GRU WASM train", () => {
    const model = Architect.GRU(2, 8, 1, { seed: 9 }).useWasm(kernel);
    return model.trainSequences(trainSet, { optimizer: "sgd", iterations: 4, rate: 0.01 }).error;
  }, profileConfig),
]);

function sequence(steps: number, features: number): Float32Array {
  const values = new Float32Array(steps * features);
  for (let i = 0; i < values.length; i++) values[i] = Math.sin(i / 13);
  return values;
}

function offsets(lengths: number[], features: number): Uint32Array {
  const offsets = new Uint32Array(lengths.length + 1);
  let cursor = 0;
  for (let i = 0; i < lengths.length; i++) {
    cursor += (lengths[i] as number) * features;
    offsets[i + 1] = cursor;
  }
  return offsets;
}
