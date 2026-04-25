import { Architect, Network, WorkerPool, deriveDensePlan, evaluatePopulation, loadWasmKernel, packDataset } from "../src/index.js";
import { benchmarkProfile, measure, outputMetadata, printBenchmark, sumValues } from "./harness.js";

const profile = benchmarkProfile();
const sampleCount = Math.max(16, Math.floor(128 * profile.scale));
const populationSize = Math.max(32, Math.floor(500 * profile.scale));
const set = Array.from({ length: sampleCount }, (_, index) => ({
  input: [index / sampleCount, (sampleCount - index) / sampleCount],
  output: [index % 2],
}));

const population = Array.from({ length: populationSize }, () => Architect.Perceptron(2, 8, 1));
const packed = packDataset(set, 2, 1);
console.log(JSON.stringify(outputMetadata({ population: population.length, samples: set.length })));
const scalar = await loadWasmKernel({ simd: false });
const simd = await loadWasmKernel({ simd: "auto" });
console.log(JSON.stringify({ scalarSimd: scalar.simdEnabled, autoSimd: simd.simdEnabled }));

printBenchmark("POPULATION", [
  await measure("single-thread", () => {
    let score = 0;
    for (const network of population) score -= network.testPacked(packed).error;
    return score;
  }, profile),
  await measure("workers", async () => {
    const pool = new WorkerPool({ size: 2 });
    try {
      return sumValues(await pool.evaluate(population, packed));
    } finally {
      pool.terminate();
    }
  }, profile),
  await measure("direct WASM scalar grouped V2", async () => sumValues(await evaluatePopulation(population, packed, { mode: "direct", useWasm: true, kernel: scalar })), profile),
  await measure("direct WASM SIMD grouped V2", async () => sumValues(await evaluatePopulation(population, packed, { mode: "direct", useWasm: true, kernel: simd })), profile),
  await measure("evaluatePopulation workers+WASM", async () => sumValues(await evaluatePopulation(population, packed, { mode: "workers", workers: 2, useWasm: true })), profile),
]);

const sequenceModel = Architect.RNN(1, 16, 1);
const short = new Float32Array(64).fill(0.5);
const long = new Float32Array(4096).fill(0.5);
printBenchmark("DYNAMIC SEQUENCE", [
  await measure("short", () => sequenceModel.runSequence(short).length, profile),
  await measure("long", () => sequenceModel.runSequence(long).length, profile),
]);

try {
  const tsModel = Architect.Perceptron(2, 16, 1);
  const wasmModel = Network.fromJSON(tsModel.toJSON()).useWasm(simd);
  const densePlan = deriveDensePlan(tsModel.compiledGraph())!;
  const input = new Float32Array([0.2, 0.8]);
  const packedInputs = new Float32Array(10_000 * 2);
  for (let i = 0; i < packedInputs.length; i += 2) {
    packedInputs[i] = 0.2;
    packedInputs[i + 1] = 0.8;
  }
  printBenchmark("WASM VS TS", [
    await measure("WASM loop", () => {
      let checksum = 0;
      for (let i = 0; i < 10_000; i++) checksum += wasmModel.activate(input)[0] as number;
      return checksum;
    }, profile),
    await measure("WASM graph batch SIMD", () => sumValues(simd.graphForwardBatch(tsModel.compiledGraph(), packedInputs)), profile),
    await measure("WASM dense batch SIMD", () => sumValues(simd.denseForwardBatchPlan(densePlan, packedInputs)), profile),
    await measure("TS loop", () => {
      let checksum = 0;
      for (let i = 0; i < 10_000; i++) checksum += tsModel.activate(input)[0] as number;
      return checksum;
    }, profile),
    await measure("WASM graph dataset scalar", () => scalar.evaluatePackedDataset(tsModel.compiledGraph(), packed), profile),
    await measure("WASM graph dataset SIMD", () => simd.evaluatePackedDataset(tsModel.compiledGraph(), packed), profile),
    await measure("WASM dense dataset SIMD", () => simd.denseEvaluatePackedDataset(densePlan, packed), profile),
    await measure("WASM dataset handle", () => {
      const handle = simd.createDatasetHandle(packed);
      try {
        let checksum = 0;
        for (let i = 0; i < 100; i++) checksum += simd.evaluatePackedDatasetHandle(tsModel.compiledGraph(), handle);
        return checksum;
      } finally {
        simd.disposeDatasetHandle(handle);
      }
    }, profile),
  ]);
} catch (error) {
  console.log("wasm-vs-ts skipped", error instanceof Error ? error.message : String(error));
}
