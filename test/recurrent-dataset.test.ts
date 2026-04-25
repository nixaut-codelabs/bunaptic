import { describe, expect, test } from "bun:test";

import { Architect, Network, loadWasmKernel, packSequenceDataset, packedSequenceDatasetKey } from "../src/index.js";

const set = [
  { input: new Float32Array([0, 0, 1, 1]), length: 2, output: [1] },
  { input: new Float32Array([1, 0, 0, 0]), length: 2, output: [0] },
  { input: new Float32Array([1, 1, 1, 0]), length: 2, output: [1] },
  { input: new Float32Array([0, 0, 0, 0]), length: 2, output: [0] },
];

describe("packed sequence datasets", () => {
  test("packs final-output sequence datasets with stable keys", () => {
    const first = packSequenceDataset(set, 2, 1);
    const second = packSequenceDataset(set, 2, 1);
    expect(first.sampleCount).toBe(4);
    expect(first.values.length).toBe(16);
    expect(first.targets.length).toBe(4);
    expect(first.targetMode).toBe("final");
    expect(packedSequenceDatasetKey(first)).toBe(packedSequenceDatasetKey(second));
  });

  test("supports trainSequences and testSequences from packed final-output datasets", () => {
    const packed = packSequenceDataset(set, 2, 1);
    const network = Architect.GRU(2, 5, 1, { seed: 14 });
    const before = network.testSequences(packed).error;
    const result = network.trainSequences(packed, { optimizer: "adam", iterations: 30, rate: 0.01, clip: 1 });
    expect(result.error).toBeLessThan(before);
  });

  test("supports packed per-step targets for trainSequences and testSequences", () => {
    const packed = packSequenceDataset([
      { input: new Float32Array([0, 1, 1, 0]), length: 2, output: [0, 1] },
      { input: new Float32Array([1, 0, 0, 1]), length: 2, output: [1, 0] },
    ], 2, 1);
    const network = Architect.GRU(2, 3, 1, { seed: 2 });
    const before = network.testSequences(packed).error;
    const result = network.trainSequences(packed, { iterations: 20, rate: 0.03, clip: 1 });
    expect(packed.targetMode).toBe("perStep");
    expect(result.error).toBeLessThan(before);
  });

  test("trains recurrent sequences through WASM SGD path", async () => {
    const kernel = await loadWasmKernel({ simd: "auto" });
    const packed = packSequenceDataset(set, 2, 1);
    const ts = Architect.GRU(2, 4, 1, { seed: 5 });
    const wasm = Network.fromJSON(ts.toJSON()).useWasm(kernel);
    const before = wasm.testSequences(packed).error;
    const result = wasm.trainSequences(packed, { optimizer: "sgd", iterations: 10, rate: 0.02, clip: 1 });
    expect(result.iterations).toBe(10);
    expect(wasm.testSequences(packed).error).toBeLessThan(before);
  });
});
