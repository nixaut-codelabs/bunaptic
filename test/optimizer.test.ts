import { describe, expect, test } from "bun:test";

import { applyOptimizer, createOptimizerState, optimizerConfig } from "../src/core/optimizer.js";
import { Architect, Network, packDataset } from "../src/index.js";

const xor = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

describe("optimizers", () => {
  test("applies SGD, RMSProp, and Adam finite updates", () => {
    for (const optimizer of ["sgd", "rmsprop", "adam"] as const) {
      const values = new Float32Array([1, -1]);
      const gradients = new Float32Array([0.5, -0.25]);
      const config = optimizerConfig({ optimizer, rate: 0.01 }, 0.01);
      applyOptimizer(values, gradients, createOptimizerState(values.length, optimizer), config, "minimize");
      expect(values.every((value) => Number.isFinite(value))).toBe(true);
      expect(Array.from(values)).not.toEqual([1, -1]);
    }
  });

  test("feed-forward Adam training lowers error", () => {
    const dataset = packDataset(xor, 2, 1);
    const network = Architect.Perceptron(2, 6, 1);
    const before = network.testPacked(dataset).error;
    const result = network.train(dataset, { optimizer: "adam", iterations: 80, error: -1, rate: 0.03 });
    expect(result.error).toBeLessThan(before);
  });

  test("WASM feed-forward training falls back for Adam and lowers error", async () => {
    const { loadWasmKernel } = await import("../src/index.js");
    const kernel = await loadWasmKernel();
    const dataset = packDataset(xor, 2, 1);
    const network = Architect.Perceptron(2, 6, 1).useWasm(kernel);
    const before = network.testPacked(dataset).error;
    const result = network.train(dataset, { optimizer: "adam", iterations: 80, error: -1, rate: 0.03 });
    expect(result.error).toBeLessThan(before);
  });

  test("recurrent Adam training lowers error", () => {
    const set = [
      { input: new Float32Array([0, 0, 1, 1]), length: 2, output: [1] },
      { input: new Float32Array([1, 0, 0, 0]), length: 2, output: [0] },
      { input: new Float32Array([1, 1, 1, 0]), length: 2, output: [1] },
      { input: new Float32Array([0, 0, 0, 0]), length: 2, output: [0] },
    ];
    const network = Network.fromJSON(Architect.GRU(2, 5, 1, { seed: 14 }).toJSON());
    const before = network.testSequences(set).error;
    const result = network.trainSequences(set, { optimizer: "adam", iterations: 30, rate: 0.01, clip: 1 });
    expect(result.error).toBeLessThan(before);
  });
});
