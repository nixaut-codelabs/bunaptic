import { describe, expect, test } from "bun:test";

import { Architect, WorkerPool, evaluatePopulation, loadWasmKernel, packDataset } from "../src/index.js";
import { TOLERANCE, expectCloseArray } from "./helpers/numeric.js";

const xor = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

describe("worker long-run scoring", () => {
  test("keeps cached worker WASM scores stable across repeated evaluations", async () => {
    const dataset = packDataset(Array.from({ length: 64 }, (_, index) => xor[index % xor.length]!), 2, 1);
    const population = Array.from({ length: 24 }, (_, index) => {
      const network = Architect.Perceptron(2, 5, 1);
      const parameters = network.getParameters();
      parameters.weights[0] = (parameters.weights[0] as number) + index * 0.001;
      return network.setParameters(parameters);
    });
    const kernel = await loadWasmKernel();
    const expected = await evaluatePopulation(population, dataset, { mode: "direct", useWasm: true, kernel });
    const pool = new WorkerPool({ size: 2, useWasm: true });

    try {
      await pool.cacheDataset(dataset);
      for (let i = 0; i < 20; i++) {
        const scores = await pool.evaluate(population, dataset);
        expectCloseArray(scores, expected, TOLERANCE.wasm);
      }
      pool.clearWorkerCache();
      const afterClear = await pool.evaluate(population, dataset);
      expectCloseArray(afterClear, expected, TOLERANCE.wasm);
    } finally {
      pool.terminate();
    }
  });

  test("runs repeated evaluatePopulation worker calls without score drift", async () => {
    const dataset = packDataset(xor, 2, 1);
    const population = Array.from({ length: 16 }, () => Architect.Perceptron(2, 4, 1));
    const first = await evaluatePopulation(population, dataset, { mode: "workers", workers: 2, useWasm: true });
    for (let i = 0; i < 8; i++) {
      const next = await evaluatePopulation(population, dataset, { mode: "workers", workers: 2, useWasm: true });
      expectCloseArray(next, first, TOLERANCE.wasm);
    }
  });
});
