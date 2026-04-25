import { describe, expect, test } from "bun:test";

import { Architect, evaluatePopulation, loadWasmKernel, packDataset } from "../src/index.js";
import { TOLERANCE, expectCloseArray } from "./helpers/numeric.js";

const xor = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

describe("evaluatePopulation", () => {
  test("uses grouped WASM scoring for same-topology populations", async () => {
    const kernel = await loadWasmKernel();
    const dataset = packDataset(xor, 2, 1);
    const population = Array.from({ length: 16 }, (_, index) => {
      const network = Architect.Perceptron(2, 5, 1);
      const parameters = network.getParameters();
      parameters.weights[0] = (parameters.weights[0] as number) + index * 0.01;
      return network.setParameters(parameters);
    });

    const grouped = await evaluatePopulation(population, dataset, { mode: "direct", useWasm: true, kernel });
    const expected = population.map((network) => -kernel.evaluatePackedDataset(network.compiledGraph(), dataset));

    expectCloseArray(grouped, expected, TOLERANCE.wasm);
  });

  test("preserves result order for mixed topologies", async () => {
    const kernel = await loadWasmKernel();
    const dataset = packDataset(xor, 2, 1);
    const population = [
      Architect.Perceptron(2, 3, 1),
      Architect.Perceptron(2, 5, 1),
      Architect.Perceptron(2, 3, 1),
      Architect.Perceptron(2, 4, 1),
    ];

    const scores = await evaluatePopulation(population, dataset, { mode: "direct", useWasm: true, kernel });
    const expected = population.map((network) => -kernel.evaluatePackedDataset(network.compiledGraph(), dataset));

    expectCloseArray(scores, expected, TOLERANCE.wasm);
  });

  test("uses grouped WASM scoring through workers", async () => {
    const kernel = await loadWasmKernel();
    const dataset = packDataset(xor, 2, 1);
    const population = Array.from({ length: 12 }, (_, index) => {
      const network = Architect.Perceptron(2, 4, 1);
      const parameters = network.getParameters();
      parameters.biases[2] = (parameters.biases[2] as number) + index * 0.01;
      return network.setParameters(parameters);
    });

    const scores = await evaluatePopulation(population, dataset, { mode: "workers", workers: 2, useWasm: true });
    const expected = population.map((network) => -kernel.evaluatePackedDataset(network.compiledGraph(), dataset));

    expectCloseArray(scores, expected, TOLERANCE.wasm);
  });

  test("does not collapse worker model cache for tiny parameter differences", async () => {
    const kernel = await loadWasmKernel();
    const dataset = packDataset(xor, 2, 1);
    const first = Architect.Perceptron(2, 4, 1);
    const second = Architect.Perceptron(2, 4, 1);
    const parameters = second.getParameters();
    parameters.weights[0] = (parameters.weights[0] as number) + 0.0000004;
    second.setParameters(parameters);
    const population = [first, second];

    const scores = await evaluatePopulation(population, dataset, { mode: "workers", workers: 1, useWasm: true });
    const expected = population.map((network) => -kernel.evaluatePackedDataset(network.compiledGraph(), dataset));

    expectCloseArray(scores, expected, TOLERANCE.wasm);
  });
});
