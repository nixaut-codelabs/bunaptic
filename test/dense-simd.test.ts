import { describe, expect, test } from "bun:test";

import { Architect, MutationMethod, Network, denseEvaluateDataset, denseForward, deriveDensePlan, evaluatePopulation, loadWasmKernel, packDataset } from "../src/index.js";
import { TOLERANCE, expectClose, expectCloseArray } from "./helpers/numeric.js";

const xor = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

describe("dense specialized path", () => {
  test("derives dense plans for perceptrons and matches graph output", () => {
    const network = Architect.Perceptron(2, 4, 3, 1);
    const plan = deriveDensePlan(network.compiledGraph());
    expect(plan).not.toBeNull();
    const input = new Float32Array([0.25, 0.75]);
    expectCloseArray(denseForward(plan!, input), network.engine().activate(input), TOLERANCE.scalar);
  });

  test("keeps dense dataset evaluation in parity with graph evaluation", () => {
    const network = Architect.Perceptron(2, 5, 1);
    const dataset = packDataset(xor, 2, 1);
    const plan = deriveDensePlan(network.compiledGraph())!;
    expect(denseEvaluateDataset(plan, dataset)).toBeCloseTo(network.engine().test(xor).error, 6);
  });

  test("keeps optimized dense dataset evaluation repeatable", () => {
    const network = Architect.Perceptron(2, 5, 3, 1);
    const dataset = packDataset([...xor, ...xor, ...xor], 2, 1);
    const plan = deriveDensePlan(network.compiledGraph())!;
    const first = denseEvaluateDataset(plan, dataset);
    const second = denseEvaluateDataset(plan, dataset);
    expectClose(first, second, TOLERANCE.scalar);
    expectClose(first, network.engine().test(Array.from({ length: dataset.sampleCount }, (_, index) => ({
      input: dataset.inputs.subarray(index * dataset.inputSize, (index + 1) * dataset.inputSize),
      output: dataset.targets.subarray(index * dataset.outputSize, (index + 1) * dataset.outputSize),
    }))).error, TOLERANCE.scalar);
  });

  test("falls back after topology mutation invalidates dense eligibility", () => {
    const network = Architect.Perceptron(2, 4, 1);
    expect(deriveDensePlan(network.compiledGraph())).not.toBeNull();
    network.mutate(MutationMethod.AddNode);
    const output = network.activate([0.2, 0.8]);
    expect(output.length).toBe(1);
    expect(Array.from(output).every(Number.isFinite)).toBe(true);
  });

  test("runs dense WASM plan in parity with TS dense plan", async () => {
    const scalar = await loadWasmKernel({ simd: false });
    const simd = await loadWasmKernel({ simd: "auto" });
    const network = Architect.Perceptron(2, 6, 3, 1);
    const plan = deriveDensePlan(network.compiledGraph())!;
    const input = new Float32Array([0.4, 0.6]);
    const expected = denseForward(plan, input);
    expectCloseArray(scalar.denseForwardPlan(plan, input), expected, TOLERANCE.wasm);
    expectCloseArray(simd.denseForwardPlan(plan, input), expected, TOLERANCE.simd);
  });
});

describe("SIMD kernel parity", () => {
  test("reports SIMD capability for auto-loaded SIMD artifact when available", async () => {
    const scalar = await loadWasmKernel({ simd: false });
    const auto = await loadWasmKernel({ simd: "auto" });
    expect(scalar.simdEnabled).toBe(false);
    expect(typeof auto.simdEnabled).toBe("boolean");
  });

  test("keeps scalar and SIMD dataset scoring in parity", async () => {
    const scalar = await loadWasmKernel({ simd: false });
    const simd = await loadWasmKernel({ simd: "auto" });
    const dataset = packDataset(xor, 2, 1);
    const network = Architect.Perceptron(2, 8, 1);
    const graph = network.compiledGraph();
    expectCloseArray([simd.evaluatePackedDataset(graph, dataset)], [scalar.evaluatePackedDataset(graph, dataset)], TOLERANCE.simd);
  });

  test("keeps population V2 scoring in parity with V1 and preserves order", async () => {
    const kernel = await loadWasmKernel({ simd: "auto" });
    const dataset = packDataset(xor, 2, 1);
    const population = Array.from({ length: 10 }, (_, index) => {
      const network = Architect.Perceptron(2, 5, 1);
      const params = network.getParameters();
      params.weights[0] = (params.weights[0] as number) + index * 0.02;
      return network.setParameters(params);
    });
    const handle = kernel.createDatasetHandle(dataset);
    try {
      const graphs = population.map((network) => network.compiledGraph());
      const v1 = kernel.evaluateSameTopologyPopulationHandle(graphs, handle);
      const v2 = kernel.evaluateSameTopologyPopulationV2Handle(graphs, handle);
      const facade = await evaluatePopulation(population, dataset, { mode: "direct", useWasm: true, kernel });
      expectCloseArray(v2, v1, TOLERANCE.simd);
      expectCloseArray(facade, v1, TOLERANCE.simd);
    } finally {
      kernel.disposeDatasetHandle(handle);
    }
  });

  test("keeps population V2 remainder sizes in parity with V1", async () => {
    const kernel = await loadWasmKernel({ simd: "auto" });
    const dataset = packDataset(xor, 2, 1);
    for (const size of [1, 2, 3, 5, 6, 7, 9]) {
      const population = Array.from({ length: size }, (_, index) => {
        const network = Architect.Perceptron(2, 4, 1);
        const params = network.getParameters();
        params.weights[0] = (params.weights[0] as number) + index * 0.03125;
        params.biases[2] = (params.biases[2] as number) - index * 0.0125;
        return network.setParameters(params);
      });
      const handle = kernel.createDatasetHandle(dataset);
      try {
        const graphs = population.map((network) => network.compiledGraph());
        expectCloseArray(kernel.evaluateSameTopologyPopulationV2Handle(graphs, handle), kernel.evaluateSameTopologyPopulationHandle(graphs, handle), TOLERANCE.simd);
      } finally {
        kernel.disposeDatasetHandle(handle);
      }
    }
  });

  test("uses dense WASM path through Network.useWasm", async () => {
    const kernel = await loadWasmKernel({ simd: "auto" });
    const original = Architect.Perceptron(2, 5, 1);
    const wasm = Network.fromJSON(original.toJSON()).useWasm(kernel);
    expectCloseArray(Array.from(wasm.activate([0.25, 0.75])), Array.from(original.activate([0.25, 0.75])), TOLERANCE.simd);
  });
});
