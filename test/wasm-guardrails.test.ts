import { describe, expect, test } from "bun:test";

import { Architect, Network, loadWasmKernel, packDataset } from "../src/index.js";
import { TOLERANCE, expectCloseArray } from "./helpers/numeric.js";

const xor = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

describe("WASM guardrails", () => {
  test("loads scalar and auto SIMD kernel artifacts", async () => {
    const scalar = await loadWasmKernel({ simd: false });
    const auto = await loadWasmKernel({ simd: "auto" });
    expect(scalar.exports.bunaptic_version()).toBe(1);
    expect(auto.exports.bunaptic_version()).toBe(1);
  });

  test("keeps no-state recurrent sequence parity after memory churn", async () => {
    const kernel = await loadWasmKernel();
    const churn = Architect.Perceptron(4, 32, 2).useWasm(kernel);
    const churnInputs = new Float32Array(1024 * churn.input);
    for (let i = 0; i < churnInputs.length; i++) churnInputs[i] = (i % 17) / 17;
    for (let i = 0; i < 8; i++) kernel.graphForwardBatch(churn.compiledGraph(), churnInputs);

    const values = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1, 0.5, 0.25]);
    for (const network of [Architect.LSTM(2, 4, 1, { seed: 31 }), Architect.GRU(2, 4, 1, { seed: 32 })]) {
      const expected = network.runSequence(values).flatMap((item) => Array.from(item));
      const actual = Network.fromJSON(network.toJSON()).useWasm(kernel).runSequence(values).flatMap((item) => Array.from(item));
      expectCloseArray(actual, expected, TOLERANCE.recurrent);
    }
  });

  test("starts WASM momentum training with zeroed previous deltas", async () => {
    const kernel = await loadWasmKernel();
    const dataset = packDataset([xor[1]!], 2, 1);
    const original = Architect.Perceptron(2, 4, 1);
    const withoutMomentum = Network.fromJSON(original.toJSON()).useWasm(kernel);
    const withMomentum = Network.fromJSON(original.toJSON()).useWasm(kernel);

    withoutMomentum.train(dataset, { iterations: 1, error: -1, rate: 0.2, momentum: 0 });
    withMomentum.train(dataset, { iterations: 1, error: -1, rate: 0.2, momentum: 0.9 });

    expectCloseArray(Array.from(withMomentum.getParameters().weights), Array.from(withoutMomentum.getParameters().weights), TOLERANCE.wasm);
    expectCloseArray(Array.from(withMomentum.getParameters().biases), Array.from(withoutMomentum.getParameters().biases), TOLERANCE.wasm);
  });

  test("resets fused recurrent state through runSequence options", async () => {
    const kernel = await loadWasmKernel();
    const network = Architect.GRU(1, 4, 1, { seed: 33 }).useWasm(kernel);
    const state = network.createRecurrentState();
    network.runSequence(new Float32Array([1, 2]), { recurrentState: state });

    const continued = network.runSequence(new Float32Array([3]), { recurrentState: state })[0] as Float32Array;
    const reset = network.runSequence(new Float32Array([3]), { recurrentState: state, reset: true })[0] as Float32Array;
    const fresh = Network.fromJSON(network.toJSON()).useWasm(kernel).runSequence(new Float32Array([3]))[0] as Float32Array;

    expect(Array.from(continued).map((value) => Number(value.toFixed(6)))).not.toEqual(Array.from(fresh).map((value) => Number(value.toFixed(6))));
    expectCloseArray(Array.from(reset), Array.from(fresh), TOLERANCE.recurrent);
  });

  test("reuses and invalidates persistent recurrent handles after training", async () => {
    const kernel = await loadWasmKernel();
    const network = Architect.GRU(2, 4, 1, { seed: 41 }).useWasm(kernel);
    const values = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]);
    const before = network.runSequence(values).flatMap((item) => Array.from(item));
    const repeated = network.runSequence(values).flatMap((item) => Array.from(item));
    expectCloseArray(repeated, before, TOLERANCE.recurrent);

    network.trainSequences([
      { input: values, length: 4, output: [1] },
    ], { iterations: 2, optimizer: "adam", rate: 0.01 });
    const after = network.runSequence(values).flatMap((item) => Array.from(item));

    expect(after.some((value, index) => Math.abs(value - (before[index] as number)) > 1e-6)).toBe(true);
  });
});
