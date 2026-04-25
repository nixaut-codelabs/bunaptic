import { describe, expect, test } from "bun:test";

import { Architect, Network, packSequenceDataset } from "../src/index.js";

const finalSet = [
  { input: new Float32Array([0, 0, 1, 1]), length: 2, output: [1] },
  { input: new Float32Array([1, 0, 0, 0]), length: 2, output: [0] },
  { input: new Float32Array([1, 1, 1, 0]), length: 2, output: [1] },
  { input: new Float32Array([0, 0, 0, 0]), length: 2, output: [0] },
];

const perStepSet = [
  { input: new Float32Array([0, 1, 1, 0]), length: 2, output: [0, 1] },
  { input: new Float32Array([1, 0, 0, 1]), length: 2, output: [1, 0] },
];

describe("recurrent gradient hardening", () => {
  test("LSTM SGD step moves parameters and does not increase final-output loss", () => {
    const network = Architect.LSTM(2, 3, 1, { seed: 101 });
    const before = network.testSequences(finalSet).error;
    const paramsBefore = network.toJSON().recurrentSpec!.inputWeights.slice();
    const result = network.trainSequences(finalSet, { optimizer: "sgd", iterations: 1, rate: 0.01, clip: 1 });
    const paramsAfter = network.toJSON().recurrentSpec!.inputWeights;
    expect(paramsAfter.some((value, index) => Math.abs(value - (paramsBefore[index] as number)) > 1e-8)).toBe(true);
    expect(result.error).toBeLessThanOrEqual(before + 0.02);
  });

  test("GRU SGD step moves parameters and supports truncated BPTT", () => {
    const network = Architect.GRU(2, 3, 1, { seed: 102 });
    const before = network.testSequences(finalSet).error;
    const result = network.trainSequences(finalSet, { optimizer: "sgd", iterations: 2, rate: 0.01, clip: 0.5, truncatedSteps: 1 });
    expect(result.iterations).toBe(2);
    expect(Number.isFinite(result.error)).toBe(true);
    expect(network.testSequences(finalSet).error).toBeLessThan(before + 0.05);
  });

  test("Adam and RMSProp recurrent optimizers keep finite updates", () => {
    for (const optimizer of ["adam", "rmsprop"] as const) {
      const network = Architect.GRU(2, 3, 1, { seed: optimizer === "adam" ? 103 : 104 });
      const result = network.trainSequences(finalSet, { optimizer, iterations: 3, rate: 0.01, clip: 1 });
      expect(Number.isFinite(result.error)).toBe(true);
      expect(network.toJSON().recurrentSpec!.inputWeights.every(Number.isFinite)).toBe(true);
    }
  });

  test("per-step packed dataset roundtrips through JSON copy and trains", () => {
    const packed = packSequenceDataset(perStepSet, 2, 1);
    const original = Architect.GRU(2, 3, 1, { seed: 105 });
    const network = Network.fromJSON(original.toJSON());
    const before = network.testSequences(packed).error;
    const result = network.trainSequences(packed, { optimizer: "momentum", momentum: 0.2, iterations: 4, rate: 0.01, clip: 1 });
    expect(result.error).toBeLessThan(before + 0.05);
  });
});
