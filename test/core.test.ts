import { describe, expect, test } from "bun:test";

import { Architect, ConnectionKind, MutationMethod, Network, Neat, WorkerPool, cost, decodeModel, evaluatePopulation, loadWasmKernel, packDataset, packedDatasetKey } from "../src/index.js";
import { TOLERANCE, expectCloseArray } from "./helpers/numeric.js";

const xor = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

describe("Network", () => {
  test("activates dense networks", () => {
    const network = Architect.Perceptron(2, 3, 1);
    const output = network.activate([1, 0]);
    expect(output.length).toBe(1);
    expect(Number.isFinite(output[0] as number)).toBe(true);
  });

  test("roundtrips JSON and binary", () => {
    const original = Architect.Perceptron(2, 4, 1);
    const input = [0.25, 0.75];
    const expected = Array.from(original.activate(input));
    const fromJson = Network.fromJSON(original.toJSON());
    const fromBinary = Network.fromBinary(original.toBinary());
    expect(Array.from(fromJson.activate(input))).toEqual(expected);
    expect(Array.from(fromBinary.activate(input))).toEqual(expected);
  });

  test("binary roundtrip preserves connection innovations", () => {
    const original = Architect.Random(2, 5, 1, { seed: 12, connections: 10, recurrent: 2, self: 1 });
    const restored = Network.fromBinary(original.toBinary());
    expect(restored.toJSON().connections.map((connection) => connection.innovation)).toEqual(
      original.toJSON().connections.map((connection) => connection.innovation),
    );
  });

  test("binary roundtrip preserves fused recurrent specs", () => {
    const input = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]);
    for (const original of [Architect.LSTM(2, 4, 1, { seed: 21 }), Architect.GRU(2, 4, 1, { seed: 22 })]) {
      const restored = Network.fromBinary(original.toBinary());
      expect(restored.toJSON().recurrentSpec?.kind).toBe(original.toJSON().recurrentSpec?.kind);
      const expected = original.runSequence(input).flatMap((item) => Array.from(item));
      const actual = restored.runSequence(input).flatMap((item) => Array.from(item));
      expectCloseArray(actual, expected, TOLERANCE.recurrent);
    }
  });

  test("decodes legacy V1 binary models", () => {
    const bytes = legacyBinaryFixture();
    const model = decodeModel(bytes);
    const network = Network.fromBinary(bytes);
    expect(model.connections.length).toBe(1);
    expect(network.activate([1]).length).toBe(1);
  });

  test("exports and imports flat parameters", () => {
    const original = Architect.Perceptron(2, 4, 1);
    const copy = Architect.Perceptron(2, 4, 1).setParameters(original.getParameters());
    expect(Array.from(copy.activate([0.2, 0.8]))).toEqual(Array.from(original.activate([0.2, 0.8])));
  });

  test("imports supported Neataptic JSON", () => {
    const network = Network.fromNeatapticJSON({
      input: 2,
      output: 1,
      nodes: [
        { index: 0, type: "input", bias: 0, squash: "IDENTITY" },
        { index: 1, type: "input", bias: 0, squash: "IDENTITY" },
        { index: 2, type: "output", bias: 0.1, squash: "LOGISTIC" },
      ],
      connections: [
        { from: 0, to: 2, weight: 0.5, gater: null },
        { from: 1, to: 2, weight: -0.25, gater: null },
      ],
    });
    expect(network.activate([1, 0]).length).toBe(1);
  });

  test("trains XOR below loose threshold", () => {
    const network = Architect.Perceptron(2, 6, 1);
    const packed = packDataset(xor, 2, 1);
    const before = network.testPacked(packed).error;
    const result = network.train(packed, { iterations: 800, rate: 0.5, momentum: 0.1, shuffle: true });
    expect(result.error).toBeLessThan(before);
  });

  test("packs datasets with stable keys", () => {
    const first = packDataset(xor, 2, 1);
    const second = packDataset(xor, 2, 1);
    expect(first.sampleCount).toBe(4);
    expect(packedDatasetKey(first)).toBe(packedDatasetKey(second));
  });

  test("creates deterministic networks from the same seed", () => {
    const a = Architect.Random(2, 5, 1, { seed: 123, connections: 12, recurrent: 2, self: 1 });
    const b = Architect.Random(2, 5, 1, { seed: 123, connections: 12, recurrent: 2, self: 1 });
    expect(a.toJSON()).toEqual(b.toJSON());
  });

  test("compiled ranges cover every enabled connection", () => {
    const network = Architect.Random(3, 6, 2, { seed: 9, connections: 20, recurrent: 3, self: 2 });
    const graph = network.compiledGraph();
    expect(graph.incoming.length).toBe(graph.weights.length);
    expect(graph.outgoing.length).toBe(graph.weights.length);
    expect(graph.incomingStarts.length).toBe(graph.nodeCount + 1);
    expect(graph.outgoingStarts.length).toBe(graph.nodeCount + 1);
    expect(graph.incomingStarts[graph.incomingStarts.length - 1]).toBe(graph.weights.length);
    expect(graph.outgoingStarts[graph.outgoingStarts.length - 1]).toBe(graph.weights.length);
  });
});

describe("Dynamic sequences", () => {
  test("runs only actual sequence steps", () => {
    const network = Architect.RNN(1, 3, 1);
    const short = network.runSequence(new Float32Array([1, 2, 3]));
    const long = network.runSequence(new Float32Array([1, 2, 3, 4, 5, 6, 7]));
    expect(short.length).toBe(3);
    expect(long.length).toBe(7);
  });

  test("runs ragged batches without padding", () => {
    const network = Architect.GRU(1, 3, 1);
    const outputs = network.runBatchRagged({
      values: new Float32Array([1, 2, 3, 10, 11]),
      offsets: new Uint32Array([0, 3, 5]),
      featureSize: 1,
    });
    expect(outputs.length).toBe(2);
    expect(outputs[0]?.length).toBe(1);
    expect(outputs[1]?.length).toBe(1);
  });

  test("LSTM fused WASM matches TS reference", async () => {
    const kernel = await loadWasmKernel();
    const tsNetwork = Architect.LSTM(2, 4, 1, { seed: 123 });
    const wasmNetwork = Network.fromJSON(tsNetwork.toJSON()).useWasm(kernel);
    const values = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]);
    const ts = tsNetwork.runSequence(values).flatMap((item) => Array.from(item));
    const wasm = wasmNetwork.runSequence(values).flatMap((item) => Array.from(item));
    expectCloseArray(wasm, ts, TOLERANCE.recurrent);
  });

  test("GRU fused WASM matches TS reference", async () => {
    const kernel = await loadWasmKernel();
    const tsNetwork = Architect.GRU(2, 4, 1, { seed: 456 });
    const wasmNetwork = Network.fromJSON(tsNetwork.toJSON()).useWasm(kernel);
    const values = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]);
    const ts = tsNetwork.runSequence(values).flatMap((item) => Array.from(item));
    const wasm = wasmNetwork.runSequence(values).flatMap((item) => Array.from(item));
    expectCloseArray(wasm, ts, TOLERANCE.recurrent);
  });

  test("fused recurrent ragged output matches TS reference", async () => {
    const kernel = await loadWasmKernel();
    const batch = { values: new Float32Array([0, 0, 0, 1, 1, 0, 1, 1, 0, 1]), offsets: new Uint32Array([0, 4, 10]), featureSize: 2 };
    for (const network of [Architect.LSTM(2, 3, 1, { seed: 7 }), Architect.GRU(2, 3, 1, { seed: 8 })]) {
      const wasmNetwork = Network.fromJSON(network.toJSON()).useWasm(kernel);
      const ts = network.runBatchRagged(batch).flatMap((item) => Array.from(item));
      const wasm = wasmNetwork.runBatchRagged(batch).flatMap((item) => Array.from(item));
      expectCloseArray(wasm, ts, TOLERANCE.recurrent);
    }
  });

  test("supports recurrent state continuation helpers", () => {
    const network = Architect.LSTM(1, 3, 1, { seed: 19 });
    const state = network.createRecurrentState();
    const first = network.runSequence(new Float32Array([1, 2]), { recurrentState: state });
    const snapshot = network.cloneRecurrentState(state);
    const continued = network.runSequence(new Float32Array([3]), { recurrentState: state });
    network.resetRecurrentState(state);
    const restarted = network.runSequence(new Float32Array([3]), { recurrentState: state });
    expect(first.length).toBe(2);
    expect(snapshot.hidden.length).toBe(3);
    expect(continued.length).toBe(1);
    expect(restarted.length).toBe(1);
    expect(Array.from(continued[0] as Float32Array)).not.toEqual(Array.from(restarted[0] as Float32Array));
  });

  test("trains fused recurrent sequence output with TS reference BPTT", () => {
    const set = [
      { input: new Float32Array([0, 0, 1, 1]), length: 2, output: [1] },
      { input: new Float32Array([1, 0, 0, 0]), length: 2, output: [0] },
      { input: new Float32Array([1, 1, 1, 0]), length: 2, output: [1] },
      { input: new Float32Array([0, 0, 0, 0]), length: 2, output: [0] },
    ];
    const network = Architect.GRU(2, 5, 1, { seed: 14 });
    const before = network.testSequences(set).error;
    const result = network.trainSequences(set, { iterations: 30, rate: 0.05, clip: 1 });
    expect(result.error).toBeLessThan(before);
  });
});

describe("Architect", () => {
  test("creates all required architectures", () => {
    const networks = [
      Architect.Perceptron(2, 3, 1),
      Architect.Random(2, 4, 1),
      Architect.RNN(2, 3, 1),
      Architect.LSTM(2, 3, 1),
      Architect.GRU(2, 3, 1),
      Architect.NARX(2, 3, 1),
      Architect.Hopfield(3),
    ];
    for (const network of networks) {
      const input = new Float32Array(network.input);
      expect(network.activate(input).length).toBe(network.output);
    }
  });
});

describe("Neat", () => {
  test("evolves a population", async () => {
    const neat = new Neat(2, 1, (network) => -network.test(xor, cost.MSE).error, {
      seed: 42,
      popsize: 10,
      elitism: 2,
      mutationRate: 0.5,
    });
    const best = await neat.evolve();
    expect(best.input).toBe(2);
    expect(neat.generation).toBe(1);
    expect(neat.scores.length).toBe(10);
  });

  test("can evaluate through packed dataset population path", async () => {
    const neat = new Neat(2, 1, () => 0, {
      seed: 12,
      popsize: 12,
      elitism: 2,
      dataset: packDataset(xor, 2, 1),
      evaluator: { mode: "direct" },
    });
    const scores = await neat.evaluate();
    expect(scores.length).toBe(12);
    expect(scores.every((entry) => Number.isFinite(entry.score))).toBe(true);
  });

  test("preserves structural innovations across mutation and crossover", () => {
    const neat = new Neat(1, 1, () => 0, { seed: 44, popsize: 2, mutation: [MutationMethod.AddNode], mutationRate: 1 });
    const first = neat.population[0] as Network;
    const second = neat.population[1] as Network;
    first.mutate(MutationMethod.AddNode, neat.innovation);
    second.mutate(MutationMethod.AddNode, neat.innovation);
    expect(first.toJSON().connections.map((connection) => connection.innovation)).toEqual(second.toJSON().connections.map((connection) => connection.innovation));
    const child = Network.crossOver(first, second, { seed: 9 });
    const innovations = child.toJSON().connections.map((connection) => connection.innovation);
    expect(innovations.every((innovation) => typeof innovation === "number" && innovation > 0)).toBe(true);
  });

  test("keeps dataset-based evolution deterministic for the same seed", async () => {
    const options = { seed: 5, popsize: 10, elitism: 2, dataset: packDataset(xor, 2, 1), evaluator: { mode: "direct" as const } };
    const first = new Neat(2, 1, () => 0, options);
    const second = new Neat(2, 1, () => 0, options);
    await first.evolve();
    await second.evolve();
    expect(first.scores.map((entry) => Number(entry.score.toFixed(8)))).toEqual(second.scores.map((entry) => Number(entry.score.toFixed(8))));
    expect(first.export()).toEqual(second.export());
  });

  test("mutations keep activations finite", () => {
    const network = Architect.Random(3, 6, 2, { seed: 71, connections: 16, recurrent: 2, self: 2 });
    const neat = new Neat(3, 2, () => 0, { seed: 71, popsize: 2 });
    for (let i = 0; i < 40; i++) network.mutate(neat.rng.pick([MutationMethod.AddNode, MutationMethod.AddConnection, MutationMethod.AddSelfConnection, MutationMethod.ModWeight, MutationMethod.ModBias]), neat.innovation);
    const output = network.activate([0.2, 0.4, 0.6]);
    expect(Array.from(output).every(Number.isFinite)).toBe(true);
  });
});

describe("WASM", () => {
  test("matches TS graph forward", async () => {
    const kernel = await loadWasmKernel();
    const tsNetwork = Architect.Perceptron(2, 4, 1);
    const wasmNetwork = Network.fromJSON(tsNetwork.toJSON()).useWasm(kernel);
    const input = [0.4, 0.7];
    const expected = Array.from(tsNetwork.activate(input));
    const actual = Array.from(wasmNetwork.activate(input));
    expectCloseArray(actual, expected, TOLERANCE.wasm);
  });

  test("matches TS batch forward", async () => {
    const kernel = await loadWasmKernel();
    const network = Architect.Perceptron(2, 4, 1);
    const graph = network.compiledGraph();
    const inputs = new Float32Array([0, 0, 0, 1, 1, 0, 1, 1]);
    const actual = Array.from(kernel.graphForwardBatch(graph, inputs));
    const expected = xor.flatMap((sample) => Array.from(network.activate(sample.input)));
    expectCloseArray(actual, expected, TOLERANCE.wasm);
  });

  test("matches TS ragged sequence length", async () => {
    const kernel = await loadWasmKernel();
    const tsNetwork = Architect.RNN(1, 3, 1);
    const wasmNetwork = Network.fromJSON(tsNetwork.toJSON()).useWasm(kernel);
    const values = new Float32Array([1, 2, 3, 4]);
    const ts = tsNetwork.runSequence(values).flatMap((item) => Array.from(item));
    const wasm = wasmNetwork.runSequence(values).flatMap((item) => Array.from(item));
    expectCloseArray(wasm, ts, TOLERANCE.wasm);
  });

  test("matches TS ragged final outputs", async () => {
    const kernel = await loadWasmKernel();
    const tsNetwork = Architect.RNN(1, 3, 1);
    const wasmNetwork = Network.fromJSON(tsNetwork.toJSON()).useWasm(kernel);
    const batch = {
      values: new Float32Array([1, 2, 3, 10, 11]),
      offsets: new Uint32Array([0, 3, 5]),
      featureSize: 1,
    };
    const ts = tsNetwork.runBatchRagged(batch).flatMap((item) => Array.from(item));
    const wasm = wasmNetwork.runBatchRagged(batch).flatMap((item) => Array.from(item));
    expectCloseArray(wasm, ts, TOLERANCE.wasm);
  });

  test("matches TS dataset MSE", async () => {
    const kernel = await loadWasmKernel();
    const tsNetwork = Architect.Perceptron(2, 4, 1);
    const wasmNetwork = Network.fromJSON(tsNetwork.toJSON()).useWasm(kernel);
    expect(Number(wasmNetwork.test(xor).error.toFixed(6))).toBe(Number(tsNetwork.test(xor).error.toFixed(6)));
  });

  test("WASM graph training lowers error", async () => {
    const kernel = await loadWasmKernel();
    const network = Architect.Perceptron(2, 6, 1).useWasm(kernel);
    const and = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [0] },
      { input: [1, 0], output: [0] },
      { input: [1, 1], output: [1] },
    ];
    const before = network.test(and).error;
    const result = network.train([...and], { iterations: 100, error: -1, rate: 0.3, momentum: 0.1 });
    expect(result.iterations).toBe(100);
    expect(network.test(and).error).toBeLessThan(before);
  });
});

describe("Workers", () => {
  test("evaluates populations through the direct facade path", async () => {
    const population = [Architect.Perceptron(2, 3, 1), Architect.Perceptron(2, 4, 1), Architect.Perceptron(2, 5, 1)];
    const packed = packDataset(xor, 2, 1);
    const direct = population.map((network) => -network.testPacked(packed).error).map((value) => Number(value.toFixed(8)));
    const scores = (await evaluatePopulation(population, packed, { mode: "direct" })).map((value) => Number(value.toFixed(8)));
    expect(scores).toEqual(direct);
  });

  test("evaluates populations through worker facade path", async () => {
    const population = [Architect.Perceptron(2, 3, 1), Architect.Perceptron(2, 4, 1), Architect.Perceptron(2, 5, 1)];
    const packed = packDataset(xor, 2, 1);
    const direct = population.map((network) => -network.testPacked(packed).error).map((value) => Number(value.toFixed(8)));
    const scores = (await evaluatePopulation(population, packed, { mode: "workers", workers: 2 })).map((value) => Number(value.toFixed(8)));
    expect(scores).toEqual(direct);
  });

  test("evaluates populations through worker WASM facade path", async () => {
    const population = [Architect.Perceptron(2, 3, 1), Architect.Perceptron(2, 4, 1)];
    const packed = packDataset(xor, 2, 1);
    const direct = population.map((network) => -network.testPacked(packed).error).map((value) => Number(value.toFixed(6)));
    const scores = (await evaluatePopulation(population, packed, { mode: "workers", workers: 2, useWasm: true })).map((value) => Number(value.toFixed(6)));
    expect(scores).toEqual(direct);
  });

  test("evaluates direct populations through shared WASM kernel", async () => {
    const kernel = await loadWasmKernel();
    const population = [Architect.Perceptron(2, 3, 1), Architect.Perceptron(2, 4, 1)];
    const packed = packDataset(xor, 2, 1);
    const direct = population.map((network) => -network.testPacked(packed).error).map((value) => Number(value.toFixed(6)));
    const scores = (await evaluatePopulation(population, packed, { mode: "direct", useWasm: true, kernel })).map((value) => Number(value.toFixed(6)));
    expect(scores).toEqual(direct);
  });

  test("reports population shape mismatches with member index", async () => {
    const population = [Architect.Perceptron(2, 3, 1), Architect.Perceptron(3, 4, 1)];
    const packed = packDataset(xor, 2, 1);
    await expect(evaluatePopulation(population, packed, { mode: "direct" })).rejects.toThrow("Population member 1");
  });

  test("evaluates population with cached binary model payloads", async () => {
    const population = [Architect.Perceptron(2, 3, 1), Architect.Perceptron(2, 4, 1), Architect.Perceptron(2, 5, 1)];
    const packed = packDataset(xor, 2, 1);
    const direct = population.map((network) => -network.testPacked(packed).error).map((value) => Number(value.toFixed(8)));
    const pool = new WorkerPool({ size: 2, binaryModels: true });
    try {
      await pool.cacheDataset(packed);
      const first = (await pool.evaluate(population, packed)).map((value) => Number(value.toFixed(8)));
      const second = (await pool.evaluate(population, packed)).map((value) => Number(value.toFixed(8)));
      expect(first).toEqual(direct);
      expect(second).toEqual(direct);
    } finally {
      pool.terminate();
    }
  });

  test("evaluates population through worker WASM path", async () => {
    const population = [Architect.Perceptron(2, 3, 1), Architect.Perceptron(2, 4, 1)];
    const packed = packDataset(xor, 2, 1);
    const direct = population.map((network) => -network.testPacked(packed).error).map((value) => Number(value.toFixed(6)));
    const pool = new WorkerPool({ size: 2, binaryModels: true, useWasm: true });
    try {
      await pool.cacheDataset(packed);
      const scores = (await pool.evaluate(population, packed)).map((value) => Number(value.toFixed(6)));
      expect(scores).toEqual(direct);
    } finally {
      pool.terminate();
    }
  });
});

function legacyBinaryFixture(): Uint8Array {
  const buffer = new ArrayBuffer(7 * 4 + 2 * 6 + 21);
  const view = new DataView(buffer);
  let offset = 0;
  view.setUint32(offset, 0x42505443, true); offset += 4;
  view.setUint32(offset, 1, true); offset += 4;
  view.setUint32(offset, 1, true); offset += 4;
  view.setUint32(offset, 1, true); offset += 4;
  view.setUint32(offset, 0, true); offset += 4;
  view.setUint32(offset, 2, true); offset += 4;
  view.setUint32(offset, 1, true); offset += 4;
  view.setUint8(offset++, 0);
  view.setUint8(offset++, 2);
  view.setFloat32(offset, 0, true); offset += 4;
  view.setUint8(offset++, 2);
  view.setUint8(offset++, 0);
  view.setFloat32(offset, 0.1, true); offset += 4;
  view.setUint32(offset, 0, true); offset += 4;
  view.setUint32(offset, 1, true); offset += 4;
  view.setInt32(offset, -1, true); offset += 4;
  view.setFloat32(offset, 0.5, true); offset += 4;
  view.setUint8(offset++, ConnectionKind.Forward);
  view.setUint8(offset++, 1);
  offset += 3;
  return new Uint8Array(buffer);
}
