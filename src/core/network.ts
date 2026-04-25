import { encodeModel, decodeModel } from "./binary.js";
import { cost, type CostFunction } from "./cost.js";
import { assertPackedDatasetShape, packDataset } from "./dataset.js";
import { denseEvaluateDataset, denseForward, deriveDensePlan, trainDensePlan, type DensePlan } from "./dense.js";
import { ReferenceEngine, trainOneDenseLayer } from "./engine.js";
import { GraphBuilder } from "./graph.js";
import { fromNeatapticJSON, type NeatapticJSON } from "./neataptic-import.js";
import { createOptimizerState, optimizerConfig, shouldUseWasmTrainOptimizer } from "./optimizer.js";
import { Rng } from "./rng.js";
import { cloneRecurrentState, createRecurrentState, recurrentSpecFromJSON, recurrentSpecToJSON, resetRecurrentState, runRecurrentRagged, runRecurrentSequence } from "../recurrent/spec.js";
import { packSequenceDataset } from "../recurrent/dataset.js";
import { testRecurrentSequences, trainRecurrentSequences } from "../recurrent/train.js";
import { ActivationKind, ConnectionKind, NodeKind, type InnovationSource, type ModelJSON, type NetworkParameters, type NetworkState, type PackedDataset, type PackedSequenceDataset, type RaggedBatch, type RecurrentSpec, type RecurrentState, type RunSequenceOptions, type SequenceSample, type TestResult, type TrainingSample, type TrainOptions, type TrainResult, type TrainSequenceOptions } from "./types.js";
import type { WasmKernel, WasmRecurrentHandle } from "../wasm/kernel.js";

export interface NetworkOptions {
  rng?: Rng;
  seed?: number;
}

export class Network {
  graph: GraphBuilder;
  rng: Rng;
  recurrentSpec: RecurrentSpec | null = null;
  private engineCache: ReferenceEngine | null = null;
  private densePlanCache: DensePlan | null | undefined = undefined;
  private wasmKernel: WasmKernel | null = null;
  private wasmRecurrentHandle: WasmRecurrentHandle | null = null;

  constructor(input: number, output: number, options: NetworkOptions = {}) {
    this.rng = options.rng ?? new Rng(options.seed);
    this.graph = new GraphBuilder(input, output, this.rng);
    this.graph.connectDense(this.graph.inputNodes(), this.graph.outputNodes(), this.rng);
  }

  get input(): number {
    return this.graph.input;
  }

  get output(): number {
    return this.graph.output;
  }

  get nodeCount(): number {
    return this.graph.nodeKinds.length;
  }

  get connectionCount(): number {
    return this.graph.connections.filter((connection) => connection.enabled).length;
  }

  compile(): ReferenceEngine {
    this.engineCache = new ReferenceEngine(this.graph.compile());
    this.densePlanCache = undefined;
    return this.engineCache;
  }

  compiledGraph(): ReturnType<GraphBuilder["compile"]> {
    return this.engine().graph;
  }

  useWasm(kernel: WasmKernel | null): this {
    if (this.wasmKernel !== kernel) this.disposeWasmRecurrentHandle();
    this.wasmKernel = kernel;
    return this;
  }

  engine(): ReferenceEngine {
    return this.engineCache ?? this.compile();
  }

  activate(input: ArrayLike<number>, state?: NetworkState): Float32Array {
    if (!state && !this.recurrentSpec) {
      const densePlan = this.densePlan();
      if (densePlan) {
        const packed = input instanceof Float32Array ? input : Float32Array.from(input);
        return this.wasmKernel ? this.wasmKernel.denseForwardPlan(densePlan, packed) : denseForward(densePlan, packed);
      }
    }
    if (this.wasmKernel) {
      const networkState = state ?? this.createState();
      return this.wasmKernel.graphForward(this.compiledGraph(), Float32Array.from(input), networkState);
    }
    return this.engine().activate(input, state);
  }

  createState(): NetworkState {
    return this.engine().createState();
  }

  cloneState(state: NetworkState): NetworkState {
    return this.engine().cloneState(state);
  }

  clearState(state: NetworkState): void {
    this.engine().clearState(state);
  }

  createRecurrentState(): RecurrentState {
    if (!this.recurrentSpec) throw new Error("Network does not have a fused recurrent spec");
    return createRecurrentState(this.recurrentSpec);
  }

  cloneRecurrentState(state: RecurrentState): RecurrentState {
    return cloneRecurrentState(state);
  }

  resetRecurrentState(state: RecurrentState): void {
    resetRecurrentState(state);
  }

  step(input: ArrayLike<number>, state: NetworkState): Float32Array {
    return this.engine().step(input, state);
  }

  runSequence(sequence: ArrayLike<number>, options?: RunSequenceOptions): Float32Array[] {
    if (this.recurrentSpec) {
      const packed = sequence instanceof Float32Array ? sequence : Float32Array.from(sequence);
      if (options?.reset && options.recurrentState) this.resetRecurrentState(options.recurrentState);
      const flat = this.wasmKernel
        ? this.wasmKernel.recurrentForwardSequenceHandle(this.getWasmRecurrentHandle(), packed, options?.recurrentState)
        : runRecurrentSequence(this.recurrentSpec, packed, options?.recurrentState);
      return splitOutputs(flat, this.output);
    }
    if (this.wasmKernel) {
      if (sequence.length % this.input !== 0) throw new Error("Sequence value count must be divisible by network input size");
      const state = options?.state;
      if (state && options?.reset) this.clearState(state);
      const packed = sequence instanceof Float32Array ? sequence : Float32Array.from(sequence);
      const flat = this.wasmKernel.graphForwardSequence(this.compiledGraph(), packed, state);
      const outputs: Float32Array[] = [];
      for (let offset = 0; offset < flat.length; offset += this.output) {
        outputs.push(Float32Array.from(flat.subarray(offset, offset + this.output)));
      }
      return outputs;
    }
    return this.engine().runSequence(sequence, options);
  }

  runBatchRagged(batch: RaggedBatch): Float32Array[] {
    if (this.recurrentSpec) {
      const flat = this.wasmKernel
        ? this.wasmKernel.recurrentForwardRaggedHandle(this.getWasmRecurrentHandle(), batch)
        : runRecurrentRagged(this.recurrentSpec, batch);
      return splitOutputs(flat, this.output);
    }
    if (this.wasmKernel) {
      if (batch.featureSize !== this.input) throw new Error(`Feature size ${batch.featureSize} does not match network input size ${this.input}`);
      const flat = this.wasmKernel.graphForwardRagged(this.compiledGraph(), batch);
      const outputs: Float32Array[] = [];
      for (let offset = 0; offset < flat.length; offset += this.output) {
        outputs.push(Float32Array.from(flat.subarray(offset, offset + this.output)));
      }
      return outputs;
    }
    return this.engine().runBatchRagged(batch);
  }

  test(set: readonly TrainingSample[], costFunction: CostFunction = cost.MSE): TestResult {
    return this.testPacked(packDataset(set, this.input, this.output), costFunction);
  }

  testPacked(dataset: PackedDataset, costFunction: CostFunction = cost.MSE): TestResult {
    assertPackedDatasetShape(dataset, this.input, this.output);
    if (this.wasmKernel && costFunction === cost.MSE) {
      const start = Date.now();
      const densePlan = this.densePlan();
      if (densePlan) return { error: this.wasmKernel.denseEvaluatePackedDataset(densePlan, dataset), time: Date.now() - start };
      return { error: this.wasmKernel.evaluatePackedDataset(this.compiledGraph(), dataset), time: Date.now() - start };
    }
    if (costFunction === cost.MSE) {
      const densePlan = this.densePlan();
      if (densePlan) return { error: denseEvaluateDataset(densePlan, dataset), time: 0 };
    }
    return this.engine().test(unpackDataset(dataset), costFunction);
  }

  testSequences(set: readonly SequenceSample[] | PackedSequenceDataset): TestResult {
    if (!this.recurrentSpec) throw new Error("Network does not have a fused recurrent spec");
    const start = Date.now();
    return { error: testRecurrentSequences(this.recurrentSpec, set), time: Date.now() - start };
  }

  train(set: TrainingSample[] | PackedDataset, options: TrainOptions = {}): TrainResult {
    const dataset = Array.isArray(set) ? packDataset(set, this.input, this.output) : set;
    assertPackedDatasetShape(dataset, this.input, this.output);
    const start = Date.now();
    const targetError = options.error ?? 0.05;
    const iterations = options.iterations ?? 1000;
    const optimizer = optimizerConfig(options, 0.3);
    const rate = optimizer.rate;
    const momentum = optimizer.kind === "momentum" ? optimizer.momentum : options.momentum ?? 0;
    const graph = this.graph.compile();
    const densePlan = this.recurrentSpec ? null : deriveDensePlan(graph);
    if (densePlan && (!this.wasmKernel || !shouldUseWasmTrainOptimizer(optimizer.kind))) {
      const result = trainDensePlan(densePlan, dataset, optimizer, { iterations, targetError, momentum, clip: Infinity });
      this.applyCompiledWeights(result.weights, result.biases);
      if (options.clear) this.engineCache = null;
      return { error: result.error, iterations: result.iterations, time: Date.now() - start };
    }
    if (this.wasmKernel && !graph.recurrent && shouldUseWasmTrainOptimizer(optimizer.kind)) {
      const result = this.wasmKernel.trainGraph(graph, dataset, {
        iterations,
        targetError,
        rate,
        momentum,
      });
      this.applyCompiledWeights(result.weights, result.biases);
      return { error: result.error, iterations: result.iterations, time: Date.now() - start };
    }
    const weights = new Float32Array(graph.weights);
    const previousDeltas = new Float32Array(graph.weights.length);
    const optimizerState = optimizer.kind === "adam" || optimizer.kind === "rmsprop"
      ? { config: optimizer, weights: createOptimizerState(graph.weights.length, optimizer.kind), biases: createOptimizerState(graph.biases.length, optimizer.kind) }
      : undefined;
    const unpacked = unpackDataset(dataset);
    let error = Infinity;
    let iteration = 0;

    while (iteration < iterations && error > targetError) {
      if (options.shuffle) this.rng.shuffle(unpacked);
      error = trainOneDenseLayer(graph, unpacked, { rate, momentum }, weights, previousDeltas, optimizerState);
      iteration++;
    }

    this.applyCompiledWeights(graph.weights, graph.biases);
    if (options.clear) this.engineCache = null;
    return { error, iterations: iteration, time: Date.now() - start };
  }

  trainSequences(set: readonly SequenceSample[] | PackedSequenceDataset, options: TrainSequenceOptions = {}): TrainResult {
    if (!this.recurrentSpec) throw new Error("Network does not have a fused recurrent spec");
    this.disposeWasmRecurrentHandle();
    const start = Date.now();
    const optimizer = optimizerConfig(options, 0.01);
    if (this.wasmKernel && shouldUseWasmTrainOptimizer(optimizer.kind)) {
      const dataset: PackedSequenceDataset = Array.isArray(set)
        ? packSequenceDataset(set, this.recurrentSpec.inputSize, this.recurrentSpec.outputSize, { targetMode: "auto" })
        : set as PackedSequenceDataset;
      const trainOptions = {
        iterations: options.iterations ?? 100,
        targetError: options.error ?? 0.01,
        rate: optimizer.rate,
        momentum: optimizer.kind === "momentum" ? optimizer.momentum : options.momentum ?? 0,
        clip: options.clip ?? 1,
      };
      const result = this.wasmKernel.trainRecurrentSequences(this.recurrentSpec, dataset, typeof options.truncatedSteps === "number" ? { ...trainOptions, truncatedSteps: options.truncatedSteps } : trainOptions);
      return { error: result.error, iterations: result.iterations, time: Date.now() - start };
    }
    const result = trainRecurrentSequences(this.recurrentSpec, set, options);
    return { error: result.error, iterations: result.iterations, time: Date.now() - start };
  }

  mutate(method: MutationMethod = MutationMethod.ModWeight, innovationSource?: InnovationSource): this {
    switch (method) {
      case MutationMethod.AddNode:
        this.addHiddenNode(innovationSource);
        break;
      case MutationMethod.AddConnection:
        this.addRandomConnection(innovationSource);
        break;
      case MutationMethod.ModWeight:
        this.modWeight();
        break;
      case MutationMethod.ModBias:
        this.modBias();
        break;
      case MutationMethod.ModActivation:
        this.modActivation();
        break;
      case MutationMethod.AddSelfConnection:
        this.addSelfConnection(innovationSource);
        break;
    }
    this.engineCache = null;
    this.densePlanCache = undefined;
    return this;
  }

  toJSON(): ModelJSON {
    const json = this.graph.toJSON();
    if (this.recurrentSpec) json.recurrentSpec = recurrentSpecToJSON(this.recurrentSpec);
    return json;
  }

  toBinary(): Uint8Array {
    return encodeModel(this.toJSON());
  }

  getParameters(): NetworkParameters {
    const graph = this.graph.compile();
    return {
      weights: new Float32Array(graph.weights),
      biases: new Float32Array(graph.biases),
    };
  }

  setParameters(parameters: NetworkParameters): this {
    const graph = this.graph.compile();
    if (parameters.weights.length !== graph.weights.length) throw new Error("Weight parameter length mismatch");
    if (parameters.biases.length !== graph.biases.length) throw new Error("Bias parameter length mismatch");
    this.applyCompiledWeights(parameters.weights, parameters.biases);
    return this;
  }

  copy(): Network {
    return Network.fromJSON(this.toJSON(), { rng: this.rng.fork() });
  }

  static fromJSON(json: ModelJSON, options: NetworkOptions = {}): Network {
    const network = Object.create(Network.prototype) as Network;
    network.rng = options.rng ?? new Rng(options.seed);
    network.graph = GraphBuilder.fromJSON(json);
    network.recurrentSpec = json.recurrentSpec ? recurrentSpecFromJSON(json.recurrentSpec) : null;
    network.engineCache = null;
    network.densePlanCache = undefined;
    network.wasmKernel = null;
    network.wasmRecurrentHandle = null;
    return network;
  }

  static fromBinary(bytes: Uint8Array, options: NetworkOptions = {}): Network {
    return Network.fromJSON(decodeModel(bytes), options);
  }

  static fromNeatapticJSON(json: NeatapticJSON, options: NetworkOptions = {}): Network {
    return Network.fromJSON(fromNeatapticJSON(json), options);
  }

  static crossOver(parentA: Network, parentB: Network, options: NetworkOptions = {}): Network {
    if (parentA.input !== parentB.input || parentA.output !== parentB.output) {
      throw new Error("Cannot crossover networks with different input/output sizes");
    }
    const rng = options.rng ?? new Rng(options.seed);
    const jsonA = parentA.toJSON();
    const jsonB = parentB.toJSON();
    const nodes = jsonA.nodes.length >= jsonB.nodes.length ? jsonA.nodes : jsonB.nodes;
    const byInnovation = new Map<number, { a?: typeof jsonA.connections[number]; b?: typeof jsonB.connections[number] }>();
    for (const connection of jsonA.connections) byInnovation.set(connectionInnovation(connection), { ...(byInnovation.get(connectionInnovation(connection)) ?? {}), a: connection });
    for (const connection of jsonB.connections) byInnovation.set(connectionInnovation(connection), { ...(byInnovation.get(connectionInnovation(connection)) ?? {}), b: connection });
    const connections = [...byInnovation.entries()]
      .sort(([left], [right]) => left - right)
      .map(([, pair]) => {
        if (pair.a && pair.b) return rng.chance(0.5) ? pair.a : pair.b;
        return pair.a ?? pair.b;
      })
      .filter((connection): connection is typeof jsonA.connections[number] => Boolean(connection))
      .filter((connection) => connection.from < nodes.length && connection.to < nodes.length && (connection.gater < 0 || connection.gater < nodes.length));
    return Network.fromJSON({
      version: 1,
      input: jsonA.input,
      output: jsonA.output,
      recurrent: jsonA.recurrent || jsonB.recurrent,
      nodes: nodes.map((node) => ({ ...node })),
      connections: connections.map((connection) => ({ ...connection })),
    }, { rng });
  }

  private addHiddenNode(innovationSource?: InnovationSource): void {
    const candidates = this.graph.connections.filter((connection) => connection.enabled && connection.kind === ConnectionKind.Forward);
    if (candidates.length === 0) return;
    const connection = this.rng.pick(candidates);
    connection.enabled = false;
    const node = this.graph.addNode(NodeKind.Hidden, ActivationKind.Logistic, this.rng.range(-0.1, 0.1));
    const split = innovationSource?.split(connectionInnovation(connection));
    this.graph.addConnection(connection.from, node, 1, ConnectionKind.Forward, -1, split?.input);
    this.graph.addConnection(node, connection.to, connection.weight, ConnectionKind.Forward, -1, split?.output);
  }

  private addRandomConnection(innovationSource?: InnovationSource): void {
    for (let attempts = 0; attempts < 32; attempts++) {
      const from = this.rng.int(this.graph.nodeKinds.length - this.output);
      const to = this.rng.range(this.input, this.graph.nodeKinds.length) | 0;
      if (from === to) continue;
      if (this.graph.connections.some((connection) => connection.enabled && connection.from === from && connection.to === to)) continue;
      const kind = from < to ? ConnectionKind.Forward : ConnectionKind.Recurrent;
      this.graph.addConnection(from, to, this.rng.range(-1, 1), kind, -1, innovationSource?.connection(from, to, kind));
      return;
    }
  }

  private modWeight(): void {
    const enabled = this.graph.connections.filter((connection) => connection.enabled);
    if (enabled.length === 0) return;
    this.rng.pick(enabled).weight += this.rng.range(-0.5, 0.5);
  }

  private modBias(): void {
    const index = this.input + this.rng.int(Math.max(1, this.graph.nodeKinds.length - this.input));
    this.graph.biases[index] = (this.graph.biases[index] as number) + this.rng.range(-0.2, 0.2);
  }

  private modActivation(): void {
    const activations = [ActivationKind.Logistic, ActivationKind.Tanh, ActivationKind.Relu, ActivationKind.Identity, ActivationKind.Softsign] as const;
    const span = Math.max(1, this.graph.nodeKinds.length - this.input - this.output);
    const index = this.input + this.rng.int(span);
    this.graph.activations[index] = this.rng.pick(activations);
  }

  private addSelfConnection(innovationSource?: InnovationSource): void {
    const index = this.input + this.rng.int(Math.max(1, this.graph.nodeKinds.length - this.input));
    this.graph.addConnection(index, index, this.rng.range(-1, 1), ConnectionKind.Self, -1, innovationSource?.connection(index, index, ConnectionKind.Self));
  }

  private applyCompiledWeights(weights: Float32Array, biases: Float32Array): void {
    let cursor = 0;
    for (const connection of this.graph.connections) {
      if (!connection.enabled) continue;
      connection.weight = weights[cursor] as number;
      cursor++;
    }
    for (let i = 0; i < this.graph.biases.length; i++) this.graph.biases[i] = biases[i] as number;
    this.engineCache = null;
    this.densePlanCache = undefined;
  }

  private densePlan(): DensePlan | null {
    if (this.recurrentSpec) return null;
    if (typeof this.densePlanCache !== "undefined") return this.densePlanCache;
    this.densePlanCache = deriveDensePlan(this.compiledGraph());
    return this.densePlanCache;
  }

  private getWasmRecurrentHandle(): WasmRecurrentHandle {
    if (!this.wasmKernel) throw new Error("WASM kernel is not attached");
    if (!this.recurrentSpec) throw new Error("Network does not have a fused recurrent spec");
    this.wasmRecurrentHandle ??= this.wasmKernel.createRecurrentHandle(this.recurrentSpec);
    return this.wasmRecurrentHandle;
  }

  private disposeWasmRecurrentHandle(): void {
    if (this.wasmKernel && this.wasmRecurrentHandle) this.wasmKernel.disposeRecurrentHandle(this.wasmRecurrentHandle);
    this.wasmRecurrentHandle = null;
  }

  private assertSetShape(set: readonly TrainingSample[]): void {
    if (set.length === 0) throw new Error("Training/test set cannot be empty");
    for (const sample of set) {
      if (sample.input.length !== this.input) throw new Error("Dataset input size does not match network input size");
      if (sample.output.length !== this.output) throw new Error("Dataset output size does not match network output size");
    }
  }
}

function connectionInnovation(connection: { innovation?: number; from: number; to: number; kind: ConnectionKind }): number {
  return connection.innovation ?? ((connection.from + 1) * 73856093 ^ (connection.to + 1) * 19349663 ^ connection.kind * 83492791) >>> 0;
}

function unpackDataset(dataset: PackedDataset): TrainingSample[] {
  return Array.from({ length: dataset.sampleCount }, (_, sample) => ({
    input: dataset.inputs.subarray(sample * dataset.inputSize, (sample + 1) * dataset.inputSize),
    output: dataset.targets.subarray(sample * dataset.outputSize, (sample + 1) * dataset.outputSize),
  }));
}

function splitOutputs(flat: Float32Array, outputSize: number): Float32Array[] {
  const outputs: Float32Array[] = [];
  for (let offset = 0; offset < flat.length; offset += outputSize) outputs.push(Float32Array.from(flat.subarray(offset, offset + outputSize)));
  return outputs;
}

export enum MutationMethod {
  AddNode = "ADD_NODE",
  AddConnection = "ADD_CONN",
  ModWeight = "MOD_WEIGHT",
  ModBias = "MOD_BIAS",
  ModActivation = "MOD_ACTIVATION",
  AddSelfConnection = "ADD_SELF_CONN",
}
