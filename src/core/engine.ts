import { activateValue, derivativeFromActivation } from "./activation.js";
import { cost, type CostFunction } from "./cost.js";
import { applyOptimizer, type OptimizerConfig, type OptimizerState } from "./optimizer.js";
import { ActivationKind, ConnectionKind, NodeKind, type CompiledGraph, type NetworkState, type RaggedBatch, type RunSequenceOptions, type TestResult, type TrainingSample } from "./types.js";

export class ReferenceEngine {
  readonly graph: CompiledGraph;

  constructor(graph: CompiledGraph) {
    this.graph = graph;
  }

  createState(): NetworkState {
    return {
      activations: new Float32Array(this.graph.nodeCount),
      states: new Float32Array(this.graph.nodeCount),
    };
  }

  cloneState(state: NetworkState): NetworkState {
    return {
      activations: new Float32Array(state.activations),
      states: new Float32Array(state.states),
    };
  }

  clearState(state: NetworkState): void {
    state.activations.fill(0);
    state.states.fill(0);
  }

  activate(input: ArrayLike<number>, state = this.createState()): Float32Array {
    if (input.length !== this.graph.input) {
      throw new Error(`Input length ${input.length} does not match network input size ${this.graph.input}`);
    }
    return this.forward(input, state, false);
  }

  step(input: ArrayLike<number>, state: NetworkState): Float32Array {
    return this.activate(input, state);
  }

  runSequence(sequence: ArrayLike<number>, options: RunSequenceOptions = {}): Float32Array[] {
    if (sequence.length % this.graph.input !== 0) {
      throw new Error("Sequence value count must be divisible by network input size");
    }

    const state = options.state ?? this.createState();
    if (options.reset) this.clearState(state);
    const outputs: Float32Array[] = [];
    const input = new Float32Array(this.graph.input);
    const steps = sequence.length / this.graph.input;
    for (let step = 0; step < steps; step++) {
      for (let i = 0; i < this.graph.input; i++) input[i] = Number(sequence[step * this.graph.input + i]);
      outputs.push(Float32Array.from(this.step(input, state)));
    }
    return outputs;
  }

  runBatchRagged(batch: RaggedBatch): Float32Array[] {
    if (batch.featureSize !== this.graph.input) {
      throw new Error(`Feature size ${batch.featureSize} does not match network input size ${this.graph.input}`);
    }
    const outputs: Float32Array[] = [];
    for (let item = 0; item < batch.offsets.length - 1; item++) {
      const start = batch.offsets[item] as number;
      const end = batch.offsets[item + 1] as number;
      const state = this.createState();
      let last: Float32Array<ArrayBufferLike> = new Float32Array(this.graph.output);
      for (let cursor = start; cursor < end; cursor += batch.featureSize) {
        last = this.step(batch.values.subarray(cursor, cursor + batch.featureSize), state);
      }
      outputs.push(Float32Array.from(last));
    }
    return outputs;
  }

  test(set: readonly TrainingSample[], costFunction: CostFunction = cost.MSE): TestResult {
    const start = Date.now();
    let error = 0;
    for (const sample of set) error += costFunction(sample.output, this.activate(sample.input));
    return { error: error / set.length, time: Date.now() - start };
  }

  private forward(input: ArrayLike<number>, state: NetworkState, training: boolean): Float32Array {
    const graph = this.graph;
    const activations = state.activations;
    const states = state.states;

    for (let i = 0; i < graph.input; i++) activations[i] = Number(input[i]);
    const previousActivations = graph.recurrent ? new Float32Array(activations) : activations;

    for (let node = graph.input; node < graph.nodeCount; node++) {
      const kind = graph.nodeKinds[node];
      if (kind === NodeKind.Input) continue;
      if (kind === NodeKind.Constant) {
        activations[node] = 1;
        continue;
      }

      let sum = graph.biases[node] as number;
      const start = graph.incomingStarts[node] as number;
      const end = graph.incomingStarts[node + 1] as number;
      for (let cursor = start; cursor < end; cursor++) {
        const conn = graph.incoming[cursor] as number;
        const from = graph.from[conn] as number;
        const gater = graph.gater[conn] as number;
        const source = graph.connectionKinds[conn] === ConnectionKind.Forward ? activations : previousActivations;
        const gainSource = gater >= 0 && graph.connectionKinds[conn] !== ConnectionKind.Forward ? previousActivations : activations;
        const gain = gater >= 0 ? gainSource[gater] as number : 1;
        sum += (source[from] as number) * (graph.weights[conn] as number) * gain;
      }
      states[node] = sum;
      activations[node] = activateValue(states[node] as number, graph.activations[node] as ActivationKind);
    }

    const output = new Float32Array(graph.output);
    for (let i = 0; i < graph.output; i++) output[i] = activations[graph.outputStart + i] as number;
    return output;
  }
}

export function trainOneDenseLayer(
  graph: CompiledGraph,
  set: readonly TrainingSample[],
  options: { rate: number; momentum: number },
  weights: Float32Array,
  previousDeltas: Float32Array,
  optimizer?: { config: OptimizerConfig; weights: OptimizerState; biases: OptimizerState },
): number {
  const engine = new ReferenceEngine(graph);
  let totalError = 0;
  const outputStart = graph.outputStart;
  const nodeDeltas = new Float32Array(graph.nodeCount);
  const weightGradients = optimizer ? new Float32Array(graph.weights.length) : null;
  const biasGradients = optimizer ? new Float32Array(graph.biases.length) : null;
  for (const sample of set) {
    const state = engine.createState();
    const output = engine.activate(sample.input, state);
    totalError += cost.MSE(sample.output, output);
    nodeDeltas.fill(0);
    weightGradients?.fill(0);
    biasGradients?.fill(0);

    for (let out = 0; out < graph.output; out++) {
      const node = outputStart + out;
      const target = Number(sample.output[out]);
      const actual = output[out] as number;
      const activationKind = graph.activations[node] as ActivationKind;
      const derivative = derivativeFromActivation(actual, state.states[node] as number, activationKind);
      const error = target - actual;
      nodeDeltas[node] = error * derivative;
    }

    for (let node = outputStart - 1; node >= graph.input; node--) {
      if (graph.nodeKinds[node] === NodeKind.Constant) continue;
      let downstream = 0;
      const start = graph.outgoingStarts[node] as number;
      const end = graph.outgoingStarts[node + 1] as number;
      for (let cursor = start; cursor < end; cursor++) {
        const conn = graph.outgoing[cursor] as number;
        if (graph.connectionKinds[conn] !== ConnectionKind.Forward) continue;
        downstream += (nodeDeltas[graph.to[conn] as number] as number) * (weights[conn] as number);
      }
      const activationKind = graph.activations[node] as ActivationKind;
      nodeDeltas[node] = downstream * derivativeFromActivation(state.activations[node] as number, state.states[node] as number, activationKind);
    }

    for (let node = graph.input; node < graph.nodeCount; node++) {
      if (biasGradients) biasGradients[node] = nodeDeltas[node] as number;
      else graph.biases[node] = (graph.biases[node] as number) + options.rate * (nodeDeltas[node] as number);
    }

    for (let node = graph.input; node < graph.nodeCount; node++) {
      const start = graph.incomingStarts[node] as number;
      const end = graph.incomingStarts[node + 1] as number;
      for (let cursor = start; cursor < end; cursor++) {
        const conn = graph.incoming[cursor] as number;
        const to = graph.to[conn] as number;
        const from = graph.from[conn] as number;
        const inputActivation = state.activations[from] as number;
        const gradient = (nodeDeltas[to] as number) * inputActivation;
        if (weightGradients) {
          weightGradients[conn] = gradient;
        } else {
          const delta = options.rate * gradient + options.momentum * (previousDeltas[conn] as number);
          weights[conn] = (weights[conn] as number) + delta;
          previousDeltas[conn] = delta;
        }
      }
    }
    if (optimizer && weightGradients && biasGradients) {
      applyOptimizer(weights, weightGradients, optimizer.weights, optimizer.config, "maximize");
      applyOptimizer(graph.biases, biasGradients, optimizer.biases, optimizer.config, "maximize");
    }
  }
  graph.weights.set(weights);
  return totalError / set.length;
}
