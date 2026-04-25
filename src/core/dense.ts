import { activateValue, derivativeFromActivation } from "./activation.js";
import { applyOptimizer, createOptimizerState, type OptimizerConfig, type OptimizerState } from "./optimizer.js";
import { ActivationKind, ConnectionKind, NodeKind, type CompiledGraph, type NetworkParameters, type PackedDataset } from "./types.js";

export interface DenseLayerPlan {
  inputNodes: Uint32Array;
  outputNodes: Uint32Array;
  connectionIndexes: Uint32Array;
  activation: ActivationKind;
  weights: Float32Array;
  biases: Float32Array;
}

export interface DensePlan {
  inputSize: number;
  outputSize: number;
  layers: DenseLayerPlan[];
}

export interface PackedDensePlan {
  layerSizes: Uint32Array;
  activations: Uint8Array;
  weightStarts: Uint32Array;
  biasStarts: Uint32Array;
  weights: Float32Array;
  biases: Float32Array;
}

export interface DenseTrainResult extends NetworkParameters {
  error: number;
  iterations: number;
}

export function deriveDensePlan(graph: CompiledGraph): DensePlan | null {
  if (graph.recurrent) return null;
  for (let i = 0; i < graph.connectionKinds.length; i++) {
    if (graph.connectionKinds[i] !== ConnectionKind.Forward || (graph.gater[i] as number) >= 0) return null;
  }
  for (let node = 0; node < graph.nodeCount; node++) {
    if (graph.nodeKinds[node] === NodeKind.Constant) return null;
  }

  const levels = new Int32Array(graph.nodeCount).fill(-1);
  for (let node = 0; node < graph.input; node++) levels[node] = 0;
  for (let node = graph.input; node < graph.nodeCount; node++) {
    let level = -1;
    for (let cursor = graph.incomingStarts[node] as number; cursor < (graph.incomingStarts[node + 1] as number); cursor++) {
      const conn = graph.incoming[cursor] as number;
      const sourceLevel = levels[graph.from[conn] as number] as number;
      if (sourceLevel < 0) return null;
      level = Math.max(level, sourceLevel + 1);
    }
    if (level < 1) return null;
    levels[node] = level;
  }

  const maxLevel = Math.max(...Array.from(levels));
  const outputNodes = Array.from({ length: graph.output }, (_, index) => graph.outputStart + index);
  if (!outputNodes.every((node) => levels[node] === maxLevel)) return null;

  const groups: number[][] = [];
  groups[0] = Array.from({ length: graph.input }, (_, index) => index);
  for (let level = 1; level <= maxLevel; level++) {
    const nodes: number[] = [];
    for (let node = graph.input; node < graph.nodeCount; node++) if (levels[node] === level) nodes.push(node);
    if (nodes.length === 0) return null;
    groups[level] = nodes;
  }

  const layers: DenseLayerPlan[] = [];
  for (let level = 1; level <= maxLevel; level++) {
    const previous = groups[level - 1] as number[];
    const current = groups[level] as number[];
    const activation = graph.activations[current[0] as number] as ActivationKind;
    if (!current.every((node) => graph.activations[node] === activation)) return null;

    const weights = new Float32Array(previous.length * current.length);
    const connectionIndexes = new Uint32Array(weights.length);
    for (let out = 0; out < current.length; out++) {
      const node = current[out] as number;
      const incoming = new Map<number, number>();
      for (let cursor = graph.incomingStarts[node] as number; cursor < (graph.incomingStarts[node + 1] as number); cursor++) {
        const conn = graph.incoming[cursor] as number;
        incoming.set(graph.from[conn] as number, conn);
      }
      if (incoming.size !== previous.length) return null;
      for (let input = 0; input < previous.length; input++) {
        const conn = incoming.get(previous[input] as number);
        if (typeof conn !== "number") return null;
        const index = out * previous.length + input;
        weights[index] = graph.weights[conn] as number;
        connectionIndexes[index] = conn;
      }
    }

    layers.push({
      inputNodes: Uint32Array.from(previous),
      outputNodes: Uint32Array.from(current),
      connectionIndexes,
      activation,
      weights,
      biases: Float32Array.from(current.map((node) => graph.biases[node] as number)),
    });
  }

  return { inputSize: graph.input, outputSize: graph.output, layers };
}

export function denseForward(plan: DensePlan, input: ArrayLike<number>): Float32Array {
  if (input.length !== plan.inputSize) throw new Error(`Input length ${input.length} does not match dense input size ${plan.inputSize}`);
  let previous = Float32Array.from(input);
  for (const layer of plan.layers) {
    const output = new Float32Array(layer.outputNodes.length);
    for (let out = 0; out < output.length; out++) {
      let sum = layer.biases[out] as number;
      const row = out * previous.length;
      for (let i = 0; i < previous.length; i++) sum += (previous[i] as number) * (layer.weights[row + i] as number);
      output[out] = activateValue(sum, layer.activation);
    }
    previous = output;
  }
  return previous;
}

export function denseEvaluateDataset(plan: DensePlan, dataset: PackedDataset): number {
  assertDenseDataset(plan, dataset);
  let error = 0;
  const scratch = createDenseScratch(plan);
  for (let sample = 0; sample < dataset.sampleCount; sample++) {
    const output = denseForwardInto(plan, dataset.inputs.subarray(sample * dataset.inputSize, (sample + 1) * dataset.inputSize), scratch);
    for (let out = 0; out < dataset.outputSize; out++) {
      const delta = (dataset.targets[sample * dataset.outputSize + out] as number) - (output[out] as number);
      error += delta * delta;
    }
  }
  return error / (dataset.sampleCount * dataset.outputSize);
}

export function trainDensePlan(plan: DensePlan, dataset: PackedDataset, optimizer: OptimizerConfig, options: { iterations: number; targetError: number; momentum: number; clip?: number }): DenseTrainResult {
  assertDenseDataset(plan, dataset);
  const weightVelocities = plan.layers.map((layer) => new Float32Array(layer.weights.length));
  const optimizerState = plan.layers.map((layer) => ({
    weights: createOptimizerState(layer.weights.length, optimizer.kind),
    biases: createOptimizerState(layer.biases.length, optimizer.kind),
  }));
  let error = Infinity;
  let iteration = 0;
  while (iteration < options.iterations && error > options.targetError) {
    error = trainDenseIteration(plan, dataset, optimizer, weightVelocities, optimizerState, options.clip ?? Infinity);
    iteration++;
  }
  return { ...denseParameters(plan), error, iterations: iteration };
}

export function denseParameters(plan: DensePlan): NetworkParameters {
  const maxConnection = Math.max(...plan.layers.flatMap((layer) => Array.from(layer.connectionIndexes))) + 1;
  const maxNode = Math.max(...plan.layers.flatMap((layer) => Array.from(layer.outputNodes))) + 1;
  const weights = new Float32Array(maxConnection);
  const biases = new Float32Array(maxNode);
  for (const layer of plan.layers) {
    for (let i = 0; i < layer.connectionIndexes.length; i++) weights[layer.connectionIndexes[i] as number] = layer.weights[i] as number;
    for (let i = 0; i < layer.outputNodes.length; i++) biases[layer.outputNodes[i] as number] = layer.biases[i] as number;
  }
  return { weights, biases };
}

export function packDensePlan(plan: DensePlan): PackedDensePlan {
  const layerSizes = new Uint32Array(plan.layers.length + 1);
  const activations = new Uint8Array(plan.layers.length);
  const weightStarts = new Uint32Array(plan.layers.length + 1);
  const biasStarts = new Uint32Array(plan.layers.length + 1);
  layerSizes[0] = plan.inputSize;
  let weightCount = 0;
  let biasCount = 0;
  for (let layer = 0; layer < plan.layers.length; layer++) {
    const item = plan.layers[layer] as DenseLayerPlan;
    layerSizes[layer + 1] = item.outputNodes.length;
    activations[layer] = item.activation;
    weightStarts[layer] = weightCount;
    biasStarts[layer] = biasCount;
    weightCount += item.weights.length;
    biasCount += item.biases.length;
  }
  weightStarts[plan.layers.length] = weightCount;
  biasStarts[plan.layers.length] = biasCount;
  const weights = new Float32Array(weightCount);
  const biases = new Float32Array(biasCount);
  let weightCursor = 0;
  let biasCursor = 0;
  for (const layer of plan.layers) {
    weights.set(layer.weights, weightCursor);
    biases.set(layer.biases, biasCursor);
    weightCursor += layer.weights.length;
    biasCursor += layer.biases.length;
  }
  return { layerSizes, activations, weightStarts, biasStarts, weights, biases };
}

function trainDenseIteration(plan: DensePlan, dataset: PackedDataset, optimizer: OptimizerConfig, velocities: Float32Array[], states: Array<{ weights: OptimizerState; biases: OptimizerState }>, clip: number): number {
  let totalError = 0;
  const useAdaptive = optimizer.kind === "adam" || optimizer.kind === "rmsprop";
  for (let sample = 0; sample < dataset.sampleCount; sample++) {
    const forward = denseForwardTrace(plan, dataset.inputs.subarray(sample * dataset.inputSize, (sample + 1) * dataset.inputSize));
    const output = forward.activations[forward.activations.length - 1] as Float32Array;
    const layerDeltas = plan.layers.map((layer) => new Float32Array(layer.outputNodes.length));
    for (let out = 0; out < plan.outputSize; out++) {
      const actual = output[out] as number;
      const target = dataset.targets[sample * dataset.outputSize + out] as number;
      const error = target - actual;
      totalError += error * error;
      const lastLayer = plan.layers[plan.layers.length - 1] as DenseLayerPlan;
      (layerDeltas[layerDeltas.length - 1] as Float32Array)[out] = error * derivativeFromActivation(actual, (forward.states[forward.states.length - 1] as Float32Array)[out] as number, lastLayer.activation);
    }

    for (let layerIndex = plan.layers.length - 2; layerIndex >= 0; layerIndex--) {
      const layer = plan.layers[layerIndex] as DenseLayerPlan;
      const next = plan.layers[layerIndex + 1] as DenseLayerPlan;
      const delta = layerDeltas[layerIndex] as Float32Array;
      const nextDelta = layerDeltas[layerIndex + 1] as Float32Array;
      const activation = forward.activations[layerIndex + 1] as Float32Array;
      const state = forward.states[layerIndex] as Float32Array;
      for (let node = 0; node < delta.length; node++) {
        let downstream = 0;
        for (let nextNode = 0; nextNode < nextDelta.length; nextNode++) downstream += (nextDelta[nextNode] as number) * (next.weights[nextNode * delta.length + node] as number);
        delta[node] = downstream * derivativeFromActivation(activation[node] as number, state[node] as number, layer.activation);
      }
    }

    for (let layerIndex = 0; layerIndex < plan.layers.length; layerIndex++) {
      const layer = plan.layers[layerIndex] as DenseLayerPlan;
      const input = forward.activations[layerIndex] as Float32Array;
      const delta = layerDeltas[layerIndex] as Float32Array;
      const weightGradients = new Float32Array(layer.weights.length);
      const biasGradients = new Float32Array(layer.biases.length);
      for (let out = 0; out < delta.length; out++) {
        biasGradients[out] = delta[out] as number;
        const row = out * input.length;
        for (let i = 0; i < input.length; i++) weightGradients[row + i] = (delta[out] as number) * (input[i] as number);
      }
      if (useAdaptive) {
        applyOptimizer(layer.weights, weightGradients, states[layerIndex]!.weights, optimizer, "maximize", clip);
        applyOptimizer(layer.biases, biasGradients, states[layerIndex]!.biases, optimizer, "maximize", clip);
      } else {
        const velocity = velocities[layerIndex] as Float32Array;
        for (let i = 0; i < layer.weights.length; i++) {
          const update = optimizer.rate * clamp(weightGradients[i] as number, clip) + optimizer.momentum * (velocity[i] as number);
          layer.weights[i] = (layer.weights[i] as number) + update;
          velocity[i] = update;
        }
        for (let i = 0; i < layer.biases.length; i++) layer.biases[i] = (layer.biases[i] as number) + optimizer.rate * clamp(biasGradients[i] as number, clip);
      }
    }
  }
  return totalError / (dataset.sampleCount * dataset.outputSize);
}

function denseForwardTrace(plan: DensePlan, input: Float32Array): { activations: Float32Array[]; states: Float32Array[] } {
  const activations = [Float32Array.from(input)];
  const states: Float32Array[] = [];
  for (const layer of plan.layers) {
    const previous = activations[activations.length - 1] as Float32Array;
    const state = new Float32Array(layer.outputNodes.length);
    const output = new Float32Array(layer.outputNodes.length);
    for (let out = 0; out < output.length; out++) {
      let sum = layer.biases[out] as number;
      const row = out * previous.length;
      for (let i = 0; i < previous.length; i++) sum += (previous[i] as number) * (layer.weights[row + i] as number);
      state[out] = sum;
      output[out] = activateValue(sum, layer.activation);
    }
    states.push(state);
    activations.push(output);
  }
  return { activations, states };
}

function createDenseScratch(plan: DensePlan): Float32Array[] {
  return plan.layers.map((layer) => new Float32Array(layer.outputNodes.length));
}

function denseForwardInto(plan: DensePlan, input: Float32Array, scratch: Float32Array[]): Float32Array {
  let previous = input;
  for (let layerIndex = 0; layerIndex < plan.layers.length; layerIndex++) {
    const layer = plan.layers[layerIndex] as DenseLayerPlan;
    const output = scratch[layerIndex] as Float32Array;
    for (let out = 0; out < output.length; out++) {
      let sum = layer.biases[out] as number;
      const row = out * previous.length;
      for (let i = 0; i < previous.length; i++) sum += (previous[i] as number) * (layer.weights[row + i] as number);
      output[out] = activateValue(sum, layer.activation);
    }
    previous = output;
  }
  return previous;
}

function assertDenseDataset(plan: DensePlan, dataset: PackedDataset): void {
  if (dataset.inputSize !== plan.inputSize) throw new Error(`Dense dataset input size ${dataset.inputSize} does not match plan input ${plan.inputSize}`);
  if (dataset.outputSize !== plan.outputSize) throw new Error(`Dense dataset output size ${dataset.outputSize} does not match plan output ${plan.outputSize}`);
}

function clamp(value: number, clip: number): number {
  if (!Number.isFinite(clip)) return value;
  return Math.max(-clip, Math.min(clip, value));
}
