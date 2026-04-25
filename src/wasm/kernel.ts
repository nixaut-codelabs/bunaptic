import { detectRuntime } from "../runtime/runtime.js";
import { assertPackedDatasetShape, packDataset } from "../core/dataset.js";
import { packDensePlan } from "../core/dense.js";
import { hasSameTopology } from "../core/topology-key.js";
import type { CompiledGraph, NetworkParameters, NetworkState, PackedDataset, PackedSequenceDataset, RaggedBatch, RecurrentSpec, RecurrentState, TrainingSample } from "../core/types.js";
import type { DensePlan, PackedDensePlan } from "../core/dense.js";

export interface KernelExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  bunaptic_version: () => number;
  bunaptic_simd_enabled: () => number;
  bunaptic_alloc: (len: number) => number;
  bunaptic_dealloc: (ptr: number, len: number) => void;
  bunaptic_dense_forward: (
    inputPtr: number,
    weightsPtr: number,
    biasPtr: number,
    outputPtr: number,
    inputLen: number,
    outputLen: number,
    activation: number,
  ) => void;
  bunaptic_dense_forward_plan: DensePlanForwardExport;
  bunaptic_dense_forward_batch_plan: DensePlanBatchExport;
  bunaptic_dense_evaluate_dataset_plan: DensePlanEvaluateExport;
  bunaptic_mse: (targetPtr: number, outputPtr: number, len: number) => number;
  bunaptic_graph_forward: (
    inputPtr: number,
    activationsPtr: number,
    statesPtr: number,
    previousPtr: number,
    nodeKindsPtr: number,
    activationKindsPtr: number,
    biasesPtr: number,
    fromPtr: number,
    gaterPtr: number,
    weightsPtr: number,
    connectionKindsPtr: number,
    incomingStartsPtr: number,
    incomingPtr: number,
    outputPtr: number,
    inputLen: number,
    nodeCount: number,
    outputLen: number,
    connectionLen: number,
    recurrent: number,
  ) => void;
  bunaptic_graph_forward_batch: (
    inputsPtr: number,
    outputPtr: number,
    activationsPtr: number,
    statesPtr: number,
    previousPtr: number,
    nodeKindsPtr: number,
    activationKindsPtr: number,
    biasesPtr: number,
    fromPtr: number,
    gaterPtr: number,
    weightsPtr: number,
    connectionKindsPtr: number,
    incomingStartsPtr: number,
    incomingPtr: number,
    inputLen: number,
    nodeCount: number,
    outputLen: number,
    connectionLen: number,
    batchSize: number,
    recurrent: number,
  ) => void;
  bunaptic_graph_forward_sequence: (
    valuesPtr: number,
    outputPtr: number,
    activationsPtr: number,
    statesPtr: number,
    previousPtr: number,
    nodeKindsPtr: number,
    activationKindsPtr: number,
    biasesPtr: number,
    fromPtr: number,
    gaterPtr: number,
    weightsPtr: number,
    connectionKindsPtr: number,
    incomingStartsPtr: number,
    incomingPtr: number,
    inputLen: number,
    nodeCount: number,
    outputLen: number,
    connectionLen: number,
    steps: number,
    recurrent: number,
  ) => void;
  bunaptic_graph_forward_ragged: (
    valuesPtr: number,
    offsetsPtr: number,
    outputPtr: number,
    activationsPtr: number,
    statesPtr: number,
    previousPtr: number,
    nodeKindsPtr: number,
    activationKindsPtr: number,
    biasesPtr: number,
    fromPtr: number,
    gaterPtr: number,
    weightsPtr: number,
    connectionKindsPtr: number,
    incomingStartsPtr: number,
    incomingPtr: number,
    inputLen: number,
    nodeCount: number,
    outputLen: number,
    connectionLen: number,
    sequenceCount: number,
    valueCount: number,
    recurrent: number,
  ) => void;
  bunaptic_evaluate_dataset: (
    inputsPtr: number,
    targetsPtr: number,
    activationsPtr: number,
    statesPtr: number,
    previousPtr: number,
    nodeKindsPtr: number,
    activationKindsPtr: number,
    biasesPtr: number,
    fromPtr: number,
    gaterPtr: number,
    weightsPtr: number,
    connectionKindsPtr: number,
    incomingStartsPtr: number,
    incomingPtr: number,
    inputLen: number,
    nodeCount: number,
    outputLen: number,
    connectionLen: number,
    sampleCount: number,
    recurrent: number,
  ) => number;
  bunaptic_evaluate_population: (
    inputsPtr: number,
    targetsPtr: number,
    scoresPtr: number,
    nodeKindsPtr: number,
    activationKindsPtr: number,
    fromPtr: number,
    gaterPtr: number,
    connectionKindsPtr: number,
    incomingStartsPtr: number,
    incomingPtr: number,
    weightsPtr: number,
    biasesPtr: number,
    inputLen: number,
    nodeCount: number,
    outputLen: number,
    connectionLen: number,
    sampleCount: number,
    populationSize: number,
  ) => void;
  bunaptic_evaluate_population_v2: (
    inputsPtr: number,
    targetsPtr: number,
    scoresPtr: number,
    nodeKindsPtr: number,
    activationKindsPtr: number,
    fromPtr: number,
    gaterPtr: number,
    connectionKindsPtr: number,
    incomingStartsPtr: number,
    incomingPtr: number,
    weightsPtr: number,
    biasesPtr: number,
    inputLen: number,
    nodeCount: number,
    outputLen: number,
    connectionLen: number,
    sampleCount: number,
    populationSize: number,
  ) => void;
  bunaptic_train_graph: (
    inputsPtr: number,
    targetsPtr: number,
    weightsPtr: number,
    biasesPtr: number,
    previousDeltasPtr: number,
    activationsPtr: number,
    statesPtr: number,
    previousPtr: number,
    nodeDeltasPtr: number,
    nodeKindsPtr: number,
    activationKindsPtr: number,
    fromPtr: number,
    toPtr: number,
    gaterPtr: number,
    connectionKindsPtr: number,
    incomingStartsPtr: number,
    incomingPtr: number,
    outgoingStartsPtr: number,
    outgoingPtr: number,
    resultPtr: number,
    inputLen: number,
    nodeCount: number,
    outputLen: number,
    connectionLen: number,
    sampleCount: number,
    iterations: number,
    targetError: number,
    rate: number,
    momentum: number,
  ) => void;
  bunaptic_lstm_forward_sequence: RecurrentSequenceExport;
  bunaptic_lstm_forward_ragged: RecurrentRaggedExport;
  bunaptic_gru_forward_sequence: RecurrentSequenceExport;
  bunaptic_gru_forward_ragged: RecurrentRaggedExport;
  bunaptic_train_recurrent_sequences: RecurrentTrainExport;
}

type RecurrentSequenceExport = (
  valuesPtr: number,
  outputPtr: number,
  hiddenPtr: number,
  cellOrInputWeightsPtr: number,
  inputWeightsOrHiddenWeightsPtr: number,
  hiddenWeightsOrBiasesPtr: number,
  biasesOrOutputWeightsPtr: number,
  outputWeightsOrBiasesPtr: number,
  outputBiasesOrInputLen: number,
  inputLenOrHiddenLen: number,
  hiddenLenOrOutputLen: number,
  outputLenOrSteps: number,
  stepsOrActivation: number,
  outputActivation?: number,
) => void;

type DensePlanForwardExport = (
  inputPtr: number,
  outputPtr: number,
  layerSizesPtr: number,
  activationsPtr: number,
  weightStartsPtr: number,
  biasStartsPtr: number,
  weightsPtr: number,
  biasesPtr: number,
  layerCount: number,
) => void;

type DensePlanBatchExport = (
  inputsPtr: number,
  outputPtr: number,
  layerSizesPtr: number,
  activationsPtr: number,
  weightStartsPtr: number,
  biasStartsPtr: number,
  weightsPtr: number,
  biasesPtr: number,
  layerCount: number,
  batchSize: number,
) => void;

type DensePlanEvaluateExport = (
  inputsPtr: number,
  targetsPtr: number,
  layerSizesPtr: number,
  activationsPtr: number,
  weightStartsPtr: number,
  biasStartsPtr: number,
  weightsPtr: number,
  biasesPtr: number,
  layerCount: number,
  sampleCount: number,
) => number;

type RecurrentRaggedExport = (
  valuesPtr: number,
  offsetsPtr: number,
  outputPtr: number,
  hiddenPtr: number,
  cellOrInputWeightsPtr: number,
  inputWeightsOrHiddenWeightsPtr: number,
  hiddenWeightsOrBiasesPtr: number,
  biasesOrOutputWeightsPtr: number,
  outputWeightsOrBiasesPtr: number,
  outputBiasesOrInputLen: number,
  inputLenOrHiddenLen: number,
  hiddenLenOrOutputLen: number,
  outputLenOrSequenceCount: number,
  sequenceCountOrValueCount: number,
  valueCountOrActivation: number,
  outputActivation?: number,
) => void;

type RecurrentTrainExport = (
  valuesPtr: number,
  offsetsPtr: number,
  targetsPtr: number,
  targetOffsetsPtr: number,
  inputWeightsPtr: number,
  hiddenWeightsPtr: number,
  biasesPtr: number,
  outputWeightsPtr: number,
  outputBiasesPtr: number,
  resultPtr: number,
  kind: number,
  inputLen: number,
  hiddenLen: number,
  outputLen: number,
  sampleCount: number,
  valueCount: number,
  targetCount: number,
  targetMode: number,
  outputActivation: number,
  iterations: number,
  targetError: number,
  rate: number,
  momentum: number,
  clip: number,
  truncatedSteps: number,
) => void;

interface WasmGraphBuffers {
  nodeKinds: WasmAllocation;
  activations: WasmAllocation;
  biases: WasmAllocation;
  from: WasmAllocation;
  to: WasmAllocation;
  gater: WasmAllocation;
  weights: WasmAllocation;
  connectionKinds: WasmAllocation;
  incomingStarts: WasmAllocation;
  incoming: WasmAllocation;
  outgoingStarts: WasmAllocation;
  outgoing: WasmAllocation;
}

interface WasmDensePlanBuffers {
  layerSizes: WasmAllocation;
  activations: WasmAllocation;
  weightStarts: WasmAllocation;
  biasStarts: WasmAllocation;
  weights: WasmAllocation;
  biases: WasmAllocation;
  layerCount: number;
  inputSize: number;
  outputSize: number;
}

export interface WasmTrainOptions {
  iterations: number;
  targetError: number;
  rate: number;
  momentum: number;
}

export interface WasmTrainResult extends NetworkParameters {
  error: number;
  iterations: number;
}

export interface WasmRecurrentTrainOptions {
  iterations: number;
  targetError: number;
  rate: number;
  momentum: number;
  clip: number;
  truncatedSteps?: number;
}

export interface WasmRecurrentTrainResult {
  error: number;
  iterations: number;
}

export interface LoadWasmKernelOptions {
  path?: string | URL;
  simd?: "auto" | boolean;
}

interface NormalizedLoadWasmKernelOptions {
  path?: string | URL;
  simd: "auto" | boolean;
}

export interface WasmAllocation {
  ptr: number;
  bytes: number;
}

export interface WasmDatasetHandle {
  inputSize: number;
  outputSize: number;
  sampleCount: number;
  inputs: WasmAllocation;
  targets: WasmAllocation;
  disposed: boolean;
}

export interface WasmRecurrentHandle {
  kind: RecurrentSpec["kind"];
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
  outputActivation: number;
  inputWeights: WasmAllocation;
  hiddenWeights: WasmAllocation;
  biases: WasmAllocation;
  outputWeights: WasmAllocation;
  outputBiases: WasmAllocation;
  disposed: boolean;
}

export class WasmKernel {
  readonly exports: KernelExports;
  private graphCache = new WeakMap<CompiledGraph, WasmGraphBuffers>();
  private densePlanCache = new WeakMap<DensePlan, WasmDensePlanBuffers>();

  constructor(exports: KernelExports) {
    this.exports = exports;
  }

  get simdEnabled(): boolean {
    return this.exports.bunaptic_simd_enabled() === 1;
  }

  denseForward(input: Float32Array, weights: Float32Array, bias: Float32Array, activation: number): Float32Array {
    const output = new Float32Array(bias.length);
    const inputAlloc = this.copy(input);
    const weightsAlloc = this.copy(weights);
    const biasAlloc = this.copy(bias);
    const outputAlloc = this.alloc(output.byteLength);
    try {
      this.exports.bunaptic_dense_forward(inputAlloc.ptr, weightsAlloc.ptr, biasAlloc.ptr, outputAlloc.ptr, input.length, output.length, activation);
      output.set(new Float32Array(this.exports.memory.buffer, outputAlloc.ptr, output.length));
      return output;
    } finally {
      this.free(inputAlloc);
      this.free(weightsAlloc);
      this.free(biasAlloc);
      this.free(outputAlloc);
    }
  }

  denseForwardPlan(plan: DensePlan, input: Float32Array): Float32Array {
    if (input.length !== plan.inputSize) throw new Error(`Input length ${input.length} does not match dense plan input ${plan.inputSize}`);
    const buffers = this.getDensePlanBuffers(plan);
    const output = new Float32Array(plan.outputSize);
    const inputAlloc = this.copy(input);
    const outputAlloc = this.allocZeroed(output.byteLength);
    try {
      this.exports.bunaptic_dense_forward_plan(
        inputAlloc.ptr,
        outputAlloc.ptr,
        buffers.layerSizes.ptr,
        buffers.activations.ptr,
        buffers.weightStarts.ptr,
        buffers.biasStarts.ptr,
        buffers.weights.ptr,
        buffers.biases.ptr,
        buffers.layerCount,
      );
      output.set(new Float32Array(this.exports.memory.buffer, outputAlloc.ptr, output.length));
      return output;
    } finally {
      this.free(inputAlloc);
      this.free(outputAlloc);
    }
  }

  denseForwardBatchPlan(plan: DensePlan, inputs: Float32Array): Float32Array {
    if (inputs.length % plan.inputSize !== 0) throw new Error("Dense batch input length must be divisible by plan input size");
    const batchSize = inputs.length / plan.inputSize;
    const buffers = this.getDensePlanBuffers(plan);
    const output = new Float32Array(batchSize * plan.outputSize);
    const inputAlloc = this.copy(inputs);
    const outputAlloc = this.allocZeroed(output.byteLength);
    try {
      this.exports.bunaptic_dense_forward_batch_plan(
        inputAlloc.ptr,
        outputAlloc.ptr,
        buffers.layerSizes.ptr,
        buffers.activations.ptr,
        buffers.weightStarts.ptr,
        buffers.biasStarts.ptr,
        buffers.weights.ptr,
        buffers.biases.ptr,
        buffers.layerCount,
        batchSize,
      );
      output.set(new Float32Array(this.exports.memory.buffer, outputAlloc.ptr, output.length));
      return output;
    } finally {
      this.free(inputAlloc);
      this.free(outputAlloc);
    }
  }

  denseEvaluatePackedDataset(plan: DensePlan, dataset: PackedDataset): number {
    assertPackedDatasetShape(dataset, plan.inputSize, plan.outputSize);
    const buffers = this.getDensePlanBuffers(plan);
    const inputAlloc = this.copy(dataset.inputs);
    const targetAlloc = this.copy(dataset.targets);
    try {
      return this.exports.bunaptic_dense_evaluate_dataset_plan(
        inputAlloc.ptr,
        targetAlloc.ptr,
        buffers.layerSizes.ptr,
        buffers.activations.ptr,
        buffers.weightStarts.ptr,
        buffers.biasStarts.ptr,
        buffers.weights.ptr,
        buffers.biases.ptr,
        buffers.layerCount,
        dataset.sampleCount,
      );
    } finally {
      this.free(inputAlloc);
      this.free(targetAlloc);
    }
  }

  mse(target: Float32Array, output: Float32Array): number {
    const targetAlloc = this.copy(target);
    const outputAlloc = this.copy(output);
    try {
      return this.exports.bunaptic_mse(targetAlloc.ptr, outputAlloc.ptr, target.length);
    } finally {
      this.free(targetAlloc);
      this.free(outputAlloc);
    }
  }

  graphForward(graph: CompiledGraph, input: Float32Array, state: NetworkState): Float32Array {
    if (input.length !== graph.input) throw new Error(`Input length ${input.length} does not match graph input ${graph.input}`);
    const buffers = this.getGraphBuffers(graph);
    const output = new Float32Array(graph.output);
    const inputAlloc = this.copy(input);
    const activationAlloc = this.copy(state.activations);
    const stateAlloc = this.copy(state.states);
    const previousAlloc = this.allocZeroed(state.activations.byteLength);
    const outputAlloc = this.allocZeroed(output.byteLength);
    try {
      this.exports.bunaptic_graph_forward(
        inputAlloc.ptr,
        activationAlloc.ptr,
        stateAlloc.ptr,
        previousAlloc.ptr,
        buffers.nodeKinds.ptr,
        buffers.activations.ptr,
        buffers.biases.ptr,
        buffers.from.ptr,
        buffers.gater.ptr,
        buffers.weights.ptr,
        buffers.connectionKinds.ptr,
        buffers.incomingStarts.ptr,
        buffers.incoming.ptr,
        outputAlloc.ptr,
        graph.input,
        graph.nodeCount,
        graph.output,
        graph.weights.length,
        graph.recurrent ? 1 : 0,
      );
      state.activations.set(new Float32Array(this.exports.memory.buffer, activationAlloc.ptr, state.activations.length));
      state.states.set(new Float32Array(this.exports.memory.buffer, stateAlloc.ptr, state.states.length));
      output.set(new Float32Array(this.exports.memory.buffer, outputAlloc.ptr, output.length));
      return output;
    } finally {
      this.free(inputAlloc);
      this.free(activationAlloc);
      this.free(stateAlloc);
      this.free(previousAlloc);
      this.free(outputAlloc);
    }
  }

  graphForwardBatch(graph: CompiledGraph, inputs: Float32Array): Float32Array {
    if (inputs.length % graph.input !== 0) throw new Error("Batch input length must be divisible by graph input size");
    const batchSize = inputs.length / graph.input;
    const output = new Float32Array(batchSize * graph.output);
    const buffers = this.getGraphBuffers(graph);
    const inputAlloc = this.copy(inputs);
    const outputAlloc = this.allocZeroed(output.byteLength);
    const activationAlloc = this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    const stateAlloc = this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    const previousAlloc = this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    try {
      this.exports.bunaptic_graph_forward_batch(
        inputAlloc.ptr,
        outputAlloc.ptr,
        activationAlloc.ptr,
        stateAlloc.ptr,
        previousAlloc.ptr,
        buffers.nodeKinds.ptr,
        buffers.activations.ptr,
        buffers.biases.ptr,
        buffers.from.ptr,
        buffers.gater.ptr,
        buffers.weights.ptr,
        buffers.connectionKinds.ptr,
        buffers.incomingStarts.ptr,
        buffers.incoming.ptr,
        graph.input,
        graph.nodeCount,
        graph.output,
        graph.weights.length,
        batchSize,
        graph.recurrent ? 1 : 0,
      );
      output.set(new Float32Array(this.exports.memory.buffer, outputAlloc.ptr, output.length));
      return output;
    } finally {
      this.free(inputAlloc);
      this.free(outputAlloc);
      this.free(activationAlloc);
      this.free(stateAlloc);
      this.free(previousAlloc);
    }
  }

  graphForwardSequence(graph: CompiledGraph, values: Float32Array, state?: NetworkState): Float32Array {
    if (values.length % graph.input !== 0) throw new Error("Sequence value count must be divisible by graph input size");
    const steps = values.length / graph.input;
    const output = new Float32Array(steps * graph.output);
    const buffers = this.getGraphBuffers(graph);
    const valueAlloc = this.copy(values);
    const outputAlloc = this.allocZeroed(output.byteLength);
    const activationAlloc = state ? this.copy(state.activations) : this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    const stateAlloc = state ? this.copy(state.states) : this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    const previousAlloc = this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    try {
      this.exports.bunaptic_graph_forward_sequence(
        valueAlloc.ptr,
        outputAlloc.ptr,
        activationAlloc.ptr,
        stateAlloc.ptr,
        previousAlloc.ptr,
        buffers.nodeKinds.ptr,
        buffers.activations.ptr,
        buffers.biases.ptr,
        buffers.from.ptr,
        buffers.gater.ptr,
        buffers.weights.ptr,
        buffers.connectionKinds.ptr,
        buffers.incomingStarts.ptr,
        buffers.incoming.ptr,
        graph.input,
        graph.nodeCount,
        graph.output,
        graph.weights.length,
        steps,
        graph.recurrent ? 1 : 0,
      );
      output.set(new Float32Array(this.exports.memory.buffer, outputAlloc.ptr, output.length));
      if (state) {
        state.activations.set(new Float32Array(this.exports.memory.buffer, activationAlloc.ptr, state.activations.length));
        state.states.set(new Float32Array(this.exports.memory.buffer, stateAlloc.ptr, state.states.length));
      }
      return output;
    } finally {
      this.free(valueAlloc);
      this.free(outputAlloc);
      this.free(activationAlloc);
      this.free(stateAlloc);
      this.free(previousAlloc);
    }
  }

  graphForwardRagged(graph: CompiledGraph, batch: RaggedBatch): Float32Array {
    if (batch.featureSize !== graph.input) throw new Error(`Feature size ${batch.featureSize} does not match graph input ${graph.input}`);
    const sequenceCount = batch.offsets.length - 1;
    const output = new Float32Array(sequenceCount * graph.output);
    const buffers = this.getGraphBuffers(graph);
    const valueAlloc = this.copy(batch.values);
    const offsetAlloc = this.copyBytes(batch.offsets);
    const outputAlloc = this.allocZeroed(output.byteLength);
    const activationAlloc = this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    const stateAlloc = this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    const previousAlloc = this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    try {
      this.exports.bunaptic_graph_forward_ragged(
        valueAlloc.ptr,
        offsetAlloc.ptr,
        outputAlloc.ptr,
        activationAlloc.ptr,
        stateAlloc.ptr,
        previousAlloc.ptr,
        buffers.nodeKinds.ptr,
        buffers.activations.ptr,
        buffers.biases.ptr,
        buffers.from.ptr,
        buffers.gater.ptr,
        buffers.weights.ptr,
        buffers.connectionKinds.ptr,
        buffers.incomingStarts.ptr,
        buffers.incoming.ptr,
        graph.input,
        graph.nodeCount,
        graph.output,
        graph.weights.length,
        sequenceCount,
        batch.values.length,
        graph.recurrent ? 1 : 0,
      );
      output.set(new Float32Array(this.exports.memory.buffer, outputAlloc.ptr, output.length));
      return output;
    } finally {
      this.free(valueAlloc);
      this.free(offsetAlloc);
      this.free(outputAlloc);
      this.free(activationAlloc);
      this.free(stateAlloc);
      this.free(previousAlloc);
    }
  }

  evaluateDataset(graph: CompiledGraph, set: readonly TrainingSample[]): number {
    return this.evaluatePackedDataset(graph, packDataset(set, graph.input, graph.output));
  }

  evaluatePackedDataset(graph: CompiledGraph, dataset: PackedDataset): number {
    assertPackedDatasetShape(dataset, graph.input, graph.output);
    const handle = this.createDatasetHandle(dataset);
    try {
      return this.evaluatePackedDatasetHandle(graph, handle);
    } finally {
      this.disposeDatasetHandle(handle);
    }
  }

  createDatasetHandle(dataset: PackedDataset): WasmDatasetHandle {
    return {
      inputSize: dataset.inputSize,
      outputSize: dataset.outputSize,
      sampleCount: dataset.sampleCount,
      inputs: this.copy(dataset.inputs),
      targets: this.copy(dataset.targets),
      disposed: false,
    };
  }

  disposeDatasetHandle(handle: WasmDatasetHandle): void {
    if (handle.disposed) return;
    this.free(handle.inputs);
    this.free(handle.targets);
    handle.disposed = true;
  }

  evaluatePackedDatasetHandle(graph: CompiledGraph, handle: WasmDatasetHandle): number {
    assertDatasetHandle(handle, graph.input, graph.output);
    const buffers = this.getGraphBuffers(graph);
    const activationAlloc = this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    const stateAlloc = this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    const previousAlloc = this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    try {
      return this.exports.bunaptic_evaluate_dataset(
        handle.inputs.ptr,
        handle.targets.ptr,
        activationAlloc.ptr,
        stateAlloc.ptr,
        previousAlloc.ptr,
        buffers.nodeKinds.ptr,
        buffers.activations.ptr,
        buffers.biases.ptr,
        buffers.from.ptr,
        buffers.gater.ptr,
        buffers.weights.ptr,
        buffers.connectionKinds.ptr,
        buffers.incomingStarts.ptr,
        buffers.incoming.ptr,
        graph.input,
        graph.nodeCount,
        graph.output,
        graph.weights.length,
        handle.sampleCount,
        graph.recurrent ? 1 : 0,
      );
    } finally {
      this.free(activationAlloc);
      this.free(stateAlloc);
      this.free(previousAlloc);
    }
  }

  evaluateSameTopologyPopulation(graphs: readonly CompiledGraph[], dataset: PackedDataset): number[] {
    const handle = this.createDatasetHandle(dataset);
    try {
      return this.evaluateSameTopologyPopulationHandle(graphs, handle);
    } finally {
      this.disposeDatasetHandle(handle);
    }
  }

  evaluateSameTopologyPopulationHandle(graphs: readonly CompiledGraph[], handle: WasmDatasetHandle): number[] {
    if (graphs.length === 0) return [];
    const base = graphs[0] as CompiledGraph;
    if (base.recurrent) throw new Error("Same-topology WASM population evaluation supports feed-forward graphs only");
    assertDatasetHandle(handle, base.input, base.output);
    for (let i = 1; i < graphs.length; i++) {
      const graph = graphs[i] as CompiledGraph;
      assertDatasetHandle(handle, graph.input, graph.output);
      if (!hasSameTopology(base, graph)) throw new Error(`Population member ${i} does not share the same topology`);
    }

    const buffers = this.getGraphBuffers(base);
    const packedWeights = new Float32Array(graphs.length * base.weights.length);
    const packedBiases = new Float32Array(graphs.length * base.nodeCount);
    for (let i = 0; i < graphs.length; i++) {
      const graph = graphs[i] as CompiledGraph;
      packedWeights.set(graph.weights, i * base.weights.length);
      packedBiases.set(graph.biases, i * base.nodeCount);
    }

    const scores = new Float32Array(graphs.length);
    const scoresAlloc = this.allocZeroed(scores.byteLength);
    const weightsAlloc = this.copy(packedWeights);
    const biasesAlloc = this.copy(packedBiases);
    try {
      this.exports.bunaptic_evaluate_population(
        handle.inputs.ptr,
        handle.targets.ptr,
        scoresAlloc.ptr,
        buffers.nodeKinds.ptr,
        buffers.activations.ptr,
        buffers.from.ptr,
        buffers.gater.ptr,
        buffers.connectionKinds.ptr,
        buffers.incomingStarts.ptr,
        buffers.incoming.ptr,
        weightsAlloc.ptr,
        biasesAlloc.ptr,
        base.input,
        base.nodeCount,
        base.output,
        base.weights.length,
        handle.sampleCount,
        graphs.length,
      );
      scores.set(new Float32Array(this.exports.memory.buffer, scoresAlloc.ptr, scores.length));
      return Array.from(scores);
    } finally {
      this.free(scoresAlloc);
      this.free(weightsAlloc);
      this.free(biasesAlloc);
    }
  }

  evaluateSameTopologyPopulationV2Handle(graphs: readonly CompiledGraph[], handle: WasmDatasetHandle): number[] {
    if (graphs.length === 0) return [];
    const base = graphs[0] as CompiledGraph;
    if (base.recurrent) throw new Error("Same-topology WASM population evaluation supports feed-forward graphs only");
    assertDatasetHandle(handle, base.input, base.output);
    for (let i = 1; i < graphs.length; i++) {
      const graph = graphs[i] as CompiledGraph;
      assertDatasetHandle(handle, graph.input, graph.output);
      if (!hasSameTopology(base, graph)) throw new Error(`Population member ${i} does not share the same topology`);
    }

    const buffers = this.getGraphBuffers(base);
    const packedWeights = new Float32Array(graphs.length * base.weights.length);
    const packedBiases = new Float32Array(graphs.length * base.nodeCount);
    for (let conn = 0; conn < base.weights.length; conn++) {
      for (let genome = 0; genome < graphs.length; genome++) packedWeights[conn * graphs.length + genome] = (graphs[genome] as CompiledGraph).weights[conn] as number;
    }
    for (let node = 0; node < base.nodeCount; node++) {
      for (let genome = 0; genome < graphs.length; genome++) packedBiases[node * graphs.length + genome] = (graphs[genome] as CompiledGraph).biases[node] as number;
    }

    const scores = new Float32Array(graphs.length);
    const scoresAlloc = this.allocZeroed(scores.byteLength);
    const weightsAlloc = this.copy(packedWeights);
    const biasesAlloc = this.copy(packedBiases);
    try {
      this.exports.bunaptic_evaluate_population_v2(
        handle.inputs.ptr,
        handle.targets.ptr,
        scoresAlloc.ptr,
        buffers.nodeKinds.ptr,
        buffers.activations.ptr,
        buffers.from.ptr,
        buffers.gater.ptr,
        buffers.connectionKinds.ptr,
        buffers.incomingStarts.ptr,
        buffers.incoming.ptr,
        weightsAlloc.ptr,
        biasesAlloc.ptr,
        base.input,
        base.nodeCount,
        base.output,
        base.weights.length,
        handle.sampleCount,
        graphs.length,
      );
      scores.set(new Float32Array(this.exports.memory.buffer, scoresAlloc.ptr, scores.length));
      return Array.from(scores);
    } finally {
      this.free(scoresAlloc);
      this.free(weightsAlloc);
      this.free(biasesAlloc);
    }
  }

  trainGraph(graph: CompiledGraph, dataset: PackedDataset, options: WasmTrainOptions): WasmTrainResult {
    if (graph.recurrent) throw new Error("WASM graph training currently supports feed-forward graphs only");
    assertPackedDatasetShape(dataset, graph.input, graph.output);
    const buffers = this.getGraphBuffers(graph);
    const inputAlloc = this.copy(dataset.inputs);
    const targetAlloc = this.copy(dataset.targets);
    const weightsAlloc = this.copy(graph.weights);
    const biasesAlloc = this.copy(graph.biases);
    const previousDeltasAlloc = this.allocZeroed(graph.weights.byteLength);
    const activationAlloc = this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    const stateAlloc = this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    const previousAlloc = this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    const nodeDeltasAlloc = this.allocZeroed(graph.nodeCount * Float32Array.BYTES_PER_ELEMENT);
    const resultAlloc = this.allocZeroed(2 * Float32Array.BYTES_PER_ELEMENT);
    try {
      this.exports.bunaptic_train_graph(
        inputAlloc.ptr,
        targetAlloc.ptr,
        weightsAlloc.ptr,
        biasesAlloc.ptr,
        previousDeltasAlloc.ptr,
        activationAlloc.ptr,
        stateAlloc.ptr,
        previousAlloc.ptr,
        nodeDeltasAlloc.ptr,
        buffers.nodeKinds.ptr,
        buffers.activations.ptr,
        buffers.from.ptr,
        buffers.to.ptr,
        buffers.gater.ptr,
        buffers.connectionKinds.ptr,
        buffers.incomingStarts.ptr,
        buffers.incoming.ptr,
        buffers.outgoingStarts.ptr,
        buffers.outgoing.ptr,
        resultAlloc.ptr,
        graph.input,
        graph.nodeCount,
        graph.output,
        graph.weights.length,
        dataset.sampleCount,
        options.iterations,
        options.targetError,
        options.rate,
        options.momentum,
      );
      const result = new Float32Array(this.exports.memory.buffer, resultAlloc.ptr, 2);
      return {
        error: result[0] as number,
        iterations: Math.round(result[1] as number),
        weights: Float32Array.from(new Float32Array(this.exports.memory.buffer, weightsAlloc.ptr, graph.weights.length)),
        biases: Float32Array.from(new Float32Array(this.exports.memory.buffer, biasesAlloc.ptr, graph.biases.length)),
      };
    } finally {
      this.free(inputAlloc);
      this.free(targetAlloc);
      this.free(weightsAlloc);
      this.free(biasesAlloc);
      this.free(previousDeltasAlloc);
      this.free(activationAlloc);
      this.free(stateAlloc);
      this.free(previousAlloc);
      this.free(nodeDeltasAlloc);
      this.free(resultAlloc);
    }
  }

  trainRecurrentSequences(spec: RecurrentSpec, dataset: PackedSequenceDataset, options: WasmRecurrentTrainOptions): WasmRecurrentTrainResult {
    if (dataset.featureSize !== spec.inputSize) throw new Error(`Sequence feature size ${dataset.featureSize} does not match recurrent input size ${spec.inputSize}`);
    if (dataset.outputSize !== spec.outputSize) throw new Error(`Sequence output size ${dataset.outputSize} does not match recurrent output size ${spec.outputSize}`);
    const valuesAlloc = this.copy(dataset.values);
    const offsetsAlloc = this.copyBytes(dataset.offsets);
    const targetsAlloc = this.copy(dataset.targets);
    const targetOffsetsAlloc = dataset.targetOffsets ? this.copyBytes(dataset.targetOffsets) : this.allocZeroed(Uint32Array.BYTES_PER_ELEMENT);
    const inputWeightsAlloc = this.copy(spec.inputWeights);
    const hiddenWeightsAlloc = this.copy(spec.hiddenWeights);
    const biasesAlloc = this.copy(spec.biases);
    const outputWeightsAlloc = this.copy(spec.outputWeights);
    const outputBiasesAlloc = this.copy(spec.outputBiases);
    const resultAlloc = this.allocZeroed(2 * Float32Array.BYTES_PER_ELEMENT);
    try {
      this.exports.bunaptic_train_recurrent_sequences(
        valuesAlloc.ptr,
        offsetsAlloc.ptr,
        targetsAlloc.ptr,
        targetOffsetsAlloc.ptr,
        inputWeightsAlloc.ptr,
        hiddenWeightsAlloc.ptr,
        biasesAlloc.ptr,
        outputWeightsAlloc.ptr,
        outputBiasesAlloc.ptr,
        resultAlloc.ptr,
        spec.kind === "lstm" ? 1 : 2,
        spec.inputSize,
        spec.hiddenSize,
        spec.outputSize,
        dataset.sampleCount,
        dataset.values.length,
        dataset.targets.length,
        dataset.targetMode === "perStep" ? 1 : 0,
        spec.outputActivation,
        options.iterations,
        options.targetError,
        options.rate,
        options.momentum,
        options.clip,
        options.truncatedSteps ?? 0,
      );
      spec.inputWeights.set(new Float32Array(this.exports.memory.buffer, inputWeightsAlloc.ptr, spec.inputWeights.length));
      spec.hiddenWeights.set(new Float32Array(this.exports.memory.buffer, hiddenWeightsAlloc.ptr, spec.hiddenWeights.length));
      spec.biases.set(new Float32Array(this.exports.memory.buffer, biasesAlloc.ptr, spec.biases.length));
      spec.outputWeights.set(new Float32Array(this.exports.memory.buffer, outputWeightsAlloc.ptr, spec.outputWeights.length));
      spec.outputBiases.set(new Float32Array(this.exports.memory.buffer, outputBiasesAlloc.ptr, spec.outputBiases.length));
      const result = new Float32Array(this.exports.memory.buffer, resultAlloc.ptr, 2);
      return { error: result[0] as number, iterations: Math.round(result[1] as number) };
    } finally {
      this.free(valuesAlloc);
      this.free(offsetsAlloc);
      this.free(targetsAlloc);
      this.free(targetOffsetsAlloc);
      this.free(inputWeightsAlloc);
      this.free(hiddenWeightsAlloc);
      this.free(biasesAlloc);
      this.free(outputWeightsAlloc);
      this.free(outputBiasesAlloc);
      this.free(resultAlloc);
    }
  }

  recurrentForwardSequence(spec: RecurrentSpec, values: Float32Array, state?: RecurrentState): Float32Array {
    if (values.length % spec.inputSize !== 0) throw new Error("Sequence value count must be divisible by recurrent input size");
    if (state) assertRecurrentStateShape(spec, state);
    const handle = this.createRecurrentHandle(spec);
    try {
      return this.recurrentForwardSequenceHandle(handle, values, state);
    } finally {
      this.disposeRecurrentHandle(handle);
    }
  }

  createRecurrentHandle(spec: RecurrentSpec): WasmRecurrentHandle {
    return {
      kind: spec.kind,
      inputSize: spec.inputSize,
      hiddenSize: spec.hiddenSize,
      outputSize: spec.outputSize,
      outputActivation: spec.outputActivation,
      inputWeights: this.copy(spec.inputWeights),
      hiddenWeights: this.copy(spec.hiddenWeights),
      biases: this.copy(spec.biases),
      outputWeights: this.copy(spec.outputWeights),
      outputBiases: this.copy(spec.outputBiases),
      disposed: false,
    };
  }

  disposeRecurrentHandle(handle: WasmRecurrentHandle): void {
    if (handle.disposed) return;
    this.free(handle.inputWeights);
    this.free(handle.hiddenWeights);
    this.free(handle.biases);
    this.free(handle.outputWeights);
    this.free(handle.outputBiases);
    handle.disposed = true;
  }

  recurrentForwardSequenceHandle(handle: WasmRecurrentHandle, values: Float32Array, state?: RecurrentState): Float32Array {
    if (values.length % handle.inputSize !== 0) throw new Error("Sequence value count must be divisible by recurrent input size");
    if (state) assertRecurrentHandleStateShape(handle, state);
    const steps = values.length / handle.inputSize;
    const output = new Float32Array(steps * handle.outputSize);
    const valueAlloc = this.copy(values);
    const outputAlloc = this.allocZeroed(output.byteLength);
    const hiddenAlloc = state ? this.copy(state.hidden) : this.allocZeroed(handle.hiddenSize * Float32Array.BYTES_PER_ELEMENT);
    const cellAlloc = handle.kind === "lstm"
      ? state?.cell ? this.copy(state.cell) : this.allocZeroed(handle.hiddenSize * Float32Array.BYTES_PER_ELEMENT)
      : null;
    try {
      if (handle.kind === "lstm") {
        this.exports.bunaptic_lstm_forward_sequence(
          valueAlloc.ptr,
          outputAlloc.ptr,
          hiddenAlloc.ptr,
          cellAlloc!.ptr,
          handle.inputWeights.ptr,
          handle.hiddenWeights.ptr,
          handle.biases.ptr,
          handle.outputWeights.ptr,
          handle.outputBiases.ptr,
          handle.inputSize,
          handle.hiddenSize,
          handle.outputSize,
          steps,
          handle.outputActivation,
        );
      } else {
        this.exports.bunaptic_gru_forward_sequence(
          valueAlloc.ptr,
          outputAlloc.ptr,
          hiddenAlloc.ptr,
          handle.inputWeights.ptr,
          handle.hiddenWeights.ptr,
          handle.biases.ptr,
          handle.outputWeights.ptr,
          handle.outputBiases.ptr,
          handle.inputSize,
          handle.hiddenSize,
          handle.outputSize,
          steps,
          handle.outputActivation,
        );
      }
      output.set(new Float32Array(this.exports.memory.buffer, outputAlloc.ptr, output.length));
      if (state) {
        state.hidden.set(new Float32Array(this.exports.memory.buffer, hiddenAlloc.ptr, handle.hiddenSize));
        if (state.cell && cellAlloc) state.cell.set(new Float32Array(this.exports.memory.buffer, cellAlloc.ptr, handle.hiddenSize));
      }
      return output;
    } finally {
      this.free(valueAlloc);
      this.free(outputAlloc);
      this.free(hiddenAlloc);
      if (cellAlloc) this.free(cellAlloc);
    }
  }

  recurrentForwardRagged(spec: RecurrentSpec, batch: RaggedBatch): Float32Array {
    if (batch.featureSize !== spec.inputSize) throw new Error("Ragged feature size does not match recurrent input size");
    const handle = this.createRecurrentHandle(spec);
    try {
      return this.recurrentForwardRaggedHandle(handle, batch);
    } finally {
      this.disposeRecurrentHandle(handle);
    }
  }

  recurrentForwardRaggedHandle(handle: WasmRecurrentHandle, batch: RaggedBatch): Float32Array {
    if (batch.featureSize !== handle.inputSize) throw new Error("Ragged feature size does not match recurrent input size");
    const sequenceCount = batch.offsets.length - 1;
    const output = new Float32Array(sequenceCount * handle.outputSize);
    const valueAlloc = this.copy(batch.values);
    const outputAlloc = this.allocZeroed(output.byteLength);
    const offsetAlloc = this.copyBytes(batch.offsets);
    const hiddenAlloc = this.allocZeroed(handle.hiddenSize * Float32Array.BYTES_PER_ELEMENT);
    const cellAlloc = handle.kind === "lstm" ? this.allocZeroed(handle.hiddenSize * Float32Array.BYTES_PER_ELEMENT) : null;
    try {
      if (handle.kind === "lstm") {
        this.exports.bunaptic_lstm_forward_ragged(
          valueAlloc.ptr,
          offsetAlloc.ptr,
          outputAlloc.ptr,
          hiddenAlloc.ptr,
          cellAlloc!.ptr,
          handle.inputWeights.ptr,
          handle.hiddenWeights.ptr,
          handle.biases.ptr,
          handle.outputWeights.ptr,
          handle.outputBiases.ptr,
          handle.inputSize,
          handle.hiddenSize,
          handle.outputSize,
          sequenceCount,
          batch.values.length,
          handle.outputActivation,
        );
      } else {
        this.exports.bunaptic_gru_forward_ragged(
          valueAlloc.ptr,
          offsetAlloc.ptr,
          outputAlloc.ptr,
          hiddenAlloc.ptr,
          handle.inputWeights.ptr,
          handle.hiddenWeights.ptr,
          handle.biases.ptr,
          handle.outputWeights.ptr,
          handle.outputBiases.ptr,
          handle.inputSize,
          handle.hiddenSize,
          handle.outputSize,
          sequenceCount,
          batch.values.length,
          handle.outputActivation,
        );
      }
      output.set(new Float32Array(this.exports.memory.buffer, outputAlloc.ptr, output.length));
      return output;
    } finally {
      this.free(valueAlloc);
      this.free(outputAlloc);
      this.free(offsetAlloc);
      this.free(hiddenAlloc);
      if (cellAlloc) this.free(cellAlloc);
    }
  }

  disposeGraph(graph: CompiledGraph): void {
    const buffers = this.graphCache.get(graph);
    if (!buffers) return;
    for (const allocation of Object.values(buffers)) this.free(allocation);
    this.graphCache.delete(graph);
  }

  disposeDensePlan(plan: DensePlan): void {
    const buffers = this.densePlanCache.get(plan);
    if (!buffers) return;
    for (const allocation of Object.values(buffers)) {
      if (typeof allocation === "object") this.free(allocation as WasmAllocation);
    }
    this.densePlanCache.delete(plan);
  }

  private getGraphBuffers(graph: CompiledGraph): WasmGraphBuffers {
    const existing = this.graphCache.get(graph);
    if (existing) return existing;
    const buffers = {
      nodeKinds: this.copyBytes(graph.nodeKinds),
      activations: this.copyBytes(graph.activations),
      biases: this.copy(graph.biases),
      from: this.copyBytes(graph.from),
      to: this.copyBytes(graph.to),
      gater: this.copyBytes(graph.gater),
      weights: this.copy(graph.weights),
      connectionKinds: this.copyBytes(graph.connectionKinds),
      incomingStarts: this.copyBytes(graph.incomingStarts),
      incoming: this.copyBytes(graph.incoming),
      outgoingStarts: this.copyBytes(graph.outgoingStarts),
      outgoing: this.copyBytes(graph.outgoing),
    };
    this.graphCache.set(graph, buffers);
    return buffers;
  }

  private getDensePlanBuffers(plan: DensePlan): WasmDensePlanBuffers {
    const existing = this.densePlanCache.get(plan);
    if (existing) return existing;
    const packed = packDensePlan(plan);
    const buffers = this.createDensePlanBuffers(packed, plan.inputSize, plan.outputSize);
    this.densePlanCache.set(plan, buffers);
    return buffers;
  }

  private createDensePlanBuffers(packed: PackedDensePlan, inputSize: number, outputSize: number): WasmDensePlanBuffers {
    return {
      layerSizes: this.copyBytes(packed.layerSizes),
      activations: this.copyBytes(packed.activations),
      weightStarts: this.copyBytes(packed.weightStarts),
      biasStarts: this.copyBytes(packed.biasStarts),
      weights: this.copy(packed.weights),
      biases: this.copy(packed.biases),
      layerCount: packed.activations.length,
      inputSize,
      outputSize,
    };
  }

  private alloc(bytes: number): WasmAllocation {
    const allocatedBytes = Math.max(1, bytes);
    return { ptr: this.exports.bunaptic_alloc(allocatedBytes), bytes: allocatedBytes };
  }

  private allocZeroed(bytes: number): WasmAllocation {
    const allocation = this.alloc(bytes);
    new Uint8Array(this.exports.memory.buffer, allocation.ptr, allocation.bytes).fill(0);
    return allocation;
  }

  private free(allocation: WasmAllocation): void {
    this.exports.bunaptic_dealloc(allocation.ptr, allocation.bytes);
  }

  private copy(values: Float32Array): WasmAllocation {
    const allocation = this.alloc(values.byteLength);
    new Uint8Array(this.exports.memory.buffer, allocation.ptr, values.byteLength).set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
    return allocation;
  }

  private copyBytes(values: Uint8Array | Uint32Array | Int32Array): WasmAllocation {
    const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    const allocation = this.alloc(bytes.byteLength);
    new Uint8Array(this.exports.memory.buffer, allocation.ptr, bytes.byteLength).set(bytes);
    return allocation;
  }

}

function assertDatasetHandle(handle: WasmDatasetHandle, inputSize: number, outputSize: number): void {
  if (handle.disposed) throw new Error("WASM dataset handle has been disposed");
  if (handle.inputSize !== inputSize) throw new Error(`Dataset handle input size ${handle.inputSize} does not match graph input ${inputSize}`);
  if (handle.outputSize !== outputSize) throw new Error(`Dataset handle output size ${handle.outputSize} does not match graph output ${outputSize}`);
}

function assertRecurrentStateShape(spec: RecurrentSpec, state: RecurrentState): void {
  if (state.hidden.length !== spec.hiddenSize) {
    throw new Error(`Recurrent hidden state length ${state.hidden.length} does not match hidden size ${spec.hiddenSize}`);
  }
  if (spec.kind === "lstm" && state.cell && state.cell.length !== spec.hiddenSize) {
    throw new Error(`Recurrent cell state length ${state.cell.length} does not match hidden size ${spec.hiddenSize}`);
  }
}

function assertRecurrentHandleStateShape(handle: WasmRecurrentHandle, state: RecurrentState): void {
  if (handle.disposed) throw new Error("WASM recurrent handle has been disposed");
  if (state.hidden.length !== handle.hiddenSize) {
    throw new Error(`Recurrent hidden state length ${state.hidden.length} does not match hidden size ${handle.hiddenSize}`);
  }
  if (handle.kind === "lstm" && state.cell && state.cell.length !== handle.hiddenSize) {
    throw new Error(`Recurrent cell state length ${state.cell.length} does not match hidden size ${handle.hiddenSize}`);
  }
}

export async function loadWasmKernel(source?: string | URL | LoadWasmKernelOptions): Promise<WasmKernel> {
  const bytes = await readWasmBytes(source);
  const instance = await WebAssembly.instantiate(bytes, {});
  return new WasmKernel(instance.instance.exports as KernelExports);
}

async function readWasmBytes(source?: string | URL | LoadWasmKernelOptions): Promise<ArrayBuffer> {
  const options = normalizeLoadOptions(source);
  const sources = options.path ? [normalizeWasmSource(options.path)] : defaultWasmSources(options.simd);
  const errors: string[] = [];
  for (const source of sources) {
    try {
      if (detectRuntime() === "bun") {
        const file = Bun.file(source);
        if (await file.exists()) return file.arrayBuffer();
        errors.push(`${source.toString()}: not found`);
        continue;
      }
      const { readFile } = await import("node:fs/promises");
      const buffer = await readFile(source);
      return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
    } catch (error) {
      errors.push(`${source.toString()}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  throw new Error(`Unable to load Bunaptic WASM kernel. Tried: ${errors.join("; ")}`);
}

function defaultWasmSources(simd: "auto" | boolean): Array<string | URL> {
  const scalar = [
    new URL("./bunaptic_kernel.wasm", import.meta.url),
    new URL("../bunaptic_kernel.wasm", import.meta.url),
    new URL("../../dist/bunaptic_kernel.wasm", import.meta.url),
    new URL("../../rust/kernel/target/wasm32-unknown-unknown/release/bunaptic_kernel.wasm", import.meta.url),
  ];
  const simdSources = [
    new URL("./bunaptic_kernel_simd.wasm", import.meta.url),
    new URL("../bunaptic_kernel_simd.wasm", import.meta.url),
    new URL("../../dist/bunaptic_kernel_simd.wasm", import.meta.url),
  ];
  if (simd === true) return simdSources;
  if (simd === false) return scalar;
  return [...simdSources, ...scalar];
}

function normalizeLoadOptions(source?: string | URL | LoadWasmKernelOptions): NormalizedLoadWasmKernelOptions {
  if (!source) return { simd: "auto" };
  if (typeof source === "string" || source instanceof URL) return { path: source, simd: false };
  const simd = source.simd ?? "auto";
  return source.path ? { path: source.path, simd } : { simd };
}

function normalizeWasmSource(path: string | URL): string | URL {
  if (path instanceof URL) return path;
  return path.startsWith("file:") ? new URL(path) : path;
}
