import { activateValue, derivativeFromActivation } from "../core/activation.js";
import { applyOptimizer, createOptimizerState, optimizerConfig, type OptimizerConfig, type OptimizerState } from "../core/optimizer.js";
import { type PackedSequenceDataset, type RecurrentSpec, type SequenceSample, type TrainSequenceOptions } from "../core/types.js";
import { assertPackedSequenceDatasetShape, unpackSequenceDataset } from "./dataset.js";

const LSTM_GATES = 4;
const GRU_GATES = 3;

export function testRecurrentSequences(spec: RecurrentSpec, set: readonly SequenceSample[] | PackedSequenceDataset): number {
  const samples = resolveSequenceSet(spec, set);
  let error = 0;
  for (const sample of samples) error += sampleError(spec, sample).error;
  return error / samples.length;
}

export function trainRecurrentSequences(spec: RecurrentSpec, set: readonly SequenceSample[] | PackedSequenceDataset, options: TrainSequenceOptions = {}): { error: number; iterations: number } {
  const samples = resolveSequenceSet(spec, set);
  if (samples.length === 0) throw new Error("Sequence training set cannot be empty");
  const iterations = options.iterations ?? 100;
  const targetError = options.error ?? 0.01;
  const optimizer = optimizerConfig(options, 0.01);
  const clip = options.clip ?? 1;
  const state = createRecurrentOptimizerState(spec, optimizer);
  let error = Infinity;
  let iteration = 0;
  while (iteration < iterations && error > targetError) {
    error = 0;
    for (const sample of samples) error += spec.kind === "lstm" ? trainLstm(spec, sample, optimizer, state, clip, options.truncatedSteps) : trainGru(spec, sample, optimizer, state, clip, options.truncatedSteps);
    error /= samples.length;
    iteration++;
  }
  return { error, iterations: iteration };
}

function resolveSequenceSet(spec: RecurrentSpec, set: readonly SequenceSample[] | PackedSequenceDataset): readonly SequenceSample[] {
  if (!isPackedSequenceDataset(set)) return set;
  assertPackedSequenceDatasetShape(set, spec.inputSize, spec.outputSize);
  return unpackSequenceDataset(set);
}

function isPackedSequenceDataset(set: readonly SequenceSample[] | PackedSequenceDataset): set is PackedSequenceDataset {
  return !Array.isArray(set) && "offsets" in set && "targets" in set;
}

function trainLstm(spec: RecurrentSpec, sample: SequenceSample, optimizer: OptimizerConfig, state: RecurrentOptimizerState, clip: number, truncatedSteps?: number): number {
  const cache = forwardLstm(spec, sample);
  const grad = gradients(spec);
  const dHiddenByStep = outputGradients(spec, cache, sample, grad);
  const dHidden = new Float32Array(spec.hiddenSize);
  let dCell = new Float32Array(spec.hiddenSize);
  const start = backStart(cache.steps, truncatedSteps);
  for (let step = cache.steps - 1; step >= start; step--) {
    addInto(dHidden, dHiddenByStep.subarray(step * spec.hiddenSize, (step + 1) * spec.hiddenSize));
    const prevHidden = cache.hidden.subarray(step * spec.hiddenSize, (step + 1) * spec.hiddenSize);
    const prevCell = cache.cell.subarray(step * spec.hiddenSize, (step + 1) * spec.hiddenSize);
    const cell = cache.cell.subarray((step + 1) * spec.hiddenSize, (step + 2) * spec.hiddenSize);
    const inputOffset = step * spec.inputSize;
    const nextHidden = new Float32Array(spec.hiddenSize);
    const nextCell = new Float32Array(spec.hiddenSize);
    for (let h = 0; h < spec.hiddenSize; h++) {
      const i = cache.gates[(step * LSTM_GATES + 0) * spec.hiddenSize + h] as number;
      const f = cache.gates[(step * LSTM_GATES + 1) * spec.hiddenSize + h] as number;
      const g = cache.gates[(step * LSTM_GATES + 2) * spec.hiddenSize + h] as number;
      const o = cache.gates[(step * LSTM_GATES + 3) * spec.hiddenSize + h] as number;
      const tanhCell = Math.tanh(cell[h] as number);
      const dc = (dCell[h] as number) + (dHidden[h] as number) * o * (1 - tanhCell * tanhCell);
      const dz = [dc * g * i * (1 - i), dc * (prevCell[h] as number) * f * (1 - f), dc * i * (1 - g * g), (dHidden[h] as number) * tanhCell * o * (1 - o)];
      nextCell[h] = (nextCell[h] as number) + dc * f;
      for (let gate = 0; gate < LSTM_GATES; gate++) addGateGrad(spec, grad, gate, h, dz[gate] as number, sample.input, inputOffset, prevHidden, nextHidden);
    }
    dHidden.set(nextHidden);
    dCell = nextCell;
  }
  applyGrad(spec, grad, optimizer, state, clip);
  return cache.error;
}

function trainGru(spec: RecurrentSpec, sample: SequenceSample, optimizer: OptimizerConfig, state: RecurrentOptimizerState, clip: number, truncatedSteps?: number): number {
  const cache = forwardGru(spec, sample);
  const grad = gradients(spec);
  const dHiddenByStep = outputGradients(spec, cache, sample, grad);
  const dHidden = new Float32Array(spec.hiddenSize);
  const start = backStart(cache.steps, truncatedSteps);
  for (let step = cache.steps - 1; step >= start; step--) {
    addInto(dHidden, dHiddenByStep.subarray(step * spec.hiddenSize, (step + 1) * spec.hiddenSize));
    const prevHidden = cache.hidden.subarray(step * spec.hiddenSize, (step + 1) * spec.hiddenSize);
    const inputOffset = step * spec.inputSize;
    const nextHidden = new Float32Array(spec.hiddenSize);
    const resetGrad = new Float32Array(spec.hiddenSize);
    const updatePre = new Float32Array(spec.hiddenSize);
    const resetPre = new Float32Array(spec.hiddenSize);
    const candidatePre = new Float32Array(spec.hiddenSize);
    for (let h = 0; h < spec.hiddenSize; h++) {
      const z = cache.gates[(step * GRU_GATES + 0) * spec.hiddenSize + h] as number;
      const r = cache.gates[(step * GRU_GATES + 1) * spec.hiddenSize + h] as number;
      const n = cache.gates[(step * GRU_GATES + 2) * spec.hiddenSize + h] as number;
      nextHidden[h] = (nextHidden[h] as number) + (dHidden[h] as number) * z;
      updatePre[h] = (dHidden[h] as number) * ((prevHidden[h] as number) - n) * z * (1 - z);
      candidatePre[h] = (dHidden[h] as number) * (1 - z) * (1 - n * n);
      addGateGrad(spec, grad, 2, h, candidatePre[h] as number, sample.input, inputOffset, scaled(prevHidden, r), nextHidden, resetGrad);
    }
    for (let h = 0; h < spec.hiddenSize; h++) {
      const r = cache.gates[(step * GRU_GATES + 1) * spec.hiddenSize + h] as number;
      resetPre[h] = (resetGrad[h] as number) * r * (1 - r);
      addGateGrad(spec, grad, 0, h, updatePre[h] as number, sample.input, inputOffset, prevHidden, nextHidden);
      addGateGrad(spec, grad, 1, h, resetPre[h] as number, sample.input, inputOffset, prevHidden, nextHidden);
    }
    dHidden.set(nextHidden);
  }
  applyGrad(spec, grad, optimizer, state, clip);
  return cache.error;
}

function forwardLstm(spec: RecurrentSpec, sample: SequenceSample): ForwardCache {
  validate(spec, sample);
  const hidden = new Float32Array((sample.length + 1) * spec.hiddenSize);
  const cell = new Float32Array((sample.length + 1) * spec.hiddenSize);
  const gates = new Float32Array(sample.length * LSTM_GATES * spec.hiddenSize);
  for (let step = 0; step < sample.length; step++) {
    const prevHidden = hidden.subarray(step * spec.hiddenSize, (step + 1) * spec.hiddenSize);
    const prevCell = cell.subarray(step * spec.hiddenSize, (step + 1) * spec.hiddenSize);
    const nextHidden = hidden.subarray((step + 1) * spec.hiddenSize, (step + 2) * spec.hiddenSize);
    const nextCell = cell.subarray((step + 1) * spec.hiddenSize, (step + 2) * spec.hiddenSize);
    for (let h = 0; h < spec.hiddenSize; h++) {
      const i = sigmoid(gateSum(spec, 0, h, sample.input, step, prevHidden));
      const f = sigmoid(gateSum(spec, 1, h, sample.input, step, prevHidden));
      const g = Math.tanh(gateSum(spec, 2, h, sample.input, step, prevHidden));
      const o = sigmoid(gateSum(spec, 3, h, sample.input, step, prevHidden));
      gates[(step * LSTM_GATES + 0) * spec.hiddenSize + h] = i;
      gates[(step * LSTM_GATES + 1) * spec.hiddenSize + h] = f;
      gates[(step * LSTM_GATES + 2) * spec.hiddenSize + h] = g;
      gates[(step * LSTM_GATES + 3) * spec.hiddenSize + h] = o;
      nextCell[h] = f * (prevCell[h] as number) + i * g;
      nextHidden[h] = o * Math.tanh(nextCell[h] as number);
    }
  }
  return finishForward(spec, sample, hidden, cell, gates);
}

function forwardGru(spec: RecurrentSpec, sample: SequenceSample): ForwardCache {
  validate(spec, sample);
  const hidden = new Float32Array((sample.length + 1) * spec.hiddenSize);
  const gates = new Float32Array(sample.length * GRU_GATES * spec.hiddenSize);
  for (let step = 0; step < sample.length; step++) {
    const prevHidden = hidden.subarray(step * spec.hiddenSize, (step + 1) * spec.hiddenSize);
    const nextHidden = hidden.subarray((step + 1) * spec.hiddenSize, (step + 2) * spec.hiddenSize);
    const reset = new Float32Array(spec.hiddenSize);
    const update = new Float32Array(spec.hiddenSize);
    for (let h = 0; h < spec.hiddenSize; h++) {
      update[h] = sigmoid(gateSum(spec, 0, h, sample.input, step, prevHidden));
      reset[h] = sigmoid(gateSum(spec, 1, h, sample.input, step, prevHidden));
    }
    for (let h = 0; h < spec.hiddenSize; h++) {
      const n = Math.tanh(gateSum(spec, 2, h, sample.input, step, prevHidden, reset));
      gates[(step * GRU_GATES + 0) * spec.hiddenSize + h] = update[h] as number;
      gates[(step * GRU_GATES + 1) * spec.hiddenSize + h] = reset[h] as number;
      gates[(step * GRU_GATES + 2) * spec.hiddenSize + h] = n;
      nextHidden[h] = (1 - (update[h] as number)) * n + (update[h] as number) * (prevHidden[h] as number);
    }
  }
  return finishForward(spec, sample, hidden, new Float32Array(0), gates);
}

function finishForward(spec: RecurrentSpec, sample: SequenceSample, hidden: Float32Array, cell: Float32Array, gates: Float32Array): ForwardCache {
  const targetMode = sample.output.length === spec.outputSize ? "final" : "perStep";
  const outputCount = targetMode === "final" ? 1 : sample.length;
  const preOutput = new Float32Array(outputCount * spec.outputSize);
  const output = new Float32Array(outputCount * spec.outputSize);
  let error = 0;
  for (let item = 0; item < outputCount; item++) {
    const hiddenStep = targetMode === "final" ? sample.length : item + 1;
    const hiddenValues = hidden.subarray(hiddenStep * spec.hiddenSize, (hiddenStep + 1) * spec.hiddenSize);
    for (let out = 0; out < spec.outputSize; out++) {
      let sum = spec.outputBiases[out] as number;
      for (let h = 0; h < spec.hiddenSize; h++) sum += (hiddenValues[h] as number) * (spec.outputWeights[out * spec.hiddenSize + h] as number);
      const offset = item * spec.outputSize + out;
      preOutput[offset] = sum;
      output[offset] = activateValue(sum, spec.outputActivation);
      const delta = (output[offset] as number) - Number(sample.output[offset]);
      error += delta * delta;
    }
  }
  return { steps: sample.length, targetMode, hidden, cell, gates, preOutput, output, error: error / (outputCount * spec.outputSize) };
}

function outputGradients(spec: RecurrentSpec, cache: ForwardCache, sample: SequenceSample, grad: Gradients): Float32Array {
  const dHiddenByStep = new Float32Array(cache.steps * spec.hiddenSize);
  const outputCount = cache.targetMode === "final" ? 1 : cache.steps;
  const scale = 2 / (outputCount * spec.outputSize);
  for (let item = 0; item < outputCount; item++) {
    const hiddenStep = cache.targetMode === "final" ? cache.steps : item + 1;
    const hidden = cache.hidden.subarray(hiddenStep * spec.hiddenSize, (hiddenStep + 1) * spec.hiddenSize);
    const dHidden = dHiddenByStep.subarray((hiddenStep - 1) * spec.hiddenSize, hiddenStep * spec.hiddenSize);
    for (let out = 0; out < spec.outputSize; out++) {
      const offset = item * spec.outputSize + out;
      const dOut = (scale * ((cache.output[offset] as number) - Number(sample.output[offset]))) * derivativeFromActivation(cache.output[offset] as number, cache.preOutput[offset] as number, spec.outputActivation);
      grad.outputBiases[out] = (grad.outputBiases[out] as number) + dOut;
      for (let h = 0; h < spec.hiddenSize; h++) {
        grad.outputWeights[out * spec.hiddenSize + h] = (grad.outputWeights[out * spec.hiddenSize + h] as number) + dOut * (hidden[h] as number);
        dHidden[h] = (dHidden[h] as number) + dOut * (spec.outputWeights[out * spec.hiddenSize + h] as number);
      }
    }
  }
  return dHiddenByStep;
}

function addGateGrad(spec: RecurrentSpec, grad: Gradients, gate: number, h: number, delta: number, input: ArrayLike<number>, inputOffset: number, prevHidden: Float32Array, nextHidden: Float32Array, resetGrad?: Float32Array): void {
  grad.biases[gate * spec.hiddenSize + h] = (grad.biases[gate * spec.hiddenSize + h] as number) + delta;
  const inputWeightOffset = gate * spec.hiddenSize * spec.inputSize + h * spec.inputSize;
  for (let i = 0; i < spec.inputSize; i++) grad.inputWeights[inputWeightOffset + i] = (grad.inputWeights[inputWeightOffset + i] as number) + delta * Number(input[inputOffset + i]);
  const hiddenWeightOffset = gate * spec.hiddenSize * spec.hiddenSize + h * spec.hiddenSize;
  for (let j = 0; j < spec.hiddenSize; j++) {
    grad.hiddenWeights[hiddenWeightOffset + j] = (grad.hiddenWeights[hiddenWeightOffset + j] as number) + delta * (prevHidden[j] as number);
    const back = delta * (spec.hiddenWeights[hiddenWeightOffset + j] as number);
    if (resetGrad) resetGrad[j] = (resetGrad[j] as number) + back * (prevHidden[j] as number);
    nextHidden[j] = (nextHidden[j] as number) + back;
  }
}

function applyGrad(spec: RecurrentSpec, grad: Gradients, optimizer: OptimizerConfig, state: RecurrentOptimizerState, clip: number): void {
  applyOptimizer(spec.inputWeights, grad.inputWeights, state.inputWeights, optimizer, "minimize", clip);
  applyOptimizer(spec.hiddenWeights, grad.hiddenWeights, state.hiddenWeights, optimizer, "minimize", clip);
  applyOptimizer(spec.biases, grad.biases, state.biases, optimizer, "minimize", clip);
  applyOptimizer(spec.outputWeights, grad.outputWeights, state.outputWeights, optimizer, "minimize", clip);
  applyOptimizer(spec.outputBiases, grad.outputBiases, state.outputBiases, optimizer, "minimize", clip);
}

function gradients(spec: RecurrentSpec): Gradients {
  return { inputWeights: new Float32Array(spec.inputWeights.length), hiddenWeights: new Float32Array(spec.hiddenWeights.length), biases: new Float32Array(spec.biases.length), outputWeights: new Float32Array(spec.outputWeights.length), outputBiases: new Float32Array(spec.outputBiases.length) };
}

function createRecurrentOptimizerState(spec: RecurrentSpec, optimizer: OptimizerConfig): RecurrentOptimizerState {
  return {
    inputWeights: createOptimizerState(spec.inputWeights.length, optimizer.kind),
    hiddenWeights: createOptimizerState(spec.hiddenWeights.length, optimizer.kind),
    biases: createOptimizerState(spec.biases.length, optimizer.kind),
    outputWeights: createOptimizerState(spec.outputWeights.length, optimizer.kind),
    outputBiases: createOptimizerState(spec.outputBiases.length, optimizer.kind),
  };
}

function gateSum(spec: RecurrentSpec, gate: number, h: number, values: ArrayLike<number>, step: number, hidden: Float32Array, reset?: Float32Array): number {
  let sum = spec.biases[gate * spec.hiddenSize + h] as number;
  const inputOffset = gate * spec.hiddenSize * spec.inputSize + h * spec.inputSize;
  for (let i = 0; i < spec.inputSize; i++) sum += Number(values[step * spec.inputSize + i]) * (spec.inputWeights[inputOffset + i] as number);
  const hiddenOffset = gate * spec.hiddenSize * spec.hiddenSize + h * spec.hiddenSize;
  for (let j = 0; j < spec.hiddenSize; j++) sum += (reset ? (reset[j] as number) * (hidden[j] as number) : hidden[j] as number) * (spec.hiddenWeights[hiddenOffset + j] as number);
  return sum;
}

function sampleError(spec: RecurrentSpec, sample: SequenceSample): { error: number } {
  return spec.kind === "lstm" ? forwardLstm(spec, sample) : forwardGru(spec, sample);
}

function validate(spec: RecurrentSpec, sample: SequenceSample): void {
  if (sample.length <= 0) throw new Error("Sequence sample length must be positive");
  if (sample.input.length !== sample.length * spec.inputSize) throw new Error("Sequence sample input length does not match recurrent input size");
  if (sample.output.length !== spec.outputSize && sample.output.length !== sample.length * spec.outputSize) throw new Error("Sequence sample output length must match final or per-step recurrent output size");
}

function addInto(target: Float32Array, source: Float32Array): void {
  for (let i = 0; i < target.length; i++) target[i] = (target[i] as number) + (source[i] as number);
}

function scaled(values: Float32Array, scale: number): Float32Array {
  const output = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) output[i] = (values[i] as number) * scale;
  return output;
}

function backStart(steps: number, truncatedSteps?: number): number {
  return typeof truncatedSteps === "number" ? Math.max(0, steps - truncatedSteps) : 0;
}

function clamp(value: number, clip: number): number {
  return Math.max(-clip, Math.min(clip, value));
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

interface ForwardCache {
  steps: number;
  targetMode: "final" | "perStep";
  hidden: Float32Array;
  cell: Float32Array;
  gates: Float32Array;
  preOutput: Float32Array;
  output: Float32Array;
  error: number;
}

interface Gradients {
  inputWeights: Float32Array;
  hiddenWeights: Float32Array;
  biases: Float32Array;
  outputWeights: Float32Array;
  outputBiases: Float32Array;
}

interface RecurrentOptimizerState {
  inputWeights: OptimizerState;
  hiddenWeights: OptimizerState;
  biases: OptimizerState;
  outputWeights: OptimizerState;
  outputBiases: OptimizerState;
}
