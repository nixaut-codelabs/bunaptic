import { activateValue } from "../core/activation.js";
import { Rng } from "../core/rng.js";
import { ActivationKind, type RaggedBatch, type RecurrentSpec, type RecurrentSpecJSON, type RecurrentState } from "../core/types.js";

const LSTM_GATES = 4;
const GRU_GATES = 3;

export function createLstmSpec(inputSize: number, hiddenSize: number, outputSize: number, rng: Rng, outputActivation = ActivationKind.Logistic): RecurrentSpec {
  return createSpec("lstm", LSTM_GATES, inputSize, hiddenSize, outputSize, rng, outputActivation);
}

export function createGruSpec(inputSize: number, hiddenSize: number, outputSize: number, rng: Rng, outputActivation = ActivationKind.Logistic): RecurrentSpec {
  return createSpec("gru", GRU_GATES, inputSize, hiddenSize, outputSize, rng, outputActivation);
}

export function recurrentSpecToJSON(spec: RecurrentSpec): RecurrentSpecJSON {
  return {
    kind: spec.kind,
    inputSize: spec.inputSize,
    hiddenSize: spec.hiddenSize,
    outputSize: spec.outputSize,
    outputActivation: spec.outputActivation,
    inputWeights: Array.from(spec.inputWeights),
    hiddenWeights: Array.from(spec.hiddenWeights),
    biases: Array.from(spec.biases),
    outputWeights: Array.from(spec.outputWeights),
    outputBiases: Array.from(spec.outputBiases),
  };
}

export function recurrentSpecFromJSON(json: RecurrentSpecJSON): RecurrentSpec {
  return {
    kind: json.kind,
    inputSize: json.inputSize,
    hiddenSize: json.hiddenSize,
    outputSize: json.outputSize,
    outputActivation: json.outputActivation,
    inputWeights: Float32Array.from(json.inputWeights),
    hiddenWeights: Float32Array.from(json.hiddenWeights),
    biases: Float32Array.from(json.biases),
    outputWeights: Float32Array.from(json.outputWeights),
    outputBiases: Float32Array.from(json.outputBiases),
  };
}

export function createRecurrentState(spec: RecurrentSpec): RecurrentState {
  const state: RecurrentState = { hidden: new Float32Array(spec.hiddenSize) };
  if (spec.kind === "lstm") state.cell = new Float32Array(spec.hiddenSize);
  return state;
}

export function cloneRecurrentState(state: RecurrentState): RecurrentState {
  const cloned: RecurrentState = { hidden: new Float32Array(state.hidden) };
  if (state.cell) cloned.cell = new Float32Array(state.cell);
  return cloned;
}

export function resetRecurrentState(state: RecurrentState): void {
  state.hidden.fill(0);
  state.cell?.fill(0);
}

export function runRecurrentSequence(spec: RecurrentSpec, values: ArrayLike<number>, state = createRecurrentState(spec)): Float32Array {
  if (values.length % spec.inputSize !== 0) throw new Error("Sequence value count must be divisible by recurrent input size");
  assertRecurrentStateShape(spec, state);
  return spec.kind === "lstm" ? runLstmSequence(spec, values, state) : runGruSequence(spec, values, state);
}

export function runRecurrentRagged(spec: RecurrentSpec, batch: RaggedBatch): Float32Array {
  if (batch.featureSize !== spec.inputSize) throw new Error("Ragged feature size does not match recurrent input size");
  const sequenceCount = batch.offsets.length - 1;
  const output = new Float32Array(sequenceCount * spec.outputSize);
  for (let sequence = 0; sequence < sequenceCount; sequence++) {
    const start = batch.offsets[sequence] as number;
    const end = batch.offsets[sequence + 1] as number;
    const state = createRecurrentState(spec);
    const sequenceOutput = runRecurrentSequence(spec, batch.values.subarray(start, end), state);
    output.set(sequenceOutput.subarray(sequenceOutput.length - spec.outputSize), sequence * spec.outputSize);
  }
  return output;
}

function createSpec(kind: "lstm" | "gru", gates: number, inputSize: number, hiddenSize: number, outputSize: number, rng: Rng, outputActivation: ActivationKind): RecurrentSpec {
  const inputScale = Math.sqrt(2 / Math.max(1, inputSize));
  const hiddenScale = Math.sqrt(2 / Math.max(1, hiddenSize));
  return {
    kind,
    inputSize,
    hiddenSize,
    outputSize,
    outputActivation,
    inputWeights: randomArray(gates * hiddenSize * inputSize, rng, inputScale),
    hiddenWeights: randomArray(gates * hiddenSize * hiddenSize, rng, hiddenScale),
    biases: randomArray(gates * hiddenSize, rng, 0.1),
    outputWeights: randomArray(outputSize * hiddenSize, rng, hiddenScale),
    outputBiases: randomArray(outputSize, rng, 0.1),
  };
}

function assertRecurrentStateShape(spec: RecurrentSpec, state: RecurrentState): void {
  if (state.hidden.length !== spec.hiddenSize) throw new Error(`Recurrent hidden state length ${state.hidden.length} does not match hidden size ${spec.hiddenSize}`);
  if (spec.kind === "lstm" && state.cell && state.cell.length !== spec.hiddenSize) throw new Error(`Recurrent cell state length ${state.cell.length} does not match hidden size ${spec.hiddenSize}`);
}

function runLstmSequence(spec: RecurrentSpec, values: ArrayLike<number>, state: RecurrentState): Float32Array {
  const steps = values.length / spec.inputSize;
  const outputs = new Float32Array(steps * spec.outputSize);
  const hidden = state.hidden;
  const cell = state.cell ?? new Float32Array(spec.hiddenSize);
  const nextHidden = new Float32Array(spec.hiddenSize);
  const nextCell = new Float32Array(spec.hiddenSize);
  for (let step = 0; step < steps; step++) {
    for (let h = 0; h < spec.hiddenSize; h++) {
      const i = sigmoid(gateSum(spec, 0, h, values, step, hidden));
      const f = sigmoid(gateSum(spec, 1, h, values, step, hidden));
      const g = Math.tanh(gateSum(spec, 2, h, values, step, hidden));
      const o = sigmoid(gateSum(spec, 3, h, values, step, hidden));
      nextCell[h] = f * (cell[h] as number) + i * g;
      nextHidden[h] = o * Math.tanh(nextCell[h] as number);
    }
    hidden.set(nextHidden);
    cell.set(nextCell);
    projectOutput(spec, hidden, outputs, step * spec.outputSize);
  }
  return outputs;
}

function runGruSequence(spec: RecurrentSpec, values: ArrayLike<number>, state: RecurrentState): Float32Array {
  const steps = values.length / spec.inputSize;
  const outputs = new Float32Array(steps * spec.outputSize);
  const hidden = state.hidden;
  const nextHidden = new Float32Array(spec.hiddenSize);
  const reset = new Float32Array(spec.hiddenSize);
  const update = new Float32Array(spec.hiddenSize);
  for (let step = 0; step < steps; step++) {
    for (let h = 0; h < spec.hiddenSize; h++) {
      update[h] = sigmoid(gateSum(spec, 0, h, values, step, hidden));
      reset[h] = sigmoid(gateSum(spec, 1, h, values, step, hidden));
    }
    for (let h = 0; h < spec.hiddenSize; h++) {
      const n = Math.tanh(gateSum(spec, 2, h, values, step, hidden, reset));
      nextHidden[h] = (1 - (update[h] as number)) * n + (update[h] as number) * (hidden[h] as number);
    }
    hidden.set(nextHidden);
    projectOutput(spec, hidden, outputs, step * spec.outputSize);
  }
  return outputs;
}

function gateSum(spec: RecurrentSpec, gate: number, hiddenIndex: number, values: ArrayLike<number>, step: number, hidden: Float32Array, reset?: Float32Array): number {
  let sum = spec.biases[gate * spec.hiddenSize + hiddenIndex] as number;
  const inputOffset = gate * spec.hiddenSize * spec.inputSize + hiddenIndex * spec.inputSize;
  const valueOffset = step * spec.inputSize;
  for (let i = 0; i < spec.inputSize; i++) sum += Number(values[valueOffset + i]) * (spec.inputWeights[inputOffset + i] as number);
  const hiddenOffset = gate * spec.hiddenSize * spec.hiddenSize + hiddenIndex * spec.hiddenSize;
  for (let h = 0; h < spec.hiddenSize; h++) {
    const value = reset ? (reset[h] as number) * (hidden[h] as number) : hidden[h] as number;
    sum += value * (spec.hiddenWeights[hiddenOffset + h] as number);
  }
  return sum;
}

function projectOutput(spec: RecurrentSpec, hidden: Float32Array, output: Float32Array, offset: number): void {
  for (let out = 0; out < spec.outputSize; out++) {
    let sum = spec.outputBiases[out] as number;
    const weightOffset = out * spec.hiddenSize;
    for (let h = 0; h < spec.hiddenSize; h++) sum += (hidden[h] as number) * (spec.outputWeights[weightOffset + h] as number);
    output[offset + out] = activateValue(sum, spec.outputActivation);
  }
}

function randomArray(length: number, rng: Rng, scale: number): Float32Array {
  const values = new Float32Array(length);
  for (let i = 0; i < values.length; i++) values[i] = rng.range(-scale, scale);
  return values;
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}
