import type { PackedDataset, TrainingSample } from "./types.js";

export function packDataset(set: readonly TrainingSample[], inputSize: number, outputSize: number): PackedDataset {
  if (set.length === 0) throw new Error("Dataset cannot be empty");
  const inputs = new Float32Array(set.length * inputSize);
  const targets = new Float32Array(set.length * outputSize);
  for (let sample = 0; sample < set.length; sample++) {
    const item = set[sample] as TrainingSample;
    if (item.input.length !== inputSize) throw new Error(`Dataset input size ${item.input.length} does not match ${inputSize}`);
    if (item.output.length !== outputSize) throw new Error(`Dataset output size ${item.output.length} does not match ${outputSize}`);
    for (let i = 0; i < inputSize; i++) inputs[sample * inputSize + i] = Number(item.input[i]);
    for (let i = 0; i < outputSize; i++) targets[sample * outputSize + i] = Number(item.output[i]);
  }
  return { inputs, targets, inputSize, outputSize, sampleCount: set.length };
}

export function assertPackedDatasetShape(dataset: PackedDataset, inputSize: number, outputSize: number): void {
  if (dataset.inputSize !== inputSize) throw new Error(`Packed dataset input size ${dataset.inputSize} does not match ${inputSize}`);
  if (dataset.outputSize !== outputSize) throw new Error(`Packed dataset output size ${dataset.outputSize} does not match ${outputSize}`);
  if (dataset.inputs.length !== dataset.sampleCount * inputSize) throw new Error("Packed dataset inputs length is invalid");
  if (dataset.targets.length !== dataset.sampleCount * outputSize) throw new Error("Packed dataset targets length is invalid");
}

export function packedDatasetKey(dataset: PackedDataset): string {
  let hash = 2166136261;
  hash = hashString(hash, `${dataset.inputSize}:${dataset.outputSize}:${dataset.sampleCount}`);
  hash = hashFloats(hash, dataset.inputs);
  hash = hashFloats(hash, dataset.targets);
  return hash.toString(36);
}

function hashFloats(hash: number, values: Float32Array): number {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function hashString(hash: number, value: string): number {
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}
