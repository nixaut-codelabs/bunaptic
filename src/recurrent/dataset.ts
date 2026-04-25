import type { PackedSequenceDataset, SequenceSample, SequenceTargetMode } from "../core/types.js";

export interface PackSequenceDatasetOptions {
  targetMode?: SequenceTargetMode | "auto";
}

export type SequenceDatasetInput = readonly SequenceSample[] | PackedSequenceDataset;

export function packSequenceDataset(
  set: readonly SequenceSample[],
  featureSize: number,
  outputSize: number,
  options: PackSequenceDatasetOptions = {},
): PackedSequenceDataset {
  if (set.length === 0) throw new Error("Sequence dataset cannot be empty");
  const offsets = new Uint32Array(set.length + 1);
  let valueCount = 0;
  let targetCount = 0;
  let targetMode: SequenceTargetMode | null = options.targetMode === "auto" ? null : options.targetMode ?? null;

  for (let i = 0; i < set.length; i++) {
    const sample = set[i] as SequenceSample;
    validateSequenceSample(sample, featureSize, outputSize);
    const sampleMode = sample.output.length === outputSize ? "final" : "perStep";
    targetMode ??= sampleMode;
    if (sampleMode !== targetMode) throw new Error("Sequence dataset cannot mix final and per-step targets");
    offsets[i] = valueCount;
    valueCount += sample.input.length;
    targetCount += sample.output.length;
  }
  offsets[set.length] = valueCount;

  const values = new Float32Array(valueCount);
  const targets = new Float32Array(targetCount);
  const targetOffsets = targetMode === "perStep" ? new Uint32Array(set.length + 1) : undefined;
  let valueCursor = 0;
  let targetCursor = 0;
  for (let i = 0; i < set.length; i++) {
    const sample = set[i] as SequenceSample;
    values.set(Float32Array.from(sample.input), valueCursor);
    targets.set(Float32Array.from(sample.output), targetCursor);
    if (targetOffsets) targetOffsets[i] = targetCursor;
    valueCursor += sample.input.length;
    targetCursor += sample.output.length;
  }
  if (targetOffsets) targetOffsets[set.length] = targetCursor;

  const dataset: PackedSequenceDataset = {
    values,
    offsets,
    targets,
    featureSize,
    outputSize,
    sampleCount: set.length,
    targetMode: targetMode ?? "final",
  };
  if (targetOffsets) dataset.targetOffsets = targetOffsets;
  assertPackedSequenceDatasetShape(dataset, featureSize, outputSize);
  return dataset;
}

export function assertPackedSequenceDatasetShape(dataset: PackedSequenceDataset, featureSize: number, outputSize: number): void {
  if (dataset.featureSize !== featureSize) throw new Error(`Sequence feature size ${dataset.featureSize} does not match expected ${featureSize}`);
  if (dataset.outputSize !== outputSize) throw new Error(`Sequence output size ${dataset.outputSize} does not match expected ${outputSize}`);
  if (dataset.sampleCount <= 0) throw new Error("Sequence dataset cannot be empty");
  if (dataset.offsets.length !== dataset.sampleCount + 1) throw new Error("Sequence offsets length must equal sampleCount + 1");
  if ((dataset.offsets[0] as number) !== 0) throw new Error("Sequence offsets must start at 0");
  if ((dataset.offsets[dataset.offsets.length - 1] as number) !== dataset.values.length) throw new Error("Sequence offsets must end at values length");
  for (let i = 0; i < dataset.sampleCount; i++) {
    const start = dataset.offsets[i] as number;
    const end = dataset.offsets[i + 1] as number;
    if (end <= start) throw new Error("Sequence sample length must be positive");
    if ((end - start) % featureSize !== 0) throw new Error("Sequence value span must be divisible by feature size");
  }
  if (dataset.targetMode === "final") {
    if (dataset.targets.length !== dataset.sampleCount * outputSize) throw new Error("Final sequence target length must equal sampleCount * outputSize");
  } else {
    if (!dataset.targetOffsets) throw new Error("Per-step sequence targets require targetOffsets");
    if (dataset.targetOffsets.length !== dataset.sampleCount + 1) throw new Error("Sequence targetOffsets length must equal sampleCount + 1");
    if ((dataset.targetOffsets[0] as number) !== 0) throw new Error("Sequence targetOffsets must start at 0");
    if ((dataset.targetOffsets[dataset.targetOffsets.length - 1] as number) !== dataset.targets.length) throw new Error("Sequence targetOffsets must end at targets length");
  }
}

export function unpackSequenceDataset(dataset: PackedSequenceDataset): SequenceSample[] {
  assertPackedSequenceDatasetShape(dataset, dataset.featureSize, dataset.outputSize);
  const set: SequenceSample[] = [];
  for (let i = 0; i < dataset.sampleCount; i++) {
    const start = dataset.offsets[i] as number;
    const end = dataset.offsets[i + 1] as number;
    const targetStart = dataset.targetMode === "final" ? i * dataset.outputSize : dataset.targetOffsets![i] as number;
    const targetEnd = dataset.targetMode === "final" ? targetStart + dataset.outputSize : dataset.targetOffsets![i + 1] as number;
    set.push({
      input: dataset.values.subarray(start, end),
      length: (end - start) / dataset.featureSize,
      output: dataset.targets.subarray(targetStart, targetEnd),
    });
  }
  return set;
}

export function packedSequenceDatasetKey(dataset: PackedSequenceDataset): string {
  let hash = 2166136261;
  hash = hashString(hash, `${dataset.featureSize}:${dataset.outputSize}:${dataset.sampleCount}:${dataset.targetMode};`);
  hash = hashBytes(hash, dataset.values);
  hash = hashBytes(hash, dataset.offsets);
  hash = hashBytes(hash, dataset.targets);
  if (dataset.targetOffsets) hash = hashBytes(hash, dataset.targetOffsets);
  return hash.toString(36);
}

function validateSequenceSample(sample: SequenceSample, featureSize: number, outputSize: number): void {
  if (sample.length <= 0) throw new Error("Sequence sample length must be positive");
  if (sample.input.length !== sample.length * featureSize) throw new Error("Sequence sample input length does not match feature size");
  const finalTarget = sample.output.length === outputSize;
  const perStepTarget = sample.output.length === sample.length * outputSize;
  if (!finalTarget && !perStepTarget) throw new Error("Sequence sample output length must match final or per-step target size");
}

function hashBytes(hash: number, values: Float32Array | Uint32Array): number {
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
