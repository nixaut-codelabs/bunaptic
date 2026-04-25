export enum NodeKind {
  Input = 0,
  Hidden = 1,
  Output = 2,
  Constant = 3,
}

export enum ActivationKind {
  Logistic = 0,
  Tanh = 1,
  Identity = 2,
  Step = 3,
  Relu = 4,
  Softsign = 5,
  Sinusoid = 6,
  Gaussian = 7,
  BentIdentity = 8,
  Bipolar = 9,
  BipolarSigmoid = 10,
  HardTanh = 11,
  Absolute = 12,
  Inverse = 13,
  Selu = 14,
}

export enum GateKind {
  None = 0,
  Input = 1,
  Output = 2,
  Self = 3,
}

export enum ConnectionKind {
  Forward = 0,
  Recurrent = 1,
  Self = 2,
}

export type FloatArray = Float32Array;
export type IndexArray = Uint32Array;
export type SignedIndexArray = Int32Array;

export interface ConnectionGene {
  from: number;
  to: number;
  weight: number;
  gater: number;
  kind: ConnectionKind;
  enabled: boolean;
  innovation?: number;
}

export interface InnovationSource {
  connection(from: number, to: number, kind: ConnectionKind): number;
  split(connectionInnovation: number): { input: number; output: number };
}

export interface ModelJSON {
  version: 1;
  input: number;
  output: number;
  recurrent: boolean;
  nodes: Array<{
    kind: NodeKind;
    bias: number;
    activation: ActivationKind;
  }>;
  connections: ConnectionGene[];
  recurrentSpec?: RecurrentSpecJSON;
}

export type RecurrentKind = "lstm" | "gru";

export interface RecurrentSpecJSON {
  kind: RecurrentKind;
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
  outputActivation: ActivationKind;
  inputWeights: number[];
  hiddenWeights: number[];
  biases: number[];
  outputWeights: number[];
  outputBiases: number[];
}

export interface RecurrentSpec {
  kind: RecurrentKind;
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
  outputActivation: ActivationKind;
  inputWeights: Float32Array;
  hiddenWeights: Float32Array;
  biases: Float32Array;
  outputWeights: Float32Array;
  outputBiases: Float32Array;
}

export interface RecurrentState {
  hidden: Float32Array;
  cell?: Float32Array;
}

export interface CompiledGraph {
  input: number;
  output: number;
  recurrent: boolean;
  nodeCount: number;
  nodeKinds: Uint8Array;
  activations: Uint8Array;
  biases: Float32Array;
  from: Uint32Array;
  to: Uint32Array;
  gater: Int32Array;
  weights: Float32Array;
  connectionKinds: Uint8Array;
  incomingStarts: Uint32Array;
  incoming: Uint32Array;
  outgoingStarts: Uint32Array;
  outgoing: Uint32Array;
  outputStart: number;
}

export interface TrainingSample {
  input: ArrayLike<number>;
  output: ArrayLike<number>;
}

export interface PackedDataset {
  inputs: Float32Array;
  targets: Float32Array;
  inputSize: number;
  outputSize: number;
  sampleCount: number;
}

export interface SequenceSample {
  input: ArrayLike<number>;
  length: number;
  output: ArrayLike<number>;
}

export type SequenceTargetMode = "final" | "perStep";

export interface PackedSequenceDataset {
  values: Float32Array;
  offsets: Uint32Array;
  targets: Float32Array;
  targetOffsets?: Uint32Array;
  featureSize: number;
  outputSize: number;
  sampleCount: number;
  targetMode: SequenceTargetMode;
}

export interface RaggedBatch {
  values: Float32Array;
  offsets: Uint32Array;
  featureSize: number;
}

export interface RunSequenceOptions {
  state?: NetworkState;
  recurrentState?: RecurrentState;
  reset?: boolean;
}

export interface NetworkState {
  activations: Float32Array;
  states: Float32Array;
}

export type OptimizerKind = "sgd" | "momentum" | "rmsprop" | "adam";

export interface TrainOptions {
  iterations?: number;
  error?: number;
  rate?: number;
  momentum?: number;
  optimizer?: OptimizerKind;
  beta1?: number;
  beta2?: number;
  rho?: number;
  epsilon?: number;
  weightDecay?: number;
  shuffle?: boolean;
  clear?: boolean;
  batchSize?: number;
}

export interface TrainSequenceOptions {
  iterations?: number;
  error?: number;
  rate?: number;
  momentum?: number;
  optimizer?: OptimizerKind;
  beta1?: number;
  beta2?: number;
  rho?: number;
  epsilon?: number;
  weightDecay?: number;
  clip?: number;
  truncatedSteps?: number;
}

export interface TrainResult {
  error: number;
  iterations: number;
  time: number;
}

export interface TestResult {
  error: number;
  time: number;
}

export interface NetworkParameters {
  weights: Float32Array;
  biases: Float32Array;
}

export type FitnessFunction<TGenome> = (genome: TGenome) => number | Promise<number>;
