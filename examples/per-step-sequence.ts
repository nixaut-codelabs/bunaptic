import { Architect, packSequenceDataset } from "../src/index.js";

const set = [
  { input: new Float32Array([0, 1, 1, 0]), length: 2, output: [0, 1] },
  { input: new Float32Array([1, 0, 0, 1]), length: 2, output: [1, 0] },
];

const packed = packSequenceDataset(set, 2, 1);
const model = Architect.GRU(2, 4, 1, { seed: 3 });
const before = model.testSequences(packed).error;
const result = model.trainSequences(packed, { optimizer: "momentum", momentum: 0.2, iterations: 40, rate: 0.02, clip: 1 });

console.log({ targetMode: packed.targetMode, before, after: result.error, iterations: result.iterations });
