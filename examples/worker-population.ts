import { Architect, evaluatePopulation, packDataset } from "../src/index.js";

const set = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

const population = Array.from({ length: 128 }, () => Architect.Perceptron(2, 8, 1));
const packed = packDataset(set, 2, 1);

const scores = await evaluatePopulation(population, packed, {
  mode: "workers",
  workers: 4,
  useWasm: true,
});

console.log(scores.slice(0, 5));
