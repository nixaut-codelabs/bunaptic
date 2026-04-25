import { Neat, packDataset } from "../src/index.js";

const set = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

const neat = new Neat(2, 1, () => 0, {
  seed: 42,
  popsize: 64,
  elitism: 4,
  mutationRate: 0.6,
  mutationAmount: 2,
  dataset: packDataset(set, 2, 1),
  evaluator: { mode: "auto", useWasm: true },
});

for (let generation = 0; generation < 10; generation++) {
  const best = await neat.evolve();
  const error = best.test(set).error;
  console.log({ generation, error, average: neat.getAverage() });
}
