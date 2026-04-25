import { Architect } from "../src/index.js";

const set = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

const network = Architect.Perceptron(2, 6, 1);
const result = network.train(set, { iterations: 1000, rate: 0.5, momentum: 0.1, shuffle: true });

console.log(result);
for (const sample of set) console.log(sample.input, Array.from(network.activate(sample.input)));
