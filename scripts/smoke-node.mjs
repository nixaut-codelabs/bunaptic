import { Architect, evaluatePopulation, loadWasmKernel, packDataset, runtimeTarget } from "../dist/node.js";

const set = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

const population = Array.from({ length: 4 }, () => Architect.Perceptron(2, 4, 1));
const dataset = packDataset(set, 2, 1);
const kernel = await loadWasmKernel();
const directScores = await evaluatePopulation(population, dataset, { mode: "direct", useWasm: true, kernel });
const workerScores = await evaluatePopulation(population, dataset, { mode: "workers", workers: 2, useWasm: true });

if (runtimeTarget !== "node") throw new Error(`Unexpected runtime target: ${runtimeTarget}`);
if (!validScores(directScores, population.length)) {
  throw new Error("Node smoke failed: invalid scores");
}
if (!validScores(workerScores, population.length)) {
  throw new Error("Node worker smoke failed: invalid scores");
}

console.log(JSON.stringify({ runtime: "node", direct: directScores.length, workers: workerScores.length }));

function validScores(scores, count) {
  return scores.length === count && scores.every((score) => Number.isFinite(score));
}
