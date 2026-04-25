import { Architect, deriveDensePlan, loadWasmKernel, packDataset } from "../src/index.js";

const kernel = await loadWasmKernel({ simd: "auto" });
const model = Architect.Perceptron(2, 8, 4, 1);
const plan = deriveDensePlan(model.compiledGraph());
if (!plan) throw new Error("Expected Architect.Perceptron to produce a dense plan");

const dataset = packDataset([
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
], 2, 1);

const input = new Float32Array([0.25, 0.75]);
const output = kernel.denseForwardPlan(plan, input);
const error = kernel.denseEvaluatePackedDataset(plan, dataset);

console.log({ simd: kernel.simdEnabled, output: Array.from(output), error });
