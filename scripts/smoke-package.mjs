import { $ } from "bun";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const temp = await mkdtemp(join(tmpdir(), "bunaptic-package-smoke-"));

try {
  const packDir = join(temp, "pack");
  await mkdir(packDir);
  const packedName = (await $`bun pm pack --quiet --destination ${packDir}`.cwd(root).text()).trim().split(/\s+/).pop();
  if (!packedName) throw new Error("Package smoke failed: bun pm pack did not return a tarball name");

  const tarball = resolve(packDir, packedName);
  await Bun.write(join(temp, "package.json"), JSON.stringify({
    type: "module",
    private: true,
    dependencies: {
      bunaptic: `file:${tarball}`,
    },
    devDependencies: {
      typescript: "latest",
    },
  }, null, 2));

  await $`bun install`.cwd(temp).quiet();

  await Bun.write(join(temp, "bun-smoke.mjs"), runtimeSmoke("bunaptic", "bunaptic/bun", "bun"));
  await Bun.write(join(temp, "node-smoke.mjs"), runtimeSmoke("bunaptic", "bunaptic/node", "node"));
  await Bun.write(join(temp, "type-smoke.ts"), typeSmoke());

  await $`bun ${join(temp, "bun-smoke.mjs")}`.cwd(temp);
  await $`node ${join(temp, "node-smoke.mjs")}`.cwd(temp);
  await $`bunx tsc --module NodeNext --moduleResolution NodeNext --target ES2022 --strict --skipLibCheck --noEmit ${join(temp, "type-smoke.ts")}`.cwd(temp);

  const packageRoot = join(temp, "node_modules", "bunaptic");
  for (const expected of [
    "dist/index.js",
    "dist/bun.js",
    "dist/node.js",
    "dist/index.d.ts",
    "dist/bun.d.ts",
    "dist/node.d.ts",
    "dist/runtime/evaluate-worker.js",
    "dist/runtime/evaluate-worker.d.ts",
    "dist/bunaptic_kernel.wasm",
    "dist/bunaptic_kernel_simd.wasm",
    "README.md",
    "LICENSE",
  ]) {
    if (!await Bun.file(join(packageRoot, expected)).exists()) {
      throw new Error(`Package smoke failed: missing ${expected}`);
    }
  }

  console.log(JSON.stringify({ runtime: "package", tarball: packedName }));
} finally {
  await rm(temp, { recursive: true, force: true });
}

function runtimeSmoke(rootImport, runtimeImport, expectedRuntime) {
  return `
import { Architect, evaluatePopulation, loadWasmKernel, packDataset } from ${JSON.stringify(rootImport)};
import { runtimeTarget } from ${JSON.stringify(runtimeImport)};

const set = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

const dataset = packDataset(set, 2, 1);
const population = Array.from({ length: 6 }, (_, index) => Architect.Perceptron(2, 3 + (index % 2), 1));
const kernel = await loadWasmKernel();
const directScores = await evaluatePopulation(population, dataset, { mode: "direct", useWasm: true, kernel });
const workerScores = await evaluatePopulation(population, dataset, { mode: "workers", workers: 2, useWasm: true });

if (runtimeTarget !== ${JSON.stringify(expectedRuntime)}) throw new Error(\`Unexpected runtime target: \${runtimeTarget}\`);
if (!validScores(directScores, population.length)) throw new Error("Invalid direct scores");
if (!validScores(workerScores, population.length)) throw new Error("Invalid worker scores");

function validScores(scores, count) {
  return scores.length === count && scores.every((score) => Number.isFinite(score));
}
`;
}

function typeSmoke() {
  return `
import { Architect, evaluatePopulation, packDataset, type PackedDataset, type TrainOptions } from "bunaptic";

const set = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
];
const dataset: PackedDataset = packDataset(set, 2, 1);
const options: TrainOptions = { iterations: 1, rate: 0.1 };
const population = [Architect.Perceptron(2, 3, 1)];
await evaluatePopulation(population, dataset, { mode: "direct" });
void options;
`;
}
