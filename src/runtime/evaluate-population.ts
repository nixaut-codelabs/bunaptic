import { cost, type CostFunction } from "../core/cost.js";
import { assertPackedDatasetShape, packDataset } from "../core/dataset.js";
import type { Network } from "../core/network.js";
import { topologyKey } from "../core/topology-key.js";
import type { PackedDataset, TrainingSample } from "../core/types.js";
import { loadWasmKernel, type WasmDatasetHandle, type WasmKernel } from "../wasm/kernel.js";
import { cpuCount } from "./runtime.js";
import { WorkerPool, type WorkerPoolOptions } from "./worker-pool.js";

export type PopulationEvaluationMode = "auto" | "direct" | "workers";

export interface EvaluatePopulationOptions {
  mode?: PopulationEvaluationMode;
  workers?: number;
  minWorkerPopulation?: number;
  binaryModels?: boolean;
  useWasm?: boolean;
  wasmPath?: string | URL;
  kernel?: WasmKernel;
  pool?: WorkerPool;
  cost?: CostFunction;
}

export async function evaluatePopulation(
  population: readonly Network[],
  set: readonly TrainingSample[] | PackedDataset,
  options: EvaluatePopulationOptions = {},
): Promise<number[]> {
  if (population.length === 0) return [];
  const first = population[0] as Network;
  const dataset = isPackedDataset(set) ? set : packDataset(set, first.input, first.output);
  assertPopulationShape(population, dataset);

  const mode = options.mode ?? "auto";
  if (mode === "direct" || shouldUseDirect(population.length, mode, options)) {
    return evaluateDirect(population, dataset, options);
  }

  return evaluateWithWorkers(population, dataset, options);
}

function isPackedDataset(value: readonly TrainingSample[] | PackedDataset): value is PackedDataset {
  return !Array.isArray(value);
}

function shouldUseDirect(count: number, mode: PopulationEvaluationMode, options: EvaluatePopulationOptions): boolean {
  if (mode === "workers") return false;
  return count < (options.minWorkerPopulation ?? 16);
}

async function evaluateDirect(population: readonly Network[], dataset: PackedDataset, options: EvaluatePopulationOptions): Promise<number[]> {
  const kernel = options.useWasm ? options.kernel ?? await loadWasmKernel(options.wasmPath) : null;
  const costFunction = options.cost ?? cost.MSE;
  if (kernel && costFunction === cost.MSE) {
    return evaluateDirectWasm(population, dataset, kernel);
  }
  return population.map((network) => {
    return -network.testPacked(dataset, costFunction).error;
  });
}

function evaluateDirectWasm(population: readonly Network[], dataset: PackedDataset, kernel: WasmKernel): number[] {
  const handle = kernel.createDatasetHandle(dataset);
  try {
    return evaluateDirectWasmHandle(population, handle, kernel);
  } finally {
    kernel.disposeDatasetHandle(handle);
  }
}

function evaluateDirectWasmHandle(population: readonly Network[], dataset: WasmDatasetHandle, kernel: WasmKernel): number[] {
  const scores = new Array<number>(population.length);
  const groups = new Map<string, Array<{ index: number; graph: ReturnType<Network["compiledGraph"]> }>>();
  for (let index = 0; index < population.length; index++) {
    const network = population[index] as Network;
    const graph = network.compiledGraph();
    if (graph.recurrent) {
      scores[index] = -kernel.evaluatePackedDatasetHandle(graph, dataset);
      continue;
    }
    const key = topologyKey(graph);
    const group = groups.get(key) ?? [];
    group.push({ index, graph });
    groups.set(key, group);
  }

  for (const group of groups.values()) {
    if (group.length === 1) {
      const item = group[0] as { index: number; graph: ReturnType<Network["compiledGraph"]> };
      scores[item.index] = -kernel.evaluatePackedDatasetHandle(item.graph, dataset);
      continue;
    }
    const groupScores = kernel.evaluateSameTopologyPopulationV2Handle(group.map((item) => item.graph), dataset);
    for (let i = 0; i < group.length; i++) scores[(group[i] as { index: number }).index] = groupScores[i] as number;
  }

  return scores;
}

async function evaluateWithWorkers(population: readonly Network[], dataset: PackedDataset, options: EvaluatePopulationOptions): Promise<number[]> {
  if (options.cost && options.cost !== cost.MSE) return evaluateDirect(population, dataset, options);
  const pool = options.pool ?? new WorkerPool(workerOptions(options));
  try {
    return await pool.evaluate(population, dataset);
  } finally {
    if (!options.pool) pool.terminate();
  }
}

function workerOptions(options: EvaluatePopulationOptions): WorkerPoolOptions {
  const workerOptions: WorkerPoolOptions = {
    size: options.workers ?? cpuCount(),
    binaryModels: options.binaryModels ?? true,
    useWasm: options.useWasm ?? false,
  };
  if (options.wasmPath) workerOptions.wasmPath = options.wasmPath;
  return workerOptions;
}

function assertPopulationShape(population: readonly Network[], dataset: PackedDataset): void {
  for (let i = 0; i < population.length; i++) {
    const network = population[i] as Network;
    try {
      assertPackedDatasetShape(dataset, network.input, network.output);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Population member ${i} is incompatible with dataset: ${message}`);
    }
  }
}
