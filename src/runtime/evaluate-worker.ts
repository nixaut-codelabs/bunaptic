import { Network } from "../core/network.js";
import { cost } from "../core/cost.js";
import { decodeModel } from "../core/binary.js";
import { packDataset } from "../core/dataset.js";
import { topologyKey } from "../core/topology-key.js";
import { loadWasmKernel, type WasmDatasetHandle, type WasmKernel } from "../wasm/kernel.js";
import type { ModelJSON, PackedDataset, TrainingSample } from "../core/types.js";

type ModelPayload = ModelJSON | Uint8Array;

interface CachedModelPayload {
  key: string;
  model?: ModelPayload;
}

interface EvaluateMessage {
  id: number;
  models: CachedModelPayload[];
  dataset: { key: string; value?: PackedDataset | TrainingSample[]; inputSize?: number; outputSize?: number };
  clearCache?: boolean;
  useWasm?: boolean;
  wasmPath?: string;
}

interface EvaluateResult {
  id: number;
  scores: number[];
}

interface EvaluateError {
  id: number;
  error: string;
}

const modelCache = new Map<string, Network>();
const datasetCache = new Map<string, PackedDataset>();
const wasmDatasetCache = new Map<string, WasmDatasetHandle>();
let kernelPromise: Promise<WasmKernel> | null = null;

async function evaluate(message: EvaluateMessage): Promise<EvaluateResult> {
  if (message.clearCache) {
    modelCache.clear();
    datasetCache.clear();
    await disposeWasmDatasetCache();
  }
  const kernel = message.useWasm ? await getKernel(message.wasmPath) : null;
  const dataset = resolveDataset(message.dataset);
  const networks = message.models.map(resolveModel);
  const scores = kernel ? evaluateWasm(networks, dataset, message.dataset.key, kernel) : evaluateReference(networks, dataset);
  return { id: message.id, scores };
}

function evaluateReference(networks: readonly Network[], dataset: PackedDataset): number[] {
  return networks.map((network) => -network.testPacked(dataset, cost.MSE).error);
}

function evaluateWasm(networks: readonly Network[], dataset: PackedDataset, datasetKey: string, kernel: WasmKernel): number[] {
  const datasetHandle = resolveWasmDatasetHandle(datasetKey, dataset, kernel);
  const scores = new Array<number>(networks.length);
  const groups = new Map<string, Array<{ index: number; network: Network }>>();
  for (let index = 0; index < networks.length; index++) {
    const network = networks[index] as Network;
    const graph = network.compiledGraph();
    if (graph.recurrent) {
      scores[index] = -kernel.evaluatePackedDatasetHandle(graph, datasetHandle);
      continue;
    }
    const key = topologyKey(graph);
    const group = groups.get(key) ?? [];
    group.push({ index, network });
    groups.set(key, group);
  }

  for (const group of groups.values()) {
    if (group.length === 1) {
      const item = group[0] as { index: number; network: Network };
      scores[item.index] = -kernel.evaluatePackedDatasetHandle(item.network.compiledGraph(), datasetHandle);
      continue;
    }
    const groupScores = kernel.evaluateSameTopologyPopulationV2Handle(group.map((item) => item.network.compiledGraph()), datasetHandle);
    for (let i = 0; i < group.length; i++) scores[(group[i] as { index: number }).index] = groupScores[i] as number;
  }

  return scores;
}

function resolveWasmDatasetHandle(key: string, dataset: PackedDataset, kernel: WasmKernel): WasmDatasetHandle {
  const cached = wasmDatasetCache.get(key);
  if (cached) return cached;
  const handle = kernel.createDatasetHandle(dataset);
  wasmDatasetCache.set(key, handle);
  return handle;
}

async function disposeWasmDatasetCache(): Promise<void> {
  const kernel = kernelPromise ? await kernelPromise : null;
  if (!kernel) {
    wasmDatasetCache.clear();
    return;
  }
  for (const handle of wasmDatasetCache.values()) kernel.disposeDatasetHandle(handle);
  wasmDatasetCache.clear();
}

function getKernel(path?: string): Promise<WasmKernel> {
  kernelPromise ??= loadWasmKernel(path);
  return kernelPromise;
}

function resolveModel(payload: CachedModelPayload): Network {
  const cached = modelCache.get(payload.key);
  if (cached) return cached;
  if (!payload.model) throw new Error(`Worker model cache miss: ${payload.key}`);
  const network = payload.model instanceof Uint8Array
    ? Network.fromJSON(decodeModel(payload.model))
    : Network.fromJSON(payload.model);
  modelCache.set(payload.key, network);
  return network;
}

function resolveDataset(payload: EvaluateMessage["dataset"]): PackedDataset {
  const cached = datasetCache.get(payload.key);
  if (cached) return cached;
  if (!payload.value) throw new Error(`Worker dataset cache miss: ${payload.key}`);
  const dataset = Array.isArray(payload.value)
    ? packDataset(payload.value, payload.inputSize ?? 0, payload.outputSize ?? 0)
    : payload.value;
  datasetCache.set(payload.key, dataset);
  return dataset;
}

async function handleMessage(message: EvaluateMessage): Promise<EvaluateResult | EvaluateError> {
  try {
    return await evaluate(message);
  } catch (error) {
    return { id: message.id, error: error instanceof Error ? error.message : String(error) };
  }
}

if (typeof self !== "undefined" && "postMessage" in self) {
  self.onmessage = (event: MessageEvent<EvaluateMessage>) => {
    void handleMessage(event.data).then((result) => postMessage(result));
  };
} else {
  const { parentPort } = await import("node:worker_threads");
  parentPort?.on("message", (message: EvaluateMessage) => {
    void handleMessage(message).then((result) => parentPort.postMessage(result));
  });
}
