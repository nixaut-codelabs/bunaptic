import type { Network } from "../core/network.js";
import { packDataset, packedDatasetKey } from "../core/dataset.js";
import type { PackedDataset, TrainingSample } from "../core/types.js";
import { cpuCount } from "./runtime.js";

interface WorkerJob {
  id: number;
  models: Array<{ key: string; model?: ReturnType<Network["toJSON"]> | Uint8Array }>;
  dataset: { key: string; value?: PackedDataset; inputSize?: number; outputSize?: number };
  clearCache?: boolean;
  useWasm?: boolean;
  wasmPath?: string;
}

interface WorkerResponse {
  id: number;
  scores?: number[];
  error?: string;
}

export interface WorkerPoolOptions {
  size?: number;
  workerUrl?: URL | string;
  binaryModels?: boolean;
  useWasm?: boolean;
  wasmPath?: string | URL;
}

export class WorkerPool {
  private workers: WorkerLike[] = [];
  private cursor = 0;
  private nextId = 1;
  private binaryModels: boolean;
  private useWasm: boolean;
  private wasmPath: string | URL | undefined;
  private workerModels = new Map<WorkerLike, Set<string>>();
  private workerDatasets = new Map<WorkerLike, Set<string>>();
  private pending = new Map<number, { resolve: (scores: number[]) => void; reject: (error: Error) => void }>();

  constructor(options: WorkerPoolOptions = {}) {
    const size = options.size ?? cpuCount();
    const workerUrl = options.workerUrl ?? defaultWorkerUrl();
    this.binaryModels = options.binaryModels ?? true;
    this.useWasm = options.useWasm ?? false;
    this.wasmPath = options.wasmPath;
    for (let i = 0; i < size; i++) this.addWorker(workerUrl);
  }

  async evaluate(population: readonly Network[], set: TrainingSample[] | PackedDataset): Promise<number[]> {
    if (population.length === 0) return [];
    const dataset = Array.isArray(set) ? packDataset(set, population[0]!.input, population[0]!.output) : set;
    const chunks = chunk(population, this.workers.length);
    const results = await Promise.all(chunks.map((items) => this.runEvaluate(items, dataset)));
    return results.flat();
  }

  async cacheDataset(dataset: PackedDataset): Promise<string> {
    const key = packedDatasetKey(dataset);
    await Promise.all(this.workers.map((worker) => {
      const known = this.workerDatasets.get(worker) ?? new Set<string>();
      this.workerDatasets.set(worker, known);
      known.add(key);
      return this.run({ id: this.nextId++, models: [], dataset: { key, value: dataset } }, worker);
    }));
    return key;
  }

  clearWorkerCache(): void {
    for (let i = 0; i < this.workers.length; i++) {
      const worker = this.workers[i] as WorkerLike;
      this.workerModels.get(worker)?.clear();
      this.workerDatasets.get(worker)?.clear();
      void this.run({ id: this.nextId++, models: [], dataset: { key: "empty", value: emptyPackedDataset() }, clearCache: true }, worker).catch(() => undefined);
    }
  }

  terminate(): void {
    for (const worker of this.workers) worker.terminate();
    this.workers = [];
    for (const pending of this.pending.values()) pending.reject(new Error("Worker pool terminated"));
    this.pending.clear();
  }

  private addWorker(url: URL | string): void {
    const worker = createWorker(url);
    worker.onMessage((data) => {
      const pending = this.pending.get(data.id);
      if (!pending) return;
      this.pending.delete(data.id);
      if (data.error) pending.reject(new Error(data.error));
      else pending.resolve(data.scores ?? []);
    });
    worker.onError((error) => {
      for (const pending of this.pending.values()) pending.reject(error);
      this.pending.clear();
    });
    this.workers.push(worker);
    this.workerModels.set(worker, new Set());
    this.workerDatasets.set(worker, new Set());
  }

  private runEvaluate(items: readonly Network[], dataset: PackedDataset): Promise<number[]> {
    if (this.workers.length === 0) return Promise.resolve([]);
    const worker = this.workers[this.cursor++ % this.workers.length] as WorkerLike;
    const known = this.workerModels.get(worker) ?? new Set<string>();
    this.workerModels.set(worker, known);
    const knownDatasets = this.workerDatasets.get(worker) ?? new Set<string>();
    this.workerDatasets.set(worker, knownDatasets);
    const job: WorkerJob = {
      id: this.nextId++,
      models: items.map((network) => this.modelPayload(network, known)),
      dataset: this.datasetPayload(dataset, knownDatasets),
      useWasm: this.useWasm,
    };
    if (this.wasmPath) job.wasmPath = this.wasmPath.toString();
    return this.run(job, worker);
  }

  private run(job: WorkerJob, selectedWorker?: WorkerLike): Promise<number[]> {
    if (this.workers.length === 0) return Promise.resolve([]);
    const worker = selectedWorker ?? this.workers[this.cursor++ % this.workers.length] as WorkerLike;
    return new Promise((resolve, reject) => {
      this.pending.set(job.id, { resolve, reject });
      worker.postMessage(job);
    });
  }

  private modelPayload(network: Network, known: Set<string>): WorkerJob["models"][number] {
    const key = modelKey(network);
    if (known.has(key)) return { key };
    known.add(key);
    return { key, model: this.binaryModels ? network.toBinary() : network.toJSON() };
  }

  private datasetPayload(dataset: PackedDataset, known: Set<string>): WorkerJob["dataset"] {
    const key = packedDatasetKey(dataset);
    if (known.has(key)) return { key };
    known.add(key);
    return { key, value: dataset, inputSize: dataset.inputSize, outputSize: dataset.outputSize };
  }
}

function defaultWorkerUrl(): URL {
  const suffix = import.meta.url.endsWith(".ts") ? "./evaluate-worker.ts" : "./runtime/evaluate-worker.js";
  return new URL(suffix, import.meta.url);
}

function emptyPackedDataset(): PackedDataset {
  return { inputs: new Float32Array(1), targets: new Float32Array(1), inputSize: 1, outputSize: 1, sampleCount: 1 };
}

function modelKey(network: Network): string {
  const bytes = network.toBinary();
  let hash = 2166136261;
  for (let i = 0; i < bytes.length; i++) {
    hash ^= bytes[i] as number;
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

interface WorkerLike {
  postMessage(message: WorkerJob): void;
  terminate(): void;
  onMessage(handler: (data: WorkerResponse) => void): void;
  onError(handler: (error: Error) => void): void;
}

function createWorker(url: URL | string): WorkerLike {
  if (typeof Worker !== "undefined") {
    const worker = new Worker(url, { type: "module" });
    return {
      postMessage: (message) => worker.postMessage(message),
      terminate: () => worker.terminate(),
      onMessage: (handler) => { worker.onmessage = (event: MessageEvent<WorkerResponse>) => handler(event.data); },
      onError: (handler) => { worker.onerror = (event) => handler(new Error(event.message)); },
    };
  }
  return createNodeWorker(url);
}

function createNodeWorker(url: URL | string): WorkerLike {
  let workerThreads: typeof import("node:worker_threads") | null = null;
  const load = async () => {
    workerThreads ??= await import("node:worker_threads");
    return workerThreads;
  };
  let worker: import("node:worker_threads").Worker | null = null;
  const callbacks: Array<(data: WorkerResponse) => void> = [];
  const errors: Array<(error: Error) => void> = [];
  const ready = load().then(({ Worker: NodeWorker }) => {
    const workerUrl = typeof url === "string" ? new URL(url, import.meta.url) : url;
    worker = new NodeWorker(workerUrl);
    worker.on("message", (data: WorkerResponse) => callbacks.forEach((handler) => handler(data)));
    worker.on("error", (error: unknown) => errors.forEach((handler) => handler(error instanceof Error ? error : new Error(String(error)))));
  });
  return {
    postMessage: (message) => { void ready.then(() => worker?.postMessage(message)); },
    terminate: () => { void worker?.terminate(); },
    onMessage: (handler) => { callbacks.push(handler); },
    onError: (handler) => { errors.push(handler); },
  };
}

function chunk<T>(items: readonly T[], count: number): T[][] {
  const chunks: T[][] = [];
  const size = Math.ceil(items.length / Math.max(1, count));
  for (let i = 0; i < items.length; i += size) chunks.push(items.slice(i, i + size) as T[]);
  return chunks;
}
