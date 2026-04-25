export interface BenchmarkOptions {
  warmup?: number;
  repeats?: number;
}

export interface BenchmarkProfile {
  name: string;
  warmup: number;
  repeats: number;
  scale: number;
}

export interface BenchmarkResult {
  name: string;
  warmup: number;
  repeats: number;
  min: number;
  median: number;
  p95: number;
  max: number;
  checksum: number;
}

export async function measure(name: string, fn: () => number | Promise<number>, options: BenchmarkOptions = {}): Promise<BenchmarkResult> {
  const warmup = options.warmup ?? 2;
  const repeats = options.repeats ?? 7;
  let checksum = 0;
  for (let i = 0; i < warmup; i++) checksum = await fn();

  const times: number[] = [];
  for (let i = 0; i < repeats; i++) {
    const start = performance.now();
    checksum = await fn();
    times.push(performance.now() - start);
  }

  const sorted = [...times].sort((left, right) => left - right);
  return {
    name,
    warmup,
    repeats,
    min: round(sorted[0] ?? 0),
    median: round(percentile(sorted, 0.5)),
    p95: round(percentile(sorted, 0.95)),
    max: round(sorted[sorted.length - 1] ?? 0),
    checksum: round(checksum),
  };
}

export function printBenchmark(title: string, results: BenchmarkResult[]): void {
  console.log(`\n${title}`);
  console.table(results);
  maybeWriteMachineOutput(title, results);
}

export function runtimeLabel(): string {
  if (typeof Bun !== "undefined") return `bun ${Bun.version}`;
  if (typeof process !== "undefined" && process.versions?.node) return `node ${process.versions.node}`;
  return "unknown";
}

export function benchmarkProfile(defaultWarmup = 2, defaultRepeats = 7): BenchmarkProfile {
  const name = String(env("BENCH_PROFILE") ?? "standard");
  if (name === "ci") return { name, warmup: 1, repeats: 3, scale: 0.25 };
  if (name === "quick") return { name, warmup: 1, repeats: 5, scale: 0.5 };
  if (name === "large") return { name, warmup: defaultWarmup, repeats: Math.max(defaultRepeats, 10), scale: 2 };
  return { name, warmup: defaultWarmup, repeats: defaultRepeats, scale: 1 };
}

export function outputMetadata(extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    runtime: runtimeLabel(),
    profile: String(env("BENCH_PROFILE") ?? "standard"),
    cpu: typeof navigator !== "undefined" ? navigator.hardwareConcurrency : undefined,
    ...extra,
  };
}

export function sumValues(values: readonly number[] | Float32Array): number {
  let sum = 0;
  for (const value of values) sum += value;
  return sum;
}

export function sumNestedOutputs(outputs: Float32Array[] | Float32Array): number {
  if (outputs instanceof Float32Array) return sumValues(outputs);
  let sum = 0;
  for (const output of outputs) sum += sumValues(output);
  return sum;
}

export function round(value: number): number {
  return Number(value.toFixed(4));
}

function maybeWriteMachineOutput(title: string, results: BenchmarkResult[]): void {
  const format = String(env("BENCH_FORMAT") ?? "").toLowerCase();
  if (format === "json") {
    console.log(JSON.stringify({ title, results }));
    return;
  }
  if (format === "csv") {
    console.log("title,name,warmup,repeats,min,median,p95,max,checksum");
    for (const result of results) console.log([title, result.name, result.warmup, result.repeats, result.min, result.median, result.p95, result.max, result.checksum].join(","));
  }
}

function env(name: string): string | undefined {
  if (typeof Bun !== "undefined") return Bun.env[name];
  if (typeof process !== "undefined") return process.env[name];
  return undefined;
}

function percentile(sorted: number[], value: number): number {
  if (sorted.length === 0) return 0;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * value) - 1));
  return sorted[index] as number;
}
