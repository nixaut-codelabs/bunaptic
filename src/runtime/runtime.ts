import { availableParallelism } from "node:os";

export type RuntimeKind = "bun" | "node" | "unknown";

export function detectRuntime(): RuntimeKind {
  if (typeof Bun !== "undefined") return "bun";
  if (typeof process !== "undefined" && process.versions?.node) return "node";
  return "unknown";
}

export function cpuCount(): number {
  const runtime = detectRuntime();
  if (runtime === "bun" || runtime === "node") {
    const count = Number(process.env.BUNAPTIC_WORKERS || "0");
    if (count > 0) return count;
    return Math.max(1, availableParallelism());
  }
  return 1;
}
