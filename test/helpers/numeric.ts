import { expect } from "bun:test";

export interface NumericTolerance {
  abs?: number;
  rel?: number;
}

export const TOLERANCE = {
  scalar: { abs: 1e-6, rel: 1e-6 },
  wasm: { abs: 1e-5, rel: 1e-5 },
  simd: { abs: 2e-5, rel: 2e-5 },
  recurrent: { abs: 5e-5, rel: 5e-5 },
} as const;

export function expectClose(actual: number, expected: number, tolerance: NumericTolerance = TOLERANCE.scalar): void {
  const abs = tolerance.abs ?? 0;
  const rel = tolerance.rel ?? 0;
  const delta = Math.abs(actual - expected);
  const scale = Math.max(1, Math.abs(actual), Math.abs(expected));
  expect(delta).toBeLessThanOrEqual(Math.max(abs, rel * scale));
}

export function expectCloseArray(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance: NumericTolerance = TOLERANCE.scalar): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) expectClose(Number(actual[i]), Number(expected[i]), tolerance);
}

export function checksum(values: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += Number(values[i]);
  return sum;
}
