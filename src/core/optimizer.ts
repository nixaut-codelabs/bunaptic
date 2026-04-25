import type { OptimizerKind, TrainOptions, TrainSequenceOptions } from "./types.js";

export interface OptimizerConfig {
  kind: OptimizerKind;
  rate: number;
  momentum: number;
  beta1: number;
  beta2: number;
  rho: number;
  epsilon: number;
  weightDecay: number;
}

export interface OptimizerState {
  kind: OptimizerKind;
  step: number;
  first: Float32Array;
  second: Float32Array;
}

export type OptimizerDirection = "minimize" | "maximize";

export function optimizerConfig(options: TrainOptions | TrainSequenceOptions, defaultRate: number): OptimizerConfig {
  const kind = options.optimizer ?? ((options.momentum ?? 0) > 0 ? "momentum" : "sgd");
  return {
    kind,
    rate: options.rate ?? defaultRate,
    momentum: options.momentum ?? (kind === "momentum" ? 0.9 : 0),
    beta1: options.beta1 ?? 0.9,
    beta2: options.beta2 ?? 0.999,
    rho: options.rho ?? 0.9,
    epsilon: options.epsilon ?? 1e-8,
    weightDecay: options.weightDecay ?? 0,
  };
}

export function createOptimizerState(length: number, kind: OptimizerKind): OptimizerState {
  return {
    kind,
    step: 0,
    first: new Float32Array(length),
    second: new Float32Array(length),
  };
}

export function applyOptimizer(
  values: Float32Array,
  gradients: Float32Array,
  state: OptimizerState,
  config: OptimizerConfig,
  direction: OptimizerDirection,
  clip = Infinity,
): void {
  state.step++;
  const sign = direction === "minimize" ? -1 : 1;
  for (let i = 0; i < values.length; i++) {
    const gradient = clamp((gradients[i] as number) + config.weightDecay * (values[i] as number), clip);
    const update = optimizerStep(gradient, i, state, config);
    values[i] = (values[i] as number) + sign * update;
  }
}

export function shouldUseWasmTrainOptimizer(kind: OptimizerKind): boolean {
  return kind === "sgd" || kind === "momentum";
}

function optimizerStep(gradient: number, index: number, state: OptimizerState, config: OptimizerConfig): number {
  switch (config.kind) {
    case "sgd":
      return config.rate * gradient;
    case "momentum": {
      const velocity = config.momentum * (state.first[index] as number) + config.rate * gradient;
      state.first[index] = velocity;
      return velocity;
    }
    case "rmsprop": {
      const average = config.rho * (state.second[index] as number) + (1 - config.rho) * gradient * gradient;
      state.second[index] = average;
      return config.rate * gradient / (Math.sqrt(average) + config.epsilon);
    }
    case "adam": {
      const first = config.beta1 * (state.first[index] as number) + (1 - config.beta1) * gradient;
      const second = config.beta2 * (state.second[index] as number) + (1 - config.beta2) * gradient * gradient;
      state.first[index] = first;
      state.second[index] = second;
      const correctedFirst = first / (1 - config.beta1 ** state.step);
      const correctedSecond = second / (1 - config.beta2 ** state.step);
      return config.rate * correctedFirst / (Math.sqrt(correctedSecond) + config.epsilon);
    }
  }
}

function clamp(value: number, clip: number): number {
  if (!Number.isFinite(clip)) return value;
  return Math.max(-clip, Math.min(clip, value));
}
