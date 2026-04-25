export type CostFunction = (target: ArrayLike<number>, output: ArrayLike<number>) => number;

export const cost = {
  MSE(target: ArrayLike<number>, output: ArrayLike<number>): number {
    let sum = 0;
    for (let i = 0; i < target.length; i++) {
      const delta = Number(target[i]) - Number(output[i]);
      sum += delta * delta;
    }
    return sum / target.length;
  },

  MAE(target: ArrayLike<number>, output: ArrayLike<number>): number {
    let sum = 0;
    for (let i = 0; i < target.length; i++) sum += Math.abs(Number(target[i]) - Number(output[i]));
    return sum / target.length;
  },

  CROSS_ENTROPY(target: ArrayLike<number>, output: ArrayLike<number>): number {
    let sum = 0;
    for (let i = 0; i < target.length; i++) {
      const value = Math.min(Math.max(Number(output[i]), 1e-15), 1 - 1e-15);
      sum -= Number(target[i]) * Math.log(value) + (1 - Number(target[i])) * Math.log(1 - value);
    }
    return sum / target.length;
  },
} as const;
