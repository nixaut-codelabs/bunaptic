import { ActivationKind } from "./types.js";

const SELU_ALPHA = 1.6732632423543772;
const SELU_SCALE = 1.0507009873554805;

export function activateValue(value: number, kind: ActivationKind): number {
  switch (kind) {
    case ActivationKind.Logistic:
      return 1 / (1 + Math.exp(-value));
    case ActivationKind.Tanh:
      return Math.tanh(value);
    case ActivationKind.Identity:
      return value;
    case ActivationKind.Step:
      return value > 0 ? 1 : 0;
    case ActivationKind.Relu:
      return value > 0 ? value : 0;
    case ActivationKind.Softsign:
      return value / (1 + Math.abs(value));
    case ActivationKind.Sinusoid:
      return Math.sin(value);
    case ActivationKind.Gaussian:
      return Math.exp(-(value * value));
    case ActivationKind.BentIdentity:
      return (Math.sqrt(value * value + 1) - 1) / 2 + value;
    case ActivationKind.Bipolar:
      return value > 0 ? 1 : -1;
    case ActivationKind.BipolarSigmoid:
      return 2 / (1 + Math.exp(-value)) - 1;
    case ActivationKind.HardTanh:
      return Math.max(-1, Math.min(1, value));
    case ActivationKind.Absolute:
      return Math.abs(value);
    case ActivationKind.Inverse:
      return 1 - value;
    case ActivationKind.Selu:
      return (value > 0 ? value : SELU_ALPHA * Math.exp(value) - SELU_ALPHA) * SELU_SCALE;
  }
}

export function derivativeFromActivation(activation: number, state: number, kind: ActivationKind): number {
  switch (kind) {
    case ActivationKind.Logistic:
      return activation * (1 - activation);
    case ActivationKind.Tanh:
      return 1 - activation * activation;
    case ActivationKind.Identity:
      return 1;
    case ActivationKind.Step:
      return 0;
    case ActivationKind.Relu:
      return state > 0 ? 1 : 0;
    case ActivationKind.Softsign: {
      const denom = 1 + Math.abs(state);
      return 1 / (denom * denom);
    }
    case ActivationKind.Sinusoid:
      return Math.cos(state);
    case ActivationKind.Gaussian:
      return -2 * state * activation;
    case ActivationKind.BentIdentity:
      return state / (2 * Math.sqrt(state * state + 1)) + 1;
    case ActivationKind.Bipolar:
      return 0;
    case ActivationKind.BipolarSigmoid:
      return 0.5 * (1 + activation) * (1 - activation);
    case ActivationKind.HardTanh:
      return state > -1 && state < 1 ? 1 : 0;
    case ActivationKind.Absolute:
      return state >= 0 ? 1 : -1;
    case ActivationKind.Inverse:
      return -1;
    case ActivationKind.Selu:
      return state > 0 ? SELU_SCALE : SELU_SCALE * SELU_ALPHA * Math.exp(state);
  }
}

export const activation = {
  LOGISTIC: ActivationKind.Logistic,
  TANH: ActivationKind.Tanh,
  IDENTITY: ActivationKind.Identity,
  STEP: ActivationKind.Step,
  RELU: ActivationKind.Relu,
  SOFTSIGN: ActivationKind.Softsign,
  SINUSOID: ActivationKind.Sinusoid,
  GAUSSIAN: ActivationKind.Gaussian,
  BENT_IDENTITY: ActivationKind.BentIdentity,
  BIPOLAR: ActivationKind.Bipolar,
  BIPOLAR_SIGMOID: ActivationKind.BipolarSigmoid,
  HARD_TANH: ActivationKind.HardTanh,
  ABSOLUTE: ActivationKind.Absolute,
  INVERSE: ActivationKind.Inverse,
  SELU: ActivationKind.Selu,
} as const;
