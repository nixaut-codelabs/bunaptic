import { activation } from "../core/activation.js";
import { cost } from "../core/cost.js";
import { MutationMethod } from "../core/network.js";

export const selection = {
  POWER: { name: "POWER", power: 4 },
  FITNESS_PROPORTIONATE: { name: "FITNESS_PROPORTIONATE" },
  TOURNAMENT: { name: "TOURNAMENT", size: 5, probability: 0.5 },
} as const;

export const crossover = {
  SINGLE_POINT: "SINGLE_POINT",
  TWO_POINT: "TWO_POINT",
  UNIFORM: "UNIFORM",
  AVERAGE: "AVERAGE",
} as const;

export const mutation = {
  ADD_NODE: MutationMethod.AddNode,
  ADD_CONN: MutationMethod.AddConnection,
  MOD_WEIGHT: MutationMethod.ModWeight,
  MOD_BIAS: MutationMethod.ModBias,
  MOD_ACTIVATION: MutationMethod.ModActivation,
  ADD_SELF_CONN: MutationMethod.AddSelfConnection,
  FFW: [MutationMethod.AddNode, MutationMethod.AddConnection, MutationMethod.ModWeight, MutationMethod.ModBias, MutationMethod.ModActivation],
  ALL: [
    MutationMethod.AddNode,
    MutationMethod.AddConnection,
    MutationMethod.ModWeight,
    MutationMethod.ModBias,
    MutationMethod.ModActivation,
    MutationMethod.AddSelfConnection,
  ],
} as const;

export const methods = {
  activation,
  cost,
  selection,
  crossover,
  mutation,
} as const;
