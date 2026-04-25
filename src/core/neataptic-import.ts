import { ActivationKind, ConnectionKind, NodeKind, type ModelJSON } from "./types.js";

export interface NeatapticJSON {
  input: number;
  output: number;
  dropout?: number;
  nodes: Array<{
    index: number;
    bias: number;
    type: "input" | "hidden" | "output" | "constant";
    squash: string;
    mask?: number;
  }>;
  connections: Array<{
    from: number;
    to: number;
    weight: number;
    gater: number | null;
  }>;
}

export function fromNeatapticJSON(json: NeatapticJSON): ModelJSON {
  const nodes = [...json.nodes]
    .sort((a, b) => a.index - b.index)
    .map((node) => ({
      kind: mapNodeKind(node.type),
      bias: node.bias,
      activation: mapActivation(node.squash),
    }));

  return {
    version: 1,
    input: json.input,
    output: json.output,
    recurrent: json.connections.some((connection) => connection.to <= connection.from),
    nodes,
    connections: json.connections.map((connection, index) => ({
      from: connection.from,
      to: connection.to,
      weight: connection.weight,
      gater: connection.gater ?? -1,
      kind: connection.from === connection.to ? ConnectionKind.Self : connection.to > connection.from ? ConnectionKind.Forward : ConnectionKind.Recurrent,
      enabled: true,
      innovation: index + 1,
    })),
  };
}

function mapNodeKind(type: NeatapticJSON["nodes"][number]["type"]): NodeKind {
  switch (type) {
    case "input":
      return NodeKind.Input;
    case "output":
      return NodeKind.Output;
    case "constant":
      return NodeKind.Constant;
    case "hidden":
    default:
      return NodeKind.Hidden;
  }
}

function mapActivation(name: string): ActivationKind {
  switch (name) {
    case "LOGISTIC":
      return ActivationKind.Logistic;
    case "TANH":
      return ActivationKind.Tanh;
    case "IDENTITY":
      return ActivationKind.Identity;
    case "STEP":
      return ActivationKind.Step;
    case "RELU":
      return ActivationKind.Relu;
    case "SOFTSIGN":
      return ActivationKind.Softsign;
    case "SINUSOID":
      return ActivationKind.Sinusoid;
    case "GAUSSIAN":
      return ActivationKind.Gaussian;
    case "BENT_IDENTITY":
      return ActivationKind.BentIdentity;
    case "BIPOLAR":
      return ActivationKind.Bipolar;
    case "BIPOLAR_SIGMOID":
      return ActivationKind.BipolarSigmoid;
    case "HARD_TANH":
      return ActivationKind.HardTanh;
    case "ABSOLUTE":
      return ActivationKind.Absolute;
    case "INVERSE":
      return ActivationKind.Inverse;
    case "SELU":
      return ActivationKind.Selu;
    default:
      throw new Error(`Unsupported Neataptic activation: ${name}`);
  }
}
