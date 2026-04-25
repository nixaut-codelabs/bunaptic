import { ActivationKind, ConnectionKind, NodeKind, type CompiledGraph, type ConnectionGene, type ModelJSON } from "./types.js";
import { Rng } from "./rng.js";

export class GraphBuilder {
  readonly input: number;
  readonly output: number;
  readonly nodeKinds: NodeKind[] = [];
  readonly activations: ActivationKind[] = [];
  readonly biases: number[] = [];
  readonly connections: ConnectionGene[] = [];
  recurrent = false;
  private innovationCounter = 1;

  constructor(input: number, output: number, rng = new Rng()) {
    if (!Number.isInteger(input) || input <= 0) throw new Error("Input size must be a positive integer");
    if (!Number.isInteger(output) || output <= 0) throw new Error("Output size must be a positive integer");
    this.input = input;
    this.output = output;

    for (let i = 0; i < input; i++) this.addNode(NodeKind.Input, ActivationKind.Identity, 0);
    for (let i = 0; i < output; i++) this.addNode(NodeKind.Output, ActivationKind.Logistic, rng.range(-0.1, 0.1));
  }

  addNode(kind = NodeKind.Hidden, activation = ActivationKind.Logistic, bias = 0): number {
    const shouldInsertBeforeOutputs = kind !== NodeKind.Input && kind !== NodeKind.Output && this.nodeKinds.length >= this.input + this.output;
    const index = shouldInsertBeforeOutputs ? this.nodeKinds.length - this.output : this.nodeKinds.length;
    this.nodeKinds.splice(index, 0, kind);
    this.activations.splice(index, 0, activation);
    this.biases.splice(index, 0, kind === NodeKind.Input ? 0 : bias);

    for (const connection of this.connections) {
      if (connection.from >= index) connection.from++;
      if (connection.to >= index) connection.to++;
      if (connection.gater >= index) connection.gater++;
    }
    return index;
  }

  inputNodes(): number[] {
    return Array.from({ length: this.input }, (_, index) => index);
  }

  outputNodes(): number[] {
    const start = this.nodeKinds.length - this.output;
    return Array.from({ length: this.output }, (_, index) => start + index);
  }

  addConnection(from: number, to: number, weight: number, kind = ConnectionKind.Forward, gater = -1, innovation?: number): number {
    this.ensureNode(from);
    this.ensureNode(to);
    if (gater >= 0) this.ensureNode(gater);
    if (kind !== ConnectionKind.Forward) this.recurrent = true;
    const existing = this.connections.findIndex((conn) => conn.from === from && conn.to === to && conn.kind === kind);
    if (existing >= 0) {
      const previous = this.connections[existing] as ConnectionGene;
      this.connections[existing] = withInnovation({ from, to, weight, gater, kind, enabled: true }, innovation ?? previous.innovation);
      return existing;
    }
    const nextInnovation = innovation ?? this.innovationCounter++;
    this.connections.push(withInnovation({ from, to, weight, gater, kind, enabled: true }, nextInnovation));
    this.innovationCounter = Math.max(this.innovationCounter, nextInnovation + 1);
    return this.connections.length - 1;
  }

  connectDense(from: readonly number[], to: readonly number[], rng: Rng, scale = Math.sqrt(2 / Math.max(1, from.length))): void {
    for (const target of to) {
      for (const source of from) this.addConnection(source, target, rng.range(-scale, scale));
    }
  }

  setNode(index: number, values: { kind?: NodeKind; activation?: ActivationKind; bias?: number }): void {
    this.ensureNode(index);
    if (typeof values.kind !== "undefined") this.nodeKinds[index] = values.kind;
    if (typeof values.activation !== "undefined") this.activations[index] = values.activation;
    if (typeof values.bias !== "undefined") this.biases[index] = values.bias;
  }

  compile(): CompiledGraph {
    const nodeCount = this.nodeKinds.length;
    const enabled = this.connections.filter((conn) => conn.enabled);
    const from = Uint32Array.from(enabled.map((conn) => conn.from));
    const to = Uint32Array.from(enabled.map((conn) => conn.to));
    return {
      input: this.input,
      output: this.output,
      recurrent: this.recurrent,
      nodeCount,
      nodeKinds: Uint8Array.from(this.nodeKinds),
      activations: Uint8Array.from(this.activations),
      biases: Float32Array.from(this.biases),
      from,
      to,
      gater: Int32Array.from(enabled.map((conn) => conn.gater)),
      weights: Float32Array.from(enabled.map((conn) => conn.weight)),
      connectionKinds: Uint8Array.from(enabled.map((conn) => conn.kind)),
      incomingStarts: buildRangeStarts(nodeCount, to),
      incoming: buildRangeIndices(nodeCount, to),
      outgoingStarts: buildRangeStarts(nodeCount, from),
      outgoing: buildRangeIndices(nodeCount, from),
      outputStart: nodeCount - this.output,
    };
  }

  toJSON(): ModelJSON {
    return {
      version: 1,
      input: this.input,
      output: this.output,
      recurrent: this.recurrent,
      nodes: this.nodeKinds.map((kind, index) => ({
        kind,
        activation: this.activations[index] as ActivationKind,
        bias: this.biases[index] as number,
      })),
      connections: this.connections.map((conn) => ({ ...conn })),
    };
  }

  static fromJSON(json: ModelJSON): GraphBuilder {
    if (json.version !== 1) throw new Error(`Unsupported Bunaptic model version: ${json.version}`);
    const graph = Object.create(GraphBuilder.prototype) as GraphBuilder;
    Object.assign(graph, {
      input: json.input,
      output: json.output,
      nodeKinds: json.nodes.map((node) => node.kind),
      activations: json.nodes.map((node) => node.activation),
      biases: json.nodes.map((node) => node.bias),
      connections: json.connections.map((conn) => ({ ...conn })),
      recurrent: json.recurrent,
      innovationCounter: Math.max(1, ...json.connections.map((conn) => conn.innovation ?? 0)) + 1,
    });
    return graph;
  }

  private ensureNode(index: number): void {
    if (!Number.isInteger(index) || index < 0 || index >= this.nodeKinds.length) {
      throw new Error(`Node index out of range: ${index}`);
    }
  }
}

function withInnovation(connection: Omit<ConnectionGene, "innovation">, innovation?: number): ConnectionGene {
  return typeof innovation === "number" ? { ...connection, innovation } : connection;
}

function buildRangeStarts(nodeCount: number, owners: Uint32Array): Uint32Array {
  const starts = new Uint32Array(nodeCount + 1);
  for (let i = 0; i < owners.length; i++) {
    const index = (owners[i] as number) + 1;
    starts[index] = (starts[index] as number) + 1;
  }
  for (let i = 1; i < starts.length; i++) starts[i] = (starts[i] as number) + (starts[i - 1] as number);
  return starts;
}

function buildRangeIndices(nodeCount: number, owners: Uint32Array): Uint32Array {
  const starts = buildRangeStarts(nodeCount, owners);
  const cursor = new Uint32Array(starts);
  const indices = new Uint32Array(owners.length);
  for (let i = 0; i < owners.length; i++) {
    const owner = owners[i] as number;
    const index = cursor[owner] as number;
    indices[index] = i;
    cursor[owner] = index + 1;
  }
  return indices;
}
