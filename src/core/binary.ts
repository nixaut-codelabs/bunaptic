import { GraphBuilder } from "./graph.js";
import { ConnectionKind, NodeKind, ActivationKind, type ModelJSON } from "./types.js";

const MAGIC = 0x42505443;
const LEGACY_VERSION = 1;
const VERSION = 2;
const RECURRENT_KIND = { lstm: 1, gru: 2 } as const;

export function encodeModel(json: ModelJSON): Uint8Array {
  const nodeCount = json.nodes.length;
  const connectionCount = json.connections.length;
  const recurrentSpec = json.recurrentSpec;
  const headerBytes = 8 * 4;
  const nodeBytes = nodeCount * 6;
  const connectionBytes = connectionCount * 24;
  const recurrentBytes = recurrentSpec
    ? 10 * 4 + totalFloatCount(recurrentSpec) * 4
    : 0;
  const buffer = new ArrayBuffer(headerBytes + nodeBytes + connectionBytes + recurrentBytes);
  const view = new DataView(buffer);
  let offset = 0;

  view.setUint32(offset, MAGIC, true); offset += 4;
  view.setUint32(offset, VERSION, true); offset += 4;
  view.setUint32(offset, json.input, true); offset += 4;
  view.setUint32(offset, json.output, true); offset += 4;
  view.setUint32(offset, json.recurrent ? 1 : 0, true); offset += 4;
  view.setUint32(offset, nodeCount, true); offset += 4;
  view.setUint32(offset, connectionCount, true); offset += 4;
  view.setUint32(offset, recurrentSpec ? 1 : 0, true); offset += 4;

  for (const node of json.nodes) {
    view.setUint8(offset++, node.kind);
    view.setUint8(offset++, node.activation);
    view.setFloat32(offset, node.bias, true); offset += 4;
  }

  for (const connection of json.connections) {
    view.setUint32(offset, connection.from, true); offset += 4;
    view.setUint32(offset, connection.to, true); offset += 4;
    view.setInt32(offset, connection.gater, true); offset += 4;
    view.setFloat32(offset, connection.weight, true); offset += 4;
    view.setUint32(offset, connection.innovation ?? 0, true); offset += 4;
    view.setUint8(offset++, connection.kind);
    view.setUint8(offset++, connection.enabled ? 1 : 0);
    offset += 2;
  }

  if (recurrentSpec) {
    view.setUint32(offset, RECURRENT_KIND[recurrentSpec.kind], true); offset += 4;
    view.setUint32(offset, recurrentSpec.inputSize, true); offset += 4;
    view.setUint32(offset, recurrentSpec.hiddenSize, true); offset += 4;
    view.setUint32(offset, recurrentSpec.outputSize, true); offset += 4;
    view.setUint32(offset, recurrentSpec.outputActivation, true); offset += 4;
    view.setUint32(offset, recurrentSpec.inputWeights.length, true); offset += 4;
    view.setUint32(offset, recurrentSpec.hiddenWeights.length, true); offset += 4;
    view.setUint32(offset, recurrentSpec.biases.length, true); offset += 4;
    view.setUint32(offset, recurrentSpec.outputWeights.length, true); offset += 4;
    view.setUint32(offset, recurrentSpec.outputBiases.length, true); offset += 4;
    offset = writeFloat32Array(view, offset, recurrentSpec.inputWeights);
    offset = writeFloat32Array(view, offset, recurrentSpec.hiddenWeights);
    offset = writeFloat32Array(view, offset, recurrentSpec.biases);
    offset = writeFloat32Array(view, offset, recurrentSpec.outputWeights);
    offset = writeFloat32Array(view, offset, recurrentSpec.outputBiases);
  }

  return new Uint8Array(buffer);
}

export function decodeModel(bytes: Uint8Array): ModelJSON {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;
  const magic = view.getUint32(offset, true); offset += 4;
  if (magic !== MAGIC) throw new Error("Invalid Bunaptic binary model");
  const version = view.getUint32(offset, true); offset += 4;
  if (version !== LEGACY_VERSION && version !== VERSION) throw new Error(`Unsupported Bunaptic binary version: ${version}`);
  const input = view.getUint32(offset, true); offset += 4;
  const output = view.getUint32(offset, true); offset += 4;
  const recurrent = view.getUint32(offset, true) === 1; offset += 4;
  const nodeCount = view.getUint32(offset, true); offset += 4;
  const connectionCount = view.getUint32(offset, true); offset += 4;
  const hasRecurrentSpec = version >= VERSION && view.getUint32(offset, true) === 1; offset += version >= VERSION ? 4 : 0;

  const nodes: ModelJSON["nodes"] = [];
  for (let i = 0; i < nodeCount; i++) {
    const kind = view.getUint8(offset++) as NodeKind;
    const activation = view.getUint8(offset++) as ActivationKind;
    const bias = view.getFloat32(offset, true); offset += 4;
    nodes.push({ kind, activation, bias });
  }

  const connections: ModelJSON["connections"] = [];
  for (let i = 0; i < connectionCount; i++) {
    const from = view.getUint32(offset, true); offset += 4;
    const to = view.getUint32(offset, true); offset += 4;
    const gater = view.getInt32(offset, true); offset += 4;
    const weight = view.getFloat32(offset, true); offset += 4;
    const innovation = version >= VERSION ? view.getUint32(offset, true) : 0; offset += version >= VERSION ? 4 : 0;
    const kind = view.getUint8(offset++) as ConnectionKind;
    const enabled = view.getUint8(offset++) === 1;
    offset += version >= VERSION ? 2 : 3;
    connections.push(innovation > 0 ? { from, to, gater, weight, kind, enabled, innovation } : { from, to, gater, weight, kind, enabled });
  }

  const model: ModelJSON = { version: 1, input, output, recurrent, nodes, connections };
  if (hasRecurrentSpec) {
    const kindCode = view.getUint32(offset, true); offset += 4;
    const inputSize = view.getUint32(offset, true); offset += 4;
    const hiddenSize = view.getUint32(offset, true); offset += 4;
    const outputSize = view.getUint32(offset, true); offset += 4;
    const outputActivation = view.getUint32(offset, true) as ActivationKind; offset += 4;
    const inputWeightCount = view.getUint32(offset, true); offset += 4;
    const hiddenWeightCount = view.getUint32(offset, true); offset += 4;
    const biasCount = view.getUint32(offset, true); offset += 4;
    const outputWeightCount = view.getUint32(offset, true); offset += 4;
    const outputBiasCount = view.getUint32(offset, true); offset += 4;
    const inputWeights = readFloat32Array(view, offset, inputWeightCount); offset += inputWeightCount * 4;
    const hiddenWeights = readFloat32Array(view, offset, hiddenWeightCount); offset += hiddenWeightCount * 4;
    const biases = readFloat32Array(view, offset, biasCount); offset += biasCount * 4;
    const outputWeights = readFloat32Array(view, offset, outputWeightCount); offset += outputWeightCount * 4;
    const outputBiases = readFloat32Array(view, offset, outputBiasCount); offset += outputBiasCount * 4;
    if (kindCode !== RECURRENT_KIND.lstm && kindCode !== RECURRENT_KIND.gru) {
      throw new Error(`Unsupported recurrent binary spec kind: ${kindCode}`);
    }
    model.recurrentSpec = {
      kind: kindCode === RECURRENT_KIND.lstm ? "lstm" : "gru",
      inputSize,
      hiddenSize,
      outputSize,
      outputActivation,
      inputWeights,
      hiddenWeights,
      biases,
      outputWeights,
      outputBiases,
    };
  }

  return model;
}

export function graphFromBinary(bytes: Uint8Array): GraphBuilder {
  return GraphBuilder.fromJSON(decodeModel(bytes));
}

function totalFloatCount(spec: NonNullable<ModelJSON["recurrentSpec"]>): number {
  return spec.inputWeights.length + spec.hiddenWeights.length + spec.biases.length + spec.outputWeights.length + spec.outputBiases.length;
}

function writeFloat32Array(view: DataView, offset: number, values: readonly number[]): number {
  for (const value of values) {
    view.setFloat32(offset, value, true);
    offset += 4;
  }
  return offset;
}

function readFloat32Array(view: DataView, offset: number, length: number): number[] {
  const values: number[] = [];
  for (let i = 0; i < length; i++) {
    values.push(view.getFloat32(offset, true));
    offset += 4;
  }
  return values;
}
