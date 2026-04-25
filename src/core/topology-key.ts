import type { CompiledGraph } from "./types.js";

export function topologyKey(graph: CompiledGraph): string {
  let hash = 2166136261;
  hash = hashString(hash, `${graph.input}:${graph.output}:${graph.recurrent ? 1 : 0}:${graph.nodeCount}:${graph.weights.length}:${graph.outputStart};`);
  hash = hashBytes(hash, graph.nodeKinds);
  hash = hashBytes(hash, graph.activations);
  hash = hashBytes(hash, graph.from);
  hash = hashBytes(hash, graph.to);
  hash = hashBytes(hash, graph.gater);
  hash = hashBytes(hash, graph.connectionKinds);
  hash = hashBytes(hash, graph.incomingStarts);
  hash = hashBytes(hash, graph.incoming);
  hash = hashBytes(hash, graph.outgoingStarts);
  hash = hashBytes(hash, graph.outgoing);
  return hash.toString(36);
}

export function hasSameTopology(left: CompiledGraph, right: CompiledGraph): boolean {
  if (left === right) return true;
  if (left.input !== right.input || left.output !== right.output || left.recurrent !== right.recurrent) return false;
  if (left.nodeCount !== right.nodeCount || left.weights.length !== right.weights.length || left.outputStart !== right.outputStart) return false;
  return equalBytes(left.nodeKinds, right.nodeKinds)
    && equalBytes(left.activations, right.activations)
    && equalBytes(left.from, right.from)
    && equalBytes(left.to, right.to)
    && equalBytes(left.gater, right.gater)
    && equalBytes(left.connectionKinds, right.connectionKinds)
    && equalBytes(left.incomingStarts, right.incomingStarts)
    && equalBytes(left.incoming, right.incoming)
    && equalBytes(left.outgoingStarts, right.outgoingStarts)
    && equalBytes(left.outgoing, right.outgoing);
}

function hashBytes(hash: number, values: Uint8Array | Uint32Array | Int32Array): number {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function hashString(hash: number, value: string): number {
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function equalBytes(left: Uint8Array | Uint32Array | Int32Array, right: Uint8Array | Uint32Array | Int32Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  const leftBytes = new Uint8Array(left.buffer, left.byteOffset, left.byteLength);
  const rightBytes = new Uint8Array(right.buffer, right.byteOffset, right.byteLength);
  for (let i = 0; i < leftBytes.length; i++) if (leftBytes[i] !== rightBytes[i]) return false;
  return true;
}
