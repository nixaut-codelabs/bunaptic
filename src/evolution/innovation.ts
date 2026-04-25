import { ConnectionKind, type InnovationSource } from "../core/types.js";

export class InnovationRegistry implements InnovationSource {
  private next = 1;
  private connections = new Map<string, number>();
  private splits = new Map<number, { input: number; output: number }>();

  connection(from: number, to: number, kind: ConnectionKind): number {
    const key = `${from}:${to}:${kind}`;
    const existing = this.connections.get(key);
    if (existing) return existing;
    const innovation = this.next++;
    this.connections.set(key, innovation);
    return innovation;
  }

  split(connectionInnovation: number): { input: number; output: number } {
    const existing = this.splits.get(connectionInnovation);
    if (existing) return existing;
    const pair = { input: this.next++, output: this.next++ };
    this.splits.set(connectionInnovation, pair);
    return pair;
  }

  seedFromModels(models: ReadonlyArray<{ connections: ReadonlyArray<{ from: number; to: number; kind: ConnectionKind; innovation?: number }> }>): void {
    for (const model of models) {
      for (const connection of model.connections) {
        const innovation = connection.innovation ?? this.connection(connection.from, connection.to, connection.kind);
        this.connections.set(`${connection.from}:${connection.to}:${connection.kind}`, innovation);
        this.next = Math.max(this.next, innovation + 1);
      }
    }
  }
}
