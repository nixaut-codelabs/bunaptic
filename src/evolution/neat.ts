import { Network, MutationMethod } from "../core/network.js";
import { Rng } from "../core/rng.js";
import { evaluatePopulation, type EvaluatePopulationOptions } from "../runtime/evaluate-population.js";
import type { FitnessFunction, PackedDataset, TrainingSample } from "../core/types.js";
import { InnovationRegistry } from "./innovation.js";
import { mutation, selection } from "./methods.js";

export type SelectionMethod = typeof selection[keyof typeof selection];

export interface GenomeScore {
  genome: Network;
  score: number;
  species: number;
}

export interface NeatOptions {
  seed?: number;
  network?: Network;
  popsize?: number;
  elitism?: number;
  mutationRate?: number;
  mutationAmount?: number;
  mutation?: readonly MutationMethod[];
  selection?: SelectionMethod;
  survivalThreshold?: number;
  maxStagnation?: number;
  equal?: boolean;
  speciation?: boolean;
  speciesThreshold?: number;
  islands?: number;
  dataset?: readonly TrainingSample[] | PackedDataset;
  evaluator?: EvaluatePopulationOptions;
}

export class Neat {
  readonly input: number;
  readonly output: number;
  readonly fitness: FitnessFunction<Network>;
  readonly rng: Rng;
  population: Network[] = [];
  generation = 0;
  popsize: number;
  elitism: number;
  mutationRate: number;
  mutationAmount: number;
  mutation: readonly MutationMethod[];
  selection: SelectionMethod;
  survivalThreshold: number;
  maxStagnation: number;
  equal: boolean;
  speciation: boolean;
  speciesThreshold: number;
  islands: number;
  dataset: readonly TrainingSample[] | PackedDataset | null;
  evaluator: EvaluatePopulationOptions;
  innovation: InnovationRegistry;
  scores: GenomeScore[] = [];
  speciesStats: SpeciesStats[] = [];

  constructor(input: number, output: number, fitness: FitnessFunction<Network>, options: NeatOptions = {}) {
    this.input = input;
    this.output = output;
    this.fitness = fitness;
    this.rng = new Rng(options.seed);
    this.popsize = options.popsize ?? 50;
    this.elitism = options.elitism ?? Math.max(1, Math.floor(this.popsize * 0.1));
    this.mutationRate = options.mutationRate ?? 0.3;
    this.mutationAmount = options.mutationAmount ?? 1;
    this.mutation = options.mutation ?? mutation.FFW;
    this.selection = options.selection ?? selection.POWER;
    this.survivalThreshold = options.survivalThreshold ?? 0.5;
    this.maxStagnation = options.maxStagnation ?? 15;
    this.equal = options.equal ?? false;
    this.speciation = options.speciation ?? true;
    this.speciesThreshold = options.speciesThreshold ?? 3;
    this.islands = Math.max(1, options.islands ?? 1);
    this.dataset = options.dataset ?? null;
    this.evaluator = options.evaluator ?? {};
    this.innovation = new InnovationRegistry();
    this.createPool(options.network);
  }

  createPool(template?: Network): void {
    this.population = [];
    for (let i = 0; i < this.popsize; i++) {
      const genome = template ? template.copy() : new Network(this.input, this.output, { rng: this.rng.fork() });
      this.population.push(genome);
    }
    this.innovation.seedFromModels(this.population.map((genome) => genome.toJSON()));
  }

  async evaluate(): Promise<GenomeScore[]> {
    const values = this.dataset
      ? await evaluatePopulation(this.population, this.dataset, this.evaluator)
      : await Promise.all(this.population.map((genome) => this.fitness(genome)));
    const scores = this.population.map((genome, index) => {
      const score = Number(values[index]);
      return { genome, score: Number.isFinite(score) ? score : -Infinity, species: 0 };
    });
    scores.sort((a, b) => b.score - a.score);
    if (this.speciation) this.assignSpecies(scores);
    this.updateSpeciesStats(scores);
    this.scores = scores;
    return scores;
  }

  async evolve(): Promise<Network> {
    const scores = await this.evaluate();
    const best = scores[0]?.genome.copy();
    if (!best) throw new Error("Cannot evolve an empty population");

    const breedingPool = this.survivors(scores);
    const next: Network[] = scores.slice(0, this.elitism).map((entry) => entry.genome.copy());
    while (next.length < this.popsize) {
      const island = this.rng.int(this.islands);
      const parentA = this.selectParent(breedingPool, island);
      const parentB = this.selectParent(breedingPool, island);
      const child = Network.crossOver(parentA, parentB, { rng: this.rng.fork() });
      if (this.rng.chance(this.mutationRate)) {
        for (let i = 0; i < this.mutationAmount; i++) child.mutate(this.rng.pick(this.mutation), this.innovation);
      }
      next.push(child);
    }
    this.population = next;
    this.generation++;
    return best;
  }

  getFittest(): Network | null {
    return this.scores[0]?.genome ?? null;
  }

  getAverage(): number {
    if (this.scores.length === 0) return 0;
    return this.scores.reduce((sum, entry) => sum + entry.score, 0) / this.scores.length;
  }

  export(): ReturnType<Network["toJSON"]>[] {
    return this.population.map((genome) => genome.toJSON());
  }

  import(json: ReturnType<Network["toJSON"]>[]): void {
    this.population = json.map((item) => Network.fromJSON(item, { rng: this.rng.fork() }));
    this.popsize = this.population.length;
    this.innovation.seedFromModels(json);
  }

  private survivors(scores: GenomeScore[]): GenomeScore[] {
    if (!this.speciation) return scores.slice(0, Math.max(1, Math.ceil(scores.length * this.survivalThreshold)));
    const stale = new Set(this.speciesStats.filter((species) => species.stagnation > this.maxStagnation).map((species) => species.id));
    const bySpecies = new Map<number, GenomeScore[]>();
    for (const entry of scores) {
      if (stale.has(entry.species) && entry !== scores[0]) continue;
      const entries = bySpecies.get(entry.species) ?? [];
      entries.push(entry);
      bySpecies.set(entry.species, entries);
    }
    const survivors: GenomeScore[] = [];
    for (const entries of bySpecies.values()) survivors.push(...entries.slice(0, Math.max(1, Math.ceil(entries.length * this.survivalThreshold))));
    return survivors.length > 0 ? survivors.sort((a, b) => b.score - a.score) : scores;
  }

  private selectParent(scores: GenomeScore[], island: number): Network {
    const pool = this.islands > 1
      ? scores.filter((_, index) => index % this.islands === island)
      : scores;
    const candidates = pool.length > 0 ? pool : scores;

    switch (this.selection.name) {
      case "FITNESS_PROPORTIONATE": {
        const min = Math.min(...candidates.map((entry) => entry.score));
        const offset = min < 0 ? Math.abs(min) : 0;
        const total = candidates.reduce((sum, entry) => sum + entry.score + offset + 1e-9, 0);
        let cursor = this.rng.next() * total;
        for (const entry of candidates) {
          cursor -= entry.score + offset + 1e-9;
          if (cursor <= 0) return entry.genome;
        }
        return candidates[candidates.length - 1]!.genome;
      }
      case "TOURNAMENT": {
        const tournament = Array.from({ length: Math.min(this.selection.size, candidates.length) }, () => this.rng.pick(candidates));
        tournament.sort((a, b) => b.score - a.score);
        for (const entry of tournament) if (this.rng.chance(this.selection.probability)) return entry.genome;
        return tournament[tournament.length - 1]!.genome;
      }
      case "POWER":
      default: {
        const index = Math.floor(Math.pow(this.rng.next(), this.selection.power) * candidates.length);
        return candidates[Math.min(index, candidates.length - 1)]!.genome;
      }
    }
  }

  private assignSpecies(scores: GenomeScore[]): void {
    const representatives: Network[] = [];
    for (const entry of scores) {
      let assigned = false;
      for (let i = 0; i < representatives.length; i++) {
        if (compatibilityDistance(entry.genome, representatives[i]!) <= this.speciesThreshold) {
          entry.species = i;
          assigned = true;
          break;
        }
      }
      if (!assigned) {
        entry.species = representatives.length;
        representatives.push(entry.genome);
      }
    }
  }

  private updateSpeciesStats(scores: GenomeScore[]): void {
    const previous = new Map(this.speciesStats.map((species) => [species.id, species]));
    const bestBySpecies = new Map<number, number>();
    for (const entry of scores) bestBySpecies.set(entry.species, Math.max(bestBySpecies.get(entry.species) ?? -Infinity, entry.score));
    this.speciesStats = [...bestBySpecies.entries()].map(([id, best]) => {
      const old = previous.get(id);
      const improved = !old || best > old.bestScore;
      return { id, bestScore: improved ? best : old.bestScore, stagnation: improved ? 0 : old.stagnation + 1 };
    });
  }
}

interface SpeciesStats {
  id: number;
  bestScore: number;
  stagnation: number;
}

export function compatibilityDistance(a: Network, b: Network): number {
  const aj = a.toJSON();
  const bj = b.toJSON();
  const aGenes = new Map(aj.connections.map((conn) => [connectionInnovation(conn), conn]));
  const bGenes = new Map(bj.connections.map((conn) => [connectionInnovation(conn), conn]));
  const innovations = new Set([...aGenes.keys(), ...bGenes.keys()]);
  let disjoint = 0;
  let weightDelta = 0;
  let shared = 0;
  for (const innovation of innovations) {
    const left = aGenes.get(innovation);
    const right = bGenes.get(innovation);
    if (!left || !right) {
      disjoint++;
      continue;
    }
    shared++;
    weightDelta += Math.abs(left.weight - right.weight);
  }
  const normalizer = Math.max(1, Math.max(aGenes.size, bGenes.size));
  const nodeDelta = Math.abs(aj.nodes.length - bj.nodes.length) / Math.max(1, Math.max(aj.nodes.length, bj.nodes.length));
  return nodeDelta + disjoint / normalizer + (shared ? weightDelta / shared : 0);
}

function connectionInnovation(connection: { innovation?: number; from: number; to: number; kind: number }): number {
  return connection.innovation ?? ((connection.from + 1) * 73856093 ^ (connection.to + 1) * 19349663 ^ connection.kind * 83492791) >>> 0;
}
