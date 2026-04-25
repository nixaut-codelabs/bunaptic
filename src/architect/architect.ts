import { Network } from "../core/network.js";
import { Rng } from "../core/rng.js";
import { ActivationKind, ConnectionKind, NodeKind } from "../core/types.js";
import { createGruSpec, createLstmSpec } from "../recurrent/spec.js";

export interface ArchitectOptions {
  seed?: number;
  rng?: Rng;
}

export interface RandomOptions extends ArchitectOptions {
  connections?: number;
  recurrent?: number;
  self?: number;
}

export interface RecurrentOptions extends ArchitectOptions {
  outputActivation?: ActivationKind;
  recurrentScale?: number;
}

function network(input: number, output: number, options: ArchitectOptions = {}): Network {
  if (options.rng) return new Network(input, output, { rng: options.rng });
  if (typeof options.seed === "number") return new Network(input, output, { seed: options.seed });
  return new Network(input, output);
}

function replaceDefaultConnections(net: Network): void {
  net.graph.connections.length = 0;
  net.compile();
}

function addLayer(net: Network, size: number, activation: ActivationKind, rng: Rng, bias = 0): number[] {
  return Array.from({ length: size }, () => net.graph.addNode(NodeKind.Hidden, activation, bias || rng.range(-0.1, 0.1)));
}

export const Architect = {
  Perceptron(input: number, ...rest: number[]): Network {
    if (rest.length < 1) throw new Error("Perceptron requires at least an output layer");
    const output = rest[rest.length - 1] as number;
    const hidden = rest.slice(0, -1);
    const net = network(input, output);
    replaceDefaultConnections(net);
    const rng = net.rng;
    let previous = net.graph.inputNodes();
    for (const size of hidden) {
      const layer = addLayer(net, size, ActivationKind.Logistic, rng);
      net.graph.connectDense(previous, layer, rng);
      previous = layer;
    }
    net.graph.connectDense(previous, net.graph.outputNodes(), rng);
    net.compile();
    return net;
  },

  Random(input: number, hidden: number, output: number, options: RandomOptions = {}): Network {
    const net = network(input, output, options);
    replaceDefaultConnections(net);
    const hiddenNodes = addLayer(net, hidden, ActivationKind.Logistic, net.rng);
    const nodes = [...net.graph.inputNodes(), ...hiddenNodes, ...net.graph.outputNodes()];
    const connectionCount = options.connections ?? Math.max(hidden * 2, input * output);
    for (let i = 0; i < connectionCount; i++) {
      const from = net.rng.pick(nodes.slice(0, -output));
      const to = net.rng.pick(nodes.slice(input));
      if (from !== to) net.graph.addConnection(from, to, net.rng.range(-1, 1), from < to ? ConnectionKind.Forward : ConnectionKind.Recurrent);
    }
    for (let i = 0; i < (options.recurrent ?? 0); i++) {
      const from = net.rng.pick(nodes.slice(input));
      const to = net.rng.pick(nodes.slice(input));
      net.graph.addConnection(from, to, net.rng.range(-1, 1), ConnectionKind.Recurrent);
    }
    for (let i = 0; i < (options.self ?? 0); i++) {
      const node = net.rng.pick(nodes.slice(input));
      net.graph.addConnection(node, node, net.rng.range(-1, 1), ConnectionKind.Self);
    }
    net.compile();
    return net;
  },

  RNN(input: number, hidden: number, output: number, options: RecurrentOptions = {}): Network {
    const net = network(input, output, options);
    replaceDefaultConnections(net);
    const rng = net.rng;
    const memory = addLayer(net, hidden, ActivationKind.Tanh, rng);
    net.graph.connectDense(net.graph.inputNodes(), memory, rng);
    net.graph.connectDense(memory, net.graph.outputNodes(), rng);
    for (const from of memory) {
      for (const to of memory) net.graph.addConnection(from, to, rng.range(-0.5, 0.5) * (options.recurrentScale ?? 1), ConnectionKind.Recurrent);
    }
    for (const out of net.graph.outputNodes()) net.graph.activations[out] = options.outputActivation ?? ActivationKind.Logistic;
    net.compile();
    return net;
  },

  LSTM(input: number, hidden: number, output: number, options: RecurrentOptions = {}): Network {
    const net = network(input, output, options);
    replaceDefaultConnections(net);
    const rng = net.rng;
    const inputGate = addLayer(net, hidden, ActivationKind.Logistic, rng, 1);
    const forgetGate = addLayer(net, hidden, ActivationKind.Logistic, rng, 1);
    const cell = addLayer(net, hidden, ActivationKind.Tanh, rng);
    const outputGate = addLayer(net, hidden, ActivationKind.Logistic, rng, 1);
    const block = addLayer(net, hidden, ActivationKind.Tanh, rng);
    const inputs = net.graph.inputNodes();
    const outputs = net.graph.outputNodes();

    net.graph.connectDense(inputs, inputGate, rng);
    net.graph.connectDense(inputs, forgetGate, rng);
    net.graph.connectDense(inputs, cell, rng);
    net.graph.connectDense(inputs, outputGate, rng);

    for (let i = 0; i < hidden; i++) {
      net.graph.addConnection(cell[i] as number, cell[i] as number, 1, ConnectionKind.Self, forgetGate[i] as number);
      net.graph.addConnection(cell[i] as number, block[i] as number, 1, ConnectionKind.Forward, outputGate[i] as number);
      net.graph.addConnection(inputGate[i] as number, cell[i] as number, rng.range(-0.5, 0.5));
    }
    net.graph.connectDense(block, outputs, rng);
    for (const out of outputs) net.graph.activations[out] = options.outputActivation ?? ActivationKind.Logistic;
    net.recurrentSpec = createLstmSpec(input, hidden, output, rng, options.outputActivation ?? ActivationKind.Logistic);
    net.compile();
    return net;
  },

  GRU(input: number, hidden: number, output: number, options: RecurrentOptions = {}): Network {
    const net = network(input, output, options);
    replaceDefaultConnections(net);
    const rng = net.rng;
    const updateGate = addLayer(net, hidden, ActivationKind.Logistic, rng, 1);
    const resetGate = addLayer(net, hidden, ActivationKind.Logistic, rng, 1);
    const candidate = addLayer(net, hidden, ActivationKind.Tanh, rng);
    const state = addLayer(net, hidden, ActivationKind.Tanh, rng);
    const inputs = net.graph.inputNodes();

    net.graph.connectDense(inputs, updateGate, rng);
    net.graph.connectDense(inputs, resetGate, rng);
    net.graph.connectDense(inputs, candidate, rng);
    for (let i = 0; i < hidden; i++) {
      net.graph.addConnection(state[i] as number, candidate[i] as number, rng.range(-0.5, 0.5), ConnectionKind.Recurrent, resetGate[i] as number);
      net.graph.addConnection(candidate[i] as number, state[i] as number, 1, ConnectionKind.Forward, updateGate[i] as number);
      net.graph.addConnection(state[i] as number, state[i] as number, 1, ConnectionKind.Self);
    }
    net.graph.connectDense(state, net.graph.outputNodes(), rng);
    for (const out of net.graph.outputNodes()) net.graph.activations[out] = options.outputActivation ?? ActivationKind.Logistic;
    net.recurrentSpec = createGruSpec(input, hidden, output, rng, options.outputActivation ?? ActivationKind.Logistic);
    net.compile();
    return net;
  },

  NARX(input: number, hidden: number, output: number, previousInput = 2, previousOutput = 2, options: RecurrentOptions = {}): Network {
    const augmentedInput = input * (previousInput + 1) + output * previousOutput;
    const net = Architect.RNN(augmentedInput, hidden, output, options);
    return net;
  },

  Hopfield(size: number, options: ArchitectOptions = {}): Network {
    const net = network(size, size, options);
    replaceDefaultConnections(net);
    const inputs = net.graph.inputNodes();
    const outputs = net.graph.outputNodes();
    for (const out of outputs) net.graph.activations[out] = ActivationKind.Step;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        if (i !== j) net.graph.addConnection(inputs[i] as number, outputs[j] as number, net.rng.range(-1, 1));
      }
    }
    net.compile();
    return net;
  },
};
