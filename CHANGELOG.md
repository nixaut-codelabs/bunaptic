# Changelog

All notable changes to Bunaptic are documented here.

This project follows a pre-1.0 release cycle. Public APIs are intended to be stable enough for alpha users, but advanced internals may still change before `1.0.0`.

## [0.1.0-alpha.1] - 2026-04-26

Initial public alpha release of Bunaptic.

Bunaptic is an independent Bun-first neural-network and neuroevolution engine inspired by Neataptic, rebuilt for modern server and desktop runtimes with typed arrays, Workers, Rust WASM, SIMD-aware kernels, dense fast paths, recurrent helpers, and Node.js fallback support.

### Added

- Project package metadata for `bunaptic` under Nixaut CodeLabs.
- ESM package exports for root, Bun, Node.js, and worker entrypoints.
- Bun-first runtime support with Node.js fallback.
- Strict TypeScript source and declaration output.
- MIT license.
- Comprehensive single-file README documentation.
- Release-oriented package smoke checks for Bun, Node.js, worker entrypoints, WASM artifacts, declarations, and tarball consumption.

### Core Engine

- `Network` API for activation, training, testing, mutation, cloning, parameter access, JSON serialization, and binary serialization.
- `GraphBuilder` for flat typed-array graph construction.
- `ReferenceEngine` TypeScript execution path using compiled incoming/outgoing connection ranges.
- Deterministic `Rng` helper for seeded runs.
- Packed feed-forward datasets via `packDataset`.
- Model JSON support with recurrent spec preservation.
- Binary model encoding and decoding with V1 decode compatibility and V2 recurrent/innovation metadata support.
- Neataptic JSON import helper for supported feed-forward models.

### Architectures

- `Architect.Perceptron` for layered feed-forward networks.
- `Architect.Random` for randomized graph construction.
- `Architect.RNN`, `Architect.LSTM`, `Architect.GRU`, `Architect.NARX`, and `Architect.Hopfield` builders.
- Fused LSTM/GRU recurrent specs attached to recurrent builders while retaining graph compatibility.
- Recurrent state helpers for create, clone, reset, and streaming-style sequence continuation.

### Training

- Feed-forward TypeScript training path.
- Feed-forward WASM graph training for `sgd` and `momentum`.
- TypeScript optimizer support for `sgd`, `momentum`, `rmsprop`, and `adam`.
- Dense TypeScript training path for eligible layered feed-forward graphs.
- TypeScript BPTT sequence training for fused LSTM/GRU specs.
- WASM recurrent sequence training for LSTM/GRU `sgd` and `momentum`.
- Packed final-output sequence datasets.
- Packed per-step sequence datasets.
- Gradient clipping and truncated BPTT options for recurrent training.

### Dense Fast Paths

- Dense plan derivation for compatible strictly layered feed-forward graphs.
- TypeScript dense forward and packed dataset evaluation helpers.
- Dense plan packing for WASM calls.
- WASM dense single forward, batch forward, and packed dataset evaluation.
- Automatic dense routing inside `Network.activate`, `Network.testPacked`, and eligible training paths.
- Safe fallback to the generic graph path for arbitrary, recurrent, gated, mutated, or non-layered topologies.

### Rust WASM Kernel

- Packaged scalar WASM artifact: `dist/bunaptic_kernel.wasm`.
- Packaged SIMD WASM artifact: `dist/bunaptic_kernel_simd.wasm`.
- WASM loader with `simd: "auto" | true | false` options.
- Runtime SIMD capability reporting via `kernel.simdEnabled`.
- WASM memory allocation/deallocation helpers.
- Graph forward, batch, sequence, ragged sequence, dataset evaluation, and graph training exports.
- Dense plan forward, batch, and dataset evaluation exports.
- Same-topology population V1 and V2 exports.
- Fused LSTM/GRU sequence and ragged inference exports.
- Fused LSTM/GRU recurrent training export.
- Rust kernel split into focused modules: `memory`, `math`, `dense`, `graph`, `population`, and `recurrent`.

### SIMD

- Real `simd128`-guarded dot-product accumulation.
- SIMD squared-error/MSE reductions.
- SIMD-backed dense dot products through the shared dot helper.
- SIMD-backed recurrent projection and gate dot products where applicable.
- Lane-major same-topology population V2 scoring path for four-genome chunks.
- Scalar fallback artifact and scalar runtime path preserved.

### Workers And Population Evaluation

- `WorkerPool` for Bun and Node.js worker-based population scoring.
- `evaluatePopulation` facade with `auto`, `direct`, and `workers` modes.
- Packed dataset caching per worker.
- Binary model payload support for worker communication.
- Worker-local WASM scoring support.
- Grouped same-topology feed-forward population scoring through WASM.
- Lane-major population V2 packing for grouped scoring.
- Stable result-order preservation for mixed topology populations.
- Worker model cache keys based on binary model byte hashes to avoid tiny-parameter cache collisions.

### NEAT

- `Neat` population loop with mutation, crossover, selection, speciation distance, elitism, survival threshold, and stagnation controls.
- Innovation registry support for structural mutations.
- Dataset-backed fitness evaluation using packed datasets and `evaluatePopulation`.
- Custom async fitness function support.
- NEAT XOR example.

### Benchmarks

- Feed-forward/population benchmark suite.
- Recurrent benchmark suite.
- Benchmark profiles: `ci`, `quick`, `standard`, and `large`.
- Human-readable tables by default.
- Machine-readable JSON and CSV output via `BENCH_FORMAT`.
- Runtime, CPU, profile, workload, checksum, and SIMD metadata in benchmark output.
- Dense, WASM, SIMD, worker, population V2, recurrent handle, and recurrent training rows.
- Neataptic comparison benchmark in the parent workspace for selected migration and performance scenarios.

### Examples

- XOR training.
- Dynamic sequence input.
- WASM graph inference.
- Worker population evaluation.
- NEAT XOR evolution.
- LSTM sequence inference.
- GRU ragged sequence inference.
- Recurrent TypeScript and WASM training.
- Dense SIMD usage.
- Per-step sequence training.

### Documentation

- Installation and runtime target notes.
- Relationship to Neataptic and migration caveats.
- API overview for networks, architects, datasets, dense plans, WASM, Workers, NEAT, recurrent models, sequence training, optimizers, serialization, and benchmarks.
- Performance snapshot with measured benchmark rows and caveats.
- Validation snapshot for the alpha release gate.
- Limitations, roadmap, FAQ, and package contents sections.

### Validation

- `bunx tsc --noEmit` passed.
- `bun test` passed with 70 tests, 0 failures, and 954 expectations across 8 files.
- `bun run build` passed and generated JavaScript bundles, declarations, scalar WASM, and SIMD WASM artifacts.
- `bun run smoke` passed for Bun, Node.js, and package tarball consumption.
- `BENCH_PROFILE=ci bun run bench` passed.
- `BENCH_PROFILE=ci bun run bench:recurrent` passed.
- All package examples passed.
- Parent Neataptic comparison CI benchmark passed.
- `bun pm pack --dry-run` passed.
- `npm pack --dry-run` passed.

### Known Limitations

- Browser, CDN, AMD, CommonJS, and global `window` builds are not release targets.
- Bunaptic is not a full drop-in replacement for every Neataptic API.
- Native BLAS and GPU acceleration are not included.
- SIMD accelerates dot-product and reduction-heavy paths, but nonlinear activations still use scalar math.
- WASM feed-forward graph training currently targets `sgd` and `momentum`.
- WASM recurrent training currently targets LSTM/GRU `sgd` and `momentum`.
- Recurrent `adam` and `rmsprop` training use the TypeScript BPTT path.
- Advanced internals remain pre-1.0 and may change.
