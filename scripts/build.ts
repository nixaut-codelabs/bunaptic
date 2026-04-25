import { $ } from "bun";

const entries = [
  "src/index.ts",
  "src/bun.ts",
  "src/node.ts",
  "src/runtime/evaluate-worker.ts",
];

const wasmSource = "rust/kernel/target/wasm32-unknown-unknown/release/bunaptic_kernel.wasm";
const wasmTarget = "dist/bunaptic_kernel.wasm";
const wasmSimdTarget = "dist/bunaptic_kernel_simd.wasm";

await $`rm -rf dist`;
await $`cargo build --manifest-path rust/kernel/Cargo.toml --release --target wasm32-unknown-unknown`;
const scalarWasm = await Bun.file(wasmSource).arrayBuffer();
await $`cargo build --manifest-path rust/kernel/Cargo.toml --release --target wasm32-unknown-unknown`.env({ ...Bun.env, RUSTFLAGS: "-C target-feature=+simd128" });
const simdWasm = await Bun.file(wasmSource).arrayBuffer();

const result = await Bun.build({
  entrypoints: entries,
  outdir: "dist",
  target: "bun",
  format: "esm",
  sourcemap: "external",
  splitting: false,
});

if (!result.success) {
  for (const log of result.logs) console.error(log);
  process.exit(1);
}

await $`bunx tsc -p tsconfig.build.json`;
await Bun.write(wasmTarget, scalarWasm);
await Bun.write(wasmSimdTarget, simdWasm);

console.log(`Built ${entries.length} entrypoints, declarations, scalar WASM, and SIMD WASM.`);
