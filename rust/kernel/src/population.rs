use core::slice;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::*;

use crate::graph::graph_forward_core;
use crate::math::{activate, squared_error_sum};

#[no_mangle]
pub unsafe extern "C" fn bunaptic_evaluate_population(
    inputs_ptr: *const f32,
    targets_ptr: *const f32,
    scores_ptr: *mut f32,
    node_kinds_ptr: *const u8,
    activation_kinds_ptr: *const u8,
    from_ptr: *const u32,
    gater_ptr: *const i32,
    connection_kinds_ptr: *const u8,
    incoming_starts_ptr: *const u32,
    incoming_ptr: *const u32,
    weights_ptr: *const f32,
    biases_ptr: *const f32,
    input_len: usize,
    node_count: usize,
    output_len: usize,
    connection_len: usize,
    sample_count: usize,
    population_size: usize,
) {
    let inputs = slice::from_raw_parts(inputs_ptr, sample_count * input_len);
    let targets = slice::from_raw_parts(targets_ptr, sample_count * output_len);
    let scores = slice::from_raw_parts_mut(scores_ptr, population_size);
    let node_kinds = slice::from_raw_parts(node_kinds_ptr, node_count);
    let activation_kinds = slice::from_raw_parts(activation_kinds_ptr, node_count);
    let from = slice::from_raw_parts(from_ptr, connection_len);
    let gater = slice::from_raw_parts(gater_ptr, connection_len);
    let connection_kinds = slice::from_raw_parts(connection_kinds_ptr, connection_len);
    let incoming_starts = slice::from_raw_parts(incoming_starts_ptr, node_count + 1);
    let incoming = slice::from_raw_parts(incoming_ptr, connection_len);
    let weights = slice::from_raw_parts(weights_ptr, population_size * connection_len);
    let biases = slice::from_raw_parts(biases_ptr, population_size * node_count);
    let mut activations = vec![0.0; node_count];
    let mut states = vec![0.0; node_count];
    let mut previous = vec![0.0; node_count];
    let mut output = vec![0.0; output_len];

    for genome in 0..population_size {
        let genome_weights = &weights[genome * connection_len..(genome + 1) * connection_len];
        let genome_biases = &biases[genome * node_count..(genome + 1) * node_count];
        let mut error = 0.0;
        for sample in 0..sample_count {
            activations.fill(0.0);
            states.fill(0.0);
            previous.fill(0.0);
            graph_forward_core(
                &inputs[sample * input_len..(sample + 1) * input_len],
                &mut activations,
                &mut states,
                &mut previous,
                node_kinds,
                activation_kinds,
                genome_biases,
                from,
                gater,
                genome_weights,
                connection_kinds,
                incoming_starts,
                incoming,
                &mut output,
                input_len,
                node_count,
                output_len,
                0,
            );
            error += squared_error_sum(&targets[sample * output_len..(sample + 1) * output_len], &output);
        }
        scores[genome] = -(error / (sample_count * output_len) as f32);
    }
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_evaluate_population_v2(
    inputs_ptr: *const f32,
    targets_ptr: *const f32,
    scores_ptr: *mut f32,
    node_kinds_ptr: *const u8,
    activation_kinds_ptr: *const u8,
    from_ptr: *const u32,
    gater_ptr: *const i32,
    connection_kinds_ptr: *const u8,
    incoming_starts_ptr: *const u32,
    incoming_ptr: *const u32,
    lane_major_weights_ptr: *const f32,
    lane_major_biases_ptr: *const f32,
    input_len: usize,
    node_count: usize,
    output_len: usize,
    connection_len: usize,
    sample_count: usize,
    population_size: usize,
) {
    let inputs = slice::from_raw_parts(inputs_ptr, sample_count * input_len);
    let targets = slice::from_raw_parts(targets_ptr, sample_count * output_len);
    let scores = slice::from_raw_parts_mut(scores_ptr, population_size);
    let node_kinds = slice::from_raw_parts(node_kinds_ptr, node_count);
    let activation_kinds = slice::from_raw_parts(activation_kinds_ptr, node_count);
    let from = slice::from_raw_parts(from_ptr, connection_len);
    let gater = slice::from_raw_parts(gater_ptr, connection_len);
    let connection_kinds = slice::from_raw_parts(connection_kinds_ptr, connection_len);
    let incoming_starts = slice::from_raw_parts(incoming_starts_ptr, node_count + 1);
    let incoming = slice::from_raw_parts(incoming_ptr, connection_len);
    let weights = slice::from_raw_parts(lane_major_weights_ptr, population_size * connection_len);
    let biases = slice::from_raw_parts(lane_major_biases_ptr, population_size * node_count);
    scores.fill(0.0);

    let mut genome = 0usize;
    while genome + 4 <= population_size {
        evaluate_population_chunk4(
            genome,
            inputs,
            targets,
            scores,
            node_kinds,
            activation_kinds,
            from,
            gater,
            connection_kinds,
            incoming_starts,
            incoming,
            weights,
            biases,
            input_len,
            node_count,
            output_len,
            connection_len,
            sample_count,
            population_size,
        );
        genome += 4;
    }

    let mut activations = vec![0.0; node_count];
    let mut states = vec![0.0; node_count];
    let mut previous = vec![0.0; node_count];
    let mut output = vec![0.0; output_len];
    let mut genome_weights = vec![0.0; connection_len];
    let mut genome_biases = vec![0.0; node_count];
    while genome < population_size {
        for conn in 0..connection_len {
            genome_weights[conn] = weights[conn * population_size + genome];
        }
        for node in 0..node_count {
            genome_biases[node] = biases[node * population_size + genome];
        }
        let mut error = 0.0;
        for sample in 0..sample_count {
            activations.fill(0.0);
            states.fill(0.0);
            previous.fill(0.0);
            graph_forward_core(
                &inputs[sample * input_len..(sample + 1) * input_len],
                &mut activations,
                &mut states,
                &mut previous,
                node_kinds,
                activation_kinds,
                &genome_biases,
                from,
                gater,
                &genome_weights,
                connection_kinds,
                incoming_starts,
                incoming,
                &mut output,
                input_len,
                node_count,
                output_len,
                0,
            );
            error += squared_error_sum(&targets[sample * output_len..(sample + 1) * output_len], &output);
        }
        scores[genome] = -(error / (sample_count * output_len) as f32);
        genome += 1;
    }
}

#[inline]
fn add_lane4_product(sums: &mut [f32; 4], sources: [f32; 4], weights: &[f32], weight_base: usize) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        let current = v128_load(sums.as_ptr() as *const v128);
        let source = v128_load(sources.as_ptr() as *const v128);
        let weight = v128_load(weights.as_ptr().add(weight_base) as *const v128);
        let next = f32x4_add(current, f32x4_mul(source, weight));
        v128_store(sums.as_mut_ptr() as *mut v128, next);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        for lane in 0..4 {
            sums[lane] += sources[lane] * weights[weight_base + lane];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_population_chunk4(
    start_genome: usize,
    inputs: &[f32],
    targets: &[f32],
    scores: &mut [f32],
    node_kinds: &[u8],
    activation_kinds: &[u8],
    from: &[u32],
    gater: &[i32],
    connection_kinds: &[u8],
    incoming_starts: &[u32],
    incoming: &[u32],
    weights: &[f32],
    biases: &[f32],
    input_len: usize,
    node_count: usize,
    output_len: usize,
    _connection_len: usize,
    sample_count: usize,
    population_size: usize,
) {
    let mut activations = vec![vec![0.0; node_count]; 4];
    let mut states = vec![vec![0.0; node_count]; 4];
    let previous = vec![0.0; node_count];
    let output_start = node_count - output_len;
    let mut errors = [0.0f32; 4];
    for sample in 0..sample_count {
        for lane in 0..4 {
            activations[lane].fill(0.0);
            states[lane].fill(0.0);
            activations[lane][..input_len].copy_from_slice(&inputs[sample * input_len..(sample + 1) * input_len]);
        }

        for node in input_len..node_count {
            if node_kinds[node] == 0 {
                continue;
            }
            if node_kinds[node] == 3 {
                for lane in 0..4 {
                    activations[lane][node] = 1.0;
                }
                continue;
            }
            let mut sums = [0.0f32; 4];
            for lane in 0..4 {
                sums[lane] = biases[node * population_size + start_genome + lane];
            }
            for cursor in incoming_starts[node] as usize..incoming_starts[node + 1] as usize {
                let conn = incoming[cursor] as usize;
                let mut source_values = [0.0f32; 4];
                for lane in 0..4 {
                    let source = if connection_kinds[conn] == 0 { &activations[lane] } else { &previous };
                    let gain = if gater[conn] >= 0 { activations[lane][gater[conn] as usize] } else { 1.0 };
                    source_values[lane] = source[from[conn] as usize] * gain;
                }
                add_lane4_product(&mut sums, source_values, weights, conn * population_size + start_genome);
            }
            for lane in 0..4 {
                states[lane][node] = sums[lane];
                activations[lane][node] = activate(sums[lane], activation_kinds[node] as u32);
            }
        }

        for lane in 0..4 {
            for out in 0..output_len {
                let delta = targets[sample * output_len + out] - activations[lane][output_start + out];
                errors[lane] += delta * delta;
            }
        }
    }
    for lane in 0..4 {
        scores[start_genome + lane] = -(errors[lane] / (sample_count * output_len) as f32);
    }
}
