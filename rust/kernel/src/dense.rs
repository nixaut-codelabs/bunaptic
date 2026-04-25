use core::slice;

use crate::math::{activate, dot, squared_error_sum};

#[no_mangle]
pub unsafe extern "C" fn bunaptic_dense_forward(
    input_ptr: *const f32,
    weights_ptr: *const f32,
    bias_ptr: *const f32,
    output_ptr: *mut f32,
    input_len: usize,
    output_len: usize,
    activation: u32,
) {
    let input = slice::from_raw_parts(input_ptr, input_len);
    let weights = slice::from_raw_parts(weights_ptr, input_len * output_len);
    let bias = slice::from_raw_parts(bias_ptr, output_len);
    let output = slice::from_raw_parts_mut(output_ptr, output_len);

    for out in 0..output_len {
        let mut sum = bias[out];
        let row = out * input_len;
        sum += dot(input, &weights[row..row + input_len]);
        output[out] = activate(sum, activation);
    }
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_dense_forward_plan(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    layer_sizes_ptr: *const u32,
    activations_ptr: *const u8,
    weight_starts_ptr: *const u32,
    bias_starts_ptr: *const u32,
    weights_ptr: *const f32,
    biases_ptr: *const f32,
    layer_count: usize,
) {
    let layer_sizes = slice::from_raw_parts(layer_sizes_ptr, layer_count + 1);
    let activations = slice::from_raw_parts(activations_ptr, layer_count);
    let weight_starts = slice::from_raw_parts(weight_starts_ptr, layer_count + 1);
    let bias_starts = slice::from_raw_parts(bias_starts_ptr, layer_count + 1);
    let weights = slice::from_raw_parts(weights_ptr, weight_starts[layer_count] as usize);
    let biases = slice::from_raw_parts(biases_ptr, bias_starts[layer_count] as usize);
    let output_len = layer_sizes[layer_count] as usize;
    let output = slice::from_raw_parts_mut(output_ptr, output_len);
    let input = slice::from_raw_parts(input_ptr, layer_sizes[0] as usize);
    dense_forward_plan_core(input, output, layer_sizes, activations, weight_starts, bias_starts, weights, biases, layer_count);
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_dense_forward_batch_plan(
    inputs_ptr: *const f32,
    output_ptr: *mut f32,
    layer_sizes_ptr: *const u32,
    activations_ptr: *const u8,
    weight_starts_ptr: *const u32,
    bias_starts_ptr: *const u32,
    weights_ptr: *const f32,
    biases_ptr: *const f32,
    layer_count: usize,
    batch_size: usize,
) {
    let layer_sizes = slice::from_raw_parts(layer_sizes_ptr, layer_count + 1);
    let activations = slice::from_raw_parts(activations_ptr, layer_count);
    let weight_starts = slice::from_raw_parts(weight_starts_ptr, layer_count + 1);
    let bias_starts = slice::from_raw_parts(bias_starts_ptr, layer_count + 1);
    let weights = slice::from_raw_parts(weights_ptr, weight_starts[layer_count] as usize);
    let biases = slice::from_raw_parts(biases_ptr, bias_starts[layer_count] as usize);
    let input_len = layer_sizes[0] as usize;
    let output_len = layer_sizes[layer_count] as usize;
    let inputs = slice::from_raw_parts(inputs_ptr, batch_size * input_len);
    let output = slice::from_raw_parts_mut(output_ptr, batch_size * output_len);
    for sample in 0..batch_size {
        dense_forward_plan_core(
            &inputs[sample * input_len..(sample + 1) * input_len],
            &mut output[sample * output_len..(sample + 1) * output_len],
            layer_sizes,
            activations,
            weight_starts,
            bias_starts,
            weights,
            biases,
            layer_count,
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_dense_evaluate_dataset_plan(
    inputs_ptr: *const f32,
    targets_ptr: *const f32,
    layer_sizes_ptr: *const u32,
    activations_ptr: *const u8,
    weight_starts_ptr: *const u32,
    bias_starts_ptr: *const u32,
    weights_ptr: *const f32,
    biases_ptr: *const f32,
    layer_count: usize,
    sample_count: usize,
) -> f32 {
    let layer_sizes = slice::from_raw_parts(layer_sizes_ptr, layer_count + 1);
    let activations = slice::from_raw_parts(activations_ptr, layer_count);
    let weight_starts = slice::from_raw_parts(weight_starts_ptr, layer_count + 1);
    let bias_starts = slice::from_raw_parts(bias_starts_ptr, layer_count + 1);
    let weights = slice::from_raw_parts(weights_ptr, weight_starts[layer_count] as usize);
    let biases = slice::from_raw_parts(biases_ptr, bias_starts[layer_count] as usize);
    let input_len = layer_sizes[0] as usize;
    let output_len = layer_sizes[layer_count] as usize;
    let inputs = slice::from_raw_parts(inputs_ptr, sample_count * input_len);
    let targets = slice::from_raw_parts(targets_ptr, sample_count * output_len);
    let mut output = vec![0.0; output_len];
    let mut error = 0.0;
    for sample in 0..sample_count {
        dense_forward_plan_core(&inputs[sample * input_len..(sample + 1) * input_len], &mut output, layer_sizes, activations, weight_starts, bias_starts, weights, biases, layer_count);
        error += squared_error_sum(&targets[sample * output_len..(sample + 1) * output_len], &output);
    }
    error / (sample_count * output_len) as f32
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_mse(target_ptr: *const f32, output_ptr: *const f32, len: usize) -> f32 {
    let target = slice::from_raw_parts(target_ptr, len);
    let output = slice::from_raw_parts(output_ptr, len);
    squared_error_sum(target, output) / len as f32
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_forward_plan_core(
    input: &[f32],
    output: &mut [f32],
    layer_sizes: &[u32],
    activations: &[u8],
    weight_starts: &[u32],
    bias_starts: &[u32],
    weights: &[f32],
    biases: &[f32],
    layer_count: usize,
) {
    if layer_count == 0 {
        output.copy_from_slice(input);
        return;
    }

    let scratch_len = layer_sizes.iter().map(|value| *value as usize).max().unwrap_or(input.len());
    let mut current = vec![0.0; scratch_len];
    let mut next = vec![0.0; scratch_len];
    current[..input.len()].copy_from_slice(input);
    let mut current_len = input.len();

    for layer in 0..layer_count {
        let input_len = layer_sizes[layer] as usize;
        let output_len = layer_sizes[layer + 1] as usize;
        let weight_start = weight_starts[layer] as usize;
        let bias_start = bias_starts[layer] as usize;
        let input_slice = &current[..current_len.min(input_len)];
        for out in 0..output_len {
            let row = weight_start + out * input_len;
            let sum = biases[bias_start + out] + dot(input_slice, &weights[row..row + input_len]);
            next[out] = activate(sum, activations[layer] as u32);
        }
        if layer + 1 == layer_count {
            output.copy_from_slice(&next[..output_len]);
        } else {
            core::mem::swap(&mut current, &mut next);
            current_len = output_len;
        }
    }
}
