use core::slice;

use crate::math::{activate, derivative, dot, sigmoid};

#[no_mangle]
pub unsafe extern "C" fn bunaptic_lstm_forward_sequence(
    values_ptr: *const f32,
    output_ptr: *mut f32,
    hidden_ptr: *mut f32,
    cell_ptr: *mut f32,
    input_weights_ptr: *const f32,
    hidden_weights_ptr: *const f32,
    biases_ptr: *const f32,
    output_weights_ptr: *const f32,
    output_biases_ptr: *const f32,
    input_len: usize,
    hidden_len: usize,
    output_len: usize,
    steps: usize,
    output_activation: u32,
) {
    let values = slice::from_raw_parts(values_ptr, steps * input_len);
    let output = slice::from_raw_parts_mut(output_ptr, steps * output_len);
    let hidden = slice::from_raw_parts_mut(hidden_ptr, hidden_len);
    let cell = slice::from_raw_parts_mut(cell_ptr, hidden_len);
    let input_weights = slice::from_raw_parts(input_weights_ptr, 4 * hidden_len * input_len);
    let hidden_weights = slice::from_raw_parts(hidden_weights_ptr, 4 * hidden_len * hidden_len);
    let biases = slice::from_raw_parts(biases_ptr, 4 * hidden_len);
    let output_weights = slice::from_raw_parts(output_weights_ptr, output_len * hidden_len);
    let output_biases = slice::from_raw_parts(output_biases_ptr, output_len);
    let mut next_hidden = vec![0.0; hidden_len];
    let mut next_cell = vec![0.0; hidden_len];

    for step in 0..steps {
        let input = &values[step * input_len..(step + 1) * input_len];
        lstm_step(input, hidden, cell, &mut next_hidden, &mut next_cell, input_weights, hidden_weights, biases, input_len, hidden_len);
        hidden.copy_from_slice(&next_hidden);
        cell.copy_from_slice(&next_cell);
        project_recurrent_output(hidden, &mut output[step * output_len..(step + 1) * output_len], output_weights, output_biases, hidden_len, output_len, output_activation);
    }
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_lstm_forward_ragged(
    values_ptr: *const f32,
    offsets_ptr: *const u32,
    output_ptr: *mut f32,
    hidden_ptr: *mut f32,
    cell_ptr: *mut f32,
    input_weights_ptr: *const f32,
    hidden_weights_ptr: *const f32,
    biases_ptr: *const f32,
    output_weights_ptr: *const f32,
    output_biases_ptr: *const f32,
    input_len: usize,
    hidden_len: usize,
    output_len: usize,
    sequence_count: usize,
    value_count: usize,
    output_activation: u32,
) {
    let values = slice::from_raw_parts(values_ptr, value_count);
    let offsets = slice::from_raw_parts(offsets_ptr, sequence_count + 1);
    let output = slice::from_raw_parts_mut(output_ptr, sequence_count * output_len);
    let hidden = slice::from_raw_parts_mut(hidden_ptr, hidden_len);
    let cell = slice::from_raw_parts_mut(cell_ptr, hidden_len);
    let input_weights = slice::from_raw_parts(input_weights_ptr, 4 * hidden_len * input_len);
    let hidden_weights = slice::from_raw_parts(hidden_weights_ptr, 4 * hidden_len * hidden_len);
    let biases = slice::from_raw_parts(biases_ptr, 4 * hidden_len);
    let output_weights = slice::from_raw_parts(output_weights_ptr, output_len * hidden_len);
    let output_biases = slice::from_raw_parts(output_biases_ptr, output_len);
    let mut next_hidden = vec![0.0; hidden_len];
    let mut next_cell = vec![0.0; hidden_len];

    for sequence in 0..sequence_count {
        hidden.fill(0.0);
        cell.fill(0.0);
        let mut cursor = offsets[sequence] as usize;
        let end = offsets[sequence + 1] as usize;
        while cursor < end {
            lstm_step(&values[cursor..cursor + input_len], hidden, cell, &mut next_hidden, &mut next_cell, input_weights, hidden_weights, biases, input_len, hidden_len);
            hidden.copy_from_slice(&next_hidden);
            cell.copy_from_slice(&next_cell);
            cursor += input_len;
        }
        project_recurrent_output(hidden, &mut output[sequence * output_len..(sequence + 1) * output_len], output_weights, output_biases, hidden_len, output_len, output_activation);
    }
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_gru_forward_sequence(
    values_ptr: *const f32,
    output_ptr: *mut f32,
    hidden_ptr: *mut f32,
    input_weights_ptr: *const f32,
    hidden_weights_ptr: *const f32,
    biases_ptr: *const f32,
    output_weights_ptr: *const f32,
    output_biases_ptr: *const f32,
    input_len: usize,
    hidden_len: usize,
    output_len: usize,
    steps: usize,
    output_activation: u32,
) {
    let values = slice::from_raw_parts(values_ptr, steps * input_len);
    let output = slice::from_raw_parts_mut(output_ptr, steps * output_len);
    let hidden = slice::from_raw_parts_mut(hidden_ptr, hidden_len);
    let input_weights = slice::from_raw_parts(input_weights_ptr, 3 * hidden_len * input_len);
    let hidden_weights = slice::from_raw_parts(hidden_weights_ptr, 3 * hidden_len * hidden_len);
    let biases = slice::from_raw_parts(biases_ptr, 3 * hidden_len);
    let output_weights = slice::from_raw_parts(output_weights_ptr, output_len * hidden_len);
    let output_biases = slice::from_raw_parts(output_biases_ptr, output_len);
    let mut next_hidden = vec![0.0; hidden_len];

    for step in 0..steps {
        gru_step(&values[step * input_len..(step + 1) * input_len], hidden, &mut next_hidden, input_weights, hidden_weights, biases, input_len, hidden_len);
        hidden.copy_from_slice(&next_hidden);
        project_recurrent_output(hidden, &mut output[step * output_len..(step + 1) * output_len], output_weights, output_biases, hidden_len, output_len, output_activation);
    }
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_gru_forward_ragged(
    values_ptr: *const f32,
    offsets_ptr: *const u32,
    output_ptr: *mut f32,
    hidden_ptr: *mut f32,
    input_weights_ptr: *const f32,
    hidden_weights_ptr: *const f32,
    biases_ptr: *const f32,
    output_weights_ptr: *const f32,
    output_biases_ptr: *const f32,
    input_len: usize,
    hidden_len: usize,
    output_len: usize,
    sequence_count: usize,
    value_count: usize,
    output_activation: u32,
) {
    let values = slice::from_raw_parts(values_ptr, value_count);
    let offsets = slice::from_raw_parts(offsets_ptr, sequence_count + 1);
    let output = slice::from_raw_parts_mut(output_ptr, sequence_count * output_len);
    let hidden = slice::from_raw_parts_mut(hidden_ptr, hidden_len);
    let input_weights = slice::from_raw_parts(input_weights_ptr, 3 * hidden_len * input_len);
    let hidden_weights = slice::from_raw_parts(hidden_weights_ptr, 3 * hidden_len * hidden_len);
    let biases = slice::from_raw_parts(biases_ptr, 3 * hidden_len);
    let output_weights = slice::from_raw_parts(output_weights_ptr, output_len * hidden_len);
    let output_biases = slice::from_raw_parts(output_biases_ptr, output_len);
    let mut next_hidden = vec![0.0; hidden_len];

    for sequence in 0..sequence_count {
        hidden.fill(0.0);
        let mut cursor = offsets[sequence] as usize;
        let end = offsets[sequence + 1] as usize;
        while cursor < end {
            gru_step(&values[cursor..cursor + input_len], hidden, &mut next_hidden, input_weights, hidden_weights, biases, input_len, hidden_len);
            hidden.copy_from_slice(&next_hidden);
            cursor += input_len;
        }
        project_recurrent_output(hidden, &mut output[sequence * output_len..(sequence + 1) * output_len], output_weights, output_biases, hidden_len, output_len, output_activation);
    }
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_train_recurrent_sequences(
    values_ptr: *const f32,
    offsets_ptr: *const u32,
    targets_ptr: *const f32,
    target_offsets_ptr: *const u32,
    input_weights_ptr: *mut f32,
    hidden_weights_ptr: *mut f32,
    biases_ptr: *mut f32,
    output_weights_ptr: *mut f32,
    output_biases_ptr: *mut f32,
    result_ptr: *mut f32,
    kind: u32,
    input_len: usize,
    hidden_len: usize,
    output_len: usize,
    sample_count: usize,
    value_count: usize,
    target_count: usize,
    target_mode: u32,
    output_activation: u32,
    iterations: usize,
    target_error: f32,
    rate: f32,
    momentum: f32,
    clip: f32,
    truncated_steps: usize,
) {
    let values = slice::from_raw_parts(values_ptr, value_count);
    let offsets = slice::from_raw_parts(offsets_ptr, sample_count + 1);
    let targets = slice::from_raw_parts(targets_ptr, target_count);
    let target_offsets = if target_mode == 1 { Some(slice::from_raw_parts(target_offsets_ptr, sample_count + 1)) } else { None };
    let gates = if kind == 1 { 4 } else { 3 };
    let input_weights = slice::from_raw_parts_mut(input_weights_ptr, gates * hidden_len * input_len);
    let hidden_weights = slice::from_raw_parts_mut(hidden_weights_ptr, gates * hidden_len * hidden_len);
    let biases = slice::from_raw_parts_mut(biases_ptr, gates * hidden_len);
    let output_weights = slice::from_raw_parts_mut(output_weights_ptr, output_len * hidden_len);
    let output_biases = slice::from_raw_parts_mut(output_biases_ptr, output_len);
    let result = slice::from_raw_parts_mut(result_ptr, 2);
    let mut previous_input = vec![0.0; input_weights.len()];
    let mut previous_hidden = vec![0.0; hidden_weights.len()];
    let mut previous_biases = vec![0.0; biases.len()];
    let mut previous_output_weights = vec![0.0; output_weights.len()];
    let mut previous_output_biases = vec![0.0; output_biases.len()];
    let mut final_error = f32::INFINITY;
    let mut completed = 0usize;

    for iteration in 0..iterations {
        let mut error = 0.0;
        for sample in 0..sample_count {
            let start = offsets[sample] as usize;
            let end = offsets[sample + 1] as usize;
            let steps = (end - start) / input_len;
            let target_start = if target_mode == 1 { target_offsets.unwrap()[sample] as usize } else { sample * output_len };
            let target_end = if target_mode == 1 { target_offsets.unwrap()[sample + 1] as usize } else { target_start + output_len };
            let train_error = if kind == 1 {
                train_lstm_sample(
                    &values[start..end],
                    &targets[target_start..target_end],
                    input_weights,
                    hidden_weights,
                    biases,
                    output_weights,
                    output_biases,
                    &mut previous_input,
                    &mut previous_hidden,
                    &mut previous_biases,
                    &mut previous_output_weights,
                    &mut previous_output_biases,
                    input_len,
                    hidden_len,
                    output_len,
                    steps,
                    target_mode,
                    output_activation,
                    rate,
                    momentum,
                    clip,
                    truncated_steps,
                )
            } else {
                train_gru_sample(
                    &values[start..end],
                    &targets[target_start..target_end],
                    input_weights,
                    hidden_weights,
                    biases,
                    output_weights,
                    output_biases,
                    &mut previous_input,
                    &mut previous_hidden,
                    &mut previous_biases,
                    &mut previous_output_weights,
                    &mut previous_output_biases,
                    input_len,
                    hidden_len,
                    output_len,
                    steps,
                    target_mode,
                    output_activation,
                    rate,
                    momentum,
                    clip,
                    truncated_steps,
                )
            };
            error += train_error;
        }
        final_error = error / sample_count as f32;
        completed = iteration + 1;
        if target_error >= 0.0 && final_error <= target_error {
            break;
        }
    }
    result[0] = final_error;
    result[1] = completed as f32;
}

#[allow(clippy::too_many_arguments)]
fn train_lstm_sample(
    input: &[f32],
    target: &[f32],
    input_weights: &mut [f32],
    hidden_weights: &mut [f32],
    biases: &mut [f32],
    output_weights: &mut [f32],
    output_biases: &mut [f32],
    previous_input: &mut [f32],
    previous_hidden: &mut [f32],
    previous_biases: &mut [f32],
    previous_output_weights: &mut [f32],
    previous_output_biases: &mut [f32],
    input_len: usize,
    hidden_len: usize,
    output_len: usize,
    steps: usize,
    target_mode: u32,
    output_activation: u32,
    rate: f32,
    momentum: f32,
    clip: f32,
    truncated_steps: usize,
) -> f32 {
    let gates_count = 4;
    let mut hidden = vec![0.0; (steps + 1) * hidden_len];
    let mut cell = vec![0.0; (steps + 1) * hidden_len];
    let mut gates = vec![0.0; steps * gates_count * hidden_len];
    for step in 0..steps {
        let prev_hidden = hidden[step * hidden_len..(step + 1) * hidden_len].to_vec();
        let prev_cell = cell[step * hidden_len..(step + 1) * hidden_len].to_vec();
        for h in 0..hidden_len {
            let i = sigmoid(recurrent_gate_sum(0, h, &input[step * input_len..(step + 1) * input_len], &prev_hidden, None, input_weights, hidden_weights, biases, input_len, hidden_len));
            let f = sigmoid(recurrent_gate_sum(1, h, &input[step * input_len..(step + 1) * input_len], &prev_hidden, None, input_weights, hidden_weights, biases, input_len, hidden_len));
            let g = recurrent_gate_sum(2, h, &input[step * input_len..(step + 1) * input_len], &prev_hidden, None, input_weights, hidden_weights, biases, input_len, hidden_len).tanh();
            let o = sigmoid(recurrent_gate_sum(3, h, &input[step * input_len..(step + 1) * input_len], &prev_hidden, None, input_weights, hidden_weights, biases, input_len, hidden_len));
            gates[(step * gates_count + 0) * hidden_len + h] = i;
            gates[(step * gates_count + 1) * hidden_len + h] = f;
            gates[(step * gates_count + 2) * hidden_len + h] = g;
            gates[(step * gates_count + 3) * hidden_len + h] = o;
            cell[(step + 1) * hidden_len + h] = f * prev_cell[h] + i * g;
            hidden[(step + 1) * hidden_len + h] = o * cell[(step + 1) * hidden_len + h].tanh();
        }
    }
    let mut grad_input = vec![0.0; input_weights.len()];
    let mut grad_hidden = vec![0.0; hidden_weights.len()];
    let mut grad_biases = vec![0.0; biases.len()];
    let mut grad_output_weights = vec![0.0; output_weights.len()];
    let mut grad_output_biases = vec![0.0; output_biases.len()];
    let (error, output_hidden_grad) = recurrent_output_gradients(&hidden, target, &mut grad_output_weights, &mut grad_output_biases, output_weights, output_biases, hidden_len, output_len, steps, target_mode, output_activation);
    let mut d_hidden = vec![0.0; hidden_len];
    let mut d_cell = vec![0.0; hidden_len];
    let start = recurrent_back_start(steps, truncated_steps);
    for step in (start..steps).rev() {
        for h in 0..hidden_len {
            d_hidden[h] += output_hidden_grad[step * hidden_len + h];
        }
        let prev_hidden_values = hidden[step * hidden_len..(step + 1) * hidden_len].to_vec();
        let prev_cell_values = cell[step * hidden_len..(step + 1) * hidden_len].to_vec();
        let cell_values = cell[(step + 1) * hidden_len..(step + 2) * hidden_len].to_vec();
        let mut next_hidden = vec![0.0; hidden_len];
        let mut next_cell = vec![0.0; hidden_len];
        for h in 0..hidden_len {
            let i = gates[(step * gates_count + 0) * hidden_len + h];
            let f = gates[(step * gates_count + 1) * hidden_len + h];
            let g = gates[(step * gates_count + 2) * hidden_len + h];
            let o = gates[(step * gates_count + 3) * hidden_len + h];
            let tanh_cell = cell_values[h].tanh();
            let dc = d_cell[h] + d_hidden[h] * o * (1.0 - tanh_cell * tanh_cell);
            let deltas = [
                dc * g * i * (1.0 - i),
                dc * prev_cell_values[h] * f * (1.0 - f),
                dc * i * (1.0 - g * g),
                d_hidden[h] * tanh_cell * o * (1.0 - o),
            ];
            next_cell[h] += dc * f;
            for gate in 0..gates_count {
                add_recurrent_gate_grad(gate, h, deltas[gate], input, step * input_len, &prev_hidden_values, &prev_hidden_values, &mut grad_input, &mut grad_hidden, &mut grad_biases, hidden_weights, &mut next_hidden, None, input_len, hidden_len);
            }
        }
        d_hidden = next_hidden;
        d_cell = next_cell;
    }
    apply_recurrent_updates(input_weights, &grad_input, previous_input, rate, momentum, clip);
    apply_recurrent_updates(hidden_weights, &grad_hidden, previous_hidden, rate, momentum, clip);
    apply_recurrent_updates(biases, &grad_biases, previous_biases, rate, momentum, clip);
    apply_recurrent_updates(output_weights, &grad_output_weights, previous_output_weights, rate, momentum, clip);
    apply_recurrent_updates(output_biases, &grad_output_biases, previous_output_biases, rate, momentum, clip);
    error
}

#[allow(clippy::too_many_arguments)]
fn train_gru_sample(
    input: &[f32],
    target: &[f32],
    input_weights: &mut [f32],
    hidden_weights: &mut [f32],
    biases: &mut [f32],
    output_weights: &mut [f32],
    output_biases: &mut [f32],
    previous_input: &mut [f32],
    previous_hidden: &mut [f32],
    previous_biases: &mut [f32],
    previous_output_weights: &mut [f32],
    previous_output_biases: &mut [f32],
    input_len: usize,
    hidden_len: usize,
    output_len: usize,
    steps: usize,
    target_mode: u32,
    output_activation: u32,
    rate: f32,
    momentum: f32,
    clip: f32,
    truncated_steps: usize,
) -> f32 {
    let gates_count = 3;
    let mut hidden = vec![0.0; (steps + 1) * hidden_len];
    let mut gates = vec![0.0; steps * gates_count * hidden_len];
    for step in 0..steps {
        let prev_hidden = hidden[step * hidden_len..(step + 1) * hidden_len].to_vec();
        let mut reset = vec![0.0; hidden_len];
        let mut update = vec![0.0; hidden_len];
        for h in 0..hidden_len {
            update[h] = sigmoid(recurrent_gate_sum(0, h, &input[step * input_len..(step + 1) * input_len], &prev_hidden, None, input_weights, hidden_weights, biases, input_len, hidden_len));
            reset[h] = sigmoid(recurrent_gate_sum(1, h, &input[step * input_len..(step + 1) * input_len], &prev_hidden, None, input_weights, hidden_weights, biases, input_len, hidden_len));
        }
        for h in 0..hidden_len {
            let n = recurrent_gate_sum(2, h, &input[step * input_len..(step + 1) * input_len], &prev_hidden, Some(&reset), input_weights, hidden_weights, biases, input_len, hidden_len).tanh();
            gates[(step * gates_count + 0) * hidden_len + h] = update[h];
            gates[(step * gates_count + 1) * hidden_len + h] = reset[h];
            gates[(step * gates_count + 2) * hidden_len + h] = n;
            hidden[(step + 1) * hidden_len + h] = (1.0 - update[h]) * n + update[h] * prev_hidden[h];
        }
    }
    let mut grad_input = vec![0.0; input_weights.len()];
    let mut grad_hidden = vec![0.0; hidden_weights.len()];
    let mut grad_biases = vec![0.0; biases.len()];
    let mut grad_output_weights = vec![0.0; output_weights.len()];
    let mut grad_output_biases = vec![0.0; output_biases.len()];
    let (error, output_hidden_grad) = recurrent_output_gradients(&hidden, target, &mut grad_output_weights, &mut grad_output_biases, output_weights, output_biases, hidden_len, output_len, steps, target_mode, output_activation);
    let mut d_hidden = vec![0.0; hidden_len];
    let start = recurrent_back_start(steps, truncated_steps);
    for step in (start..steps).rev() {
        for h in 0..hidden_len {
            d_hidden[h] += output_hidden_grad[step * hidden_len + h];
        }
        let prev_hidden_values = hidden[step * hidden_len..(step + 1) * hidden_len].to_vec();
        let mut next_hidden = vec![0.0; hidden_len];
        let mut reset_grad = vec![0.0; hidden_len];
        let mut update_pre = vec![0.0; hidden_len];
        let mut reset_pre = vec![0.0; hidden_len];
        let mut candidate_pre = vec![0.0; hidden_len];
        for h in 0..hidden_len {
            let z = gates[(step * gates_count + 0) * hidden_len + h];
            let r = gates[(step * gates_count + 1) * hidden_len + h];
            let n = gates[(step * gates_count + 2) * hidden_len + h];
            next_hidden[h] += d_hidden[h] * z;
            update_pre[h] = d_hidden[h] * (prev_hidden_values[h] - n) * z * (1.0 - z);
            candidate_pre[h] = d_hidden[h] * (1.0 - z) * (1.0 - n * n);
            let scaled_hidden: Vec<f32> = prev_hidden_values.iter().map(|value| value * r).collect();
            add_recurrent_gate_grad(2, h, candidate_pre[h], input, step * input_len, &prev_hidden_values, &scaled_hidden, &mut grad_input, &mut grad_hidden, &mut grad_biases, hidden_weights, &mut next_hidden, Some(&mut reset_grad), input_len, hidden_len);
        }
        for h in 0..hidden_len {
            let r = gates[(step * gates_count + 1) * hidden_len + h];
            reset_pre[h] = reset_grad[h] * r * (1.0 - r);
            add_recurrent_gate_grad(0, h, update_pre[h], input, step * input_len, &prev_hidden_values, &prev_hidden_values, &mut grad_input, &mut grad_hidden, &mut grad_biases, hidden_weights, &mut next_hidden, None, input_len, hidden_len);
            add_recurrent_gate_grad(1, h, reset_pre[h], input, step * input_len, &prev_hidden_values, &prev_hidden_values, &mut grad_input, &mut grad_hidden, &mut grad_biases, hidden_weights, &mut next_hidden, None, input_len, hidden_len);
        }
        d_hidden = next_hidden;
    }
    apply_recurrent_updates(input_weights, &grad_input, previous_input, rate, momentum, clip);
    apply_recurrent_updates(hidden_weights, &grad_hidden, previous_hidden, rate, momentum, clip);
    apply_recurrent_updates(biases, &grad_biases, previous_biases, rate, momentum, clip);
    apply_recurrent_updates(output_weights, &grad_output_weights, previous_output_weights, rate, momentum, clip);
    apply_recurrent_updates(output_biases, &grad_output_biases, previous_output_biases, rate, momentum, clip);
    error
}

#[allow(clippy::too_many_arguments)]
fn recurrent_output_gradients(
    hidden: &[f32],
    target: &[f32],
    grad_output_weights: &mut [f32],
    grad_output_biases: &mut [f32],
    output_weights: &[f32],
    output_biases: &[f32],
    hidden_len: usize,
    output_len: usize,
    steps: usize,
    target_mode: u32,
    output_activation: u32,
) -> (f32, Vec<f32>) {
    let output_count = if target_mode == 1 { steps } else { 1 };
    let scale = 2.0 / (output_count * output_len) as f32;
    let mut error = 0.0;
    let mut hidden_grad = vec![0.0; steps * hidden_len];
    for item in 0..output_count {
        let hidden_step = if target_mode == 1 { item + 1 } else { steps };
        let hidden_slice = &hidden[hidden_step * hidden_len..(hidden_step + 1) * hidden_len];
        for out in 0..output_len {
            let mut sum = output_biases[out];
            sum += dot(hidden_slice, &output_weights[out * hidden_len..(out + 1) * hidden_len]);
            let actual = activate(sum, output_activation);
            let delta = actual - target[item * output_len + out];
            error += delta * delta;
            let d_out = scale * delta * derivative(actual, sum, output_activation);
            grad_output_biases[out] += d_out;
            for h in 0..hidden_len {
                grad_output_weights[out * hidden_len + h] += d_out * hidden_slice[h];
                hidden_grad[(hidden_step - 1) * hidden_len + h] += d_out * output_weights[out * hidden_len + h];
            }
        }
    }
    (error / (output_count * output_len) as f32, hidden_grad)
}

#[allow(clippy::too_many_arguments)]
fn add_recurrent_gate_grad(
    gate: usize,
    hidden_index: usize,
    delta: f32,
    input: &[f32],
    input_offset: usize,
    _prev_hidden: &[f32],
    hidden_input: &[f32],
    grad_input: &mut [f32],
    grad_hidden: &mut [f32],
    grad_biases: &mut [f32],
    hidden_weights: &[f32],
    next_hidden: &mut [f32],
    mut reset_grad: Option<&mut [f32]>,
    input_len: usize,
    hidden_len: usize,
) {
    grad_biases[gate * hidden_len + hidden_index] += delta;
    let input_weight_offset = gate * hidden_len * input_len + hidden_index * input_len;
    for i in 0..input_len {
        grad_input[input_weight_offset + i] += delta * input[input_offset + i];
    }
    let hidden_weight_offset = gate * hidden_len * hidden_len + hidden_index * hidden_len;
    for h in 0..hidden_len {
        grad_hidden[hidden_weight_offset + h] += delta * hidden_input[h];
        let back = delta * hidden_weights[hidden_weight_offset + h];
        if let Some(values) = reset_grad.as_deref_mut() {
            values[h] += back * hidden_input[h];
        }
        next_hidden[h] += back;
    }
}

fn apply_recurrent_updates(values: &mut [f32], gradients: &[f32], previous: &mut [f32], rate: f32, momentum: f32, clip: f32) {
    for i in 0..values.len() {
        let gradient = gradients[i].clamp(-clip, clip);
        let update = rate * gradient + momentum * previous[i];
        values[i] -= update;
        previous[i] = update;
    }
}

fn recurrent_back_start(steps: usize, truncated_steps: usize) -> usize {
    if truncated_steps == 0 { 0 } else { steps.saturating_sub(truncated_steps) }
}

#[allow(clippy::too_many_arguments)]
fn lstm_step(input: &[f32], hidden: &[f32], cell: &[f32], next_hidden: &mut [f32], next_cell: &mut [f32], input_weights: &[f32], hidden_weights: &[f32], biases: &[f32], input_len: usize, hidden_len: usize) {
    for h in 0..hidden_len {
        let i = sigmoid(recurrent_gate_sum(0, h, input, hidden, None, input_weights, hidden_weights, biases, input_len, hidden_len));
        let f = sigmoid(recurrent_gate_sum(1, h, input, hidden, None, input_weights, hidden_weights, biases, input_len, hidden_len));
        let g = recurrent_gate_sum(2, h, input, hidden, None, input_weights, hidden_weights, biases, input_len, hidden_len).tanh();
        let o = sigmoid(recurrent_gate_sum(3, h, input, hidden, None, input_weights, hidden_weights, biases, input_len, hidden_len));
        next_cell[h] = f * cell[h] + i * g;
        next_hidden[h] = o * next_cell[h].tanh();
    }
}

#[allow(clippy::too_many_arguments)]
fn gru_step(input: &[f32], hidden: &[f32], next_hidden: &mut [f32], input_weights: &[f32], hidden_weights: &[f32], biases: &[f32], input_len: usize, hidden_len: usize) {
    let mut update = vec![0.0; hidden_len];
    let mut reset = vec![0.0; hidden_len];
    for h in 0..hidden_len {
        update[h] = sigmoid(recurrent_gate_sum(0, h, input, hidden, None, input_weights, hidden_weights, biases, input_len, hidden_len));
        reset[h] = sigmoid(recurrent_gate_sum(1, h, input, hidden, None, input_weights, hidden_weights, biases, input_len, hidden_len));
    }
    for h in 0..hidden_len {
        let n = recurrent_gate_sum(2, h, input, hidden, Some(&reset), input_weights, hidden_weights, biases, input_len, hidden_len).tanh();
        next_hidden[h] = (1.0 - update[h]) * n + update[h] * hidden[h];
    }
}

#[allow(clippy::too_many_arguments)]
fn recurrent_gate_sum(gate: usize, hidden_index: usize, input: &[f32], hidden: &[f32], reset: Option<&[f32]>, input_weights: &[f32], hidden_weights: &[f32], biases: &[f32], input_len: usize, hidden_len: usize) -> f32 {
    let mut sum = biases[gate * hidden_len + hidden_index];
    let input_offset = gate * hidden_len * input_len + hidden_index * input_len;
    sum += dot(input, &input_weights[input_offset..input_offset + input_len]);
    let hidden_offset = gate * hidden_len * hidden_len + hidden_index * hidden_len;
    if let Some(reset) = reset {
        for h in 0..hidden_len {
            sum += reset[h] * hidden[h] * hidden_weights[hidden_offset + h];
        }
    } else {
        sum += dot(hidden, &hidden_weights[hidden_offset..hidden_offset + hidden_len]);
    }
    sum
}

fn project_recurrent_output(hidden: &[f32], output: &mut [f32], output_weights: &[f32], output_biases: &[f32], hidden_len: usize, output_len: usize, output_activation: u32) {
    for out in 0..output_len {
        let mut sum = output_biases[out];
        let weight_offset = out * hidden_len;
        sum += dot(hidden, &output_weights[weight_offset..weight_offset + hidden_len]);
        output[out] = activate(sum, output_activation);
    }
}
