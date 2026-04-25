use core::slice;

use crate::math::{activate, derivative, squared_error_sum};

#[no_mangle]
pub unsafe extern "C" fn bunaptic_graph_forward(
    input_ptr: *const f32,
    activations_ptr: *mut f32,
    states_ptr: *mut f32,
    previous_ptr: *mut f32,
    node_kinds_ptr: *const u8,
    activation_kinds_ptr: *const u8,
    biases_ptr: *const f32,
    from_ptr: *const u32,
    gater_ptr: *const i32,
    weights_ptr: *const f32,
    connection_kinds_ptr: *const u8,
    incoming_starts_ptr: *const u32,
    incoming_ptr: *const u32,
    output_ptr: *mut f32,
    input_len: usize,
    node_count: usize,
    output_len: usize,
    connection_len: usize,
    recurrent: u32,
) {
    let input = slice::from_raw_parts(input_ptr, input_len);
    let activations = slice::from_raw_parts_mut(activations_ptr, node_count);
    let states = slice::from_raw_parts_mut(states_ptr, node_count);
    let previous = slice::from_raw_parts_mut(previous_ptr, node_count);
    let node_kinds = slice::from_raw_parts(node_kinds_ptr, node_count);
    let activation_kinds = slice::from_raw_parts(activation_kinds_ptr, node_count);
    let biases = slice::from_raw_parts(biases_ptr, node_count);
    let from = slice::from_raw_parts(from_ptr, connection_len);
    let gater = slice::from_raw_parts(gater_ptr, connection_len);
    let weights = slice::from_raw_parts(weights_ptr, connection_len);
    let connection_kinds = slice::from_raw_parts(connection_kinds_ptr, connection_len);
    let incoming_starts = slice::from_raw_parts(incoming_starts_ptr, node_count + 1);
    let incoming = slice::from_raw_parts(incoming_ptr, connection_len);
    let output = slice::from_raw_parts_mut(output_ptr, output_len);

    graph_forward_core(
        input,
        activations,
        states,
        previous,
        node_kinds,
        activation_kinds,
        biases,
        from,
        gater,
        weights,
        connection_kinds,
        incoming_starts,
        incoming,
        output,
        input_len,
        node_count,
        output_len,
        recurrent,
    );
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_graph_forward_batch(
    inputs_ptr: *const f32,
    output_ptr: *mut f32,
    activations_ptr: *mut f32,
    states_ptr: *mut f32,
    previous_ptr: *mut f32,
    node_kinds_ptr: *const u8,
    activation_kinds_ptr: *const u8,
    biases_ptr: *const f32,
    from_ptr: *const u32,
    gater_ptr: *const i32,
    weights_ptr: *const f32,
    connection_kinds_ptr: *const u8,
    incoming_starts_ptr: *const u32,
    incoming_ptr: *const u32,
    input_len: usize,
    node_count: usize,
    output_len: usize,
    connection_len: usize,
    batch_size: usize,
    recurrent: u32,
) {
    let inputs = slice::from_raw_parts(inputs_ptr, batch_size * input_len);
    let output = slice::from_raw_parts_mut(output_ptr, batch_size * output_len);
    let activations = slice::from_raw_parts_mut(activations_ptr, node_count);
    let states = slice::from_raw_parts_mut(states_ptr, node_count);
    let previous = slice::from_raw_parts_mut(previous_ptr, node_count);
    let node_kinds = slice::from_raw_parts(node_kinds_ptr, node_count);
    let activation_kinds = slice::from_raw_parts(activation_kinds_ptr, node_count);
    let biases = slice::from_raw_parts(biases_ptr, node_count);
    let from = slice::from_raw_parts(from_ptr, connection_len);
    let gater = slice::from_raw_parts(gater_ptr, connection_len);
    let weights = slice::from_raw_parts(weights_ptr, connection_len);
    let connection_kinds = slice::from_raw_parts(connection_kinds_ptr, connection_len);
    let incoming_starts = slice::from_raw_parts(incoming_starts_ptr, node_count + 1);
    let incoming = slice::from_raw_parts(incoming_ptr, connection_len);

    for sample in 0..batch_size {
        activations.fill(0.0);
        states.fill(0.0);
        previous.fill(0.0);
        graph_forward_core(
            &inputs[sample * input_len..(sample + 1) * input_len],
            activations,
            states,
            previous,
            node_kinds,
            activation_kinds,
            biases,
            from,
            gater,
            weights,
            connection_kinds,
            incoming_starts,
            incoming,
            &mut output[sample * output_len..(sample + 1) * output_len],
            input_len,
            node_count,
            output_len,
            recurrent,
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_graph_forward_sequence(
    values_ptr: *const f32,
    output_ptr: *mut f32,
    activations_ptr: *mut f32,
    states_ptr: *mut f32,
    previous_ptr: *mut f32,
    node_kinds_ptr: *const u8,
    activation_kinds_ptr: *const u8,
    biases_ptr: *const f32,
    from_ptr: *const u32,
    gater_ptr: *const i32,
    weights_ptr: *const f32,
    connection_kinds_ptr: *const u8,
    incoming_starts_ptr: *const u32,
    incoming_ptr: *const u32,
    input_len: usize,
    node_count: usize,
    output_len: usize,
    connection_len: usize,
    steps: usize,
    recurrent: u32,
) {
    let values = slice::from_raw_parts(values_ptr, steps * input_len);
    let output = slice::from_raw_parts_mut(output_ptr, steps * output_len);
    let activations = slice::from_raw_parts_mut(activations_ptr, node_count);
    let states = slice::from_raw_parts_mut(states_ptr, node_count);
    let previous = slice::from_raw_parts_mut(previous_ptr, node_count);
    let node_kinds = slice::from_raw_parts(node_kinds_ptr, node_count);
    let activation_kinds = slice::from_raw_parts(activation_kinds_ptr, node_count);
    let biases = slice::from_raw_parts(biases_ptr, node_count);
    let from = slice::from_raw_parts(from_ptr, connection_len);
    let gater = slice::from_raw_parts(gater_ptr, connection_len);
    let weights = slice::from_raw_parts(weights_ptr, connection_len);
    let connection_kinds = slice::from_raw_parts(connection_kinds_ptr, connection_len);
    let incoming_starts = slice::from_raw_parts(incoming_starts_ptr, node_count + 1);
    let incoming = slice::from_raw_parts(incoming_ptr, connection_len);

    for step in 0..steps {
        graph_forward_core(
            &values[step * input_len..(step + 1) * input_len],
            activations,
            states,
            previous,
            node_kinds,
            activation_kinds,
            biases,
            from,
            gater,
            weights,
            connection_kinds,
            incoming_starts,
            incoming,
            &mut output[step * output_len..(step + 1) * output_len],
            input_len,
            node_count,
            output_len,
            recurrent,
        );
    }
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_graph_forward_ragged(
    values_ptr: *const f32,
    offsets_ptr: *const u32,
    output_ptr: *mut f32,
    activations_ptr: *mut f32,
    states_ptr: *mut f32,
    previous_ptr: *mut f32,
    node_kinds_ptr: *const u8,
    activation_kinds_ptr: *const u8,
    biases_ptr: *const f32,
    from_ptr: *const u32,
    gater_ptr: *const i32,
    weights_ptr: *const f32,
    connection_kinds_ptr: *const u8,
    incoming_starts_ptr: *const u32,
    incoming_ptr: *const u32,
    input_len: usize,
    node_count: usize,
    output_len: usize,
    connection_len: usize,
    sequence_count: usize,
    value_count: usize,
    recurrent: u32,
) {
    let values = slice::from_raw_parts(values_ptr, value_count);
    let offsets = slice::from_raw_parts(offsets_ptr, sequence_count + 1);
    let output = slice::from_raw_parts_mut(output_ptr, sequence_count * output_len);
    let activations = slice::from_raw_parts_mut(activations_ptr, node_count);
    let states = slice::from_raw_parts_mut(states_ptr, node_count);
    let previous = slice::from_raw_parts_mut(previous_ptr, node_count);
    let node_kinds = slice::from_raw_parts(node_kinds_ptr, node_count);
    let activation_kinds = slice::from_raw_parts(activation_kinds_ptr, node_count);
    let biases = slice::from_raw_parts(biases_ptr, node_count);
    let from = slice::from_raw_parts(from_ptr, connection_len);
    let gater = slice::from_raw_parts(gater_ptr, connection_len);
    let weights = slice::from_raw_parts(weights_ptr, connection_len);
    let connection_kinds = slice::from_raw_parts(connection_kinds_ptr, connection_len);
    let incoming_starts = slice::from_raw_parts(incoming_starts_ptr, node_count + 1);
    let incoming = slice::from_raw_parts(incoming_ptr, connection_len);

    for sequence in 0..sequence_count {
        activations.fill(0.0);
        states.fill(0.0);
        previous.fill(0.0);
        let start = offsets[sequence] as usize;
        let end = offsets[sequence + 1] as usize;
        let mut cursor = start;
        while cursor < end {
            graph_forward_core(
                &values[cursor..cursor + input_len],
                activations,
                states,
                previous,
                node_kinds,
                activation_kinds,
                biases,
                from,
                gater,
                weights,
                connection_kinds,
                incoming_starts,
                incoming,
                &mut output[sequence * output_len..(sequence + 1) * output_len],
                input_len,
                node_count,
                output_len,
                recurrent,
            );
            cursor += input_len;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_evaluate_dataset(
    inputs_ptr: *const f32,
    targets_ptr: *const f32,
    activations_ptr: *mut f32,
    states_ptr: *mut f32,
    previous_ptr: *mut f32,
    node_kinds_ptr: *const u8,
    activation_kinds_ptr: *const u8,
    biases_ptr: *const f32,
    from_ptr: *const u32,
    gater_ptr: *const i32,
    weights_ptr: *const f32,
    connection_kinds_ptr: *const u8,
    incoming_starts_ptr: *const u32,
    incoming_ptr: *const u32,
    input_len: usize,
    node_count: usize,
    output_len: usize,
    connection_len: usize,
    sample_count: usize,
    recurrent: u32,
) -> f32 {
    let inputs = slice::from_raw_parts(inputs_ptr, sample_count * input_len);
    let targets = slice::from_raw_parts(targets_ptr, sample_count * output_len);
    let activations = slice::from_raw_parts_mut(activations_ptr, node_count);
    let states = slice::from_raw_parts_mut(states_ptr, node_count);
    let previous = slice::from_raw_parts_mut(previous_ptr, node_count);
    let node_kinds = slice::from_raw_parts(node_kinds_ptr, node_count);
    let activation_kinds = slice::from_raw_parts(activation_kinds_ptr, node_count);
    let biases = slice::from_raw_parts(biases_ptr, node_count);
    let from = slice::from_raw_parts(from_ptr, connection_len);
    let gater = slice::from_raw_parts(gater_ptr, connection_len);
    let weights = slice::from_raw_parts(weights_ptr, connection_len);
    let connection_kinds = slice::from_raw_parts(connection_kinds_ptr, connection_len);
    let incoming_starts = slice::from_raw_parts(incoming_starts_ptr, node_count + 1);
    let incoming = slice::from_raw_parts(incoming_ptr, connection_len);
    let mut output = vec![0.0; output_len];
    let mut error = 0.0;

    for sample in 0..sample_count {
        activations.fill(0.0);
        states.fill(0.0);
        previous.fill(0.0);
        graph_forward_core(
            &inputs[sample * input_len..(sample + 1) * input_len],
            activations,
            states,
            previous,
            node_kinds,
            activation_kinds,
            biases,
            from,
            gater,
            weights,
            connection_kinds,
            incoming_starts,
            incoming,
            &mut output,
            input_len,
            node_count,
            output_len,
            recurrent,
        );
        error += squared_error_sum(&targets[sample * output_len..(sample + 1) * output_len], &output);
    }

    error / (sample_count * output_len) as f32
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_train_graph(
    inputs_ptr: *const f32,
    targets_ptr: *const f32,
    weights_ptr: *mut f32,
    biases_ptr: *mut f32,
    previous_deltas_ptr: *mut f32,
    activations_ptr: *mut f32,
    states_ptr: *mut f32,
    previous_ptr: *mut f32,
    node_deltas_ptr: *mut f32,
    node_kinds_ptr: *const u8,
    activation_kinds_ptr: *const u8,
    from_ptr: *const u32,
    to_ptr: *const u32,
    gater_ptr: *const i32,
    connection_kinds_ptr: *const u8,
    incoming_starts_ptr: *const u32,
    incoming_ptr: *const u32,
    outgoing_starts_ptr: *const u32,
    outgoing_ptr: *const u32,
    result_ptr: *mut f32,
    input_len: usize,
    node_count: usize,
    output_len: usize,
    connection_len: usize,
    sample_count: usize,
    iterations: usize,
    target_error: f32,
    rate: f32,
    momentum: f32,
) {
    let inputs = slice::from_raw_parts(inputs_ptr, sample_count * input_len);
    let targets = slice::from_raw_parts(targets_ptr, sample_count * output_len);
    let weights = slice::from_raw_parts_mut(weights_ptr, connection_len);
    let biases = slice::from_raw_parts_mut(biases_ptr, node_count);
    let previous_deltas = slice::from_raw_parts_mut(previous_deltas_ptr, connection_len);
    let activations = slice::from_raw_parts_mut(activations_ptr, node_count);
    let states = slice::from_raw_parts_mut(states_ptr, node_count);
    let previous = slice::from_raw_parts_mut(previous_ptr, node_count);
    let node_deltas = slice::from_raw_parts_mut(node_deltas_ptr, node_count);
    let node_kinds = slice::from_raw_parts(node_kinds_ptr, node_count);
    let activation_kinds = slice::from_raw_parts(activation_kinds_ptr, node_count);
    let from = slice::from_raw_parts(from_ptr, connection_len);
    let to = slice::from_raw_parts(to_ptr, connection_len);
    let gater = slice::from_raw_parts(gater_ptr, connection_len);
    let connection_kinds = slice::from_raw_parts(connection_kinds_ptr, connection_len);
    let incoming_starts = slice::from_raw_parts(incoming_starts_ptr, node_count + 1);
    let incoming = slice::from_raw_parts(incoming_ptr, connection_len);
    let outgoing_starts = slice::from_raw_parts(outgoing_starts_ptr, node_count + 1);
    let outgoing = slice::from_raw_parts(outgoing_ptr, connection_len);
    let result = slice::from_raw_parts_mut(result_ptr, 2);
    let mut output = vec![0.0; output_len];
    let output_start = node_count - output_len;
    let mut final_error = f32::INFINITY;
    let mut completed = 0usize;

    for iteration in 0..iterations {
        let mut error = 0.0;
        for sample in 0..sample_count {
            activations.fill(0.0);
            states.fill(0.0);
            previous.fill(0.0);
            node_deltas.fill(0.0);

            graph_forward_core(
                &inputs[sample * input_len..(sample + 1) * input_len],
                activations,
                states,
                previous,
                node_kinds,
                activation_kinds,
                biases,
                from,
                gater,
                weights,
                connection_kinds,
                incoming_starts,
                incoming,
                &mut output,
                input_len,
                node_count,
                output_len,
                0,
            );

            for out in 0..output_len {
                let node = output_start + out;
                let target = targets[sample * output_len + out];
                let actual = output[out];
                let delta = target - actual;
                error += delta * delta;
                node_deltas[node] = delta * derivative(actual, states[node], activation_kinds[node] as u32);
            }

            for node in (input_len..output_start).rev() {
                if node_kinds[node] == 3 {
                    continue;
                }
                let mut downstream = 0.0;
                for cursor in outgoing_starts[node] as usize..outgoing_starts[node + 1] as usize {
                    let conn = outgoing[cursor] as usize;
                    if connection_kinds[conn] != 0 {
                        continue;
                    }
                    downstream += node_deltas[to[conn] as usize] * weights[conn];
                }
                node_deltas[node] = downstream * derivative(activations[node], states[node], activation_kinds[node] as u32);
            }

            for node in input_len..node_count {
                biases[node] += rate * node_deltas[node];
            }

            for node in input_len..node_count {
                for cursor in incoming_starts[node] as usize..incoming_starts[node + 1] as usize {
                    let conn = incoming[cursor] as usize;
                    if connection_kinds[conn] != 0 {
                        continue;
                    }
                    let gradient = node_deltas[node] * activations[from[conn] as usize];
                    let delta = rate * gradient + momentum * previous_deltas[conn];
                    weights[conn] += delta;
                    previous_deltas[conn] = delta;
                }
            }
        }

        final_error = error / (sample_count * output_len) as f32;
        completed = iteration + 1;
        if target_error >= 0.0 && final_error <= target_error {
            break;
        }
    }

    result[0] = final_error;
    result[1] = completed as f32;
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn graph_forward_core(
    input: &[f32],
    activations: &mut [f32],
    states: &mut [f32],
    previous: &mut [f32],
    node_kinds: &[u8],
    activation_kinds: &[u8],
    biases: &[f32],
    from: &[u32],
    gater: &[i32],
    weights: &[f32],
    connection_kinds: &[u8],
    incoming_starts: &[u32],
    incoming: &[u32],
    output: &mut [f32],
    input_len: usize,
    node_count: usize,
    output_len: usize,
    recurrent: u32,
) {
    activations[..input_len].copy_from_slice(input);
    if recurrent == 1 {
        previous.copy_from_slice(activations);
    }

    for node in input_len..node_count {
        if node_kinds[node] == 0 {
            continue;
        }
        if node_kinds[node] == 3 {
            activations[node] = 1.0;
            continue;
        }

        let mut sum = biases[node];
        for cursor in incoming_starts[node] as usize..incoming_starts[node + 1] as usize {
            let conn = incoming[cursor] as usize;
            let source = if connection_kinds[conn] == 0 { &*activations } else { &*previous };
            let gain_source = if gater[conn] >= 0 && connection_kinds[conn] != 0 { &*previous } else { &*activations };
            let gain = if gater[conn] >= 0 { gain_source[gater[conn] as usize] } else { 1.0 };
            sum += source[from[conn] as usize] * weights[conn] * gain;
        }
        states[node] = sum;
        activations[node] = activate(sum, activation_kinds[node] as u32);
    }

    let output_start = node_count - output_len;
    output.copy_from_slice(&activations[output_start..node_count]);
}
