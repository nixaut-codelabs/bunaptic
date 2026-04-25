#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::*;

const SELU_ALPHA: f32 = 1.6732632;
const SELU_SCALE: f32 = 1.050701;

#[inline]
pub(crate) fn activate(value: f32, activation: u32) -> f32 {
    match activation {
        0 => 1.0 / (1.0 + (-value).exp()),
        1 => value.tanh(),
        2 => value,
        3 => {
            if value > 0.0 { 1.0 } else { 0.0 }
        }
        4 => value.max(0.0),
        5 => value / (1.0 + value.abs()),
        6 => value.sin(),
        7 => (-(value * value)).exp(),
        8 => ((value * value + 1.0).sqrt() - 1.0) / 2.0 + value,
        9 => {
            if value > 0.0 { 1.0 } else { -1.0 }
        }
        10 => 2.0 / (1.0 + (-value).exp()) - 1.0,
        11 => value.clamp(-1.0, 1.0),
        12 => value.abs(),
        13 => 1.0 - value,
        14 => (if value > 0.0 { value } else { SELU_ALPHA * value.exp() - SELU_ALPHA }) * SELU_SCALE,
        _ => value,
    }
}

#[inline]
pub(crate) fn dot(input: &[f32], weights: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        dot_simd(input, weights)
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        dot_scalar(input, weights)
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline]
fn dot_scalar(input: &[f32], weights: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..input.len() {
        sum += input[i] * weights[i];
    }
    sum
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn dot_simd(input: &[f32], weights: &[f32]) -> f32 {
    let mut acc = f32x4_splat(0.0);
    let mut i = 0usize;
    while i + 4 <= input.len() {
        let left = v128_load(input.as_ptr().add(i) as *const v128);
        let right = v128_load(weights.as_ptr().add(i) as *const v128);
        acc = f32x4_add(acc, f32x4_mul(left, right));
        i += 4;
    }
    let mut sum = f32x4_extract_lane::<0>(acc) + f32x4_extract_lane::<1>(acc) + f32x4_extract_lane::<2>(acc) + f32x4_extract_lane::<3>(acc);
    while i < input.len() {
        sum += input[i] * weights[i];
        i += 1;
    }
    sum
}

#[inline]
pub(crate) fn squared_error_sum(target: &[f32], output: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        squared_error_sum_simd(target, output)
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        squared_error_sum_scalar(target, output)
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline]
fn squared_error_sum_scalar(target: &[f32], output: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..target.len() {
        let delta = target[i] - output[i];
        sum += delta * delta;
    }
    sum
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn squared_error_sum_simd(target: &[f32], output: &[f32]) -> f32 {
    let mut acc = f32x4_splat(0.0);
    let mut i = 0usize;
    while i + 4 <= target.len() {
        let left = v128_load(target.as_ptr().add(i) as *const v128);
        let right = v128_load(output.as_ptr().add(i) as *const v128);
        let delta = f32x4_sub(left, right);
        acc = f32x4_add(acc, f32x4_mul(delta, delta));
        i += 4;
    }
    let mut sum = f32x4_extract_lane::<0>(acc) + f32x4_extract_lane::<1>(acc) + f32x4_extract_lane::<2>(acc) + f32x4_extract_lane::<3>(acc);
    while i < target.len() {
        let delta = target[i] - output[i];
        sum += delta * delta;
        i += 1;
    }
    sum
}

#[inline]
pub(crate) fn derivative(activation: f32, state: f32, kind: u32) -> f32 {
    match kind {
        0 => activation * (1.0 - activation),
        1 => 1.0 - activation * activation,
        2 => 1.0,
        3 => 0.0,
        4 => {
            if state > 0.0 { 1.0 } else { 0.0 }
        }
        5 => {
            let denom = 1.0 + state.abs();
            1.0 / (denom * denom)
        }
        6 => state.cos(),
        7 => -2.0 * state * activation,
        8 => state / (2.0 * (state * state + 1.0).sqrt()) + 1.0,
        9 => 0.0,
        10 => 0.5 * (1.0 + activation) * (1.0 - activation),
        11 => {
            if state > -1.0 && state < 1.0 { 1.0 } else { 0.0 }
        }
        12 => {
            if state >= 0.0 { 1.0 } else { -1.0 }
        }
        13 => -1.0,
        14 => {
            if state > 0.0 { SELU_SCALE } else { SELU_SCALE * SELU_ALPHA * state.exp() }
        }
        _ => 1.0,
    }
}

#[inline]
pub(crate) fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}
