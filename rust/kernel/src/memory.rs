#[no_mangle]
pub extern "C" fn bunaptic_version() -> u32 {
    1
}

#[no_mangle]
pub extern "C" fn bunaptic_simd_enabled() -> u32 {
    if cfg!(all(target_arch = "wasm32", target_feature = "simd128")) { 1 } else { 0 }
}

#[no_mangle]
pub extern "C" fn bunaptic_alloc(len: usize) -> *mut u8 {
    let words = len.div_ceil(8).max(1);
    let mut buffer = Vec::<u64>::with_capacity(words);
    let ptr = buffer.as_mut_ptr() as *mut u8;
    core::mem::forget(buffer);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn bunaptic_dealloc(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        let words = len.div_ceil(8).max(1);
        let _ = Vec::from_raw_parts(ptr as *mut u64, 0, words);
    }
}
