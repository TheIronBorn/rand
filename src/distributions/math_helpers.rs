//! Math helper functions

#[cfg(feature="simd_support")]
use stdsimd::simd::*;

// Until portable shuffles land in stdsimd, we expose and use the shuffle intrinsics directly.
#[cfg(feature="simd_support")]
extern "platform-intrinsic" {
    pub (crate) fn simd_shuffle2<T, U>(a: T, b: T, indices: [u32; 2]) -> U;
    pub (crate) fn simd_shuffle4<T, U>(a: T, b: T, indices: [u32; 4]) -> U;
    pub (crate) fn simd_shuffle8<T, U>(a: T, b: T, indices: [u32; 8]) -> U;
    pub (crate) fn simd_shuffle16<T, U>(a: T, b: T, indices: [u32; 16]) -> U;
    pub (crate) fn simd_shuffle32<T, U>(a: T, b: T, indices: [u32; 32]) -> U;
    pub (crate) fn simd_shuffle64<T, U>(a: T, b: T, indices: [u32; 64]) -> U;
}

/// Implement byte swapping for SIMD vectors
#[cfg(feature="simd_support")]
pub trait SwapBytes {
    /// `swap_bytes` for a vector (horizontally)
    fn swap_bytes(self) -> Self;
}

// `simd_shuffleX` require constant indices, making this a small pain to implement
#[cfg(feature="simd_support")]
macro_rules! impl_swap_bytes {
    ($ty:ident, $vec8:ident, $shuf:ident, $indices:expr) => (
        impl SwapBytes for $ty {
            fn swap_bytes(self) -> Self {
                let vec8 = $vec8::from_bits(self);
                let shuffled: $vec8 = unsafe { $shuf(vec8, vec8, $indices) };
                $ty::from_bits(shuffled)
            }
        }
    );

    // bulk impl for a shuffle intrinsic/vector width
    ($vec8:ident, $shuf:ident, $indices:expr, $($ty:ident,)+) => ($(
        impl_swap_bytes! { $ty, $vec8, $shuf, $indices }
    )+);
}

#[cfg(feature="simd_support")]
impl_swap_bytes! {
    u8x2,
    simd_shuffle2,
    [1, 0],
    u8x2,
}

#[cfg(feature="simd_support")]
impl_swap_bytes! {
    u8x4,
    simd_shuffle4,
    [3, 2, 1, 0],
    u8x4,
    u16x2,
}

#[cfg(feature="simd_support")]
impl_swap_bytes! {
    u8x8,
    simd_shuffle8,
    [7, 6, 5, 4, 3, 2, 1, 0],
    u8x8,
    u16x4,
    u32x2,
}

#[cfg(feature="simd_support")]
impl_swap_bytes! {
    u8x16,
    simd_shuffle16,
    [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    u8x16,
    u16x8,
    u32x4,
    u64x2,
}

#[cfg(feature="simd_support")]
impl_swap_bytes! {
    u8x32,
    simd_shuffle32,
    [31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    u8x32,
    u16x16,
    u32x8,
    u64x4,
}

#[cfg(feature="simd_support")]
impl_swap_bytes! {
    u8x64,
    simd_shuffle64,
    [63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    u8x64,
    u16x32,
    u32x16,
    u64x8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem;

    #[test]
    #[cfg(feature="simd_support")]
    fn swap_bytes_128() {
        let x: u128 = 0x2d99787926d46932a4c1f32680f70c55;
        let expected = x.swap_bytes();

        let vec: u8x16 = unsafe { mem::transmute(x) };
        let actual = unsafe { mem::transmute(vec.swap_bytes()) };

        assert_eq!(expected, actual);
    }
}
