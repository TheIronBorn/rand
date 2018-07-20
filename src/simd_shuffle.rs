//! An SIMD shuffle implementation.
//!
//! Loosely based on Daniel Lemire's [SIMDxorshift].
//!
//! [SIMDxorshift]: (https://github.com/lemire/SIMDxorshift

// use core::mem::size_of;
use core::simd::*;

use {swap_unchecked, Rng};

/// A trait for shuffling slices.
pub trait SimdShuf {
    /// Shuffle a mutable slice in place, using an SIMD implementation.
    ///
    /// To be used in the form:
    /// ```rust
    /// u16x8::simd_shuffle(&mut rng, &mut list);
    /// ```
    ///
    /// Use a vector of size greater than or equal to the PRNG output.
    /// Smaller lane widths will likely be faster for equal vector sizes.
    ///
    /// # Panics
    ///
    /// If `values.len()` is larger than the maximum value of the vector's
    /// lanes. (If `values.len()` is unknown, use a `u32xN` or `u64xN`
    /// vector depending on [`target_pointer_width`].)
    ///
    /// [`target_pointer_width`]: https://doc.rust-lang.org/reference/attributes.html#conditional-compilation
    fn simd_shuffle<R: Rng, T>(rng: &mut R, values: &mut [T]);
}

macro_rules! impl_simd_shuf {
    ($vec:ident, $scalar:ident) => {
        impl SimdShuf for $vec {
            // TODO: make this adapt when too many/few elements
            //       could match on values.len and use an appropriate
            //       lane width, based on chosen vector width
            #[inline(always)]
            fn simd_shuffle<R: Rng, T>(rng: &mut R, values: &mut [T]) {
                assert!(
                    values.len() <= $scalar::max_value() as usize,
                    "Slice length too long for the vector's lanes",
                );

                // Create a vector to hold `$vec::lanes()` range bounds at
                // once. This should be evaluated at compile-time.
                // TODO: consider making this a macro
                let mut interval = $vec::default();
                for vec_idx in 0..$vec::lanes() {
                    // (len, len - 1, len - 2, len - 3, ..., len - $vec::lanes() + 1)
                    interval = interval.replace(vec_idx, (values.len() - vec_idx) as $scalar);
                }
                let mut slice_idx = values.len();

                // shuffle a multiple of `$vec::lanes()` slice elements
                for _ in 0..values.len() / $vec::lanes() {
                    // `gen_below` is about 10% faster
                    let rand_indices = rng.gen_below(interval);

                    // swap each `rand_idx` with the next `slice_idx`
                    // TODO: could probably be optimized
                    for vec_idx in 0..$vec::lanes() {
                        slice_idx -= 1;
                        let rand_idx = rand_indices.extract(vec_idx) as usize;
                        unsafe { swap_unchecked(values, slice_idx, rand_idx); }
                    }

                    // move onto the next interval
                    interval -= $vec::lanes() as $scalar;
                }

                // shuffle the remaining elements
                // This is likely overzealous
                let remainder = values.len() % $vec::lanes();
                if remainder > 1 {
                    match remainder - 1 {
                        1...2 => rem_shuf!(u16x2, remainder, rng, values, slice_idx, u16),
                        3...4 => rem_shuf!(u16x4, remainder, rng, values, slice_idx, u16),
                        5...8 => rem_shuf!(u16x8, remainder, rng, values, slice_idx, u16),
                        9...16 => rem_shuf!(u16x16, remainder, rng, values, slice_idx, u16),
                        17...32 => rem_shuf!(u16x32, remainder, rng, values, slice_idx, u16),
                        33...64 | _ => rem_shuf!(u8x64, remainder, rng, values, slice_idx, u8),
                    }
                }
            }
        }
    };

    // bulk implementation for scalar types
    ($($vec:ident,)+, $scalar:ident) => {$(
        impl_simd_shuf!($vec, $scalar);
    )+};
}

macro_rules! rem_shuf {
    ($vec:ident, $rem:ident, $rng:ident, $values:ident, $slice_idx:ident, $scalar:ty) => {{
        // We can exit interval generation early, because we only need a few
        // indices.  We can't avoid generating unneeded random indices however,
        // so we use a default value of 1 to speed up the uniform sampling.
        let mut interval = $vec::splat(1);
        for vec_idx in 0..$rem - 1 {
            interval = interval.replace(vec_idx, ($rem - vec_idx) as $scalar);
        }

        let rand_indices = $rng.gen_below(interval);
        for vec_idx in 0..$rem - 1 {
            $slice_idx -= 1;
            let rand_idx = rand_indices.extract(vec_idx) as usize;
            unsafe { swap_unchecked($values, $slice_idx, rand_idx); }
        }
    }};
}

impl_simd_shuf!(u8x2, u8x4, u8x8, u8x16, u8x32, u8x64,, u8);
impl_simd_shuf!(u16x2, u16x4, u16x8, u16x16, u16x32,, u16);
impl_simd_shuf!(u32x2, u32x4, u32x8, u32x16,, u32);
impl_simd_shuf!(u64x2, u64x4, u64x8,, u64);
