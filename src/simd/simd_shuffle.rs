//! An SIMD shuffle implementation.
//!
//! Loosely based on Daniel Lemire's [SIMD Xorshift](https://github.com/lemire/SIMDxorshift)

use core::simd::*;
use core::mem::size_of;
use simd::sfc_32_simd::*;
use simd::SimdRng;

macro_rules! impl_simd_shuf {
    ($rng:ident, $vector:ident, $next_u:ident, $scalar:ident, $large:ident) => (
        impl $rng {
            /// Generate a vector where each lane is in the range [`0`, `upper_bound`).
            pub fn random_bound(&mut self, upper_bound: $vector) -> $vector {
                let random_vals = self.$next_u();

                assert_eq!(size_of::<$vector>() * 2, size_of::<$large>());
                assert_eq!($vector::lanes(), $large::lanes());

                // Cast to $large, i.e. each lane from u32 to u64.
                let random_vals = $large::from(random_vals);
                let upper_bound = $large::from(upper_bound);

                // u32 => 32, u64 => 64, etc
                const DIV: usize = size_of::<$vector>() * 8 / $vector::lanes();

                $vector::from((random_vals * upper_bound) >> DIV)
            }

            /// Shuffle a mutable slice in place, using an SIMD implementation.
            pub fn simd_shuffle<T>(&mut self, storage: &mut [T]) {
                assert_eq!(size_of::<$vector>() / $vector::lanes(), size_of::<$scalar>());

                // Create a vector to hold `$vector::lanes()` range bounds at once.
                let mut interval = $vector::default();
                for vec_idx in 0..$vector::lanes() {
                    // (len, len - 1, len - 2, len - 3, ..., len - $vector::lanes() + 1)
                    interval = interval.replace(vec_idx, storage.len() as $scalar - vec_idx as $scalar);
                }
                let mut slice_idx = storage.len();

                while slice_idx > 1 {
                    // Generate `$vector::lanes()` indices within `interval`.
                    let rand_indices = self.random_bound(interval);

                    // For each index in `rand_indices`, swap with `slice_idx`.
                    for vec_idx in 0..$vector::lanes() {
                        if slice_idx == 1 { return }
                        let rand_idx = rand_indices.extract(vec_idx) as usize;
                        storage.swap(slice_idx - 1, rand_idx);
                        slice_idx -= 1;
                    }

                    // Move on to the next interval.
                    interval -= $vector::splat($vector::lanes() as $scalar);
                }
            }
        }
    )
}

impl_simd_shuf!(Sfc32X8, u32x8, next_u32x8, u32, u64x8);
impl_simd_shuf!(Sfc32X4, u32x4, next_u32x4, u32, u64x4);
impl_simd_shuf!(Sfc32X2, u32x2, next_u32x2, u32, u64x2);
