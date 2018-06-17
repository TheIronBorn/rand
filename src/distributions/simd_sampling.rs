use stdsimd::simd::*;

use distributions::Distribution;
use Rng;

// We can use this trait to tune behavior for various SIMD vectors. i.e.
// for vectors with few lanes, the `mask.select` replacement method
// might actually be slower.

/// A helper trait to perform rejection sampling for SIMD types.
pub trait SimdRejectionSampling
where
    Self: Sized,
{
    /// The mask for the `Self` SIMD vector
    type Mask;

    /// Returns a SIMD vector randomly sampled from `distr` for which the SIMD mask
    /// `cmp` returns has only true lanes.
    fn sample<R: Rng + ?Sized, D: Distribution<Self>, F>(rng: &mut R, distr: &D, cmp: F) -> Self
    where
        F: FnMut(Self) -> Self::Mask;
}

macro_rules! impl_simd_rejection_sampling {
    ($ty:ty, $mask:ty) => {
        impl SimdRejectionSampling for $ty {
            type Mask = $mask;

            #[inline(always)]
            fn sample<R: Rng + ?Sized, D: Distribution<Self>, F>(
                rng: &mut R,
                distr: &D,
                mut cmp: F,
            ) -> Self
            where
                F: FnMut(Self) -> Self::Mask,
            {
                let mut random_numbers = rng.sample(distr);

                loop {
                    let mask = cmp(random_numbers);
                    if mask.all() {
                        return random_numbers;
                    }
                    random_numbers = mask.select(random_numbers, rng.sample(distr));
                }
            }
        }
    };
}

impl_simd_rejection_sampling! { u8x2, m8x2 }
impl_simd_rejection_sampling! { u8x4, m8x4 }
impl_simd_rejection_sampling! { u8x8, m8x8 }
impl_simd_rejection_sampling! { u8x16, m8x16 }
impl_simd_rejection_sampling! { u8x32, m8x32 }
impl_simd_rejection_sampling! { u8x64, m1x64 }
impl_simd_rejection_sampling! { u16x2, m16x2 }
impl_simd_rejection_sampling! { u16x4, m16x4 }
impl_simd_rejection_sampling! { u16x8, m16x8 }
impl_simd_rejection_sampling! { u16x16, m16x16 }
impl_simd_rejection_sampling! { u16x32, m1x32 }
impl_simd_rejection_sampling! { u32x2, m32x2 }
impl_simd_rejection_sampling! { u32x4, m32x4 }
impl_simd_rejection_sampling! { u32x8, m32x8 }
impl_simd_rejection_sampling! { u32x16, m1x16 }
impl_simd_rejection_sampling! { u64x2, m64x2 }
impl_simd_rejection_sampling! { u64x4, m64x4 }
impl_simd_rejection_sampling! { u64x8, m1x8 }
