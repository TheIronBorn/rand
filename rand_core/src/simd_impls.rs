use core::simd::*;
use core::mem;

/// Enables an RNG to use [`SimdRngImpls`].
///
/// # Example
///
/// A simple example, obviously not generating very *random* output:
///
/// ```rust
/// #![feature(stdsimd)]
/// use std::simd::u32x4;
/// use rand_core::simd_impls::SimdRng;
///
/// struct CountingSimdRng(u32x4);
///
/// impl SimdRng<u32x4> for CountingSimdRng {
///     fn generate(&mut self) -> u32x4 {
///         self.0 += u32x4::splat(1);
///         self.0
///     }
/// }
/// ```
///
/// [`SimdRngImpls`]: /trait.SimdRngImpls.html
pub trait SimdRng<Vector> {
    /// Return the next random vector.
    #[inline(always)]
    fn generate(&mut self) -> Vector;
}

/// Helper functions for implementing `RngCore` functions for SIMD RNGs which
/// implement [`SimdRng`].
///
/// # Example
///
/// A simple example, using `CountingSimdRng` from the [`SimdRng`] example:
///
/// ```rust
/// #![feature(stdsimd)]
/// use std::simd::u32x4;
/// use rand_core::{RngCore, Error};
/// use rand_core::simd_impls::{SimdRng, SimdRngImpls};
///
/// struct CountingSimdRng(u32x4);
///
/// impl SimdRng<u32x4> for CountingSimdRng {
///     fn generate(&mut self) -> u32x4 {
///         self.0 += u32x4::splat(1);
///         self.0
///     }
/// }
///
/// impl RngCore for CountingSimdRng {
///     fn next_u32(&mut self) -> u32 {
///         u32x4::next_u32_via_simd(self)
///     }
///
///     fn next_u64(&mut self) -> u64 {
///         u32x4::next_u64_via_simd(self)
///     }
///
///     fn fill_bytes(&mut self, dest: &mut [u8]) {
///         u32x4::fill_bytes_via_simd(self, dest)
///     }
///
///     fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
///         self.fill_bytes(dest);
///         Ok(())
///     }
/// }
/// ```
///
/// [`SimdRng`]: /trait.SimdRng.html
pub trait SimdRngImpls<V> {
    /// Implement `next_u32` via a SIMD vector.
    #[inline(always)]
    fn next_u32_via_simd<R: SimdRng<V>>(rng: &mut R) -> u32;

    /// Implement `next_u64` via a SIMD vector.
    #[inline(always)]
    fn next_u64_via_simd<R: SimdRng<V>>(rng: &mut R) -> u64;

    /// Implement `fill_bytes` via SIMD vectors.
    ///
    /// This is useful for generating other vector types.  If the code uses 
    /// it in a SIMD context, the result should stay in SIMD registers.
    #[inline(always)]
    fn fill_bytes_via_simd<R: SimdRng<V>>(rng: &mut R, dest: &mut [u8]);
}

macro_rules! impl_simd_rng {
    ($vector:ty, $v8:ident, $v32:ident, $v64:ident) => (
        impl SimdRngImpls<$vector> for $vector {
            fn next_u32_via_simd<R: SimdRng<$vector>>(rng: &mut R) -> u32 {
                $v32::from_bits(rng.generate()).extract(0)
            }

            fn next_u64_via_simd<R: SimdRng<$vector>>(rng: &mut R) -> u64 {
                $v64::from_bits(rng.generate()).extract(0)
            }

            fn fill_bytes_via_simd<R: SimdRng<$vector>>(rng: &mut R, dest: &mut [u8]) {
                // Forced inlining will keep the result in SIMD registers if
                // the code using it also uses it in a SIMD context.
                const CHUNK_SIZE: usize = mem::size_of::<$vector>();
                let mut read_len = 0;
                for _ in 0..dest.len() / CHUNK_SIZE {
                    // FIXME: on big-endian we should do byte swapping around
                    // here.
                    let results = $v8::from_bits(rng.generate());
                    results.store_aligned(&mut dest[read_len..]);
                    read_len += CHUNK_SIZE;
                }
                let remainder = dest.len() % CHUNK_SIZE;
                if remainder > 0 {
                    // This could be `ptr::copy_nonoverlapping` but I'm not
                    // sure SIMD is happy with it.
                    let results = $v8::from_bits(rng.generate());
                    let len = dest.len() - remainder;
                    let mut buf = [0_u8; $v8::lanes()];
                    results.store_aligned(&mut buf);
                    dest[len..].copy_from_slice(&buf[..remainder]);
                }
            }
        }
    )
}

impl_simd_rng!(u32x4, u8x16, u32x4, u64x2);
impl_simd_rng!(u32x8, u8x32, u32x8, u64x4);
impl_simd_rng!(u32x16, u8x64, u32x16, u64x8);

impl_simd_rng!(u64x2, u8x16, u32x4, u64x2);
impl_simd_rng!(u64x4, u8x32, u32x8, u64x4);
impl_simd_rng!(u64x8, u8x64, u32x16, u64x8);

impl SimdRngImpls<u32x2> for u32x2 {
    fn next_u32_via_simd<R: SimdRng<u32x2>>(rng: &mut R) -> u32 {
        rng.generate().extract(0)
    }

    fn next_u64_via_simd<R: SimdRng<u32x2>>(rng: &mut R) -> u64 {
        // We cannot do `u64x1::from_bits(u32x2)` so we concatenate the bits
        // manually.
        // Use LE; we explicitly generate one value before the next.
        let results = rng.generate();
        let x = u64::from(results.extract(0));
        let y = u64::from(results.extract(1));
        (y << 32) | x
    }

    fn fill_bytes_via_simd<R: SimdRng<u32x2>>(rng: &mut R, dest: &mut [u8]) {
        // Forced inlining will keep the result in SIMD registers if
        // the code using it also uses it in a SIMD context.
        let chunk_size = mem::size_of::<u32x2>();
        let remainder = dest.len() % chunk_size;
        let len = dest.len() - remainder;
        let mut read_len = 0;
        let mut results;
        loop {
            // FIXME: on big-endian we should do byte swapping around
            // here.
            results = u8x8::from_bits(rng.generate());
            if read_len < len {
                results.store_aligned(&mut dest[read_len..]);
                read_len += chunk_size;
            }
            if read_len == len { break; }
        }
        if remainder > 0 {
            // This could be `ptr::copy_nonoverlapping` but I'm not
            // sure SIMD is happy with it.
            let results = u8x8::from_bits(rng.generate());
            let len = dest.len() - remainder;
            let mut buf = [0_u8; u8x8::lanes()];
            results.store_aligned(&mut buf);
            dest[len..].copy_from_slice(&buf[..remainder]);
        }
    }
}
