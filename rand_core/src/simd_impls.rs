//! Helper functions for implementing `RngCore` functions for SIMD PRNGs.

// Some slight complexity in this module because we can't use `gen::<vector>()`

use core::mem;
use stdsimd::simd::*;

use math_helpers::ToLittleEndian;

/// Enables an RNG to use [`SimdRngImpls`].
///
/// # Example
///
/// A simple example, obviously not generating very *random* output:
///
/// ```rust
/// #![feature(stdsimd)]
/// extern crate stdsimd;
/// extern crate rand_core;
/// use stdsimd::simd::u32x4;
/// use rand_core::simd_impls::SimdRng;
///
/// #[allow(dead_code)]
/// struct CountingSimdRng(u32x4);
///
/// impl SimdRng<u32x4> for CountingSimdRng {
///     fn generate(&mut self) -> u32x4 {
///         self.0 += 1;
///         self.0
///     }
/// }
/// ```
///
/// [`SimdRngImpls`]: /trait.SimdRngImpls.html
pub trait SimdRng<Vector> {
    /// Return the next random vector.
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
/// extern crate stdsimd;
/// extern crate rand_core;
/// use stdsimd::simd::u32x4;
/// use rand_core::{RngCore, Error};
/// use rand_core::simd_impls::{SimdRng, SimdRngImpls};
///
/// #[allow(dead_code)]
/// struct CountingSimdRng(u32x4);
///
/// impl SimdRng<u32x4> for CountingSimdRng {
///     fn generate(&mut self) -> u32x4 {
///         self.0 += 1;
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
    fn next_u32_via_simd<R: SimdRng<V>>(rng: &mut R) -> u32;

    /// Implement `next_u64` via a SIMD vector.
    fn next_u64_via_simd<R: SimdRng<V>>(rng: &mut R) -> u64;

    /// Implement `fill_bytes` via SIMD vectors.
    ///
    /// This is useful for generating other vector types.  If the code uses
    /// it in a SIMD context, the result should stay in SIMD registers.
    fn fill_bytes_via_simd<R: SimdRng<V>>(rng: &mut R, dest: &mut [u8]);
}

macro_rules! impl_simd_rng {
    ($vector:ty, $v8:ident, $v32:ident, $v64:ident) => {
        impl SimdRngImpls<$vector> for $vector {
            #[inline(always)]
            fn next_u32_via_simd<R: SimdRng<$vector>>(rng: &mut R) -> u32 {
                // TODO: benchmark/optimize
                $v32::from_bits(rng.generate()).extract(0)
            }

            #[inline(always)]
            fn next_u64_via_simd<R: SimdRng<$vector>>(rng: &mut R) -> u64 {
                $v64::from_bits(rng.generate()).extract(0)
            }

            #[inline(always)]
            fn fill_bytes_via_simd<R: SimdRng<$vector>>(rng: &mut R, dest: &mut [u8]) {
                // Forced inlining will keep the result in SIMD registers if
                // the code using it also uses it in a SIMD context.
                const CHUNK_SIZE: usize = mem::size_of::<$vector>();
                let mut read_len = 0;
                for _ in 0..dest.len() / CHUNK_SIZE {
                    let mut results = $v8::from_bits(rng.generate());
                    results = results.to_le(); // TODO: look into reverse store intrinsics
                    results.store_unaligned(&mut dest[read_len..]);
                    read_len += CHUNK_SIZE;
                }
                let remainder = dest.len() % CHUNK_SIZE;
                if remainder > 0 {
                    // This could be `ptr::copy_nonoverlapping` which doubles
                    // the speed, but I'm not sure SIMD is happy with it.
                    let mut results = $v8::from_bits(rng.generate());
                    results = results.to_le();
                    let len = dest.len() - remainder;

                    // ensure `store_aligned` safety, perhaps overzealous
                    #[repr(align(16))]
                    struct Aligned([u8; $v8::lanes()]);
                    let mut buf = Aligned([0_u8; $v8::lanes()]);
                    results.store_aligned(&mut buf.0);

                    dest[len..].copy_from_slice(&buf.0[..remainder]);
                }
            }
        }
    };
}

// Some vectors cannot use `impl_simd_rng`
// impl_simd_rng!(u8x2, u8x2, u32x0.5, u64x0.25);
// impl_simd_rng!(u8x4, u8x4, u32x1, u64x0.5);
// impl_simd_rng!(u8x8, u8x8, u32x2, u64x1);
impl_simd_rng!(u8x16, u8x16, u32x4, u64x2);
impl_simd_rng!(u8x32, u8x32, u32x8, u64x4);
impl_simd_rng!(u8x64, u8x64, u32x16, u64x8);

// impl_simd_rng!(u16x2, u8x4, u32x1, u64x0.5);
// impl_simd_rng!(u16x4, u8x8, u32x2, u64x1);
impl_simd_rng!(u16x8, u8x16, u32x4, u64x2);
impl_simd_rng!(u16x16, u8x32, u32x8, u64x4);
impl_simd_rng!(u16x32, u8x64, u32x16, u64x8);

// impl_simd_rng!(u32x2, u8x8, u32x2, u64x1);
impl_simd_rng!(u32x4, u8x16, u32x4, u64x2);
impl_simd_rng!(u32x8, u8x32, u32x8, u64x4);
impl_simd_rng!(u32x16, u8x64, u32x16, u64x8);

impl_simd_rng!(u64x2, u8x16, u32x4, u64x2);
impl_simd_rng!(u64x4, u8x32, u32x8, u64x4);
impl_simd_rng!(u64x8, u8x64, u32x16, u64x8);

// We can't do `u64x1::from_bits(u32x2)` etc so we do it manually.
// Too much code/work? Could perhaps just use a generic
// `ptr::copy_nonoverlapping` in the above macro.

impl SimdRngImpls<u8x2> for u8x2 {
    #[inline(always)]
    fn next_u32_via_simd<R: SimdRng<u8x2>>(rng: &mut R) -> u32 {
        // Use LE; we explicitly generate one value before the next.
        let x: u16 = unsafe { mem::transmute(rng.generate()) };
        let y: u16 = unsafe { mem::transmute(rng.generate()) };
        (u32::from(y) << 16) | u32::from(x)
    }

    #[inline(always)]
    fn next_u64_via_simd<R: SimdRng<u8x2>>(rng: &mut R) -> u64 {
        // Use LE; we explicitly generate one value before the next.
        let x = u64::from(u8x2::next_u32_via_simd(rng));
        let y = u64::from(u8x2::next_u32_via_simd(rng));
        (y << 32) | x
    }

    #[inline(always)]
    fn fill_bytes_via_simd<R: SimdRng<u8x2>>(rng: &mut R, dest: &mut [u8]) {
        // Forced inlining will keep the result in SIMD registers if
        // the code using it also uses it in a SIMD context.
        const CHUNK_SIZE: usize = mem::size_of::<u8x2>();
        let mut read_len = 0;
        for _ in 0..dest.len() / CHUNK_SIZE {
            let mut results = u8x2::from_bits(rng.generate());
            results = results.to_le();
            results.store_unaligned(&mut dest[read_len..]);
            read_len += CHUNK_SIZE;
        }
        let remainder = dest.len() % CHUNK_SIZE;
        if remainder > 0 {
            let mut results = u8x2::from_bits(rng.generate());
            results = results.to_le();
            let len = dest.len() - remainder;
            let mut buf = [0_u8; u8x2::lanes()];
            results.store_unaligned(&mut buf);
            dest[len..].copy_from_slice(&buf[..remainder]);
        }
    }
}

impl SimdRngImpls<u16x2> for u16x2 {
    #[inline(always)]
    fn next_u32_via_simd<R: SimdRng<u16x2>>(rng: &mut R) -> u32 {
        unsafe { mem::transmute(rng.generate()) }
    }

    #[inline(always)]
    fn next_u64_via_simd<R: SimdRng<u16x2>>(rng: &mut R) -> u64 {
        // Use LE; we explicitly generate one value before the next.
        let x = u64::from(u16x2::next_u32_via_simd(rng));
        let y = u64::from(u16x2::next_u32_via_simd(rng));
        (y << 32) | x
    }

    #[inline(always)]
    fn fill_bytes_via_simd<R: SimdRng<u16x2>>(rng: &mut R, dest: &mut [u8]) {
        // Forced inlining will keep the result in SIMD registers if
        // the code using it also uses it in a SIMD context.
        const CHUNK_SIZE: usize = mem::size_of::<u16x2>();
        let mut read_len = 0;
        for _ in 0..dest.len() / CHUNK_SIZE {
            let mut results = u8x4::from_bits(rng.generate());
            results = results.to_le();
            results.store_unaligned(&mut dest[read_len..]);
            read_len += CHUNK_SIZE;
        }
        let remainder = dest.len() % CHUNK_SIZE;
        if remainder > 0 {
            let mut results = u8x4::from_bits(rng.generate());
            results = results.to_le();
            let len = dest.len() - remainder;
            let mut buf = [0_u8; u8x8::lanes()];
            results.store_unaligned(&mut buf);
            dest[len..].copy_from_slice(&buf[..remainder]);
        }
    }
}

impl SimdRngImpls<u16x4> for u16x4 {
    #[inline(always)]
    fn next_u32_via_simd<R: SimdRng<u16x4>>(rng: &mut R) -> u32 {
        u32x2::from_bits(rng.generate()).extract(0)
    }

    #[inline(always)]
    fn next_u64_via_simd<R: SimdRng<u16x4>>(rng: &mut R) -> u64 {
        unsafe { mem::transmute(rng.generate()) }
    }

    #[inline(always)]
    fn fill_bytes_via_simd<R: SimdRng<u16x4>>(rng: &mut R, dest: &mut [u8]) {
        // Forced inlining will keep the result in SIMD registers if
        // the code using it also uses it in a SIMD context.
        const CHUNK_SIZE: usize = mem::size_of::<u16x4>();
        let mut read_len = 0;
        for _ in 0..dest.len() / CHUNK_SIZE {
            let mut results = u8x8::from_bits(rng.generate());
            results = results.to_le();
            results.store_unaligned(&mut dest[read_len..]);
            read_len += CHUNK_SIZE;
        }
        let remainder = dest.len() % CHUNK_SIZE;
        if remainder > 0 {
            let mut results = u8x8::from_bits(rng.generate());
            results = results.to_le();
            let len = dest.len() - remainder;
            let mut buf = [0_u8; u8x8::lanes()];
            results.store_unaligned(&mut buf);
            dest[len..].copy_from_slice(&buf[..remainder]);
        }
    }
}

impl SimdRngImpls<u32x2> for u32x2 {
    #[inline(always)]
    fn next_u32_via_simd<R: SimdRng<u32x2>>(rng: &mut R) -> u32 {
        rng.generate().extract(0)
    }

    #[inline(always)]
    fn next_u64_via_simd<R: SimdRng<u32x2>>(rng: &mut R) -> u64 {
        // We can't do `u64x1::from_bits(u32x2)` so we do it manually.
        unsafe { mem::transmute(rng.generate()) }
    }

    #[inline(always)]
    fn fill_bytes_via_simd<R: SimdRng<u32x2>>(rng: &mut R, dest: &mut [u8]) {
        // Forced inlining will keep the result in SIMD registers if
        // the code using it also uses it in a SIMD context.
        const CHUNK_SIZE: usize = mem::size_of::<u32x2>();
        let mut read_len = 0;
        for _ in 0..dest.len() / CHUNK_SIZE {
            let mut results = u8x8::from_bits(rng.generate());
            results = results.to_le();
            results.store_unaligned(&mut dest[read_len..]);
            read_len += CHUNK_SIZE;
        }
        let remainder = dest.len() % CHUNK_SIZE;
        if remainder > 0 {
            let mut results = u8x8::from_bits(rng.generate());
            results = results.to_le();
            let len = dest.len() - remainder;
            let mut buf = [0_u8; u8x8::lanes()];
            results.store_unaligned(&mut buf);
            dest[len..].copy_from_slice(&buf[..remainder]);
        }
    }
}
