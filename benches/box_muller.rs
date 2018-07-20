#![feature(test)]
#![cfg(feature = "simd_support")]
#![feature(stdsimd)]

extern crate test;
extern crate rand;
// extern crate stdsimd;

const RAND_BENCH_N: u64 = 1 << 14;

use std::mem::{size_of, transmute};
use std::simd::*;
use test::Bencher;

use rand::{Rng, RngCore, FromEntropy};
use rand::prng::{SfcAlt64x2a, XorShiftRng};
use rand::prng::hc128::Hc128Rng;
use rand::distributions::box_muller::{BoxMuller, BoxMullerCore};
use rand::distributions::box_muller::SimdMath;
use rand::distributions::*;

macro_rules! distr_bm {
    ($fnn:ident, $ty:ident, $sample:ident) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = SfcAlt64x2a::from_entropy();

            b.iter(|| {
                let mut accum = $ty::default();
                for _ in 0..::RAND_BENCH_N {
                    let (x, y): ($ty, $ty) = BoxMuller::$sample(&mut rng);
                    accum += x;
                    accum += y;
                }
                accum
            });
            // Generates two values at once
            b.bytes = size_of::<$ty>() as u64 * 2 * ::RAND_BENCH_N;
        }
    }
}

macro_rules! distr_f {
    ($fnn:ident, $ty:ident, $scalar:ident, $rng:ident) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = $rng::from_entropy();
            const HALF: $scalar = $ty::lanes() as $scalar / 2.0;
            let inv = 1.0 / ($ty::lanes() as $scalar / 12.0).sqrt();

            b.iter(|| {
                let mut accum = $scalar::default();
                for _ in 0..::RAND_BENCH_N {
                    accum += (rng.gen::<$ty>().sum() - HALF) * inv;
                }
                accum
            });
            b.bytes = size_of::<$scalar>() as u64 * ::RAND_BENCH_N;
        }
    }
}

macro_rules! distr_fx {
    ($fnn:ident, $num:expr, $ty:ident, $scalar:ident, $rng:ident) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = $rng::from_entropy();
            const HALF: $scalar = $num as $scalar / 2.0;
            let inv = 1.0 / ($num as $scalar / 12.0).sqrt();

            b.iter(|| {
                let mut accum = $ty::default();
                for _ in 0..::RAND_BENCH_N {
                    let mut sum = $ty::default();
                    for _ in 0..$num {
                        sum += rng.gen::<$ty>();
                    }
                    accum += (sum - HALF) * inv;
                }
                accum
            });
            b.bytes = size_of::<$ty>() as u64 * ::RAND_BENCH_N;
        }
    }
}

// module structure to ease `cargo benchcmp` use

// hacked sin_cos method
mod hacked {
    use super::*;
    distr_bm!(norm_bm_f32x2, f32x2, hacked_trig);
    distr_bm!(norm_bm_f32x4, f32x4, hacked_trig);
    distr_bm!(norm_bm_f32x8, f32x8, hacked_trig);
    distr_bm!(norm_bm_f32x16, f32x16, hacked_trig);
    distr_bm!(norm_bm_f64x2, f64x2, hacked_trig);
    distr_bm!(norm_bm_f64x4, f64x4, hacked_trig);
    distr_bm!(norm_bm_f64x8, f64x8, hacked_trig);
}

// fast, approximate stdsimd sin/cos method
mod ftrig {
    use super::*;
    distr_bm!(norm_bm_f32x2, f32x2, ftrig);
    distr_bm!(norm_bm_f32x4, f32x4, ftrig);
    distr_bm!(norm_bm_f32x8, f32x8, ftrig);
    distr_bm!(norm_bm_f32x16, f32x16, ftrig);
    distr_bm!(norm_bm_f64x2, f64x2, ftrig);
    distr_bm!(norm_bm_f64x4, f64x4, ftrig);
    distr_bm!(norm_bm_f64x8, f64x8, ftrig);
}

// stdsimd sin/cos method
mod stdsimd_trig {
    use super::*;
    distr_bm!(norm_bm_f32x2, f32x2, stdsimd_trig);
    distr_bm!(norm_bm_f32x4, f32x4, stdsimd_trig);
    distr_bm!(norm_bm_f32x8, f32x8, stdsimd_trig);
    distr_bm!(norm_bm_f32x16, f32x16, stdsimd_trig);
    distr_bm!(norm_bm_f64x2, f64x2, stdsimd_trig);
    distr_bm!(norm_bm_f64x4, f64x4, stdsimd_trig);
    distr_bm!(norm_bm_f64x8, f64x8, stdsimd_trig);
}

// polar method
mod polar {
    use super::*;
    distr_bm!(norm_bm_f32x2, f32x2, polar);
    distr_bm!(norm_bm_f32x4, f32x4, polar);
    distr_bm!(norm_bm_f32x8, f32x8, polar);
    distr_bm!(norm_bm_f32x16, f32x16, polar);
    distr_bm!(norm_bm_f64x2, f64x2, polar);
    distr_bm!(norm_bm_f64x4, f64x4, polar);
    distr_bm!(norm_bm_f64x8, f64x8, polar);
}

// polar method without rejection sampling, to determine an upper bound
mod polar_no_rejection {
    use super::*;
    distr_bm!(norm_bm_f32x2, f32x2, polar_no_rejection);
    distr_bm!(norm_bm_f32x4, f32x4, polar_no_rejection);
    distr_bm!(norm_bm_f32x8, f32x8, polar_no_rejection);
    distr_bm!(norm_bm_f32x16, f32x16, polar_no_rejection);
    distr_bm!(norm_bm_f64x2, f64x2, polar_no_rejection);
    distr_bm!(norm_bm_f64x4, f64x4, polar_no_rejection);
    distr_bm!(norm_bm_f64x8, f64x8, polar_no_rejection);
}

mod sfc64x2 {
    use super::*;
    distr_f! { norm_clt_f32x2_f32, f32x2, f32, SfcAlt64x2a }
    distr_f! { norm_clt_f32x4_f32, f32x4, f32, SfcAlt64x2a }
    distr_f! { norm_clt_f32x8_f32, f32x8, f32, SfcAlt64x2a }
    distr_f! { norm_clt_f32x16_f32, f32x16, f32, SfcAlt64x2a }
    distr_f! { norm_clt_f64x2_f64, f64x2, f64, SfcAlt64x2a }
    distr_f! { norm_clt_f64x4_f64, f64x4, f64, SfcAlt64x2a }
    distr_f! { norm_clt_f64x8_f64, f64x8, f64, SfcAlt64x2a }

    distr_fx! { norm_clt_f32x2_fx, 4, f32x2, f32, SfcAlt64x2a }
    distr_fx! { norm_clt_f32x4_fx, 4, f32x4, f32, SfcAlt64x2a }
    distr_fx! { norm_clt_f32x8_fx, 4, f32x8, f32, SfcAlt64x2a }
    distr_fx! { norm_clt_f32x16_fx, 4, f32x16, f32, SfcAlt64x2a }
    distr_fx! { norm_clt_f64x2_fx, 4, f64x2, f64, SfcAlt64x2a }
    distr_fx! { norm_clt_f64x4_fx, 4, f64x4, f64, SfcAlt64x2a }
    distr_fx! { norm_clt_f64x8_fx, 4, f64x8, f64, SfcAlt64x2a }
}

mod hc128 {
    use super::*;
    distr_f! { norm_clt_f32x2_f32, f32x2, f32, Hc128Rng }
    distr_f! { norm_clt_f32x4_f32, f32x4, f32, Hc128Rng }
    distr_f! { norm_clt_f32x8_f32, f32x8, f32, Hc128Rng }
    distr_f! { norm_clt_f32x16_f32, f32x16, f32, Hc128Rng }
    distr_f! { norm_clt_f64x2_f64, f64x2, f64, Hc128Rng }
    distr_f! { norm_clt_f64x4_f64, f64x4, f64, Hc128Rng }
    distr_f! { norm_clt_f64x8_f64, f64x8, f64, Hc128Rng }

    distr_fx! { norm_clt_f32x2_fx, 4, f32x2, f32, Hc128Rng }
    distr_fx! { norm_clt_f32x4_fx, 4, f32x4, f32, Hc128Rng }
    distr_fx! { norm_clt_f32x8_fx, 4, f32x8, f32, Hc128Rng }
    distr_fx! { norm_clt_f32x16_fx, 4, f32x16, f32, Hc128Rng }
    distr_fx! { norm_clt_f64x2_fx, 4, f64x2, f64, Hc128Rng }
    distr_fx! { norm_clt_f64x4_fx, 4, f64x4, f64, Hc128Rng }
    distr_fx! { norm_clt_f64x8_fx, 4, f64x8, f64, Hc128Rng }
}

mod xorshift {
    use super::*;
    distr_f! { norm_clt_f32x2_f32, f32x2, f32, XorShiftRng }
    distr_f! { norm_clt_f32x4_f32, f32x4, f32, XorShiftRng }
    distr_f! { norm_clt_f32x8_f32, f32x8, f32, XorShiftRng }
    distr_f! { norm_clt_f32x16_f32, f32x16, f32, XorShiftRng }
    distr_f! { norm_clt_f64x2_f64, f64x2, f64, XorShiftRng }
    distr_f! { norm_clt_f64x4_f64, f64x4, f64, XorShiftRng }
    distr_f! { norm_clt_f64x8_f64, f64x8, f64, XorShiftRng }

    distr_fx! { norm_clt_f32x2_fx, 4, f32x2, f32, XorShiftRng }
    distr_fx! { norm_clt_f32x4_fx, 4, f32x4, f32, XorShiftRng }
    distr_fx! { norm_clt_f32x8_fx, 4, f32x8, f32, XorShiftRng }
    distr_fx! { norm_clt_f32x16_fx, 4, f32x16, f32, XorShiftRng }
    distr_fx! { norm_clt_f64x2_fx, 4, f64x2, f64, XorShiftRng }
    distr_fx! { norm_clt_f64x4_fx, 4, f64x4, f64, XorShiftRng }
    distr_fx! { norm_clt_f64x8_fx, 4, f64x8, f64, XorShiftRng }
}
