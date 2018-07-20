#![feature(test)]
#![cfg_attr(all(feature="i128_support", feature="nightly"), allow(stable_features))] // stable since 2018-03-27
#![cfg_attr(all(feature="i128_support", feature="nightly"), feature(i128_type, i128))]
#![cfg_attr(all(feature="simd_support", feature="nightly"), feature(stdsimd))]

extern crate test;
extern crate rand;

const RAND_BENCH_N: u64 = 1000;

use std::mem::size_of;
use test::Bencher;

use rand::{Rng, FromEntropy, XorShiftRng};
use rand::prng::hc128::Hc128Rng;
use rand::distributions::*;

macro_rules! distr_int {
    ($fnn:ident, $ty:ident, $rng:ident, $distr:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = $rng::from_entropy();
            let distr = $distr;

            b.iter(|| {
                let mut accum = $ty::default();
                for _ in 0..::RAND_BENCH_N {
                    let x: $ty = distr.sample(&mut rng);
                    // stdsimd has no `wrapping_add`, so we must rely
                    // on the lack of overflow checks in release mode.
                    accum += x;
                }
                accum
            });
            b.bytes = size_of::<$ty>() as u64 * ::RAND_BENCH_N;
        }
    }
}

macro_rules! distr_float {
    ($fnn:ident, $ty:ident, $rng:ident, $distr:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = $rng::from_entropy();
            #[allow(unused_mut)]
            let mut distr = $distr;

            b.iter(|| {
                let mut accum = $ty::default();
                for _ in 0..::RAND_BENCH_N {
                    let x: $ty = distr.sample(&mut rng);
                    accum += x;
                }
                accum
            });
            b.bytes = size_of::<$ty>() as u64 * ::RAND_BENCH_N;
        }
    }
}

macro_rules! distr {
    ($fnn:ident, $ty:ident, $rng:ident, $distr:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = $rng::from_entropy();
            let distr = $distr;

            b.iter(|| {
                let mut accum = 0u32;
                for _ in 0..::RAND_BENCH_N {
                    let x: $ty = distr.sample(&mut rng);
                    accum = accum.wrapping_add(x as u32);
                }
                accum
            });
            b.bytes = size_of::<$ty>() as u64 * ::RAND_BENCH_N;
        }
    }
}

// uniform
distr_int!(distr_uniform_i8, i8, XorShiftRng, Uniform::new(20i8, 100));
distr_int!(distr_uniform_i16, i16, XorShiftRng, Uniform::new(-500i16, 2000));
distr_int!(distr_uniform_i32, i32, XorShiftRng, Uniform::new(-200_000_000i32, 800_000_000));
distr_int!(distr_uniform_i64, i64, XorShiftRng, Uniform::new(3i64, 123_456_789_123));
#[cfg(feature = "i128_support")]
distr_int!(distr_uniform_i128, i128, XorShiftRng, Uniform::new(-123_456_789_123i128, 123_456_789_123_456_789));

distr_float!(distr_range_f32, f32, XorShiftRng, Uniform::new(2.26f32, 2.319));
distr_float!(distr_range_f64, f64, XorShiftRng, Uniform::new(2.26f64, 2.319));

// standard
distr_int!(distr_standard_i8, i8, XorShiftRng, Standard);
distr_int!(distr_standard_i16, i16, XorShiftRng, Standard);
distr_int!(distr_standard_i32, i32, XorShiftRng, Standard);
distr_int!(distr_standard_i64, i64, XorShiftRng, Standard);
#[cfg(feature = "i128_support")]
distr_int!(distr_standard_i128, i128, XorShiftRng, Standard);

distr!(distr_standard_bool, bool, XorShiftRng, Standard);
distr!(distr_standard_alphanumeric, char, XorShiftRng, Alphanumeric);
distr!(distr_standard_codepoint, char, XorShiftRng, Standard);

// distributions
distr_float!(distr_exp, f64, XorShiftRng, Exp::new(1.23 * 4.56));
distr_float!(distr_normal, f64, XorShiftRng, Normal::new(-1.23, 4.56));
distr_float!(distr_stdnorm_xorshift, f64, XorShiftRng, StandardNormal);
distr_float!(distr_stdnorm_hc128, f64, Hc128Rng, StandardNormal);
distr_float!(distr_log_normal, f64, XorShiftRng, LogNormal::new(-1.23, 4.56));
distr_float!(distr_gamma_large_shape, f64, XorShiftRng, Gamma::new(10., 1.0));
distr_float!(distr_gamma_small_shape, f64, XorShiftRng, Gamma::new(0.1, 1.0));
distr_float!(distr_cauchy, f64, XorShiftRng, Cauchy::new(4.2, 6.9));
distr_int!(distr_binomial, u64, XorShiftRng, Binomial::new(20, 0.7));
distr_int!(distr_poisson, u64, XorShiftRng, Poisson::new(4.0));
distr!(distr_bernoulli, bool, XorShiftRng, Bernoulli::<u64>::new(0.18));


// construct and sample from a range
macro_rules! gen_range_int {
    ($fnn:ident, $ty:ident, $low:expr, $high:expr) => {
        #[bench]
        fn $fnn(b: &mut Bencher) {
            let mut rng = XorShiftRng::from_entropy();

            b.iter(|| {
                let mut high = $high;
                let mut accum: $ty = 0;
                for _ in 0..::RAND_BENCH_N {
                    accum = accum.wrapping_add(rng.gen_range($low, high));
                    // force recalculation of range each time
                    high = high.wrapping_add(1) & std::$ty::MAX;
                }
                accum
            });
            b.bytes = size_of::<$ty>() as u64 * ::RAND_BENCH_N;
        }
    }
}

gen_range_int!(gen_range_i8, i8, -20i8, 100);
gen_range_int!(gen_range_i16, i16, -500i16, 2000);
gen_range_int!(gen_range_i32, i32, -200_000_000i32, 800_000_000);
gen_range_int!(gen_range_i64, i64, 3i64, 123_456_789_123);
#[cfg(feature = "i128_support")]
gen_range_int!(gen_range_i128, i128, -12345678901234i128, 123_456_789_123_456_789);

#[bench]
fn dist_iter(b: &mut Bencher) {
    let mut rng = XorShiftRng::from_entropy();
    let distr = Normal::new(-2.71828, 3.14159);
    let mut iter = distr.sample_iter(&mut rng);

    b.iter(|| {
        let mut accum = 0.0;
        for _ in 0..::RAND_BENCH_N {
            accum += iter.next().unwrap();
        }
        accum
    });
    b.bytes = size_of::<f64>() as u64 * ::RAND_BENCH_N;
}

#[cfg(feature = "simd_support")]
mod simd {
    // extern crate stdsimd;

    use super::*;

    use std::simd::*;

    use rand::prng::*;

    distr_int!(distr_standard_codepoint_x4, u32x4, Sfc32x4Rng, SimdCharDistribution);
    distr_int!(distr_standard_alphanumeric_x8, u32x8, Sfc32x4Rng, AlphanumericSimd);

    #[bench]
    fn distr_bernoulli_x2(b: &mut Bencher) {
        let mut rng = Sfc32x4Rng::from_entropy();
        let distr = Bernoulli::<u64x2>::new(f64x2::splat(0.18));

        b.iter(|| {
            let mut accum = u64x2::splat(0);
            for _ in 0..::RAND_BENCH_N {
                let x = distr.sample(&mut rng);
                accum += u64x2::from_bits(x);
            }
            accum
        });
        b.bytes = size_of::<m64x2>() as u64 * ::RAND_BENCH_N;
    }

    macro_rules! many_int_distr {
        ($(($fnn:ident, $rng:ident, $ty:ident),)+, $low:expr, $high:expr) => ($(
            distr_int!($fnn, $ty, $rng, Uniform::new($ty::splat($low), $ty::splat($high)));
        )+)
    }

    many_int_distr! {
        (distr_uniform_i8x2, Sfc32x2Rng, i8x2),
        (distr_uniform_i8x4, Sfc32x2Rng, i8x4),
        (distr_uniform_i8x8, Sfc32x2Rng, i8x8),
        (distr_uniform_i8x16, Sfc32x4Rng, i8x16),
        (distr_uniform_i8x32, Sfc32x4Rng, i8x32),
        (distr_uniform_i8x64, Sfc32x4Rng, i8x64),,
        20, 100
    }

    many_int_distr! {
        (distr_uniform_i16x2, Sfc32x2Rng, i16x2),
        (distr_uniform_i16x4, Sfc32x2Rng, i16x4),
        (distr_uniform_i16x8, Sfc32x4Rng, i16x8),
        (distr_uniform_i16x16, Sfc32x4Rng, i16x16),
        (distr_uniform_i16x32, Sfc32x4Rng, i16x32),,
        -500, 2000
    }

    many_int_distr! {
        (distr_uniform_i32x2, Sfc32x2Rng, i32x2),
        (distr_uniform_i32x4, Sfc32x4Rng, i32x4),
        (distr_uniform_i32x8, Sfc32x4Rng, i32x8),
        (distr_uniform_i32x16, Sfc32x4Rng, i32x16),,
        -200_000_000, 800_000_000
    }

    many_int_distr! {
        (distr_uniform_i64x2, Sfc32x4Rng, i64x2),
        (distr_uniform_i64x4, Sfc32x4Rng, i64x4),
        (distr_uniform_i64x8, Sfc32x4Rng, i64x8),,
        3, 123_456_789_123
    }

    macro_rules! many_float_distr {
        ($(($fnn:ident, $rng:ident, $ty:ident),)+, $low:expr, $high:expr) => ($(
            distr_float!($fnn, $ty, $rng, Uniform::new($ty::splat($low), $ty::splat($high)));
        )+)
    }

    many_float_distr! {
        (distr_range_f32x2, Sfc32x2Rng, f32x2),
        (distr_range_f32x4, Sfc32x4Rng, f32x4),
        (distr_range_f32x8, Sfc32x4Rng, f32x8),
        (distr_range_f32x16, Sfc32x4Rng, f32x16),

        (distr_range_f64x2, Sfc32x4Rng, f64x2),
        (distr_range_f64x4, Sfc32x4Rng, f64x4),
        (distr_range_f64x8, Sfc32x4Rng, f64x8),,
        2.26, 2.319
    }
}
