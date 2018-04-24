#![feature(test)]
#![cfg_attr(feature = "simd_support", feature(stdsimd))]
#![cfg_attr(feature = "i128_support", feature(i128_type, i128))]

extern crate test;
extern crate rand;

const RAND_BENCH_N: u64 = 1 << 12;

use std::mem::size_of;
#[cfg(feature = "simd_support")]
use std::simd::*;
use test::{black_box, Bencher};

use rand::{Rng, NewRng};

#[cfg(feature = "simd_support")]
use rand::prng::Sfc32x4Rng;
use rand::distributions::*;
#[cfg(feature="simd_support")]
use rand::distributions::box_muller::*;

const TWO_PI: f32x8 = f32x8::splat(2.0 * std::f32::consts::PI);

#[bench]
fn bm_reg(b: &mut Bencher) {
    let mut rng = Sfc32x4Rng::new();

    b.iter(|| {
        let mut accum = f32x8::default();
        for _ in 0..::RAND_BENCH_N {
            let radius = (f32x8::splat(-2.0) * rng.gen::<f32x8>().ln()).sqrt();
            let (sin_theta, cos_theta) = (TWO_PI * rng.gen::<f32x8>()).sin_cos();

            accum += radius * sin_theta;
            accum += radius * cos_theta;
        }
        black_box(accum);
    });
    b.bytes = size_of::<(f32x8, f32x8)>() as u64 * ::RAND_BENCH_N;
}

#[bench]
fn bm_tuple(b: &mut Bencher) {
    let mut rng = Sfc32x4Rng::new();

    b.iter(|| {
        let mut accum = f32x8::default();
        for _ in 0..::RAND_BENCH_N {
            let (u0, u1): (f32x8, f32x8) = rng.gen();

            let radius = (f32x8::splat(-2.0) * u0.ln()).sqrt();
            let (sin_theta, cos_theta) = (TWO_PI * u1).sin_cos();

            accum += radius * sin_theta;
            accum += radius * cos_theta;
        }
        black_box(accum);
    });
    b.bytes = size_of::<(f32x8, f32x8)>() as u64 * ::RAND_BENCH_N;
}

#[bench]
fn bm_closure(b: &mut Bencher) {
    let mut rng = Sfc32x4Rng::new();

    b.iter(|| {
        let mut accum = f32x8::default();
        for _ in 0..::RAND_BENCH_N {
            let mut gen = || rng.gen::<f32x8>();

            let radius = (f32x8::splat(-2.0) * gen().ln()).sqrt();
            let (sin_theta, cos_theta) = (TWO_PI * gen()).sin_cos();

            accum += radius * sin_theta;
            accum += radius * cos_theta;
        }
        black_box(accum);
    });
    b.bytes = size_of::<(f32x8, f32x8)>() as u64 * ::RAND_BENCH_N;
}
