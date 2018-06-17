#![cfg(feature = "simd_support")]
#![feature(stdsimd)]

#![allow(unused_imports)]

extern crate rand;
extern crate stdsimd;

use std::f64;
use std::mem::{size_of, transmute};
use stdsimd::simd::*;

use rand::{Rng, thread_rng, SeedableRng, FromEntropy};
use rand::rngs::SmallRng;
use rand::prng::SfcAltSplit64x2a;
use rand::distributions::box_muller::{BoxMuller, BoxMullerCore, SimdMath};
use rand::distributions::{StandardNormal, Uniform};

/*fn float_cmp(a: f64, b: f64, epsilon: f64) -> bool {
    let abs_a = a.abs();
    let abs_b = b.abs();
    let diff = (a - b).abs();

    if a == b { // shortcut, handles infinities
        true
    } else if a == 0.0 || b == 0.0 || diff < f64::MIN_POSITIVE {
        // a or b is zero or both are extremely close to it
        // relative error is less meaningful here
        println!("new eps {:?}", epsilon * f64::MIN_POSITIVE);
        diff < epsilon * f64::MIN_POSITIVE
    } else { // use relative error
        println!("{:?}", diff / (abs_a + abs_b).min(f64::MAX));
        diff / (abs_a + abs_b).min(f64::MAX) < epsilon
    }
}*/

fn test_distr<R: Rng>(rng: &mut R, sample: fn(&mut R) -> f64, exp_m: f64, exp_sd: f64) {
    let mut arr = [0.0; 1 << 14];

    for x in arr.iter_mut() {
        *x = sample(rng);
    }

    let mean = arr.iter().sum::<f64>() / arr.len() as f64;
    assert!((mean - exp_m).abs() < 1.875e-2, "{:?}", (mean - exp_m).abs());

    let s = arr.iter().map(|&e| (e - mean) * (e - mean)).sum::<f64>();
    let std_dev = (s / (arr.len() - 1) as f64).sqrt();
    assert!((std_dev - exp_sd).abs() < 1.875e-2, "{:?}", (std_dev - exp_sd).abs());
}

#[test]
fn ziggurat_normal() {
    let mut rng = SmallRng::from_rng(thread_rng()).unwrap();
    fn sample<R: Rng>(rng: &mut R) -> f64 {
        rng.sample(StandardNormal)
    }
    test_distr(&mut rng, sample, 0.0, 1.0);
}

macro_rules! clt_normal {
    ($fnn:ident, $ty:ident, $scalar:ty) => (
        #[test]
        fn $fnn() {
            let mut rng = SfcAltSplit64x2a::from_rng(thread_rng()).unwrap();
            fn sample<R: Rng>(rng: &mut R) -> f64 {
                const HALF: $scalar = $ty::lanes() as $scalar / 2.0;
                let inv = 1.0 / ($ty::lanes() as $scalar / 12.0).sqrt();
                f64::from((rng.gen::<$ty>().sum() - HALF) * inv)
            }
            test_distr(&mut rng, sample, 0.0, 1.0);
        }
    )
}

clt_normal! { clt_normal_f32x2,  f32x2,  f32 }
clt_normal! { clt_normal_f32x4,  f32x4,  f32 }
clt_normal! { clt_normal_f32x8,  f32x8,  f32 }
clt_normal! { clt_normal_f32x16, f32x16, f32 }
clt_normal! { clt_normal_f64x2,  f64x2,  f64 }
clt_normal! { clt_normal_f64x4,  f64x4,  f64 }
clt_normal! { clt_normal_f64x8,  f64x8,  f64 }
