#![feature(test)]
#![cfg(feature = "simd_support")]
#![feature(stdsimd)]

extern crate rand;
// extern crate stdsimd;
extern crate packed_simd;
extern crate test;

const RAND_BENCH_N: usize = 1 << 10;

use packed_simd::*;
use test::Bencher;

use rand::prelude::*;
use rand::prng::SfcAlt64x2k;

mod approx {
    use super::*;

    macro_rules! gen_range_int {
        ($fnn:ident, $ty:ident, $scalar:ident, $low:expr, $high:expr) => {
            #[bench]
            fn $fnn(b: &mut Bencher) {
                let mut rng = SfcAlt64x2k::from_rng(thread_rng()).unwrap();
                let low = $ty::splat($low);

                b.iter(|| {
                    let mut high = $ty::splat($high);
                    let mut accum = $ty::splat(0);

                    for _ in 0..RAND_BENCH_N {
                        accum += rng.gen_range(low, high);
                        // force recalculation of range each time
                        high = (high + 1) & $scalar::max_value();
                    }
                    accum
                });
            }
        }
    }

    gen_range_int! { sample_single_i8x2, i8x2, i8, -20, 100}
    gen_range_int! { sample_single_i8x4, i8x4, i8, -20, 100}
    gen_range_int! { sample_single_i8x8, i8x8, i8, -20, 100}
    gen_range_int! { sample_single_i8x16, i8x16, i8, -20, 100}
    gen_range_int! { sample_single_i8x32, i8x32, i8, -20, 100}
    gen_range_int! { sample_single_i8x64, i8x64, i8, -20, 100}

    gen_range_int! { sample_single_i16x2, i16x2, i16, -500, 2000}
    gen_range_int! { sample_single_i16x4, i16x4, i16, -500, 2000}
    gen_range_int! { sample_single_i16x8, i16x8, i16, -500, 2000}
    gen_range_int! { sample_single_i16x16, i16x16, i16, -500, 2000}
    gen_range_int! { sample_single_i16x32, i16x32, i16, -500, 2000}
}

mod no_approx {
    use super::*;

    use rand::distributions::utils::WideningMultiply;

    macro_rules! gen_range_int {
        ($fnn:ident, $ty:ident, $unsigned:ident, $scalar:ident, $u_scalar:ident, $low:expr, $high:expr) => {
            #[bench]
            fn $fnn(b: &mut Bencher) {
                let mut rng = SfcAlt64x2k::from_rng(thread_rng()).unwrap();
                let low = $ty::splat($low);

                b.iter(|| {
                    let mut high = $ty::splat($high);
                    let mut accum = $ty::splat(0);

                    for _ in 0..RAND_BENCH_N {
                        // #[inline(always)] means there should be no other
                        // difference in the benchmarks

                        let range: $unsigned = (high - low).cast();
                        let unsigned_max = $u_scalar::max_value();
                        let ints_to_reject = (unsigned_max - range + 1) % range;
                        let zone = unsigned_max - ints_to_reject;

                        let mut v: $unsigned = rng.gen();
                        loop {
                            let (hi, lo) = v.wmul(range);
                            let mask = lo.le(zone);
                            if mask.all() {
                                let hi: $ty = hi.cast();
                                accum += $low + hi;
                                break;
                            }
                            // Replace only the failing lanes
                            v = mask.select(v, rng.gen());
                        }

                        // force recalculation of range each time
                        high = (high + 1) & $scalar::max_value();
                    }
                    accum
                });
            }
        }
    }

    gen_range_int! { sample_single_i8x2, i8x2, u8x2, i8, u8, -2, 100 }
    gen_range_int! { sample_single_i8x4, i8x4, u8x4, i8, u8, -2, 100 }
    gen_range_int! { sample_single_i8x8, i8x8, u8x8, i8, u8, -2, 100 }
    gen_range_int! { sample_single_i8x16, i8x16, u8x16, i8, u8, -2, 100 }
    gen_range_int! { sample_single_i8x32, i8x32, u8x32, i8, u8, -2, 100 }
    gen_range_int! { sample_single_i8x64, i8x64, u8x64, i8, u8, -2, 100 }

    gen_range_int! { sample_single_i16x2, i16x2, u16x2, i16, u16, -500, 2000 }
    gen_range_int! { sample_single_i16x4, i16x4, u16x4, i16, u16, -500, 2000 }
    gen_range_int! { sample_single_i16x8, i16x8, u16x8, i16, u16, -500, 2000 }
    gen_range_int! { sample_single_i16x16, i16x16, u16x16, i16, u16, -500, 2000 }
    gen_range_int! { sample_single_i16x32, i16x32, u16x32, i16, u16, -500, 2000 }
}
