#![feature(test)]
#![cfg(feature = "simd_support")]
#![feature(stdsimd)]

extern crate rand;
extern crate stdsimd;
extern crate test;

const RAND_BENCH_N: usize = 1 << 10;

use std::mem::*;
use stdsimd::simd::*;
use test::Bencher;

use rand::distributions::Uniform;
use rand::distributions::box_muller::LeadingZeros;
use rand::prelude::*;
use rand::prng::SfcAltSplit64x2a;

/*mod gen_range {
    use super::*;

    macro_rules! gen_range_int {
        ($fnn:ident, $ty:ident, $scalar:ident, $low:expr, $high:expr) => {
            #[bench]
            fn $fnn(b: &mut Bencher) {
                let mut rng = SfcAltSplit64x2a::from_rng(thread_rng()).unwrap();

                b.iter(|| {
                    let low = $ty::splat($low);
                    let mut high = $ty::splat($high);
                    for i in 0..$ty::lanes() {
                        high = high.replace(i, high.extract(i) + i as $scalar);
                    }
                    let mut accum = $ty::default();

                    for _ in 0..RAND_BENCH_N {
                        accum += rng.gen_range(low, high);
                        // force recalculation of range each time
                        high = (high + 1) & std::$scalar::MAX;
                    }
                    accum
                });
            }
        }
    }

    gen_range_int! { gen_range_i8x2, i8x2, i8, -20, 100}
    gen_range_int! { gen_range_u8x2, u8x2, u8, 0, 120 }
    gen_range_int! { gen_range_i8x4, i8x4, i8, -20, 100}
    gen_range_int! { gen_range_u8x4, u8x4, u8, 0, 120 }
    gen_range_int! { gen_range_i8x8, i8x8, i8, -20, 100}
    gen_range_int! { gen_range_u8x8, u8x8, u8, 0, 120 }
    gen_range_int! { gen_range_i8x16, i8x16, i8, -20, 100}
    gen_range_int! { gen_range_u8x16, u8x16, u8, 0, 120 }
    gen_range_int! { gen_range_i8x32, i8x32, i8, -20, 100}
    gen_range_int! { gen_range_u8x32, u8x32, u8, 0, 120 }
    gen_range_int! { gen_range_i8x64, i8x64, i8, -20, 100}
    gen_range_int! { gen_range_u8x64, u8x64, u8, 0, 120 }

    gen_range_int! { gen_range_i16x2, i16x2, i16, -500, 2000}
    gen_range_int! { gen_range_u16x2, u16x2, u16, 0, 2500 }
    gen_range_int! { gen_range_i16x4, i16x4, i16, -500, 2000}
    gen_range_int! { gen_range_u16x4, u16x4, u16, 0, 2500 }
    gen_range_int! { gen_range_i16x8, i16x8, i16, -500, 2000}
    gen_range_int! { gen_range_u16x8, u16x8, u16, 0, 2500 }
    gen_range_int! { gen_range_i16x16, i16x16, i16, -500, 2000}
    gen_range_int! { gen_range_u16x16, u16x16, u16, 0, 2500 }
    gen_range_int! { gen_range_i16x32, i16x32, i16, -500, 2000}
    gen_range_int! { gen_range_u16x32, u16x32, u16, 0, 2500 }
}

mod flat {
    use super::*;

    macro_rules! gen_range_int {
        ($fnn:ident, $ty:ident, $scalar:ident, $low:expr, $high:expr) => {
            #[bench]
            fn $fnn(b: &mut Bencher) {
                let mut rng = SfcAltSplit64x2a::from_rng(thread_rng()).unwrap();

                b.iter(|| {
                    let low = $ty::splat($low);
                    let mut high = $ty::splat($high);
                    let mut accum = $ty::default();

                    for _ in 0..RAND_BENCH_N {
                        accum += rng.gen_range(low, high);
                        // force recalculation of range each time
                        high = (high + 1) & std::$scalar::MAX;
                    }
                    accum
                });
            }
        }
    }

    gen_range_int! { gen_range_i8x2, i8x2, i8, -20, 100}
    gen_range_int! { gen_range_u8x2, u8x2, u8, 0, 120 }
    gen_range_int! { gen_range_i8x4, i8x4, i8, -20, 100}
    gen_range_int! { gen_range_u8x4, u8x4, u8, 0, 120 }
    gen_range_int! { gen_range_i8x8, i8x8, i8, -20, 100}
    gen_range_int! { gen_range_u8x8, u8x8, u8, 0, 120 }
    gen_range_int! { gen_range_i8x16, i8x16, i8, -20, 100}
    gen_range_int! { gen_range_u8x16, u8x16, u8, 0, 120 }
    gen_range_int! { gen_range_i8x32, i8x32, i8, -20, 100}
    gen_range_int! { gen_range_u8x32, u8x32, u8, 0, 120 }
    gen_range_int! { gen_range_i8x64, i8x64, i8, -20, 100}
    gen_range_int! { gen_range_u8x64, u8x64, u8, 0, 120 }

    gen_range_int! { gen_range_i16x2, i16x2, i16, -500, 2000}
    gen_range_int! { gen_range_u16x2, u16x2, u16, 0, 2500 }
    gen_range_int! { gen_range_i16x4, i16x4, i16, -500, 2000}
    gen_range_int! { gen_range_u16x4, u16x4, u16, 0, 2500 }
    gen_range_int! { gen_range_i16x8, i16x8, i16, -500, 2000}
    gen_range_int! { gen_range_u16x8, u16x8, u16, 0, 2500 }
    gen_range_int! { gen_range_i16x16, i16x16, i16, -500, 2000}
    gen_range_int! { gen_range_u16x16, u16x16, u16, 0, 2500 }
    gen_range_int! { gen_range_i16x32, i16x32, i16, -500, 2000}
    gen_range_int! { gen_range_u16x32, u16x32, u16, 0, 2500 }
}

mod uni_sample {
    use super::*;

    macro_rules! gen_range_int {
        ($fnn:ident, $ty:ident, $scalar:ident, $low:expr, $high:expr) => {
            #[bench]
            fn $fnn(b: &mut Bencher) {
                let mut rng = SfcAltSplit64x2a::from_rng(thread_rng()).unwrap();

                b.iter(|| {
                    let low = $ty::splat($low);
                    let mut high = $ty::splat($high);
                    for i in 0..$ty::lanes() {
                        high = high.replace(i, high.extract(i) + i as $scalar);
                    }
                    let mut accum = $ty::default();

                    for _ in 0..RAND_BENCH_N {
                        accum += rng.sample(Uniform::new(low, high));
                        // force recalculation of range each time
                        high = (high + 1) & std::$scalar::MAX;
                    }
                    accum
                });
            }
        }
    }

    gen_range_int! { gen_range_i8x2, i8x2, i8, -20, 100}
    gen_range_int! { gen_range_u8x2, u8x2, u8, 0, 120 }
    gen_range_int! { gen_range_i8x4, i8x4, i8, -20, 100}
    gen_range_int! { gen_range_u8x4, u8x4, u8, 0, 120 }
    gen_range_int! { gen_range_i8x8, i8x8, i8, -20, 100}
    gen_range_int! { gen_range_u8x8, u8x8, u8, 0, 120 }
    gen_range_int! { gen_range_i8x16, i8x16, i8, -20, 100}
    gen_range_int! { gen_range_u8x16, u8x16, u8, 0, 120 }
    gen_range_int! { gen_range_i8x32, i8x32, i8, -20, 100}
    gen_range_int! { gen_range_u8x32, u8x32, u8, 0, 120 }
    gen_range_int! { gen_range_i8x64, i8x64, i8, -20, 100}
    gen_range_int! { gen_range_u8x64, u8x64, u8, 0, 120 }

    gen_range_int! { gen_range_i16x2, i16x2, i16, -500, 2000}
    gen_range_int! { gen_range_u16x2, u16x2, u16, 0, 2500 }
    gen_range_int! { gen_range_i16x4, i16x4, i16, -500, 2000}
    gen_range_int! { gen_range_u16x4, u16x4, u16, 0, 2500 }
    gen_range_int! { gen_range_i16x8, i16x8, i16, -500, 2000}
    gen_range_int! { gen_range_u16x8, u16x8, u16, 0, 2500 }
    gen_range_int! { gen_range_i16x16, i16x16, i16, -500, 2000}
    gen_range_int! { gen_range_u16x16, u16x16, u16, 0, 2500 }
    gen_range_int! { gen_range_i16x32, i16x32, i16, -500, 2000}
    gen_range_int! { gen_range_u16x32, u16x32, u16, 0, 2500 }
}

mod modulo {
    use super::*;

    macro_rules! uniform_create {
        ($fnn:ident, $vec:ident, $scalar:ident) => {
            #[bench]
            fn $fnn(b: &mut Bencher) {
                let mut rng = SfcAltSplit64x2a::from_rng(thread_rng()).unwrap();
                let mut data = [$vec::default(); RAND_BENCH_N];
                let range = Uniform::new_inclusive($vec::splat(1), $vec::splat(!0));
                for x in data.iter_mut() {
                    *x = rng.sample(range);
                }

                b.iter(|| {
                    let mut accum = $vec::default();
                    for &range in data.iter() {
                        let unsigned_max = ::std::$scalar::MAX;
                        let ints_to_reject = (unsigned_max - range + 1) % range;
                        let lz = unsigned_max - ints_to_reject;
                        accum += lz;
                    }
                    accum
                });
            }
        };
    }

    uniform_create! { create_u8x2, u8x2, u8 }
    uniform_create! { create_u8x4, u8x4, u8 }
    uniform_create! { create_u8x8, u8x8, u8 }
    uniform_create! { create_u8x16, u8x16, u8 }
    uniform_create! { create_u8x32, u8x32, u8 }
    uniform_create! { create_u8x64, u8x64, u8 }

    uniform_create! { create_u16x2, u16x2, u16 }
    uniform_create! { create_u16x4, u16x4, u16 }
    uniform_create! { create_u16x8, u16x8, u16 }
    uniform_create! { create_u16x16, u16x16, u16 }
    uniform_create! { create_u16x32, u16x32, u16 }

    uniform_create! { create_u32x2, u32x2, u32 }
    uniform_create! { create_u32x4, u32x4, u32 }
    uniform_create! { create_u32x8, u32x8, u32 }
    uniform_create! { create_u32x16, u32x16, u32 }

    uniform_create! { create_u64x2, u64x2, u64 }
    uniform_create! { create_u64x4, u64x4, u64 }
    uniform_create! { create_u64x8, u64x8, u64 }
}

mod ctlz {
    use super::*;

    macro_rules! uniform_create {
        ($fnn:ident, $vec:ident, $scalar:ident) => {
            #[bench]
            fn $fnn(b: &mut Bencher) {
                let mut rng = SfcAltSplit64x2a::from_rng(thread_rng()).unwrap();
                let mut data = [$vec::default(); RAND_BENCH_N];
                let range = Uniform::new_inclusive($vec::splat(1), $vec::splat(!0));
                for x in data.iter_mut() {
                    *x = rng.sample(range);
                }

                b.iter(|| {
                    let mut accum = $vec::default();
                    for &range in data.iter() {
                        let mut ctlz = $vec::default();
                        for i in 0..$vec::lanes() {
                            ctlz = ctlz.replace(i, range.extract(i).leading_zeros() as $scalar);
                        }
                        accum += range << ctlz;
                    }
                    accum
                });
            }
        };
    }

    uniform_create! { create_u8x2, u8x2, u8 }
    uniform_create! { create_u8x4, u8x4, u8 }
    uniform_create! { create_u8x8, u8x8, u8 }
    uniform_create! { create_u8x16, u8x16, u8 }
    uniform_create! { create_u8x32, u8x32, u8 }
    uniform_create! { create_u8x64, u8x64, u8 }

    uniform_create! { create_u16x2, u16x2, u16 }
    uniform_create! { create_u16x4, u16x4, u16 }
    uniform_create! { create_u16x8, u16x8, u16 }
    uniform_create! { create_u16x16, u16x16, u16 }
    uniform_create! { create_u16x32, u16x32, u16 }

    uniform_create! { create_u32x2, u32x2, u32 }
    uniform_create! { create_u32x4, u32x4, u32 }
    uniform_create! { create_u32x8, u32x8, u32 }
    uniform_create! { create_u32x16, u32x16, u32 }

    uniform_create! { create_u64x2, u64x2, u64 }
    uniform_create! { create_u64x4, u64x4, u64 }
    uniform_create! { create_u64x8, u64x8, u64 }
}

// usually faster
mod float_hack {
    use super::*;

    macro_rules! uniform_create {
        ($fnn:ident, $vec:ident) => {
            #[bench]
            fn $fnn(b: &mut Bencher) {
                let mut rng = SfcAltSplit64x2a::from_rng(thread_rng()).unwrap();
                let mut data = [$vec::default(); RAND_BENCH_N];
                let range = Uniform::new_inclusive($vec::splat(1), $vec::splat(!0));
                for x in data.iter_mut() {
                    *x = rng.sample(range);
                }

                b.iter(|| {
                    let mut accum = $vec::default();
                    for &range in data.iter() {
                        let lz = range.leading_zeros();
                        accum += lz;
                    }
                    accum
                });
            }
        };
    }

    uniform_create! { create_u8x2, u8x2 }
    uniform_create! { create_u8x4, u8x4 }
    uniform_create! { create_u8x8, u8x8 }

    uniform_create! { create_u16x2, u16x2 }
    uniform_create! { create_u16x4, u16x4 }
    uniform_create! { create_u16x8, u16x8 }

    uniform_create! { create_u32x2, u32x2 }
    uniform_create! { create_u32x4, u32x4 }
    uniform_create! { create_u32x8, u32x8 }
}

mod float_hack_32 {
    use super::*;

    macro_rules! uniform_create {
        ($fnn:ident, $vec:ident, $large:ident, $bits:expr) => {
            #[bench]
            fn $fnn(b: &mut Bencher) {
                let mut rng = SfcAltSplit64x2a::from_rng(thread_rng()).unwrap();
                let mut data = [$vec::default(); RAND_BENCH_N];
                let range = Uniform::new_inclusive($vec::splat(1), $vec::splat(!0));
                for x in data.iter_mut() {
                    *x = rng.sample(range);
                }

                b.iter(|| {
                    let mut accum = $vec::default();
                    for &range in data.iter() {
                        let lz = $vec::from($large::from(range).leading_zeros()) - (32 - $bits);
                        accum += lz;
                    }
                    accum
                });
            }
        };
    }

    uniform_create! { create_u8x2, u8x2, u32x2, 8 }
    uniform_create! { create_u8x4, u8x4, u32x4, 8 }
    uniform_create! { create_u8x8, u8x8, u32x8, 8 }

    uniform_create! { create_u16x2, u16x2, u32x2, 16 }
    uniform_create! { create_u16x4, u16x4, u32x4, 16 }
    uniform_create! { create_u16x8, u16x8, u32x8, 16 }
}

mod lut_hack {
    use super::*;

    macro_rules! uniform_create {
        ($fnn:ident, $vec:ident, $table:expr, $bits:expr) => {
            #[bench]
            fn $fnn(b: &mut Bencher) {
                let mut rng = SfcAltSplit64x2a::from_rng(thread_rng()).unwrap();
                let mut data = [$vec::default(); RAND_BENCH_N];
                let range = Uniform::new_inclusive($vec::splat(1), $vec::splat(!0));
                for x in data.iter_mut() {
                    *x = rng.sample(range);
                }

                #[inline]
                fn lut(x: $vec) -> $vec {
                    let mut indices = $vec::splat(0);
                    for &upper_bound in &$table {
                        let cmp = x.gt($vec::splat(upper_bound));
                        indices -= $vec::from_bits(cmp);
                    }
                    indices
                }

                b.iter(|| {
                    let mut accum = $vec::default();
                    for &range in data.iter() {
                        let tt = range >> $bits/2;

                        let t = tt >> $bits/4;
                        let lut1 = lut(t);
                        let lut2 = lut(tt);
                        let mask = t.ne($vec::splat(0));
                        let r1 = mask.select($bits*3/4 + lut1, $bits/2 + lut2);

                        let t = range >> $bits/4;
                        let lut1 = lut(t);
                        let lut2 = lut(range);
                        let mask = t.ne($vec::splat(0));
                        let r2 = mask.select($bits/4 + lut1, lut2);

                        let mask = tt.ne($vec::splat(0));
                        let lz = mask.select(r1, r2);
                        accum += lz;
                    }
                    accum
                });
            }
        };
    }

    // for smaller lane-widths, there are likely optimizations / a better solution.
    uniform_create! { create_u8x2, u8x2, [1], 8 }
    uniform_create! { create_u8x4, u8x4, [1], 8 }
    uniform_create! { create_u8x8, u8x8, [1], 8 }
    uniform_create! { create_u8x16, u8x16, [1], 8 }
    uniform_create! { create_u8x32, u8x32, [1], 8 }
    uniform_create! { create_u8x64, u8x64, [1], 8 }

    uniform_create! { create_u16x2, u16x2, [1, 3, 7], 16 }
    uniform_create! { create_u16x4, u16x4, [1, 3, 7], 16 }
    uniform_create! { create_u16x8, u16x8, [1, 3, 7], 16 }
    uniform_create! { create_u16x16, u16x16, [1, 3, 7], 16 }
    uniform_create! { create_u16x32, u16x32, [1, 3, 7], 16 }

    uniform_create! { create_u32x2, u32x2, [1, 3, 7, 15, 31, 63, 127], 32 }
    uniform_create! { create_u32x4, u32x4, [1, 3, 7, 15, 31, 63, 127], 32 }
    uniform_create! { create_u32x8, u32x8, [1, 3, 7, 15, 31, 63, 127], 32 }
    uniform_create! { create_u32x16, u32x16, [1, 3, 7, 15, 31, 63, 127], 32 }
}

mod lut_hack2 {
    use super::*;

    macro_rules! uniform_create {
        ($fnn:ident, $vec:ident, $table:expr, $bits:expr) => {
            #[bench]
            fn $fnn(b: &mut Bencher) {
                let mut rng = SfcAltSplit64x2a::from_rng(thread_rng()).unwrap();
                let mut data = [$vec::default(); RAND_BENCH_N];
                let range = Uniform::new_inclusive($vec::splat(1), $vec::splat(!0));
                for x in data.iter_mut() {
                    *x = rng.sample(range);
                }

                #[inline]
                fn lut(x: $vec) -> $vec {
                    let mut indices = $vec::splat(0);
                    for &upper_bound in &$table {
                        let cmp = x.gt($vec::splat(upper_bound));
                        indices -= $vec::from_bits(cmp);
                    }
                    indices
                }

                b.iter(|| {
                    let mut accum = $vec::default();
                    for &range in data.iter() {
                        // if (tt = v >> 24) 24 + LogTable256[tt]
                        let tt1 = range >> $bits*3/4;
                        let r1 = $bits*3/4 + lut(tt1);

                        // else if (tt = v >> 16) 16 + LogTable256[tt]
                        let tt2 = range >> $bits/2;
                        let r2 = $bits/2 + lut(tt2);

                        // if tt1 else tt2
                        let cmp1 = tt1.ne($vec::splat(0));
                        let mut r = cmp1.select(r1, r2);

                        // else if (tt = v >> 8) 8 + LogTable256[tt]
                        let tt3 = range >> $bits/4;
                        let r3 = $bits/4 + lut(tt3);
                        // if !tt1 && !tt2
                        let cmp2 = tt2.ne($vec::splat(0));
                        r = (!cmp1 & !cmp2).select(r3, r);

                        // else LogTable256[v]
                        let r4 = lut(range);
                        // if !tt1 && !tt2 && !tt3
                        let cmp3 = tt3.ne($vec::splat(0));
                        r = (!cmp1 & !cmp2 & !cmp3).select(r4, r);

                        accum += r;
                    }
                    accum
                });
            }
        };
    }

    // for smaller lane-widths, there is likely a better solution.
    uniform_create! { create_u8x2, u8x2, [1], 8 }
    uniform_create! { create_u8x4, u8x4, [1], 8 }
    uniform_create! { create_u8x8, u8x8, [1], 8 }
    uniform_create! { create_u8x16, u8x16, [1], 8 }
    uniform_create! { create_u8x32, u8x32, [1], 8 }
    uniform_create! { create_u8x64, u8x64, [1], 8 }

    uniform_create! { create_u16x2, u16x2, [1, 3, 7], 16 }
    uniform_create! { create_u16x4, u16x4, [1, 3, 7], 16 }
    uniform_create! { create_u16x8, u16x8, [1, 3, 7], 16 }
    uniform_create! { create_u16x16, u16x16, [1, 3, 7], 16 }
    uniform_create! { create_u16x32, u16x32, [1, 3, 7], 16 }

    uniform_create! { create_u32x2, u32x2, [1, 3, 7, 15, 31, 63, 127], 32 }
    uniform_create! { create_u32x4, u32x4, [1, 3, 7, 15, 31, 63, 127], 32 }
    uniform_create! { create_u32x8, u32x8, [1, 3, 7, 15, 31, 63, 127], 32 }
    uniform_create! { create_u32x16, u32x16, [1, 3, 7, 15, 31, 63, 127], 32 }
}

mod shift_or_hack {
    use super::*;

    macro_rules! uniform_create {
        ($fnn:ident, $vec:ident) => {
            #[bench]
            fn $fnn(b: &mut Bencher) {
                let mut rng = SfcAltSplit64x2a::from_rng(thread_rng()).unwrap();
                let mut data = [$vec::default(); RAND_BENCH_N];
                let range = Uniform::new_inclusive($vec::splat(1), $vec::splat(!0));
                for x in data.iter_mut() {
                    *x = rng.sample(range);
                }

                b.iter(|| {
                    let mut accum = $vec::default();
                    #[allow(overflowing_literals)]
                    for &range in data.iter() {
                        let mut v = range;
                        let mut shift: $vec;
                        let mut r: $vec;
                        // for smaller lane-widths, the compiler should ignore the useless operations.
                        r     = $vec::from_bits(v.gt($vec::splat(0xFFFF))) & (1 << 4); v >>= r;
                        shift = $vec::from_bits(v.gt($vec::splat(0xFF)))   & (1 << 3); v >>= shift; r |= shift;
                        shift = $vec::from_bits(v.gt($vec::splat(0xF)))    & (1 << 2); v >>= shift; r |= shift;
                        shift = $vec::from_bits(v.gt($vec::splat(0x3)))    & (1 << 1); v >>= shift; r |= shift;
                        let lz = r | (v >> 1);
                        accum += lz;
                    }
                    accum
                });
            }
        };
    }

    uniform_create! { create_u8x2, u8x2 }
    uniform_create! { create_u8x4, u8x4 }
    uniform_create! { create_u8x8, u8x8 }
    uniform_create! { create_u8x16, u8x16 }
    uniform_create! { create_u8x32, u8x32 }
    uniform_create! { create_u8x64, u8x64 }

    uniform_create! { create_u16x2, u16x2 }
    uniform_create! { create_u16x4, u16x4 }
    uniform_create! { create_u16x8, u16x8 }
    uniform_create! { create_u16x16, u16x16 }
    uniform_create! { create_u16x32, u16x32 }

    uniform_create! { create_u32x2, u32x2 }
    uniform_create! { create_u32x4, u32x4 }
    uniform_create! { create_u32x8, u32x8 }
    uniform_create! { create_u32x16, u32x16 }

    uniform_create! { create_u64x2, u64x2 }
    uniform_create! { create_u64x4, u64x4 }
    uniform_create! { create_u64x8, u64x8 }
}

mod two_shift_or_hack {
    use super::*;

    macro_rules! uniform_create {
        ($fnn:ident, $vec:ident) => {
            #[bench]
            fn $fnn(b: &mut Bencher) {
                let mut rng = SfcAltSplit64x2a::from_rng(thread_rng()).unwrap();
                let mut data = [$vec::default(); RAND_BENCH_N];
                let range = Uniform::new_inclusive($vec::splat(1), $vec::splat(!0));
                for x in data.iter_mut() {
                    *x = rng.sample(range);
                }

                b.iter(|| {
                    let mut accum = $vec::default();
                    #[allow(overflowing_literals)]
                    for &range in data.iter() {
                        let mut v = range;
                        let mut r = $vec::splat(0); // result of log2(v) will go here

                        macro_rules! round {
                            ($B:expr, $S:expr) => {{
                                let cmp = (v & $B).ne($vec::splat(0));
                                let s = cmp.select($vec::splat($S), $vec::splat(0));
                                v >>= s;
                                r |= s;
                            }}
                        }

                        // size_of::<$vec::lane_width()>
                        let size = size_of::<$vec>() / $vec::lanes();
                        if size >= 8 { round!(0xFFFF_FFFF_0000_0000, 32); }
                        if size >= 4 { round!(0xFFFF_0000, 16); }
                        if size >= 2 { round!(0xFF00, 8); }
                        round!(0xF0, 4);
                        round!(0xC, 2);
                        round!(0x2, 1);

                        accum += r;
                    }
                    accum
                });
            }
        };
    }

    uniform_create! { create_u8x2, u8x2 }
    uniform_create! { create_u8x4, u8x4 }
    uniform_create! { create_u8x8, u8x8 }
    uniform_create! { create_u8x16, u8x16 }
    uniform_create! { create_u8x32, u8x32 }
    uniform_create! { create_u8x64, u8x64 }

    uniform_create! { create_u16x2, u16x2 }
    uniform_create! { create_u16x4, u16x4 }
    uniform_create! { create_u16x8, u16x8 }
    uniform_create! { create_u16x16, u16x16 }
    uniform_create! { create_u16x32, u16x32 }

    uniform_create! { create_u32x2, u32x2 }
    uniform_create! { create_u32x4, u32x4 }
    uniform_create! { create_u32x8, u32x8 }
    uniform_create! { create_u32x16, u32x16 }

    uniform_create! { create_u64x2, u64x2 }
    uniform_create! { create_u64x4, u64x4 }
    uniform_create! { create_u64x8, u64x8 }
}

mod mul_lut_hack {
    use super::*;

    macro_rules! uniform_create {
        ($fnn:ident, $vec:ident, $vec8:ident) => {
            #[bench]
            fn $fnn(b: &mut Bencher) {
                let mut rng = SfcAltSplit64x2a::from_rng(thread_rng()).unwrap();
                let mut data = [$vec::default(); RAND_BENCH_N];
                let range = Uniform::new_inclusive($vec::splat(1), $vec::splat(!0));
                for x in data.iter_mut() {
                    *x = rng.sample(range);
                }

                const LUT: u8x32 = u8x32::new(
                    0, 9,  1,  10, 13, 21, 2,  29, 11, 14, 16, 18, 22, 25, 3, 30,
                    8, 12, 20, 28, 15, 17, 24, 7,  19, 27, 23, 6,  26, 5,  4, 31,
                );

                b.iter(|| {
                    let mut accum = $vec::default();
                    for &range in data.iter() {
                        let mut v = range;
                        v |= v >> 1; // first round down to one less than a power of 2
                        v |= v >> 2;
                        v |= v >> 4;
                        v |= v >> 8;
                        v |= v >> 16;

                        let indices = $vec8::from((v * 0x07c4acdd) >> 27);
                        let lz = $vec::from($vec8::new(
                            LUT.extract(indices.extract(0) as usize),
                            LUT.extract(indices.extract(1) as usize),
                            LUT.extract(indices.extract(2) as usize),
                            LUT.extract(indices.extract(3) as usize),
                            LUT.extract(indices.extract(4) as usize),
                            LUT.extract(indices.extract(5) as usize),
                            LUT.extract(indices.extract(6) as usize),
                            LUT.extract(indices.extract(7) as usize),
                            LUT.extract(indices.extract(8) as usize),
                            LUT.extract(indices.extract(9) as usize),
                            LUT.extract(indices.extract(10) as usize),
                            LUT.extract(indices.extract(11) as usize),
                            LUT.extract(indices.extract(12) as usize),
                            LUT.extract(indices.extract(13) as usize),
                            LUT.extract(indices.extract(14) as usize),
                            LUT.extract(indices.extract(15) as usize),
                        ));
                        accum += lz;
                    }
                    accum
                });
            }
        };
    }

    uniform_create! { create_u32x16, u32x16, u8x16 }
}*/
