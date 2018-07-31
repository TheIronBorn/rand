#![cfg(feature = "simd_support")]
#![feature(test)]

extern crate packed_simd;
extern crate rand;
extern crate test;

use packed_simd::*;
use test::Bencher;
use std::mem;

use rand::prelude::*;
use rand::prng::*;
#[allow(unused_imports)]
use rand::distributions::utils::WideningMultiply;

const RAND_BENCH_N: u64 = 1 << 10;

macro_rules! benches {
    () => {
        simd_uniform_bench! {
            (simd_uniform_i8x2, simd_uniform_single_i8x2, SfcAlt64x2k, i8x2, u8x2),
            (simd_uniform_i8x4, simd_uniform_single_i8x4, SfcAlt64x2k, i8x4, u8x4),
            (simd_uniform_i8x8, simd_uniform_single_i8x8, SfcAlt64x2k, i8x8, u8x8),
            (simd_uniform_i8x16, simd_uniform_single_i8x16, SfcAlt64x4k, i8x16, u8x16),
            (simd_uniform_i8x32, simd_uniform_single_i8x32, SfcAlt64x4k, i8x32, u8x32),
            (simd_uniform_i8x64, simd_uniform_single_i8x64, SfcAlt64x4k, i8x64, u8x64),,
            i8, u8, -20, 100
        }

        simd_uniform_bench! {
            (simd_uniform_i16x2, simd_uniform_single_i16x2, SfcAlt64x2k, i16x2, u16x2),
            (simd_uniform_i16x4, simd_uniform_single_i16x4, SfcAlt64x2k, i16x4, u16x4),
            (simd_uniform_i16x8, simd_uniform_single_i16x8, SfcAlt64x4k, i16x8, u16x8),
            (simd_uniform_i16x16, simd_uniform_single_i16x16, SfcAlt64x4k, i16x16, u16x16),
            (simd_uniform_i16x32, simd_uniform_single_i16x32, SfcAlt64x4k, i16x32, u16x32),,
            i16, u16, -500, 2000
        }

        simd_uniform_bench! {
            (simd_uniform_i32x2, simd_uniform_single_i32x2, SfcAlt64x2k, i32x2, u32x2),
            (simd_uniform_i32x4, simd_uniform_single_i32x4, SfcAlt64x4k, i32x4, u32x4),
            (simd_uniform_i32x8, simd_uniform_single_i32x8, SfcAlt64x4k, i32x8, u32x8),
            (simd_uniform_i32x16, simd_uniform_single_i32x16, SfcAlt64x4k, i32x16, u32x16),,
            i32, u32, -200_000_000, 800_000_000
        }

        simd_uniform_bench! {
            (simd_uniform_i64x2, simd_uniform_single_i64x2, SfcAlt64x4k, i64x2, u64x2),
            (simd_uniform_i64x4, simd_uniform_single_i64x4, SfcAlt64x4k, i64x4, u64x4),
            (simd_uniform_i64x8, simd_uniform_single_i64x8, SfcAlt64x4k, i64x8, u64x8),,
            i64, u64, 3, 123_456_789_123
        }
    }
}

mod wide_mul {
    use super::*;

    macro_rules! simd_uniform_bench {
        ($(($uniform:ident, $gen_range:ident, $rng:ident, $ty:ident, $uty:ident),)+, $scalar:ident, $u_scalar:ident, $low:expr, $high:expr) => ($(
            #[bench]
            fn $uniform(b: &mut Bencher) {
                let mut rng = $rng::from_rng(&mut thread_rng()).unwrap();
                let low = $ty::splat($low);
                let high = $ty::splat($high);
                let range: $uty = (high - low).cast();

                // breaks if `range == 0` i.e. the full integer range
                let unsigned_max = $u_scalar::max_value();
                let ints_to_reject = (unsigned_max - range + 1) % range;
                let zone = unsigned_max - ints_to_reject;

                b.iter(|| {
                    let mut accum = $ty::default();
                    for _ in 0..::RAND_BENCH_N {
                        let mut v: $uty = rng.gen();
                        loop {
                            let (hi, lo) = v.wmul(range);
                            let mask = lo.le(zone);
                            if mask.all() {
                                let hi: $ty = hi.cast();
                                accum += low + hi;
                                break;
                            }
                            v = mask.select(v, rng.gen());
                        }
                    }
                    accum
                });
                b.bytes = mem::size_of::<$ty>() as u64 * ::RAND_BENCH_N;
            }

            // construct and sample from a range
            #[bench]
            fn $gen_range(b: &mut Bencher) {
                let mut rng = $rng::from_rng(&mut thread_rng()).unwrap();
                let mut low = $ty::splat($low);
                for i in 0..$ty::lanes() {
                    low = low.replace(i, $low - i as $scalar);
                }

                b.iter(|| {
                    let mut high = $ty::splat($high);
                    let mut accum = $ty::splat(0);
                    for _ in 0..::RAND_BENCH_N {
                        let range: $uty = (high - low).cast();
                        let mut zone = $uty::default();
                        for i in 0..$uty::lanes() {
                            let x = range.extract(i);
                            zone = zone.replace(i, x << x.leading_zeros());
                        }

                        let mut v: $uty = rng.gen();
                        loop {
                            let (hi, lo) = v.wmul(range);
                            let mask = lo.le(zone);
                            if mask.all() {
                                let hi: $ty = hi.cast();
                                accum += low + hi;
                                break;
                            }
                            v = mask.select(v, rng.gen());
                        }

                        // force recalculation of range each time
                        high = (high + 1) & std::$scalar::MAX;
                    }
                    accum
                });
                b.bytes = mem::size_of::<$ty>() as u64 * ::RAND_BENCH_N;
            }
        )+)
    }

    benches!();
}

mod bitmask_simple {
    use super::*;

    macro_rules! simd_uniform_bench {
        ($(($uniform:ident, $gen_range:ident, $rng:ident, $ty:ident, $uty:ident),)+, $scalar:ident, $u_scalar:ident, $low:expr, $high:expr) => ($(
            #[bench]
            fn $uniform(b: &mut Bencher) {
                let mut rng = $rng::from_rng(&mut thread_rng()).unwrap();

                let low = $ty::splat($low);
                let high = $ty::splat($high);

                let mut range: $uty = (high - low).cast();
                let mut mask = $uty::splat($u_scalar::max_value());
                range -= 1;
                for i in 0..$ty::lanes() {
                    mask = mask.replace(i, mask.extract(i) >> (range | 1).extract(i).leading_zeros());
                }

                b.iter(|| {
                    let mut accum = $ty::default();
                    for _ in 0..::RAND_BENCH_N {
                        let mut x = rng.gen::<$uty>() & mask;
                        // reject x > range
                        loop {
                            let cmp = x.le(range);
                            if cmp.all() {
                                break;
                            }
                            x = cmp.select(x, rng.gen::<$uty>() & mask);
                        }

                        let _x: $ty = x.cast();
                        accum += low + _x; // wrapping addition
                    }
                    accum
                });
                b.bytes = mem::size_of::<$ty>() as u64 * ::RAND_BENCH_N;
            }

            // construct and sample from a range
            #[bench]
            fn $gen_range(b: &mut Bencher) {
                let mut rng = $rng::from_rng(&mut thread_rng()).unwrap();
                let mut low = $ty::splat($low);
                for i in 0..$ty::lanes() {
                    low = low.replace(i, $low - i as $scalar);
                }

                b.iter(|| {
                    let mut high = $ty::splat($high);
                    let mut accum = $ty::splat(0);
                    for _ in 0..::RAND_BENCH_N {
                        let mut range: $uty = (high - low).cast();
                        let mut mask = $uty::splat($u_scalar::max_value());
                        range -= 1;
                        for i in 0..$ty::lanes() {
                            mask = mask.replace(i, mask.extract(i) >> (range | 1).extract(i).leading_zeros());
                        }

                        let mut x = rng.gen::<$uty>() & mask;
                        // reject x > range
                        loop {
                            let cmp = x.le(range);
                            if cmp.all() {
                                break;
                            }
                            x = cmp.select(x, rng.gen::<$uty>() & mask);
                        }

                        let _x: $ty = x.cast();
                        accum += low + _x; // wrapping addition

                        // force recalculation of range each time
                        high = (high + 1) & std::$scalar::MAX;
                    }
                    accum
                });
                b.bytes = mem::size_of::<$ty>() as u64 * ::RAND_BENCH_N;
            }
        )+)
    }

    benches!();
}

mod bitmask_alt {
    use super::*;

    macro_rules! simd_uniform_bench {
        ($(($uniform:ident, $gen_range:ident, $rng:ident, $ty:ident, $uty:ident),)+, $scalar:ident, $u_scalar:ident, $low:expr, $high:expr) => ($(
            #[bench]
            fn $uniform(b: &mut Bencher) {
                const BITS: $u_scalar = mem::size_of::<$u_scalar>() as $u_scalar * 8;

                let mut rng = $rng::from_rng(&mut thread_rng()).unwrap();

                let low = $ty::splat($low);
                let high = $ty::splat($high);

                let mut range: $uty = (high - low).cast();
                // range.next_power_of_two()
                range -= 1;
                let mut zeros = $uty::default();
                for i in 0..$ty::lanes() {
                    zeros = zeros.replace(i, (range | 1).extract(i).leading_zeros() as $u_scalar);
                }
                // may be a better impl than a scalar ctlz and variable shift
                let mask = $uty::splat($u_scalar::max_value()) >> zeros;

                b.iter(|| {
                    let mut accum = $ty::default();
                    for _ in 0..::RAND_BENCH_N {
                        let mut res = rng.gen::<$uty>();
                        let mut val = res & mask;

                        loop {
                            let mut shift = BITS / 2;
                            let mut cmp = val.le(range);
                            if cmp.all() {
                                break;
                            }

                            // try the remaining bits
                            while zeros.ge($uty::splat(shift)).any() {
                                // res >>= shift as u32;

                                // if the bits didn't pass and there are bits remaining...
                                let consume_bits = !cmp & zeros.ge($uty::splat(shift));
                                res = consume_bits.select(res >> shift as u32, res);

                                val = res & mask;
                                cmp = val.le(range);
                                if cmp.all() {
                                    break;
                                }
                                shift = BITS - (BITS - shift) / 2;
                            }

                            // get more bits from the RNG, don't replace passing lanes
                            res = cmp.select(res, rng.gen::<$uty>());
                            val = res & mask;
                        }

                        let _val: $ty = val.cast();
                        accum += low + _val; // wrapping addition
                    }
                    accum
                });
                b.bytes = mem::size_of::<$ty>() as u64 * ::RAND_BENCH_N;
            }

            // construct and sample from a range
            #[bench]
            fn $gen_range(b: &mut Bencher) {
                let mut rng = $rng::from_rng(&mut thread_rng()).unwrap();
                let mut low = $ty::splat($low);
                for i in 0..$ty::lanes() {
                    low = low.replace(i, $low - i as $scalar);
                }

                b.iter(|| {
                    let mut high = $ty::splat($high);
                    let mut accum = $ty::splat(0);
                    for _ in 0..::RAND_BENCH_N {
                        let mut range: $uty = (high - low).cast();
                        range -= 1;
                        let mut zeros = $uty::default();
                        for i in 0..$ty::lanes() {
                            zeros = zeros.replace(i, (range | 1).extract(i).leading_zeros() as $u_scalar);
                        }
                        let mask = $uty::splat($u_scalar::max_value()) >> zeros;

                        let mut res = rng.gen::<$uty>();
                        let mut val = res & mask;

                        loop {
                            let mut shift = 32;
                            let mut cmp = val.le(range);
                            if cmp.all() {
                                break;
                            }

                            // try the remaining bits
                            while zeros.ge($uty::splat(shift)).any() {
                                // res >>= shift as u32;

                                // if the bits didn't pass and there are bits remaining
                                let consume_bits = !cmp & zeros.ge($uty::splat(shift));
                                res = consume_bits.select(res >> shift as u32, res);

                                val = res & mask;
                                cmp = val.le(range);
                                if cmp.all() {
                                    break;
                                }
                                shift = 64 - (64 - shift) / 2;
                            }

                            // get more bits from the RNG, don't replace passing lanes
                            res = cmp.select(res, rng.gen::<$uty>());
                            val = res & mask;
                        }

                        let _val: $ty = val.cast();
                        accum += low + _val; // wrapping addition

                        // force recalculation of range each time
                        high = (high + 1) & std::$scalar::MAX;
                    }
                    accum
                });
                b.bytes = mem::size_of::<$ty>() as u64 * ::RAND_BENCH_N;
            }
        )+)
    }

    benches!();
}

mod double_div {
    use super::*;

    macro_rules! simd_uniform_bench {
        ($(($uniform:ident, $gen_range:ident, $rng:ident, $ty:ident, $uty:ident),)+, $scalar:ident, $u_scalar:ident, $low:expr, $high:expr) => ($(
            #[bench]
            fn $uniform(b: &mut Bencher) {
                let mut rng = $rng::from_rng(&mut thread_rng()).unwrap();

                let low = $ty::splat($low);
                let high = $ty::splat($high);

                let range: $uty = (high - low).cast();
                let mut divisor = (0 - range) / range + 1;
                // overflow, it's really 2**32
                let eq_zero = divisor.eq($uty::splat(0));
                divisor = eq_zero.select($uty::splat(1), divisor);

                b.iter(|| {
                    let mut accum = $ty::default();
                    for _ in 0..::RAND_BENCH_N {
                        let mut val = rng.gen::<$uty>() / divisor;
                        // reject val >= range
                        loop {
                            let mask = val.lt(range);
                            if mask.all() {
                                val = eq_zero.select($uty::splat(0), val);
                                break;
                            }
                            val = mask.select(val, rng.gen::<$uty>() / divisor);
                        }

                        let _val: $ty = val.cast();
                        accum += low + _val; // wrapping addition
                    }
                    accum
                });
                b.bytes = mem::size_of::<$ty>() as u64 * ::RAND_BENCH_N;
            }

            // construct and sample from a range
            #[bench]
            fn $gen_range(b: &mut Bencher) {
                let mut rng = $rng::from_rng(&mut thread_rng()).unwrap();
                let mut low = $ty::splat($low);
                for i in 0..$ty::lanes() {
                    low = low.replace(i, $low - i as $scalar);
                }

                b.iter(|| {
                    let mut high = $ty::splat($high);
                    let mut accum = $ty::splat(0);
                    for _ in 0..::RAND_BENCH_N {
                        let range: $uty = (high - low).cast();
                        let mut divisor = (0 - range) / range + 1;
                        // overflow, it's really 2^32
                        let eq_zero = divisor.eq($uty::splat(0));
                        divisor = eq_zero.select($uty::splat(1), divisor);

                        let mut val = rng.gen::<$uty>() / divisor;
                        // reject val >= range
                        loop {
                            let mask = val.lt(range);
                            if mask.all() {
                                val = eq_zero.select($uty::splat(0), val);
                                break;
                            }
                            val = mask.select(val, rng.gen::<$uty>() / divisor);
                        }

                        let _val: $ty = val.cast();
                        accum += low + _val; // wrapping addition

                        // force recalculation of range each time
                        high = (high + 1) & std::$scalar::MAX;
                    }
                    accum
                });
                b.bytes = mem::size_of::<$ty>() as u64 * ::RAND_BENCH_N;
            }
        )+)
    }

    benches!();
}

mod single_mod {
    use super::*;

    macro_rules! simd_uniform_bench {
        ($(($uniform:ident, $gen_range:ident, $rng:ident, $ty:ident, $uty:ident),)+, $scalar:ident, $u_scalar:ident, $low:expr, $high:expr) => ($(
            #[bench]
            fn $uniform(b: &mut Bencher) {
                let mut rng = $rng::from_rng(&mut thread_rng()).unwrap();

                let low = $ty::splat($low);
                let high = $ty::splat($high);
                let range: $uty = (high - low).cast();

                b.iter(|| {
                    let mut accum = $ty::default();
                    for _ in 0..::RAND_BENCH_N {
                        let mut x = rng.gen::<$uty>();
                        let mut r = x % range;
                        // reject x - r > range
                        loop {
                            let cmp = (x - r).le(0 - range);
                            if cmp.all() {
                                break;
                            }
                            x = cmp.select(x, rng.gen::<$uty>());
                            r = x % range;
                        }

                        let _r: $ty = r.cast();
                        accum += low + _r; // wrapping addition
                    }
                    accum
                });
                b.bytes = mem::size_of::<$ty>() as u64 * ::RAND_BENCH_N;
            }

            // construct and sample from a range
            #[bench]
            fn $gen_range(b: &mut Bencher) {
                let mut rng = $rng::from_rng(&mut thread_rng()).unwrap();
                let mut low = $ty::splat($low);
                for i in 0..$ty::lanes() {
                    low = low.replace(i, $low - i as $scalar);
                }

                b.iter(|| {
                    let mut high = $ty::splat($high);
                    let mut accum = $ty::splat(0);
                    for _ in 0..::RAND_BENCH_N {
                        let range: $uty = (high - low).cast();

                        let mut x = rng.gen::<$uty>();
                        let mut r = x % range;
                        // reject x - r > range
                        loop {
                            let cmp = (x - r).le(0 - range);
                            if cmp.all() {
                                break;
                            }
                            x = cmp.select(x, rng.gen::<$uty>());
                            r = x % range;
                        }

                        let _r: $ty = r.cast();
                        accum += low + _r; // wrapping addition

                        // force recalculation of range each time
                        high = (high + 1) & std::$scalar::MAX;
                    }
                    accum
                });
                b.bytes = mem::size_of::<$ty>() as u64 * ::RAND_BENCH_N;
            }
        )+)
    }

    benches!();
}

// If `range` is small enough, we can replace the `wmul` with an equivalent
// regular multiplication with only the lower half of the bits set, and
// bit-select the high and low components. This allows two samples per RNG
// output and better multiplication (with u32x4, a single `pmulld` "wmul").
//
// This is an idealized implementation which assumes the range is small enough
mod wide_mul_small {
    use super::*;

    macro_rules! simd_uniform_bench {
        ($(($uniform:ident, $gen_range:ident, $rng:ident, $ty:ident, $uty:ident),)+, $scalar:ident, $u_scalar:ident, $low:expr, $high:expr) => ($(
            #[bench]
            fn $uniform(b: &mut Bencher) {
                let mut rng = $rng::from_rng(&mut thread_rng()).unwrap();

                let low = $ty::splat($low);
                let high = $ty::splat($high);

                let unsigned_max = $u_scalar::max_value();
                let range: $uty = (high - low).cast();
                let ints_to_reject = (unsigned_max - range + 1) % range;
                let zone = unsigned_max - ints_to_reject;

                b.iter(|| {
                    let mut accum = $ty::default();
                    for _ in 0..::RAND_BENCH_N {
                        // each random sample only uses 16 random bits per
                        // lane, so we alternate between the high and low 16
                        // bits
                        let mut v_wide: $uty = rng.gen();
                        let mut v = v_wide & 0xffff; // low 16

                        loop {
                            let mul = v * range;
                            let (hi, lo) = (mul >> 16, mul & 0xffff);

                            let mask = lo.le(zone);
                            if mask.all() {
                                let hi: $ty = hi.cast();
                                accum += low + hi;
                                break;
                            }
                            v = mask.select(v, v_wide >> 16); // high 16

                            let mul = v * range;
                            let (hi, lo) = (mul >> 16, mul & 0xffff);

                            let mask = lo.le(zone);
                            if mask.all() {
                                let hi: $ty = hi.cast();
                                accum += low + hi;
                                break;
                            }

                            v_wide = rng.gen();
                            v = mask.select(v, v_wide & 0xffff); // low 16
                        }
                    }
                    accum
                });
                b.bytes = mem::size_of::<$ty>() as u64 * ::RAND_BENCH_N;
            }

            // construct and sample from a range
            #[bench]
            fn $gen_range(b: &mut Bencher) {
                let mut rng = $rng::from_rng(&mut thread_rng()).unwrap();
                let mut low = $ty::splat($low);
                for i in 0..$ty::lanes() {
                    low = low.replace(i, $low - i as $scalar);
                }

                b.iter(|| {
                    let mut high = $ty::splat($high);
                    let mut accum = $ty::splat(0);
                    for _ in 0..::RAND_BENCH_N {
                        let range: $uty = (high - low).cast();
                        let mut zone = $uty::default();
                        for i in 0..$uty::lanes() {
                            let x = range.extract(i);
                            zone = zone.replace(i, x << x.leading_zeros());
                        }

                        let mut v_wide: $uty = rng.gen();
                        let mut v = v_wide & 0xffff; // low 16

                        loop {
                            let mul = v * range;
                            let (hi, lo) = (mul >> 16, mul & 0xffff);

                            let mask = lo.le(zone);
                            if mask.all() {
                                let hi: $ty = hi.cast();
                                accum += low + hi;
                                break;
                            }
                            v = mask.select(v, v_wide >> 16); // high 16

                            let mul = v * range;
                            let (hi, lo) = (mul >> 16, mul & 0xffff);

                            let mask = lo.le(zone);
                            if mask.all() {
                                let hi: $ty = hi.cast();
                                accum += low + hi;
                                break;
                            }

                            v_wide = rng.gen();
                            v = mask.select(v, v_wide & 0xffff); // low 16
                        }

                        // force recalculation of range each time
                        high = (high + 1) & std::$scalar::MAX;
                    }
                    accum
                });
                b.bytes = mem::size_of::<$ty>() as u64 * ::RAND_BENCH_N;
            }
        )+)
    }

    simd_uniform_bench! {
        (simd_uniform_i32x2, simd_uniform_single_i32x2, SfcAlt64x2k, i32x2, u32x2),
        (simd_uniform_i32x4, simd_uniform_single_i32x4, SfcAlt64x4k, i32x4, u32x4),
        (simd_uniform_i32x8, simd_uniform_single_i32x8, SfcAlt64x4k, i32x8, u32x8),
        (simd_uniform_i32x16, simd_uniform_single_i32x16, SfcAlt64x4k, i32x16, u32x16),,
        // i32, u32, -200_000_000, 800_000_000
        i32, u32, -500, 2000
    }
}
