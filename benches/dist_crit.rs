use criterion::*;
use criterion::{criterion_group, criterion_main, Criterion};
use criterion_cycles_per_byte::CyclesPerByte;

use rand::distributions::uniform::{UniformInt, UniformSampler};
use rand::prelude::*;

// At this time, distributions are optimised for 64-bit platforms.
use rand_pcg::Pcg64Mcg;

const RAND_BENCH_N: u64 = 1000;

fn bench_funcs(c: &mut Criterion<CyclesPerByte>) {
    let mut group = c.benchmark_group("gen_range");
    group.throughput(Throughput::Elements(RAND_BENCH_N));

    let mut rng = Pcg64Mcg::from_entropy();

    macro_rules! gen_range_int {
        ($name:expr, $ty:ident, $low:expr, $high:expr, $func:ident) => {
            group.bench_function($name, |b| {
                b.iter(|| {
                    let mut high: $ty = $high;
                    let mut accum: $ty = 0;
                    for _ in 0..RAND_BENCH_N {
                        accum =
                            accum.wrapping_add(UniformInt::<$ty>::$func($low, high - 1, &mut rng));
                        // force recalculation of range each time
                        high = high.wrapping_add(1) & std::$ty::MAX;
                    }
                    accum
                })
            });
        };
    }

    for (func, name) in [
        (sample_single_inclusive, ""),
        (sample_single_inclusive_old, "_old"),
    ] {
        gen_range_int!("i8_low".to_owned() + name, i8, -1i8, 0, func);
        gen_range_int!("i8_low".to_owned() + name, i8, -1i8, 0, func);
        gen_range_int!("i16_low".to_owned() + name, i16, -1i16, 0, func);
        gen_range_int!("i32_low".to_owned() + name, i32, -1i32, 0, func);
        gen_range_int!("i64_low".to_owned() + name, i64, -1i64, 0, func);
        gen_range_int!("i128_low".to_owned() + name, i128, -1i128, 0, func);

        // These were the initially tested ranges. They are likely to see fewer
        // rejections than the low tests. 2^(N - 1) + 1
        gen_range_int!("i8_high".to_owned() + name, i8, i8::min_value(), 1, func);
        gen_range_int!("i16_high".to_owned() + name, i16, i16::min_value(), 1, func);
        gen_range_int!("i32_high".to_owned() + name, i32, i32::min_value(), 1, func);
        gen_range_int!("i64_high".to_owned() + name, i64, i64::min_value(), 1, func);
        gen_range_int!("i128_high".to_owned() + name, i128, i128::min_value(), 1, func);
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().with_measurement(CyclesPerByte);
    targets = bench_funcs
);
criterion_main!(benches);
