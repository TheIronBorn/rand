// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// https://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A distribution uniformly sampling numbers within a given range.
//!
//! [`Uniform`] is the standard distribution to sample uniformly from a range;
//! e.g. `Uniform::new_inclusive(1, 6)` can sample integers from 1 to 6, like a
//! standard die. [`Rng::gen_range`] supports any type supported by
//! [`Uniform`].
//!
//! This distribution is provided with support for several primitive types
//! (all integer and floating-point types) as well as `std::time::Duration`,
//! and supports extension to user-defined types via a type-specific *back-end*
//! implementation.
//!
//! The types [`UniformInt`], [`UniformFloat`] and [`UniformDuration`] are the
//! back-ends supporting sampling from primitive integer and floating-point
//! ranges as well as from `std::time::Duration`; these types do not normally
//! need to be used directly (unless implementing a derived back-end).
//!
//! # Example usage
//!
//! ```
//! use rand::{Rng, thread_rng};
//! use rand::distributions::Uniform;
//!
//! let mut rng = thread_rng();
//! let side = Uniform::new(-10.0, 10.0);
//!
//! // sample between 1 and 10 points
//! for _ in 0..rng.gen_range(1, 11) {
//!     // sample a point from the square with sides -10 - 10 in two dimensions
//!     let (x, y) = (rng.sample(side), rng.sample(side));
//!     println!("Point: {}, {}", x, y);
//! }
//! ```
//!
//! # Extending `Uniform` to support a custom type
//!
//! To extend [`Uniform`] to support your own types, write a back-end which
//! implements the [`UniformSampler`] trait, then implement the [`SampleUniform`]
//! helper trait to "register" your back-end. See the `MyF32` example below.
//!
//! At a minimum, the back-end needs to store any parameters needed for sampling
//! (e.g. the target range) and implement `new`, `new_inclusive` and `sample`.
//! Those methods should include an assert to check the range is valid (i.e.
//! `low < high`). The example below merely wraps another back-end.
//!
//! ```
//! use rand::prelude::*;
//! use rand::distributions::uniform::{Uniform, SampleUniform,
//!         UniformSampler, UniformFloat};
//!
//! struct MyF32(f32);
//!
//! #[derive(Clone, Copy, Debug)]
//! struct UniformMyF32 {
//!     inner: UniformFloat<f32>,
//! }
//!
//! impl UniformSampler for UniformMyF32 {
//!     type X = MyF32;
//!     fn new(low: Self::X, high: Self::X) -> Self {
//!         UniformMyF32 {
//!             inner: UniformFloat::<f32>::new(low.0, high.0),
//!         }
//!     }
//!     fn new_inclusive(low: Self::X, high: Self::X) -> Self {
//!         UniformSampler::new(low, high)
//!     }
//!     fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
//!         MyF32(self.inner.sample(rng))
//!     }
//! }
//!
//! impl SampleUniform for MyF32 {
//!     type Sampler = UniformMyF32;
//! }
//!
//! let (low, high) = (MyF32(17.0f32), MyF32(22.0f32));
//! let uniform = Uniform::new(low, high);
//! let x = uniform.sample(&mut thread_rng());
//! ```
//!
//! [`Uniform`]: struct.Uniform.html
//! [`Rng::gen_range`]: ../../trait.Rng.html#method.gen_range
//! [`SampleUniform`]: trait.SampleUniform.html
//! [`UniformSampler`]: trait.UniformSampler.html
//! [`UniformInt`]: struct.UniformInt.html
//! [`UniformFloat`]: struct.UniformFloat.html
//! [`UniformDuration`]: struct.UniformDuration.html

#[cfg(feature = "std")]
use std::time::Duration;
#[cfg(feature = "simd_support")]
use packed_simd::*;
#[cfg(feature = "simd_support")]
use core::arch::x86_64::*;
#[cfg(feature = "simd_support")]
use std::mem;

use Rng;
use distributions::Distribution;
#[cfg(feature = "simd_support")]
use distributions::{SimdRejectionSampling, Standard};
#[cfg(feature = "simd_support")]
use distributions::box_muller::LeadingZeros;
use distributions::float::IntoFloat;

/// Sample values uniformly between two bounds.
///
/// [`Uniform::new`] and [`Uniform::new_inclusive`] construct a uniform
/// distribution sampling from the given range; these functions may do extra
/// work up front to make sampling of multiple values faster.
///
/// When sampling from a constant range, many calculations can happen at
/// compile-time and all methods should be fast; for floating-point ranges and
/// the full range of integer types this should have comparable performance to
/// the `Standard` distribution.
///
/// Steps are taken to avoid bias which might be present in naive
/// implementations; for example `rng.gen::<u8>() % 170` samples from the range
/// `[0, 169]` but is twice as likely to select numbers less than 85 than other
/// values. Further, the implementations here give more weight to the high-bits
/// generated by the RNG than the low bits, since with some RNGs the low-bits
/// are of lower quality than the high bits.
///
/// Implementations should attempt to sample in `[low, high)` for
/// `Uniform::new(low, high)`, i.e., excluding `high`, but this may be very
/// difficult. All the primitive integer types satisfy this property, and the
/// float types normally satisfy it, but rounding may mean `high` can occur.
///
/// # Example
///
/// ```
/// use rand::distributions::{Distribution, Uniform};
///
/// fn main() {
///     let between = Uniform::from(10..10000);
///     let mut rng = rand::thread_rng();
///     let mut sum = 0;
///     for _ in 0..1000 {
///         sum += between.sample(&mut rng);
///     }
///     println!("{}", sum);
/// }
/// ```
///
/// [`Uniform::new`]: struct.Uniform.html#method.new
/// [`Uniform::new_inclusive`]: struct.Uniform.html#method.new_inclusive
/// [`new`]: struct.Uniform.html#method.new
/// [`new_inclusive`]: struct.Uniform.html#method.new_inclusive
#[derive(Clone, Copy, Debug)]
pub struct Uniform<X: SampleUniform> {
    inner: X::Sampler,
}

impl<X: SampleUniform> Uniform<X> {
    /// Create a new `Uniform` instance which samples uniformly from the half
    /// open range `[low, high)` (excluding `high`). Panics if `low >= high`.
    pub fn new(low: X, high: X) -> Uniform<X> {
        Uniform { inner: X::Sampler::new(low, high) }
    }

    /// Create a new `Uniform` instance which samples uniformly from the closed
    /// range `[low, high]` (inclusive). Panics if `low > high`.
    pub fn new_inclusive(low: X, high: X) -> Uniform<X> {
        Uniform { inner: X::Sampler::new_inclusive(low, high) }
    }
}

impl<X: SampleUniform> Distribution<X> for Uniform<X> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> X {
        self.inner.sample(rng)
    }
}

/// Helper trait for creating objects using the correct implementation of
/// [`UniformSampler`] for the sampling type.
///
/// See the [module documentation] on how to implement [`Uniform`] range
/// sampling for a custom type.
///
/// [`UniformSampler`]: trait.UniformSampler.html
/// [module documentation]: index.html
/// [`Uniform`]: struct.Uniform.html
pub trait SampleUniform: Sized {
    /// The `UniformSampler` implementation supporting type `X`.
    type Sampler: UniformSampler<X = Self>;
}

/// Helper trait handling actual uniform sampling.
///
/// See the [module documentation] on how to implement [`Uniform`] range
/// sampling for a custom type.
///
/// Implementation of [`sample_single`] is optional, and is only useful when
/// the implementation can be faster than `Self::new(low, high).sample(rng)`.
///
/// [module documentation]: index.html
/// [`Uniform`]: struct.Uniform.html
/// [`sample_single`]: trait.UniformSampler.html#method.sample_single
pub trait UniformSampler: Sized {
    /// The type sampled by this implementation.
    type X;

    /// Construct self, with inclusive lower bound and exclusive upper bound
    /// `[low, high)`.
    ///
    /// Usually users should not call this directly but instead use
    /// `Uniform::new`, which asserts that `low < high` before calling this.
    fn new(low: Self::X, high: Self::X) -> Self;

    /// Construct self, with inclusive bounds `[low, high]`.
    ///
    /// Usually users should not call this directly but instead use
    /// `Uniform::new_inclusive`, which asserts that `low <= high` before
    /// calling this.
    fn new_inclusive(low: Self::X, high: Self::X) -> Self;

    /// Sample a value.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X;

    /// Sample a single value uniformly from a range with inclusive lower bound
    /// and exclusive upper bound `[low, high)`.
    ///
    /// Usually users should not call this directly but instead use
    /// `Uniform::sample_single`, which asserts that `low < high` before calling
    /// this.
    ///
    /// Via this method, implementations can provide a method optimized for
    /// sampling only a single value from the specified range. The default
    /// implementation simply calls `UniformSampler::new` then `sample` on the
    /// result.
    fn sample_single<R: Rng + ?Sized>(low: Self::X, high: Self::X, rng: &mut R)
        -> Self::X
    {
        let uniform: Self = UniformSampler::new(low, high);
        uniform.sample(rng)
    }

    /// Allows optimizations for the `low = 0` case.
    fn sample_single_below<R: Rng + ?Sized>(_high: Self::X, _rng: &mut R)
        -> Self::X
    {
        unimplemented!()
    }
}

impl<X: SampleUniform> From<::core::ops::Range<X>> for Uniform<X> {
    fn from(r: ::core::ops::Range<X>) -> Uniform<X> {
        Uniform::new(r.start, r.end)
    }
}

////////////////////////////////////////////////////////////////////////////////

// What follows are all back-ends.


/// The back-end implementing [`UniformSampler`] for integer types.
///
/// Unless you are implementing [`UniformSampler`] for your own type, this type
/// should not be used directly, use [`Uniform`] instead.
///
/// # Implementation notes
///
/// For a closed range, the number of possible numbers we should generate is
/// `range = (high - low + 1)`. It is not possible to end up with a uniform
/// distribution if we map *all* the random integers that can be generated to
/// this range. We have to map integers from a `zone` that is a multiple of the
/// range. The rest of the integers, that cause a bias, are rejected.
///
/// The problem with `range` is that to cover the full range of the type, it has
/// to store `unsigned_max + 1`, which can't be represented. But if the range
/// covers the full range of the type, no modulus is needed. A range of size 0
/// can't exist, so we use that to represent this special case. Wrapping
/// arithmetic even makes representing `unsigned_max + 1` as 0 simple.
///
/// We don't calculate `zone` directly, but first calculate the number of
/// integers to reject. To handle `unsigned_max + 1` not fitting in the type,
/// we use:
/// `ints_to_reject = (unsigned_max + 1) % range;`
/// `ints_to_reject = (unsigned_max - range + 1) % range;`
///
/// The smallest integer PRNGs generate is `u32`. That is why for small integer
/// sizes (`i8`/`u8` and `i16`/`u16`) there is an optimization: don't pick the
/// largest zone that can fit in the small type, but pick the largest zone that
/// can fit in an `u32`. `ints_to_reject` is always less than half the size of
/// the small integer. This means the first bit of `zone` is always 1, and so
/// are all the other preceding bits of a larger integer. The easiest way to
/// grow the `zone` for the larger type is to simply sign extend it.
///
/// An alternative to using a modulus is widening multiply: After a widening
/// multiply by `range`, the result is in the high word. Then comparing the low
/// word against `zone` makes sure our distribution is uniform.
///
/// [`UniformSampler`]: trait.UniformSampler.html
/// [`Uniform`]: struct.Uniform.html
#[derive(Clone, Copy, Debug)]
pub struct UniformInt<X> {
    low: X,
    pub range: X,
    zone: X,
}

macro_rules! uniform_int_impl {
    ($ty:ty, $signed:ty, $unsigned:ident,
     $i_large:ident, $u_large:ident) => {
        impl SampleUniform for $ty {
            type Sampler = UniformInt<$ty>;
        }

        impl UniformSampler for UniformInt<$ty> {
            // We play free and fast with unsigned vs signed here
            // (when $ty is signed), but that's fine, since the
            // contract of this macro is for $ty and $unsigned to be
            // "bit-equal", so casting between them is a no-op.

            type X = $ty;

            #[inline] // if the range is constant, this helps LLVM to do the
                      // calculations at compile-time.
            fn new(low: Self::X, high: Self::X) -> Self {
                assert!(low < high, "Uniform::new called with `low >= high`");
                UniformSampler::new_inclusive(low, high - 1)
            }

            #[inline] // if the range is constant, this helps LLVM to do the
                      // calculations at compile-time.
            fn new_inclusive(low: Self::X, high: Self::X) -> Self {
                assert!(low <= high,
                        "Uniform::new_inclusive called with `low > high`");
                let unsigned_max = ::core::$unsigned::MAX;

                let range = high.wrapping_sub(low).wrapping_add(1) as $unsigned;
                let ints_to_reject =
                    if range > 0 {
                        (unsigned_max - range + 1) % range
                    } else {
                        0
                    };
                let zone = unsigned_max - ints_to_reject;

                UniformInt {
                    low: low,
                    // These are really $unsigned values, but store as $ty:
                    range: range as $ty,
                    zone: zone as $ty
                }
            }

            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
                let range = self.range as $unsigned as $u_large;
                if range > 0 {
                    // Grow `zone` to fit a type of at least 32 bits, by
                    // sign-extending it (the first bit is always 1, so are all
                    // the preceding bits of the larger type).
                    // For types that already have the right size, all the
                    // casting is a no-op.
                    let zone = self.zone as $signed as $i_large as $u_large;
                    loop {
                        let v: $u_large = rng.gen();
                        let (hi, lo) = v.wmul(range);
                        if lo <= zone {
                            return self.low.wrapping_add(hi as $ty);
                        }
                    }
                } else {
                    // Sample from the entire integer range.
                    rng.gen()
                }
            }

            fn sample_single<R: Rng + ?Sized>(low: Self::X,
                                              high: Self::X,
                                              rng: &mut R) -> Self::X
            {
                assert!(low < high,
                        "Uniform::sample_single called with low >= high");
                let range = high.wrapping_sub(low) as $unsigned as $u_large;
                let zone =
                    if ::core::$unsigned::MAX <= ::core::u16::MAX as $unsigned {
                        // Using a modulus is faster than the approximation for
                        // i8 and i16. I suppose we trade the cost of one
                        // modulus for near-perfect branch prediction.
                        let unsigned_max: $u_large = ::core::$u_large::MAX;
                        let ints_to_reject = (unsigned_max - range + 1) % range;
                        unsigned_max - ints_to_reject
                    } else {
                        // conservative but fast approximation
                       range << range.leading_zeros()
                    };

                loop {
                    let v: $u_large = rng.gen();
                    let (hi, lo) = v.wmul(range);
                    if lo <= zone {
                        return low.wrapping_add(hi as $ty);
                    }
                }
            }
        }
    }
}

uniform_int_impl! { i8, i8, u8, i32, u32 }
uniform_int_impl! { i16, i16, u16, i32, u32 }
uniform_int_impl! { i32, i32, u32, i32, u32 }
uniform_int_impl! { i64, i64, u64, i64, u64 }
#[cfg(feature = "i128_support")]
uniform_int_impl! { i128, i128, u128, u128, u128 }
uniform_int_impl! { isize, isize, usize, isize, usize }
uniform_int_impl! { u8, i8, u8, i32, u32 }
uniform_int_impl! { u16, i16, u16, i32, u32 }
uniform_int_impl! { u32, i32, u32, i32, u32 }
uniform_int_impl! { u64, i64, u64, i64, u64 }
uniform_int_impl! { usize, isize, usize, isize, usize }
#[cfg(feature = "i128_support")]
uniform_int_impl! { u128, u128, u128, i128, u128 }

#[cfg(feature = "simd_support")]
macro_rules! uniform_simd_int_impl {
    ($ty:ident, $unsigned:ident, $signed:ident, $u_scalar:ident) => {
        // There is little point in `u_large/i_large` here because SIMD
        // PRNGs can natively generate vectors as small as u8x16/etc.
        // Generating a u64x8 then casting to u8x8 would be very slow
        // for a PRNG which generates less than 512 bits. Generating
        // then casting a u64x2 to a u8x2 would be an optimization the
        // user must make.
        // Unless we decide on a minimum SIMD PRNG output size.

        // TODO: look into `Uniform::<u32x4>::new(0u32, 100)` functionality
        //        perhaps `impl SampleUniform for $u_scalar`?
        impl SampleUniform for $ty {
            type Sampler = UniformInt<$ty>;
        }

        impl UniformSampler for UniformInt<$ty> {
            type X = $ty;

            #[inline] // if the range is constant, this helps LLVM to do the
                      // calculations at compile-time.
                      // TODO: test this ^ for simd
            fn new(low: Self::X, high: Self::X) -> Self {
                assert!(low.lt(high).all(), "Uniform::new called with `low >= high`");
                UniformSampler::new_inclusive(low, high - 1)
            }

            #[inline] // if the range is constant, this helps LLVM to do the
                      // calculations at compile-time.
            fn new_inclusive(low: Self::X, high: Self::X) -> Self {
                assert!(low.le(high).all(),
                        "Uniform::new_inclusive called with `low > high`");
                let unsigned_max = ::core::$u_scalar::MAX;

                let range: $unsigned = ((high - low) + 1).cast();
                // the `range > 0` case is complicated. Unimplemented for now
                let ints_to_reject = (unsigned_max - range + 1) % range;
                let zone = unsigned_max - ints_to_reject;

                UniformInt {
                    low: low,
                    // These are really $unsigned values, but store as $ty:
                    range: range.cast(),
                    zone: zone.cast(),
                }
            }

            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
                let range: $unsigned = self.range.cast();
                // TODO: ensure these casts are a no-op
                let zone: $unsigned = self.zone.cast();

                // This might seem very slow, generating a whole new
                // SIMD vector for every sample rejection. However, for
                // most uses the chance of rejection is small. With many
                // lanes that chance is increased. To mitigate this, we
                // can replace only the lanes of the vector which fail,
                // reducing the chance of rejection for each iteration.
                // The replacement method does however add a little
                // overhead. Benchmarking or calculating probabilities
                // might reveal contexts where replacing is slower.
                /*let mut v: $unsigned = rng.gen();
                loop {
                    let (hi, lo) = v.wmul(range);
                    let mask = lo.le(zone);
                    if mask.all() {
                        return self.low + $ty::from(hi);
                    }
                    // Replace only the failing lanes
                    v = mask.select(v, rng.gen());
                }*/

                // strangely `SimdRejectionSampling` seems to be faster
                // than the above.
                let cmp = |v: $unsigned| {
                    let (_, lo) = v.wmul(range);
                    lo.le(zone)
                };

                let v = SimdRejectionSampling::sample(rng, &Standard, cmp);
                let (hi, _) = v.wmul(range);
                let hi: $ty = hi.cast();
                self.low + hi
            }

            #[inline(always)]
            fn sample_single<R: Rng + ?Sized>(low: Self::X,
                                              high: Self::X,
                                              rng: &mut R) -> Self::X
            {
                assert!(low.lt(high).all(),
                        "Uniform::sample_single called with low >= high");
                let range: $unsigned = (high - low).cast();
                let zone =
                    if mem::size_of::<$u_scalar>() <= 2 {
                        let unsigned_max = ::core::$u_scalar::MAX;
                        let ints_to_reject = (unsigned_max - range + 1) % range;
                        unsigned_max - ints_to_reject
                    } else {
                        // Benchmarks indicate that, unlike the scalar implementation,
                        // for i8 and i16 the approximation is faster. Likely
                        // because modulus is especially slow for SIMD.
                        range << range.leading_zeros()
                    };

                /*let cmp = |v: $unsigned| {
                    let (_, lo) = v.wmul(range);
                    lo.le(zone)
                };

                let v = SimdRejectionSampling::sample(rng, &Standard, cmp);
                let (hi, _) = v.wmul(range);
                low + $ty::from(hi)*/

                let mut v: $unsigned = rng.gen();
                loop {
                    let (hi, lo) = v.wmul(range);
                    let mask = lo.le(zone);
                    if mask.all() {
                        let hi: $ty = hi.cast();
                        return low + hi;
                    }
                    // Replace only the failing lanes
                    v = mask.select(v, rng.gen());
                }
            }

            fn sample_single_below<R: Rng + ?Sized>(high: Self::X,
                                                    rng: &mut R) -> Self::X
            {
                assert!($ty::splat(0).lt(high).all(),
                        "Uniform::sample_single_below called with 0 >= high");
                let range: $unsigned = high.cast();
                let zone =
                    if mem::size_of::<$u_scalar>() <= 2 {
                        let unsigned_max = ::core::$u_scalar::MAX;
                        let ints_to_reject = (unsigned_max - range + 1) % range;
                        unsigned_max - ints_to_reject
                    } else {
                        // Benchmarks indicate that, unlike the scalar implementation,
                        // for i8 and i16 the approximation is faster. Likely
                        // because modulus is especially slow for SIMD.
                        range << range.leading_zeros()
                    };

                let cmp = |v: $unsigned| {
                    let (_, lo) = v.wmul(range);
                    lo.le(zone)
                };

                let v = SimdRejectionSampling::sample(rng, &Standard, cmp);
                let (hi, _) = v.wmul(range);
                hi.cast()
            }
        }
    };

    // bulk implementation
    ($(($unsigned:ident, $signed:ident),)+ $u_scalar:ident) => {
        $(
            uniform_simd_int_impl!($unsigned, $unsigned, $signed, $u_scalar);
            uniform_simd_int_impl!($signed, $unsigned, $signed, $u_scalar);
        )+
    };
}

#[cfg(feature = "simd_support")]
uniform_simd_int_impl! {
    (u64x2, i64x2),
    (u64x4, i64x4),
    (u64x8, i64x8),
    u64
}

#[cfg(feature = "simd_support")]
uniform_simd_int_impl! {
    (u32x2, i32x2),
    (u32x4, i32x4),
    (u32x8, i32x8),
    (u32x16, i32x16),
    u32
}

#[cfg(feature = "simd_support")]
uniform_simd_int_impl! {
    (u16x2, i16x2),
    (u16x4, i16x4),
    (u16x8, i16x8),
    (u16x16, i16x16),
    (u16x32, i16x32),
    u16
}

#[cfg(feature = "simd_support")]
uniform_simd_int_impl! {
    (u8x2, i8x2),
    (u8x4, i8x4),
    (u8x8, i8x8),
    (u8x16, i8x16),
    (u8x32, i8x32),
    (u8x64, i8x64),
    u8
}


///
pub trait WideningMultiply<RHS = Self> {
    ///
    type Output;

    ///
    fn wmul(self, x: RHS) -> Self::Output;
}

macro_rules! wmul_impl {
    ($ty:ident, $wide:ident, $shift:expr) => {
        impl WideningMultiply for $ty {
            type Output = ($ty, $ty);

            #[inline(always)]
            fn wmul(self, x: $ty) -> Self::Output {
                let tmp = (self as $wide) * (x as $wide);
                ((tmp >> $shift) as $ty, tmp as $ty)
            }
        }
    };

    // simd bulk implementation
    ($(($ty:ident, $wide:ident),)+, $shift:expr) => {
        $(
            impl WideningMultiply for $ty {
                type Output = ($ty, $ty);

                #[inline(always)]
                fn wmul(self, x: $ty) -> Self::Output {
                    // TODO: optimize
                    //       u16xN vectors could use mulhi/mullo
                    //       perhaps better mul/shuf/blend operations for general
                    let tmp = $wide::from(self) * $wide::from(x);
                    let hi: $ty = (tmp >> $shift).cast();
                    let lo: $ty = tmp.cast();
                    (hi, lo)
                }
            }
        )+
    };
}
wmul_impl! { u8, u16, 8 }
wmul_impl! { u16, u32, 16 }
wmul_impl! { u32, u64, 32 }
#[cfg(feature = "i128_support")]
wmul_impl! { u64, u128, 64 }

#[cfg(feature = "simd_support")]
wmul_impl! {
    (u8x2, u16x2),
    (u8x4, u16x4),
    (u8x8, u16x8),
    (u8x16, u16x16),
    (u8x32, u16x32),,
    8
}
#[cfg(feature = "simd_support")]
wmul_impl! {
    (u16x2, u32x2),
    (u16x4, u32x4),
    // (u16x8, u32x8),
    (u16x16, u32x16),,
    16
}
#[cfg(feature = "simd_support")]
wmul_impl! {
    (u32x2, u64x2),
    (u32x4, u64x4),
    (u32x8, u64x8),,
    32
}

#[cfg(feature = "simd_support")]
impl WideningMultiply for u16x8 {
    type Output = (u16x8, u16x8);

    #[inline(always)]
    fn wmul(self, x: u16x8) -> Self::Output {
        let a = __m128i::from_bits(self);
        let b = __m128i::from_bits(x);
        let hi = u16x8::from_bits(unsafe { _mm_mulhi_epu16(a, b) });
        let lo = u16x8::from_bits(unsafe { _mm_mullo_epi32(a, b) });
        (hi, lo)
    }
}

// This code is a translation of the __mulddi3 function in LLVM's
// compiler-rt. It is an optimised variant of the common method
// `(a + b) * (c + d) = ac + ad + bc + bd`.
//
// For some reason LLVM can optimise the C version very well, but
// keeps shuffling registers in this Rust translation.
macro_rules! wmul_impl_large {
    ($ty:ty, $half:expr) => {
        impl WideningMultiply for $ty {
            type Output = ($ty, $ty);

            #[inline(always)]
            fn wmul(self, b: $ty) -> Self::Output {
                const LOWER_MASK: $ty = !0 >> $half;
                let mut low = (self & LOWER_MASK).wrapping_mul(b & LOWER_MASK);
                let mut t = low >> $half;
                low &= LOWER_MASK;
                t += (self >> $half).wrapping_mul(b & LOWER_MASK);
                low += (t & LOWER_MASK) << $half;
                let mut high = t >> $half;
                t = low >> $half;
                low &= LOWER_MASK;
                t += (b >> $half).wrapping_mul(self & LOWER_MASK);
                low += (t & LOWER_MASK) << $half;
                high += t >> $half;
                high += (self >> $half).wrapping_mul(b >> $half);

                (high, low)
            }
        }
    };

    // simd bulk implementation
    (($($ty:ty,)+) $scalar:ty, $half:expr) => {
        $(
            impl WideningMultiply for $ty {
                type Output = ($ty, $ty);

                #[inline(always)]
                fn wmul(self, b: $ty) -> Self::Output {
                    const LOWER_MASK: $scalar = !0 >> $half;
                    let mut low = (self & LOWER_MASK) * (b & LOWER_MASK);
                    let mut t = low >> $half;
                    low &= LOWER_MASK;
                    t += (self >> $half) * (b & LOWER_MASK);
                    low += (t & LOWER_MASK) << $half;
                    let mut high = t >> $half;
                    t = low >> $half;
                    low &= LOWER_MASK;
                    t += (b >> $half) * (self & LOWER_MASK);
                    low += (t & LOWER_MASK) << $half;
                    high += t >> $half;
                    high += (self >> $half) * (b >> $half);

                    (high, low)
                }
            }
        )+
    };
}
#[cfg(not(feature = "i128_support"))]
wmul_impl_large! { u64, 32 }
#[cfg(feature = "i128_support")]
wmul_impl_large! { u128, 64 }

// TODO: optimize, this seems to seriously slow things down
#[cfg(feature = "simd_support")]
wmul_impl_large! { (u64x2, u64x4, u64x8,) u64, 32 }
#[cfg(feature = "simd_support")]
wmul_impl_large! { (u8x64,) u8, 4 }
#[cfg(feature = "simd_support")]
wmul_impl_large! { (u16x32,) u16, 8 }
#[cfg(feature = "simd_support")]
wmul_impl_large! { (u32x16,) u32, 16 }

macro_rules! wmul_impl_usize {
    ($ty:ty) => {
        impl WideningMultiply for usize {
            type Output = (usize, usize);

            #[inline(always)]
            fn wmul(self, x: usize) -> Self::Output {
                let (high, low) = (self as $ty).wmul(x as $ty);
                (high as usize, low as usize)
            }
        }
    }
}
#[cfg(target_pointer_width = "32")]
wmul_impl_usize! { u32 }
#[cfg(target_pointer_width = "64")]
wmul_impl_usize! { u64 }



/// The back-end implementing [`UniformSampler`] for floating-point types.
///
/// Unless you are implementing [`UniformSampler`] for your own type, this type
/// should not be used directly, use [`Uniform`] instead.
///
/// # Implementation notes
///
/// Instead of generating a float in the `[0, 1)` range using [`Standard`], the
/// `UniformFloat` implementation converts the output of an PRNG itself. This
/// way one or two steps can be optimized out.
///
/// The floats are first converted to a value in the `[1, 2)` interval using a
/// transmute-based method, and then mapped to the expected range with a
/// multiply and addition. Values produced this way have what equals 22 bits of
/// random digits for an `f32`, and 52 for an `f64`.
///
/// Currently there is no difference between [`new`] and [`new_inclusive`],
/// because the boundaries of a floats range are a bit of a fuzzy concept due to
/// rounding errors.
///
/// [`UniformSampler`]: trait.UniformSampler.html
/// [`new`]: trait.UniformSampler.html#tymethod.new
/// [`new_inclusive`]: trait.UniformSampler.html#tymethod.new_inclusive
/// [`Uniform`]: struct.Uniform.html
/// [`Standard`]: ../struct.Standard.html
#[derive(Clone, Copy, Debug)]
pub struct UniformFloat<X> {
    scale: X,
    offset: X,
}

macro_rules! uniform_float_impl {
    ($ty:ty, $bits_to_discard:expr, $uty:ty) => {
        impl SampleUniform for $ty {
            type Sampler = UniformFloat<$ty>;
        }

        impl UniformSampler for UniformFloat<$ty> {
            type X = $ty;

            fn new(low: Self::X, high: Self::X) -> Self {
                assert!(low < high, "Uniform::new called with `low >= high`");
                let scale = high - low;
                let offset = low - scale;
                UniformFloat {
                    scale: scale,
                    offset: offset,
                }
            }

            fn new_inclusive(low: Self::X, high: Self::X) -> Self {
                assert!(low <= high,
                        "Uniform::new_inclusive called with `low > high`");
                let scale = high - low;
                let offset = low - scale;
                UniformFloat {
                    scale: scale,
                    offset: offset,
                }
            }

            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
                // Generate a value in the range [1, 2)
                let value1_2 = (rng.gen::<$uty>() >> $bits_to_discard)
                               .into_float_with_exponent(0);
                // We don't use `f64::mul_add`, because it is not available with
                // `no_std`. Furthermore, it is slower for some targets (but
                // faster for others). However, the order of multiplication and
                // addition is important, because on some platforms (e.g. ARM)
                // it will be optimized to a single (non-FMA) instruction.
                value1_2 * self.scale + self.offset
            }

            // explicit `sample_single` provided here to let users control
            // inputs more exactly than `UniformSampler::sample_single`.
            fn sample_single<R: Rng + ?Sized>(low: Self::X,
                                              high: Self::X,
                                              rng: &mut R) -> Self::X {
                assert!(low < high,
                        "Uniform::sample_single called with low >= high");
                let scale = high - low;
                let offset = low - scale;
                // Generate a value in the range [1, 2)
                let value1_2 = (rng.gen::<$uty>() >> $bits_to_discard)
                               .into_float_with_exponent(0);
                // Doing multiply before addition allows some architectures to
                // use a single instruction.
                value1_2 * scale + offset
            }
        }
    };

    // bulk simd implementations
    ($(($ty:ty, $uty:ident),)+, $bits_to_discard:expr) => {$(
        impl SampleUniform for $ty {
            type Sampler = UniformFloat<$ty>;
        }

        impl UniformSampler for UniformFloat<$ty> {
            type X = $ty;

            fn new(low: Self::X, high: Self::X) -> Self {
                // We must use explicit `lt/le` because SIMD comparison
                // operators use lexicographic ordering.
                assert!(low.lt(high).all(), "Uniform::new called with `low >= high`");
                let scale = high - low;
                let offset = low - scale;
                UniformFloat {
                    scale: scale,
                    offset: offset,
                }
            }

            fn new_inclusive(low: Self::X, high: Self::X) -> Self {
                assert!(low.le(high).all(),
                        "Uniform::new_inclusive called with `low > high`");
                let scale = high - low;
                let offset = low - scale;
                UniformFloat {
                    scale: scale,
                    offset: offset,
                }
            }

            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
                let value1_2 = (rng.gen::<$uty>() >> $uty::splat($bits_to_discard))
                               .into_float_with_exponent(0);
                // TODO: look into SIMD FMA
                value1_2 * self.scale + self.offset
                // value1_2.fma(self.scale, self.offset)
            }

            fn sample_single<R: Rng + ?Sized>(low: Self::X,
                                              high: Self::X,
                                              rng: &mut R) -> Self::X {
                assert!(low.lt(high).all(),
                        "Uniform::sample_single called with low >= high");
                let scale = high - low;
                let offset = low - scale;
                let value1_2 = (rng.gen::<$uty>() >> $uty::splat($bits_to_discard))
                               .into_float_with_exponent(0);
                // conditional compilation with SIMD FMA would be good here
                // target_feature = "fma"
                // value1_2 * scale + offset
                value1_2.fma(scale, offset)
            }
        }
    )+};
}

uniform_float_impl! { f32, 32 - 23, u32 }
uniform_float_impl! { f64, 64 - 52, u64 }

#[cfg(feature = "simd_support")]
uniform_float_impl! {
    (f64x2, u64x2),
    (f64x4, u64x4),
    (f64x8, u64x8),,
    64 - 52
}
#[cfg(feature = "simd_support")]
uniform_float_impl! {
    (f32x2, u32x2),
    (f32x4, u32x4),
    (f32x8, u32x8),
    (f32x16, u32x16),,
    32 - 23
}



/// The back-end implementing [`UniformSampler`] for `Duration`.
///
/// Unless you are implementing [`UniformSampler`] for your own types, this type
/// should not be used directly, use [`Uniform`] instead.
///
/// [`UniformSampler`]: trait.UniformSampler.html
/// [`Uniform`]: struct.Uniform.html
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug)]
pub struct UniformDuration {
    offset: Duration,
    mode: UniformDurationMode,
}

#[cfg(feature = "std")]
#[derive(Debug, Copy, Clone)]
enum UniformDurationMode {
    Small {
        nanos: Uniform<u64>,
    },
    Large {
        size: Duration,
        secs: Uniform<u64>,
    }
}

#[cfg(feature = "std")]
impl SampleUniform for Duration {
    type Sampler = UniformDuration;
}

#[cfg(feature = "std")]
impl UniformSampler for UniformDuration {
    type X = Duration;

    #[inline]
    fn new(low: Duration, high: Duration) -> UniformDuration {
        assert!(low < high, "Uniform::new called with `low >= high`");
        UniformDuration::new_inclusive(low, high - Duration::new(0, 1))
    }

    #[inline]
    fn new_inclusive(low: Duration, high: Duration) -> UniformDuration {
        assert!(low <= high, "Uniform::new_inclusive called with `low > high`");
        let size = high - low;
        let nanos = size
            .as_secs()
            .checked_mul(1_000_000_000)
            .and_then(|n| n.checked_add(size.subsec_nanos() as u64));

        let mode = match nanos {
            Some(nanos) => {
                UniformDurationMode::Small {
                    nanos: Uniform::new_inclusive(0, nanos),
                }
            }
            None => {
                UniformDurationMode::Large {
                    size: size,
                    secs: Uniform::new_inclusive(0, size.as_secs()),
                }
            }
        };

        UniformDuration {
            mode,
            offset: low,
        }
    }

    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Duration {
        let d = match self.mode {
            UniformDurationMode::Small { nanos } => {
                let nanos = nanos.sample(rng);
                Duration::new(nanos / 1_000_000_000, (nanos % 1_000_000_000) as u32)
            }
            UniformDurationMode::Large { size, secs } => {
                // constant folding means this is at least as fast as `gen_range`
                let nano_range = Uniform::new(0, 1_000_000_000);
                loop {
                    let d = Duration::new(secs.sample(rng), nano_range.sample(rng));
                    if d <= size {
                        break d;
                    }
                }
            }
        };

        self.offset + d
    }
}

#[cfg(test)]
mod tests {
    use Rng;
    use distributions::uniform::{Uniform, UniformSampler, UniformFloat, SampleUniform};

    #[should_panic]
    #[test]
    fn test_uniform_bad_limits_equal_int() {
        Uniform::new(10, 10);
    }

    #[should_panic]
    #[test]
    fn test_uniform_bad_limits_equal_float() {
        Uniform::new(10., 10.);
    }

    #[test]
    fn test_uniform_good_limits_equal_int() {
        let mut rng = ::test::rng(804);
        let dist = Uniform::new_inclusive(10, 10);
        for _ in 0..20 {
            assert_eq!(rng.sample(dist), 10);
        }
    }

    #[test]
    fn test_uniform_good_limits_equal_float() {
        let mut rng = ::test::rng(805);
        let dist = Uniform::new_inclusive(10., 10.);
        for _ in 0..20 {
            assert_eq!(rng.sample(dist), 10.);
        }
    }

    #[should_panic]
    #[test]
    fn test_uniform_bad_limits_flipped_int() {
        Uniform::new(10, 5);
    }

    #[should_panic]
    #[test]
    fn test_uniform_bad_limits_flipped_float() {
        Uniform::new(10., 5.);
    }

    #[test]
    fn test_integers() {
        let mut rng = ::test::rng(251);
        macro_rules! t {
            ($($ty:ident),*) => {{
                $(
                   let v: &[($ty, $ty)] = &[(0, 10),
                                            (10, 127),
                                            (::core::$ty::MIN, ::core::$ty::MAX)];
                   for &(low, high) in v.iter() {
                        let my_uniform = Uniform::new(low, high);
                        for _ in 0..1000 {
                            let v: $ty = rng.sample(my_uniform);
                            assert!(low <= v && v < high);
                        }

                        let my_uniform = Uniform::new_inclusive(low, high);
                        for _ in 0..1000 {
                            let v: $ty = rng.sample(my_uniform);
                            assert!(low <= v && v <= high);
                        }

                        for _ in 0..1000 {
                            let v: $ty = rng.gen_range(low, high);
                            assert!(low <= v && v < high);
                        }
                    }
                 )*
            }}
        }
        t!(i8, i16, i32, i64, isize,
           u8, u16, u32, u64, usize);
        #[cfg(feature = "i128_support")]
        t!(i128, u128)
    }

    #[test]
    fn test_floats() {
        let mut rng = ::test::rng(252);
        macro_rules! t {
            ($($ty:ty),*) => {{
                $(
                   let v: &[($ty, $ty)] = &[(0.0, 100.0),
                                            (-1e35, -1e25),
                                            (1e-35, 1e-25),
                                            (-1e35, 1e35)];
                   for &(low, high) in v.iter() {
                        let my_uniform = Uniform::new(low, high);
                        for _ in 0..1000 {
                            let v: $ty = rng.sample(my_uniform);
                            assert!(low <= v && v < high);
                        }
                    }
                 )*
            }}
        }

        t!(f32, f64)
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_durations() {
        use std::time::Duration;

        let mut rng = ::test::rng(253);

        let v = &[(Duration::new(10, 50000), Duration::new(100, 1234)),
                  (Duration::new(0, 100), Duration::new(1, 50)),
                  (Duration::new(0, 0), Duration::new(u64::max_value(), 999_999_999))];
        for &(low, high) in v.iter() {
            let my_uniform = Uniform::new(low, high);
            for _ in 0..1000 {
                let v = rng.sample(my_uniform);
                assert!(low <= v && v < high);
            }
        }
    }

    #[test]
    fn test_custom_uniform() {
        #[derive(Clone, Copy, PartialEq, PartialOrd)]
        struct MyF32 {
            x: f32,
        }
        #[derive(Clone, Copy, Debug)]
        struct UniformMyF32 {
            inner: UniformFloat<f32>,
        }
        impl UniformSampler for UniformMyF32 {
            type X = MyF32;
            fn new(low: Self::X, high: Self::X) -> Self {
                UniformMyF32 {
                    inner: UniformFloat::<f32>::new(low.x, high.x),
                }
            }
            fn new_inclusive(low: Self::X, high: Self::X) -> Self {
                UniformSampler::new(low, high)
            }
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
                MyF32 { x: self.inner.sample(rng) }
            }
        }
        impl SampleUniform for MyF32 {
            type Sampler = UniformMyF32;
        }

        let (low, high) = (MyF32{ x: 17.0f32 }, MyF32{ x: 22.0f32 });
        let uniform = Uniform::new(low, high);
        let mut rng = ::test::rng(804);
        for _ in 0..100 {
            let x: MyF32 = rng.sample(uniform);
            assert!(low <= x && x < high);
        }
    }

    #[test]
    fn test_uniform_from_std_range() {
        let r = Uniform::from(2u32..7);
        assert_eq!(r.inner.low, 2);
        assert_eq!(r.inner.range, 5);
        let r = Uniform::from(2.0f64..7.0);
        assert_eq!(r.inner.offset, -3.0);
        assert_eq!(r.inner.scale, 5.0);
    }
}
