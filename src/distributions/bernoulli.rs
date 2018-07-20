// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// https://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//! The Bernoulli distribution.

use Rng;
use distributions::Distribution;

/// The Bernoulli distribution.
///
/// This is a special case of the Binomial distribution where `n = 1`.
///
/// # Example
///
/// ```rust
/// use rand::distributions::{Bernoulli, Distribution};
///
/// let d = Bernoulli::new(0.3);
/// let v = d.sample(&mut rand::thread_rng());
/// println!("{} is from a Bernoulli distribution", v);
/// ```
///
/// # Precision
///
/// This `Bernoulli` distribution uses 64 bits from the RNG (a `u64`),
/// so only probabilities that are multiples of 2<sup>-64</sup> can be
/// represented.
#[derive(Clone, Copy, Debug)]
pub struct Bernoulli<U> {
    /// Probability of success, relative to the maximal integer.
    p_int: U,
}

impl Bernoulli<u64> {
    /// Construct a new `Bernoulli` with the given probability of success `p`.
    ///
    /// # Panics
    ///
    /// If `p < 0` or `p > 1`.
    ///
    /// # Precision
    ///
    /// For `p = 1.0`, the resulting distribution will always generate true.
    /// For `p = 0.0`, the resulting distribution will always generate false.
    ///
    /// This method is accurate for any input `p` in the range `[0, 1]` which is
    /// a multiple of 2<sup>-64</sup>. (Note that not all multiples of
    /// 2<sup>-64</sup> in `[0, 1]` can be represented as a `f64`.)
    #[inline]
    pub fn new(p: f64) -> Bernoulli<u64> {
        assert!((p >= 0.0) & (p <= 1.0), "Bernoulli::new not called with 0 <= p <= 0");
        // Technically, this should be 2^64 or `u64::MAX + 1` because we compare
        // using `<` when sampling. However, `u64::MAX` rounds to an `f64`
        // larger than `u64::MAX` anyway.
        const MAX_P_INT: f64 = ::core::u64::MAX as f64;
        let p_int = if p < 1.0 {
            (p * MAX_P_INT) as u64
        } else {
            // Avoid overflow: `MAX_P_INT` cannot be represented as u64.
            ::core::u64::MAX
        };
        Bernoulli { p_int }
    }
}

impl Distribution<bool> for Bernoulli<u64> {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        // Make sure to always return true for p = 1.0.
        if self.p_int == ::core::u64::MAX {
            return true;
        }
        let r: u64 = rng.gen();
        r < self.p_int
    }
}

#[cfg(feature = "simd_support")]
mod simd {
    // extern crate stdsimd;

    use super::*;

    use core::simd::*;

    macro_rules! impl_boolean_vector {
        ($fty:ident, $uty:ident, $mty:ident) => (
            impl Bernoulli<$uty> {
                #[inline]
                pub fn new(p: $fty) -> Bernoulli<$uty> {
                    assert!((p.ge($fty::splat(0.0)) & p.le($fty::splat(1.0))).all(),
                        "Bernoulli::new not called with 0 <= p <= 0");
                    // Technically, this should be 2^64 or `u64::MAX + 1` because we compare
                    // using `<` when sampling. However, `u64::MAX` rounds to an `f64`
                    // larger than `u64::MAX` anyway.
                    const MAX_P_INT: f64 = ::core::u64::MAX as f64;

                    let cmp = p.lt($fty::splat(1.0));
                    let p_int = cmp.select($uty::from(p * MAX_P_INT), $uty::splat(::core::u64::MAX));

                    Bernoulli::<$uty> { p_int }
                }
            }

            impl Distribution<$mty> for Bernoulli<$uty> {
                #[inline]
                fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $mty {
                    // Make sure to always return true for p = 1.0.
                    // Ensure no random numbers generated when all lanes of
                    // `cmp` are true? Seems low priority.
                    let cmp = self.p_int.eq($uty::splat(::core::u64::MAX));
                    let r: $uty = rng.gen();
                    cmp | r.lt(self.p_int)
                }
            }
        )
    }

    impl_boolean_vector! { f64x2, u64x2, m64x2 }
    impl_boolean_vector! { f64x4, u64x4, m64x4 }
    impl_boolean_vector! { f64x8, u64x8, m1x8 } // 512-bit mask types have strange names
}
#[cfg(feature = "simd_support")]
pub use self::simd::*;

#[cfg(test)]
mod test {
    use Rng;
    use distributions::Distribution;
    /*#[cfg(feature = "simd_support")]
    extern crate stdsimd;*/

    #[cfg(feature = "simd_support")]
    use core::simd::*;

    use super::Bernoulli;

    #[test]
    fn test_trivial() {
        let mut r = ::test::rng(1);
        let always_false = Bernoulli::<u64>::new(0.0);
        let always_true = Bernoulli::<u64>::new(1.0);
        for _ in 0..5 {
            assert_eq!(r.sample::<bool, _>(&always_false), false);
            assert_eq!(r.sample::<bool, _>(&always_true), true);
            assert_eq!(Distribution::<bool>::sample(&always_false, &mut r), false);
            assert_eq!(Distribution::<bool>::sample(&always_true, &mut r), true);
        }
    }

    #[test]
    #[cfg(feature = "simd_support")]
    fn test_trivial_simd() {
        let mut r = ::test::rng(1);
        let always_false = Bernoulli::<u64x2>::new(f64x2::splat(0.0));
        let always_true = Bernoulli::<u64x2>::new(f64x2::splat(1.0));
        for _ in 0..5 {
            assert_eq!(r.sample::<m64x2, _>(&always_false), m64x2::splat(false));
            assert_eq!(r.sample::<m64x2, _>(&always_true), m64x2::splat(true));
            assert_eq!(
                Distribution::<m64x2>::sample(&always_false, &mut r),
                m64x2::splat(false)
            );
            assert_eq!(
                Distribution::<m64x2>::sample(&always_true, &mut r),
                m64x2::splat(true)
            );
        }
    }

    #[test]
    fn test_average() {
        const P: f64 = 0.3;
        let d = Bernoulli::<u64>::new(P);
        const N: u32 = 10_000_000;

        let mut sum: u32 = 0;
        let mut rng = ::test::rng(2);
        for _ in 0..N {
            if d.sample(&mut rng) {
                sum += 1;
            }
        }
        let avg = (sum as f64) / (N as f64);

        assert!((avg - P).abs() < 1e-3);
    }

    #[test]
    #[cfg(feature = "simd_support")]
    fn test_average_simd() {
        const P: f64 = 0.3;
        let d = Bernoulli::<u64x2>::new(f64x2::splat(P));
        const N: u32 = 10_000_000;

        let mut sum = u64x2::splat(0);
        let mut rng = ::test::rng(2);
        for _ in 0..N {
            sum += u64x2::from_bits(d.sample(&mut rng)) & 1;
        }
        let avg = f64x2::from(sum) / (N as f64);

        assert!((avg - P).abs().lt(f64x2::splat(1e-3)).all());
    }
}
