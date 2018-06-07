// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// https://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! SFC generators (32-bit).

use core::fmt;
#[cfg(feature = "simd_support")]
use stdsimd::simd::*;

#[cfg(feature = "simd_support")]
use distributions::box_muller::SimdIntegerMath;
#[cfg(feature = "simd_support")]
use rand_core::simd_impls::{SimdRng, SimdRngImpls};
use rand_core::{impls, le, Error, RngCore, SeedableRng};
use Rng;

/// A Small Fast Counting RNG designed by Chris Doty-Humphrey (32-bit version).
///
/// - Author: Chris Doty-Humphrey
/// - License: Public domain
/// - Source: [PractRand](http://pracrand.sourceforge.net/)
/// - Period: avg ~ 2<sup>127</sup>, min >= 2<sup>32</sup>
/// - State: 128 bits
/// - Word size: 32 bits
/// - Seed size: 96 bits
/// - Passes BigCrush and PractRand
#[derive(Clone)]
pub struct Sfc32Rng {
    a: u32,
    b: u32,
    c: u32,
    counter: u32,
}

// Custom Debug implementation that does not expose the internal state
impl fmt::Debug for Sfc32Rng {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Sfc32Rng {{}}")
    }
}

impl SeedableRng for Sfc32Rng {
    type Seed = [u8; 12];

    #[cfg_attr(feature = "cargo-clippy", allow(let_and_return))]
    fn from_seed(seed: Self::Seed) -> Self {
        let mut seed_u32 = [0u32; 3];
        le::read_u32_into(&seed, &mut seed_u32);
        let state = Self {
            a: seed_u32[0],
            b: seed_u32[1],
            c: seed_u32[2],
            counter: 1,
        };
        // Skip the first 15 outputs, just in case we have a bad seed.
/* We are allowed to assume the seed is good. Possibly use this in `from_seed_u64`
        for _ in 0..15 {
            state.next_u32();
        }
*/
        state
    }

    fn from_rng<R: RngCore>(mut rng: R) -> Result<Self, Error> {
        // Custom `from_rng` function. Because we can assume the seed to be of
        // good quality, it is not necessary to discard the first couple of
        // rounds.
        let mut seed_u32 = [0u32; 3];
        rng.try_fill(&mut seed_u32)?;

        Ok(Self {
            a: seed_u32[0],
            b: seed_u32[1],
            c: seed_u32[2],
            counter: 1,
        })
    }
}

impl RngCore for Sfc32Rng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        // good sets include {21,9,3} and {15,8,3}
        const BARREL_SHIFT: u32 = 21;
        const RSHIFT: u32 = 9;
        const LSHIFT: u32 = 3;

        let tmp = self.a.wrapping_add(self.b).wrapping_add(self.counter);
        self.counter += 1;
        self.a = self.b ^ (self.b >> RSHIFT);
        self.b = self.c.wrapping_add(self.c << LSHIFT);
        self.c = self.c.rotate_left(BARREL_SHIFT).wrapping_add(tmp);
        tmp
    }

    fn next_u64(&mut self) -> u64 {
        impls::next_u64_via_u32(self)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        impls::fill_bytes_via_u32(self, dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        Ok(self.fill_bytes(dest))
    }
}

#[cfg(feature = "simd_support")]
macro_rules! make_sfc_simd {
    (
        $rng_name:ident,
        $vector:ident,
        $test_name:ident,
        $debug_str:expr,rot:
        $rot:expr,rsh:
        $rsh:expr,lsh:
        $lsh:expr
    ) => {
        /// A SIMD implementation of Chris Doty-Humphrey's Small Fast Counting RNG (32-bit)
        pub struct $rng_name {
            a: $vector,
            b: $vector,
            c: $vector,
            counter: $vector,
        }

        // Custom Debug implementation that does not expose the internal state
        impl fmt::Debug for $rng_name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, $debug_str)
            }
        }

        impl RngCore for $rng_name {
            #[inline(always)]
            fn next_u32(&mut self) -> u32 {
                $vector::next_u32_via_simd(self)
            }

            #[inline(always)]
            fn next_u64(&mut self) -> u64 {
                $vector::next_u64_via_simd(self)
            }

            #[inline(always)]
            fn fill_bytes(&mut self, dest: &mut [u8]) {
                $vector::fill_bytes_via_simd(self, dest)
            }

            fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
                self.fill_bytes(dest);
                Ok(())
            }
        }

        impl $rng_name {
            /// Create a new PRNG using the given vector seeds.
            pub fn from_vector(a: $vector, b: $vector, c: $vector) -> Self {
                Self {
                    a,
                    b,
                    c,
                    counter: $vector::splat(1),
                }
            }

            /*/// Create a new PRNG using the given non-SIMD PRNGs.
            pub fn from_non_simd(reg_rngs: &[Sfc32Rng]) -> Self {
                let mut a = $vector::default();
                let mut b = $vector::default();
                let mut c = $vector::default();

                for (i, rng) in reg_rngs.iter().enumerate().take($vector::lanes()) {
                    a = a.replace(i, rng.a);
                    b = b.replace(i, rng.b);
                    c = c.replace(i, rng.c);
                }

                Self::from_vector(a, b, c)
            }*/
        }

        impl SimdRng<$vector> for $rng_name {
            #[inline(always)]
            fn generate(&mut self) -> $vector {
                let tmp = self.a + self.b + self.counter;
                self.counter += 1;
                self.a = self.b ^ (self.b >> $rsh);
                self.b = self.c + (self.c << $lsh);
                self.c = self.c.rotate_left($rot) + tmp;
                tmp
            }
        }

        impl SeedableRng for $rng_name {
            type Seed = [u8; 0];

            fn from_seed(_seed: Self::Seed) -> Self {
                unimplemented!();
            }

            #[inline]
            fn from_rng<R: RngCore>(mut rng: R) -> Result<Self, Error> {
                let mut seed = [$vector::default(); 3];
                rng.try_fill(&mut seed)?;

                Ok(Self::from_vector(seed[0], seed[1], seed[2]))
            }
        }

        /*#[test]
        fn $test_name() {
            use thread_rng;

            fn test(reg_rngs: &mut [Sfc32Rng]) {
                let mut simd_rng = $rng_name::from_non_simd(reg_rngs);

                for i in 0..20 {
                    let expected = reg_rngs.iter_mut().map(|x| x.next_u32());
                    let next: $vector = simd_rng.generate();

                    for (j, exp) in expected.enumerate() {
                        let actual = next.extract(j);
                        assert_eq!(actual, exp, "{:?}", i);
                    }
                }
            }

            let mut rng = thread_rng();
            for _ in 0..20 {
                let mut reg_rngs: Vec<_> = (0..$vector::lanes())
                    .map(|_| Sfc32Rng::from_rng(&mut rng).unwrap())
                    .collect();
                test(&mut reg_rngs);
            }
        }*/
    };
}

#[cfg(feature = "simd_support")]
macro_rules! make_sfc_16_simd {
    ($($rng_name:ident, $vec:ident, $test_name:ident, $debug_str:expr,)+) => (
        $(make_sfc_simd!($rng_name, $vec, $test_name, $debug_str, rot: 3, rsh: 2, lsh: 1);)+
    )
}

#[cfg(feature = "simd_support")]
make_sfc_16_simd! {
    Sfc8x2Rng, u8x2, test_sfc_16_x2, "Sfc8x2Rng {{}}",
    // Sfc8x4Rng, u8x4, test_sfc_16_x4, "Sfc8x4Rng {{}}",
    // Sfc8x8Rng, u8x8, test_sfc_16_x8, "Sfc8x8Rng {{}}",
    // Sfc8x16Rng, u8x16, test_sfc_16_x16, "Sfc8x16Rng {{}}",
    // Sfc8x32Rng, u8x32, test_sfc_16_x32, "Sfc8x32Rng {{}}",
    // Sfc8x64Rng, u8x64, test_sfc_16_x64, "Sfc8x64Rng {{}}",
}

#[cfg(feature = "simd_support")]
macro_rules! make_sfc_16_simd {
    ($($rng_name:ident, $vec:ident, $test_name:ident, $debug_str:expr,)+) => (
        $(make_sfc_simd!($rng_name, $vec, $test_name, $debug_str, rot: 6, rsh: 5, lsh: 3);)+
    )
}

#[cfg(feature = "simd_support")]
make_sfc_16_simd! {
    Sfc16x2Rng, u16x2, test_sfc_16_x2, "Sfc16x2Rng {{}}",
    Sfc16x4Rng, u16x4, test_sfc_16_x4, "Sfc16x4Rng {{}}",
    Sfc16x8Rng, u16x8, test_sfc_16_x8, "Sfc16x8Rng {{}}",
    Sfc16x16Rng, u16x16, test_sfc_16_x16, "Sfc16x16Rng {{}}",
    Sfc16x32Rng, u16x32, test_sfc_16_x32, "Sfc16x32Rng {{}}",
}

#[cfg(feature = "simd_support")]
macro_rules! make_sfc_32_simd {
    ($($rng_name:ident, $vec:ident, $test_name:ident, $debug_str:expr,)+) => (
        $(make_sfc_simd!($rng_name, $vec, $test_name, $debug_str, rot: 21, rsh: 9, lsh: 3);)+
    )
}

#[cfg(feature = "simd_support")]
make_sfc_32_simd! {
    Sfc32x2Rng, u32x2, test_sfc_32_x2, "Sfc32x2Rng {{}}",
    Sfc32x4Rng, u32x4, test_sfc_32_x4, "Sfc32x4Rng {{}}",
    Sfc32x8Rng, u32x8, test_sfc_32_x8, "Sfc32x8Rng {{}}",
    Sfc32x16Rng, u32x16, test_sfc_32_x16, "Sfc32x16Rng {{}}",
}

#[cfg(feature = "simd_support")]
macro_rules! make_sfc_64_simd {
    ($($rng_name:ident, $vec:ident, $test_name:ident, $debug_str:expr,)+) => (
        $(make_sfc_simd!($rng_name, $vec, $test_name, $debug_str, rot: 24, rsh: 11, lsh: 3);)+
    )
}

#[cfg(feature = "simd_support")]
make_sfc_64_simd! {
    Sfc64x2Rng, u64x2, test_sfc_64_x2, "Sfc64x2Rng {{}}",
    Sfc64x4Rng, u64x4, test_sfc_64_x4, "Sfc64x4Rng {{}}",
    Sfc64x8Rng, u64x8, test_sfc_64_x8, "Sfc64x8Rng {{}}",
}
