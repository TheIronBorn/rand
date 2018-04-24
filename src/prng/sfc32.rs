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

use core::{fmt, slice, mem};


use core::simd::*;

use rand_core::{RngCore, SeedableRng, Error, impls, le};
#[cfg(feature="simd_support")]
use rand_core::simd_impls::{SimdRng, SimdRngImpls};

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

    fn from_seed(seed: Self::Seed) -> Self {
        let mut seed_u32 = [0u32; 3];
        le::read_u32_into(&seed, &mut seed_u32);
        let state = Self { a: seed_u32[0],
                               b: seed_u32[1],
                               c: seed_u32[2],
                               counter: 1};
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
        // good quality, it is not neccesary to discard the first couple of
        // rounds.
        let mut seed_u32 = [0u32; 3];
        unsafe {
            let ptr = seed_u32.as_mut_ptr() as *mut u8;

            let slice = slice::from_raw_parts_mut(ptr, 4*3);
            rng.try_fill_bytes(slice)?;
        }
        Ok(Self { a: seed_u32[0], b: seed_u32[1], c: seed_u32[2], counter: 1 })
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


#[cfg(feature="simd_support")]
macro_rules! make_sfc_32_simd {
    ($rng_name:ident, $vector:ident, $test_name:ident, $debug_str:expr, $vector32:ident, $vector8:ident) => (
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
                    a, b, c,
                    counter: $vector::splat(1),
                }
            }

            /// Create a new PRNG using the given non-SIMD PRNGs.
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
            }
        }

        impl SimdRng<$vector> for $rng_name {
            #[inline(always)]
            fn generate(&mut self) -> $vector {
                #[inline]
                fn rotate_left(x: $vector, n: u32) -> $vector {
                    const BITS: u32 = 32;
                    // Protect against undefined behaviour for over-long bit shifts
                    let n = n % BITS;
                    (x << n) | (x >> ((BITS - n) % BITS))
                }

                const BARREL_SHIFT: u32 = 21;
                const RSHIFT: u32 = 9;
                const LSHIFT: u32 = 3;

                let tmp = self.a + self.b + self.counter;
                self.counter += $vector::splat(1);
                self.a = self.b ^ (self.b >> RSHIFT);
                self.b = self.c + (self.c << LSHIFT);
                self.c = rotate_left(self.c, BARREL_SHIFT) + tmp;
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
                let mut seed_u32 = [0u32; $vector::lanes() * 3];
                unsafe {
                    let ptr = seed_u32.as_mut_ptr() as *mut u8;

                    let slice = slice::from_raw_parts_mut(ptr, mem::size_of::<$vector>() * 3);
                    rng.try_fill_bytes(slice)?;
                }

                let lanes = $vector::lanes();
                let load = |x| $vector::load_unaligned(x);

                Ok(Self::from_vector(
                    load(&seed_u32[..lanes]),
                    load(&seed_u32[lanes..(2*lanes)]),
                    load(&seed_u32[(lanes*2)..]),
                ))
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
    )
}

#[cfg(feature="simd_support")]
make_sfc_32_simd!(Sfc32x2Rng, u32x2, test_sfc_32_x2, "Sfc32x2Rng {{}}", u32x2, u8x8);
#[cfg(feature="simd_support")]
make_sfc_32_simd!(Sfc32x4Rng, u32x4, test_sfc_32_x4, "Sfc32x4Rng {{}}", u32x4, u8x16);
#[cfg(feature="simd_support")]
make_sfc_32_simd!(Sfc32x8Rng, u32x8, test_sfc_32_x8, "Sfc32x8Rng {{}}", u32x8, u8x32);
