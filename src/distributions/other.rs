// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// https://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The implementations of the `Standard` distribution for other built-in types.

use core::char;
use core::num::Wrapping;

use {Rng};
use distributions::{Distribution, Standard, Uniform};

// ----- Sampling distributions -----

/// Sample a `char`, uniformly distributed over ASCII letters and numbers:
/// a-z, A-Z and 0-9.
///
/// # Example
///
/// ```
/// use std::iter;
/// use rand::{Rng, thread_rng};
/// use rand::distributions::Alphanumeric;
///
/// let mut rng = thread_rng();
/// let chars: String = iter::repeat(())
///         .map(|()| rng.sample(Alphanumeric))
///         .take(7)
///         .collect();
/// println!("Random chars: {}", chars);
/// ```
#[derive(Debug)]
pub struct Alphanumeric;


// ----- Implementations of distributions -----

impl Distribution<char> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> char {
        let range = Uniform::new(0u32, 0x11_0000);
        loop {
            match char::from_u32(range.sample(rng)) {
                Some(c) => return c,
                // About 0.2% of numbers in the range 0..0x110000 are invalid
                // codepoints (surrogates).
                None => {}
            }
        }
    }
}

impl Distribution<char> for Alphanumeric {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> char {
        const RANGE: u32 = 26 + 26 + 10;
        const GEN_ASCII_STR_CHARSET: &[u8] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                abcdefghijklmnopqrstuvwxyz\
                0123456789";
        // We can pick from 62 characters. This is so close to a power of 2, 64,
        // that we can do better than `Uniform`. Use a simple bitshift and
        // rejection sampling. We do not use a bitmask, because for small RNGs
        // the most significant bits are usually of higher quality.
        loop {
            let var = rng.next_u32() >> (32 - 6);
            if var < RANGE {
                return GEN_ASCII_STR_CHARSET[var as usize] as char
            }
        }
    }
}

impl Distribution<bool> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        // We can compare against an arbitrary bit of an u32 to get a bool.
        // Because the least significant bits of a lower quality RNG can have
        // simple patterns, we compare against the most significant bit. This is
        // easiest done using a sign test.
        (rng.next_u32() as i32) < 0
    }
}

macro_rules! tuple_impl {
    // use variables to indicate the arity of the tuple
    ($($tyvar:ident),* ) => {
        // the trailing commas are for the 1 tuple
        impl< $( $tyvar ),* >
            Distribution<( $( $tyvar ),* , )>
            for Standard
            where $( Standard: Distribution<$tyvar> ),*
        {
            #[inline]
            fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> ( $( $tyvar ),* , ) {
                (
                    // use the $tyvar's to get the appropriate number of
                    // repeats (they're not actually needed)
                    $(
                        _rng.gen::<$tyvar>()
                    ),*
                    ,
                )
            }
        }
    }
}

impl Distribution<()> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> () { () }
}
tuple_impl!{A}
tuple_impl!{A, B}
tuple_impl!{A, B, C}
tuple_impl!{A, B, C, D}
tuple_impl!{A, B, C, D, E}
tuple_impl!{A, B, C, D, E, F}
tuple_impl!{A, B, C, D, E, F, G}
tuple_impl!{A, B, C, D, E, F, G, H}
tuple_impl!{A, B, C, D, E, F, G, H, I}
tuple_impl!{A, B, C, D, E, F, G, H, I, J}
tuple_impl!{A, B, C, D, E, F, G, H, I, J, K}
tuple_impl!{A, B, C, D, E, F, G, H, I, J, K, L}

macro_rules! array_impl {
    // recursive, given at least one type parameter:
    {$n:expr, $t:ident, $($ts:ident,)*} => {
        array_impl!{($n - 1), $($ts,)*}

        impl<T> Distribution<[T; $n]> for Standard where Standard: Distribution<T> {
            #[inline]
            fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> [T; $n] {
                [_rng.gen::<$t>(), $(_rng.gen::<$ts>()),*]
            }
        }
    };
    // empty case:
    {$n:expr,} => {
        impl<T> Distribution<[T; $n]> for Standard {
            fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> [T; $n] { [] }
        }
    };
}

array_impl!{32, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T,}

impl<T> Distribution<Option<T>> for Standard where Standard: Distribution<T> {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<T> {
        // UFCS is needed here: https://github.com/rust-lang/rust/issues/24066
        if rng.gen::<bool>() {
            Some(rng.gen())
        } else {
            None
        }
    }
}

impl<T> Distribution<Wrapping<T>> for Standard where Standard: Distribution<T> {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Wrapping<T> {
        Wrapping(rng.gen())
    }
}

#[cfg(feature = "simd_support")]
mod simd {
    // extern crate stdsimd;

    use super::*;

    use core::simd::*;

    /// Sample chars as SIMD integer vectors.
    ///
    /// Because there are no `char` vectors, these can only return an integer
    /// vector where each integer is a valid value for a char.
    /// `char::from_u32_unchecked` or `from_utf8_unchecked` will safely turn
    /// those integers into chars.
    #[derive(Debug)]
    pub struct SimdCharDistribution;

    macro_rules! impl_simd_char_sampling {
        ($ty:ident) => {
            impl Distribution<$ty> for SimdCharDistribution {
                #[inline]
                fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $ty {
                    // A valid `char` is either in the interval `[0, 0xD800)` or
                    // `(0xDFFF, 0x11_0000)`. All `char`s must therefore be in
                    // `[0, 0x11_0000)` but not in the "gap" `[0xD800, 0xDFFF]` which is
                    // reserved for surrogates. This is the size of that gap.
                    const GAP_SIZE: $ty = $ty::splat(0xDFFF - 0xD800 + 1);

                    // Uniform::new(0, 0x11_0000 - GAP_SIZE) can also be used but it
                    // seemed slower. TODO: benchmark (sample_below perhaps?)
                    let range = Uniform::new(GAP_SIZE, $ty::splat(0x11_0000));

                    let mut n = range.sample(rng);
                    let cmp = n.le($ty::splat(0xDFFF));
                    n -= cmp.select(GAP_SIZE, $ty::splat(0));

                    #[cfg(any(test, debug_assertions))]
                    for i in 0..$ty::lanes() {
                        let c = n.extract(i) as u32;
                        let expected = char::from_u32(c);
                        let actual = unsafe { char::from_u32_unchecked(c) };
                        assert_eq!(expected, Some(actual));
                    }

                    n
                }
            }
        };
    }

    impl_simd_char_sampling!(u32x2);
    impl_simd_char_sampling!(u32x4);
    impl_simd_char_sampling!(u32x8);
    impl_simd_char_sampling!(u32x16);

    impl_simd_char_sampling!(u64x2);
    impl_simd_char_sampling!(u64x4);
    impl_simd_char_sampling!(u64x8);

    /// Sample chars as SIMD integer vectors, uniformly distributed over ASCII
    /// letters and numbers: a-z, A-Z and 0-9.
    ///
    /// `char::from_u32_unchecked` or `from_utf8_unchecked` will safely turn
    /// the integers into chars.
    ///
    /// This selects bit directly for rejection sampling, so the size of the
    /// integer does not effect the rate of sampling misses.  This means that
    /// vectors with many lanes will likely produce chars more quickly.
    #[derive(Debug)]
    pub struct AlphanumericSimd;

    macro_rules! impl_simd_alnum_sampling {
        ($($ty:ident,)+, $scalar:ident, $bits:expr) => (
            $(impl Distribution<$ty> for AlphanumericSimd {
                #[cfg_attr(feature="cargo-clippy", allow(unnecessary_cast))]
                fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $ty {
                    // A bitshift is less useful here because using
                    // `u8x16::from_bits(u32x4)`, the low quality bits are
                    // only in a quarter of the lanes. Shifting is also slower
                    // in SIMD than in scalar.
                    let mut bytes = rng.gen::<$ty>() % 64;

                    let var = loop {
                        let mask = bytes.lt($ty::splat(62));
                        if mask.all() {
                            break bytes;
                        }
                        bytes = mask.select(bytes, rng.gen::<$ty>() % 64);
                    };

                    let mut adjust = $ty::splat(b'0' as $scalar);

                    // compare and adjust the index (bitwise ops are faster
                    // than arithmetic)
                    adjust ^= $ty::from_bits(var.ge($ty::splat(10))) & 7;
                    adjust ^= $ty::from_bits(var.ge($ty::splat(36))) & 10;

                    let chars = adjust + var;

                    #[cfg(any(test, debug_assertions))]
                    for i in 0..$ty::lanes() {
                        let ch = chars.extract(i);
                        match char::from_u32(ch as u32) {
                            Some(c) => match c {
                                'a'...'z' | 'A'...'Z' | '0'...'9' => (),
                                _ => panic!("not an alphanumeric character"),
                            },
                            None => panic!("not a character"),
                        }
                    }

                    chars
                }
            })+
        )
    }

    impl_simd_alnum_sampling!(u8x2, u8x4, u8x8, u8x16, u8x32, u8x64,, u8, 8);
    impl_simd_alnum_sampling!(u16x2, u16x4, u16x8, u16x16, u16x32,, u16, 16);
    impl_simd_alnum_sampling!(u32x2, u32x4, u32x8, u32x16,, u32, 32);
    impl_simd_alnum_sampling!(u64x2, u64x4, u64x8,, u64, 64);

    macro_rules! impl_boolean_vector {
        ($($ty:ty, $signed:ident,)+) => (
            $(impl Distribution<$ty> for Standard {
                #[inline]
                fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $ty {
                    // TODO: compare speed with other bit selections
                    rng.gen::<$signed>().lt($signed::splat(0))
                }
            })+
        )
    }

    impl_boolean_vector! {
        m8x2, i8x2,
        m8x4, i8x4,
        m8x8, i8x8,
        m8x16, i8x16,
        m8x32, i8x32,
        m1x64, i8x64, // 512-bit mask types have strange names

        m16x2, i16x2,
        m16x4, i16x4,
        m16x8, i16x8,
        m16x16, i16x16,
        m1x32, i16x32,

        m32x2, i32x2,
        m32x4, i32x4,
        m32x8, i32x8,
        m1x16, i32x16,

        m64x2, i64x2,
        m64x4, i64x4,
        m1x8, i64x8,
    }
}
#[cfg(feature = "simd_support")]
pub use self::simd::*;

#[cfg(test)]
mod tests {
    use {Rng, RngCore, Standard};
    use distributions::Alphanumeric;
    #[cfg(all(not(feature="std"), feature="alloc"))] use alloc::String;

    #[test]
    fn test_misc() {
        let rng: &mut RngCore = &mut ::test::rng(820);

        rng.sample::<char, _>(Standard);
        rng.sample::<bool, _>(Standard);
    }

    #[cfg(feature="alloc")]
    #[test]
    fn test_chars() {
        use core::iter;
        let mut rng = ::test::rng(805);

        // Test by generating a relatively large number of chars, so we also
        // take the rejection sampling path.
        let word: String = iter::repeat(())
                .map(|()| rng.gen::<char>()).take(1000).collect();
        assert!(word.len() != 0);
    }

    #[test]
    fn test_alphanumeric() {
        let mut rng = ::test::rng(806);

        // Test by generating a relatively large number of chars, so we also
        // take the rejection sampling path.
        let mut incorrect = false;
        for _ in 0..100 {
            let c = rng.sample(Alphanumeric);
            incorrect |= !((c >= '0' && c <= '9') ||
                           (c >= 'A' && c <= 'Z') ||
                           (c >= 'a' && c <= 'z') );
        }
        assert!(incorrect == false);
    }

    #[cfg(feature = "simd_support")]
    mod simd {
        // extern crate stdsimd;

        use core::simd::*;

        use distributions::other::*;

        // Correctness is verified within the distribution code.
        //
        // NOTE: the checks will not run with `cargo test --release`

        #[test]
        fn test_simd_alphanumeric() {
            let mut rng = ::test::rng(806);

            // Test by generating a relatively large number of chars, so we also
            // take the rejection sampling path.
            for _ in 0..100 {
                let _c: u8x16 = rng.sample(AlphanumericSimd);
            }
        }

        #[test]
        fn test_simd_chars() {
            let mut rng = ::test::rng(805);

            // Test by generating a relatively large number of chars, so we also
            // take the rejection sampling path.
            for _ in 0..1000 {
                let _c: u32x4 = rng.sample(SimdCharDistribution);
            }
        }
    }
}
