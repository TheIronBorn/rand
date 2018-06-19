//! The Box-Muller transform and derived distributions, for use with SIMD PRNGs.
//!
//! <https://en.wikipedia.org/wiki/Box-Muller_transform>

// TODO: look into more accurate math

#[cfg(feature="simd_support")]
use stdsimd::arch::x86_64::*;
#[cfg(feature="simd_support")]
use stdsimd::simd::*;
#[cfg(feature="simd_support")]
use core::f32::consts::PI as PI_32;
#[cfg(feature="simd_support")]
use core::f64::consts::PI as PI_64;
#[cfg(target_feature = "fma")]
use core::intrinsics::fmaf64;
#[cfg(feature="simd_support")]
use core::mem::*;
#[cfg(feature="simd_support")]
#[allow(unused_imports)]
use core::{f32, f64};

use Rng;
#[cfg(feature="simd_support")]
use distributions::Uniform;

/// The normal distribution `N(mean, std_dev**2)`.
///
/// This uses the [Box–Muller transform] for compatibility with SIMD PRNGs.
/// For most uses, [`Normal`] is a better choice.
///
/// # Example
///
/// ```rust
/// #![feature(stdsimd)]
/// use std::simd::f32x4;
///
/// use rand::distributions::{BoxMuller, BoxMullerCore};
/// use rand::NewRng;
/// use rand::prng::Sfc32x4Rng;
///
/// // mean 2, standard deviation 3
/// let mut normal = BoxMuller::new(f32x4::splat(2.0), f32x4::splat(3.0));
/// let v = normal.sample(&mut Sfc32x4Rng::new());
/// println!("{:?} is from a N(2, 9) distribution", v)
/// ```
///
/// [`Normal`]: ../normal/struct.Normal.html
/// [Box–Muller transform]: https://en.wikipedia.org/wiki/Box-Muller_transform
#[derive(Clone, Copy, Debug)]
#[cfg(feature="simd_support")] // necessary for doc tests?
pub struct BoxMuller<T> {
    flag: bool,
    z1: T,
    mean: T,
    std_dev: T,
}

#[cfg(feature="simd_support")] // necessary for doc tests?
impl<T: Default + PartialOrd> BoxMuller<T> {
    /// Construct a new `BoxMuller` normal distribution with the given mean
    /// and standard deviation.
    ///
    /// # Panics
    ///
    /// Panics if `std_dev < 0.0`.
    pub fn new(mean: T, std_dev: T) -> Self {
        assert!(
            std_dev >= T::default(),
            "BoxMuller::new called with `std_dev` < 0"
        );
        Self {
            flag: false,
            z1: T::default(),
            mean,
            std_dev,
        }
    }
}

/// The core of the Box–Muller transform.  We implement `sample` here because
/// `Distribution` does not allow mutable state.
pub trait BoxMullerCore<T> {
    /// The core Box-Muller transform.
    ///
    /// Returns two independent random numbers with a standard normal
    /// distribution.
    fn box_muller<R: Rng + ?Sized>(rng: &mut R) -> (T, T);

    /// uses `simd_fsin`/`simd_fcos`
    fn ftrig<R: Rng + ?Sized>(rng: &mut R) -> (T, T);

    /// uses `simd_sin`/`simd_cos`
    fn stdsimd_trig<R: Rng + ?Sized>(rng: &mut R) -> (T, T);

    /// uses my hacked together `sin_cos`
    fn hacked_trig<R: Rng + ?Sized>(rng: &mut R) -> (T, T);

    /// Generate a random value of `T`, using `rng` as the source of randomness.
    fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> T;

    /// The polar variant
    fn polar<R: Rng + ?Sized>(rng: &mut R) -> (T, T);

    /// For testing only
    fn polar_no_rejection<R: Rng + ?Sized>(rng: &mut R) -> (T, T);
}

/*#[cfg(feature="simd_support")] // necessary for doc tests?
impl BoxMullerCore<f64> for BoxMuller<f64> {
    fn box_muller<R: Rng + ?Sized>(rng: &mut R) -> (f64, f64) {
        const TWO_PI: f64 = PI_64 * 2.0;

        let (u0, u1): (f64, f64) = rng.gen();

        let radius = (-2.0 * u0.ln()).sqrt();
        let (sin_theta, cos_theta) = (TWO_PI * u1).sin_cos();

        let z0 = radius * sin_theta;
        let z1 = radius * cos_theta;
        (z0, z1)
    }

    fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> f64 {
        self.flag = !self.flag;

        if !self.flag {
            return self.z1 * self.std_dev + self.mean;
        }

        let (z0, z1) = Self::box_muller(rng);
        self.z1 = z1;
        z0 * self.std_dev + self.mean
    }

    fn polar<R: Rng + ?Sized>(_rng: &mut R) -> (f64, f64) {
        unimplemented!();
    }

    fn polar_no_rejection<R: Rng + ?Sized>(_rng: &mut R) -> (f64, f64) {
        unimplemented!();
    }
}

#[cfg(feature="simd_support")] // necessary for doc tests?
impl BoxMullerCore<f32> for BoxMuller<f32> {
    fn box_muller<R: Rng + ?Sized>(rng: &mut R) -> (f32, f32) {
        const TWO_PI: f32 = PI_32 * 2.0;

        let (u0, u1): (f32, f32) = rng.gen();

        let radius = (-2.0 * u0.ln()).sqrt();
        let (sin_theta, cos_theta) = (TWO_PI * u1).sin_cos();

        let z0 = radius * sin_theta;
        let z1 = radius * cos_theta;
        (z0, z1)
    }

    fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> f32 {
        self.flag = !self.flag;

        if !self.flag {
            return self.z1 * self.std_dev + self.mean;
        }

        let (z0, z1) = Self::box_muller(rng);
        self.z1 = z1;
        z0 * self.std_dev + self.mean
    }

    fn polar<R: Rng + ?Sized>(_rng: &mut R) -> (f32, f32) {
        unimplemented!();
    }

    fn polar_no_rejection<R: Rng + ?Sized>(_rng: &mut R) ->  (f32, f32) {
        unimplemented!();
    }
}*/

#[cfg(feature="simd_support")]
macro_rules! impl_box_muller {
    ($pi:expr, $(($vector:ident, $uty:ident)),+) => (
        $(impl BoxMullerCore<$vector> for BoxMuller<$vector> {
            #[inline(always)]
            fn box_muller<R: Rng + ?Sized>(rng: &mut R) -> ($vector, $vector) {
                const TWO_PI: $vector = $vector::splat(2.0 * $pi);

                let radius = (-2.0 * rng.gen::<$vector>().ln()).sqrte();
                let (sin_theta, cos_theta) = (TWO_PI * rng.gen::<$vector>()).sin_cos();

                (radius * sin_theta, radius * cos_theta)
            }

            #[inline(always)]
            fn ftrig<R: Rng + ?Sized>(rng: &mut R) -> ($vector, $vector) {
                const TWO_PI: $vector = $vector::splat(2.0 * $pi);

                let radius = (-2.0 * rng.gen::<$vector>().ln()).sqrte();
                let intermediate = TWO_PI * rng.gen::<$vector>();
                let sin_theta = unsafe { simd_fsin(intermediate) };
                let cos_theta = unsafe { simd_fcos(intermediate) };

                (radius * sin_theta, radius * cos_theta)
            }

            #[inline(always)]
            fn stdsimd_trig<R: Rng + ?Sized>(rng: &mut R) -> ($vector, $vector) {
                const TWO_PI: $vector = $vector::splat(2.0 * $pi);

                let radius = (-2.0 * rng.gen::<$vector>().ln()).sqrte();
                let intermediate = TWO_PI * rng.gen::<$vector>();
                let sin_theta = intermediate.sin();
                let cos_theta = intermediate.cos();

                (radius * sin_theta, radius * cos_theta)
            }

            #[inline(always)]
            fn hacked_trig<R: Rng + ?Sized>(rng: &mut R) -> ($vector, $vector) {
                const TWO_PI: $vector = $vector::splat(2.0 * $pi);

                let radius = (-2.0 * rng.gen::<$vector>().ln()).sqrte();
                let (sin_theta, cos_theta) = (TWO_PI * rng.gen::<$vector>()).sin_cos();

                (radius * sin_theta, radius * cos_theta)
            }

            #[inline(always)]
            fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> $vector {
                self.flag = !self.flag;

                if !self.flag {
                    return self.z1 * self.std_dev + self.mean;
                }

                let (z0, z1) = Self::box_muller(rng);
                self.z1 = z1;
                z0 * self.std_dev + self.mean
            }

            #[inline(always)]
            fn polar<R: Rng + ?Sized>(rng: &mut R) -> ($vector, $vector) {
                let range = Uniform::new($vector::splat(-1.0), $vector::splat(1.0));

                let mut u = rng.sample(range);
                let mut v = rng.sample(range);
                let mut s = u*u + v*v;
                loop {
                    let mask = s.eq($vector::splat(0.0)) | s.ge($vector::splat(1.0));
                    if mask.none() {
                        break;
                    }
                    u = mask.select(rng.sample(range), u);
                    v = mask.select(rng.sample(range), v);
                    s = u*u + v*v;
                }

                let intermed = (-2.0 * s.ln() / s).sqrte();
                (u * intermed, v * intermed)
            }

            #[inline(always)]
            fn polar_no_rejection<R: Rng + ?Sized>(rng: &mut R) -> ($vector, $vector) {
                let range = Uniform::new($vector::splat(-1.0), $vector::splat(1.0));

                let u = rng.sample(range);
                let v = rng.sample(range);
                let s = u*u + v*v;
                // let s = u.fma(u, v*v);

                let intermed = (-2.0 * s.ln() / s).sqrte();
                (u * intermed, v * intermed)
            }
        })+
    )
}

#[cfg(feature="simd_support")]
impl_box_muller!(
    PI_32,
    (f32x2, u32x2),
    (f32x4, u32x4),
    (f32x8, u32x8),
    (f32x16, u32x16)
);
// TODO: 64-bit versions?
#[cfg(feature="simd_support")]
macro_rules! impl_box_muller_64 {
    ($pi:expr, $(($vector:ident, $uty:ident)),+) => (
        $(impl BoxMullerCore<$vector> for BoxMuller<$vector> {
            #[allow(unused_variables)]
            #[inline(always)]
            fn box_muller<R: Rng + ?Sized>(rng: &mut R) -> ($vector, $vector) {
                const TWO_PI: $vector = $vector::splat(2.0 * $pi);

                let radius = (-2.0 * rng.gen::<$vector>().ln()).sqrte();
                let intermediate = TWO_PI * rng.gen::<$vector>();
                let sin_theta = unsafe { simd_fsin(intermediate) };
                let cos_theta = unsafe { simd_fcos(intermediate) };

                (radius * sin_theta, radius * cos_theta)
            }

            #[inline(always)]
            fn ftrig<R: Rng + ?Sized>(rng: &mut R) -> ($vector, $vector) {
                const TWO_PI: $vector = $vector::splat(2.0 * $pi);

                let radius = (-2.0 * rng.gen::<$vector>().ln()).sqrte();
                let intermediate = TWO_PI * rng.gen::<$vector>();
                let sin_theta = unsafe { simd_fsin(intermediate) };
                let cos_theta = unsafe { simd_fcos(intermediate) };

                (radius * sin_theta, radius * cos_theta)
            }

            #[inline(always)]
            fn stdsimd_trig<R: Rng + ?Sized>(rng: &mut R) -> ($vector, $vector) {
                const TWO_PI: $vector = $vector::splat(2.0 * $pi);

                let radius = (-2.0 * rng.gen::<$vector>().ln()).sqrte();
                let intermediate = TWO_PI * rng.gen::<$vector>();
                let sin_theta = intermediate.sin();
                let cos_theta = intermediate.cos();

                (radius * sin_theta, radius * cos_theta)
            }

            #[inline(always)]
            fn hacked_trig<R: Rng + ?Sized>(rng: &mut R) -> ($vector, $vector) {
                const TWO_PI: $vector = $vector::splat(2.0 * $pi);

                let radius = (-2.0 * rng.gen::<$vector>().ln()).sqrte();
                let (sin_theta, cos_theta) = (TWO_PI * rng.gen::<$vector>()).sin_cos();

                (radius * sin_theta, radius * cos_theta)
            }

            #[inline(always)]
            fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> $vector {
                self.flag = !self.flag;

                if !self.flag {
                    return self.z1 * self.std_dev + self.mean;
                }

                let (z0, z1) = Self::box_muller(rng);
                self.z1 = z1;
                z0 * self.std_dev + self.mean
            }

            #[inline(always)]
            fn polar<R: Rng + ?Sized>(rng: &mut R) -> ($vector, $vector) {
                let range = Uniform::new($vector::splat(-1.0), $vector::splat(1.0));

                let mut u = rng.sample(range);
                let mut v = rng.sample(range);
                let mut s = u*u + v*v;
                loop {
                    let mask = s.eq($vector::splat(0.0)) | s.ge($vector::splat(1.0));
                    if mask.none() {
                        break;
                    }
                    u = mask.select(rng.sample(range), u);
                    v = mask.select(rng.sample(range), v);
                    s = u*u + v*v;
                }

                let intermed = (-2.0 * s.ln() / s).sqrte();
                (u * intermed, v * intermed)
            }

            #[inline(always)]
            fn polar_no_rejection<R: Rng + ?Sized>(rng: &mut R) -> ($vector, $vector) {
                let range = Uniform::new($vector::splat(-1.0), $vector::splat(1.0));

                let u = rng.sample(range);
                let v = rng.sample(range);
                let s = u*u + v*v;
                // let s = u.fma(u, v*v);

                let intermed = (-2.0 * s.ln() / s).sqrte();
                (u * intermed, v * intermed)
            }
        })+
    )
}
#[cfg(feature="simd_support")]
impl_box_muller_64!(PI_64, (f64x2, u64x2), (f64x4, u64x4), (f64x8, u64x8));

/// The log-normal distribution `ln N(mean, std_dev**2)`.
///
/// This uses the [Box–Muller transform] for compatibility with SIMD PRNGs.
/// For most uses, [`LogNormal`] is a better choice.
///
/// If `X` is log-normal distributed, then `ln(X)` is `N(mean,
/// std_dev**2)` distributed.
///
/// # Example
///
/// ```rust
/// #![feature(stdsimd)]
/// use std::simd::f32x4;
///
/// use rand::distributions::{LogBoxMuller, BoxMullerCore};
/// use rand::NewRng;
/// use rand::prng::Sfc32x4Rng;
///
/// // mean 2, standard deviation 3
/// let mut log_normal = LogBoxMuller::new(f32x4::splat(2.0), f32x4::splat(3.0));
/// let v = log_normal.sample(&mut Sfc32x4Rng::new());
/// println!("{:?} is from an ln N(2, 9) distribution", v)
/// ```
///
/// [`LogNormal`]: ../normal/struct.LogNormal.html
/// [Box–Muller transform]: https://en.wikipedia.org/wiki/Box-Muller_transform
#[derive(Clone, Copy, Debug)]
#[cfg(feature="simd_support")] // necessary for doc tests?
pub struct LogBoxMuller<T>
where
    BoxMuller<T>: BoxMullerCore<T>,
{
    box_muller: BoxMuller<T>,
}

#[cfg(feature="simd_support")]
impl<T: Default + PartialOrd + SimdMath> LogBoxMuller<T>
where
    BoxMuller<T>: BoxMullerCore<T>,
{
    /// Construct a new `LogBoxMuller` distribution with the given mean
    /// and standard deviation.
    ///
    /// # Panics
    ///
    /// Panics if `std_dev < 0`.
    #[inline]
    pub fn new(mean: T, std_dev: T) -> Self {
        assert!(
            std_dev >= T::default(),
            "LogBoxMuller::new called with `std_dev` < 0"
        );
        Self {
            box_muller: BoxMuller::new(mean, std_dev),
        }
    }

    /// Generate a random value of `T`, using `rng` as the source of randomness.
    #[inline(always)]
    pub fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> T {
        self.box_muller.sample(rng).exp()
    }
}

// TODO: add explicit standard normal distr?

#[cfg(feature="simd_support")]
extern "platform-intrinsic" {
    fn simd_fsin<T>(x: T) -> T;
    fn simd_fcos<T>(x: T) -> T;
}

/// SIMD math functions not included in `feature(stdsimd)`.
#[cfg(feature="simd_support")]
pub trait SimdMath
where
    Self: Sized,
{
    /// Returns the natural logarithm of each lane of the vector.
    fn ln(self) -> Self;

    /// Simultaneously computes the sine and cosine of the vector. Returns
    /// (sin, cos).
    fn sin_cos(self) -> (Self, Self);

    /// Computes the tangent of a vector (in radians).
    fn tan(self) -> Self;

    /// Returns the largest integer less than or equal to each lane.
    fn floor(self) -> Self;

    /// Returns the square root of each lane of the vector.
    /// It should compile down to a single instruction.
    fn sqrt(self) -> Self;

    /// Returns `e^(self)`, (the exponential function).
    fn exp(self) -> Self;

    /// Raises the vector to a floating point power.
    fn powf(self, n: Self) -> Self;

    ///
    fn log_gamma(self) -> Self;
}

/// SIMD integer functions not included in `feature(stdsimd)`.
#[cfg(feature="simd_support")]
pub trait SimdIntegerMath<T> {
    /// Shifts the bits to the left by a specified amount, `n`,
    /// wrapping the truncated bits to the end of the resulting integer.
    ///
    /// Please note this isn't the same operation as `<<`!
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// Please note that this example is shared between integer types.
    /// Which explains why `u64` is used here.
    ///
    /// ```
    /// let n = u64x2::splat(0x0123456789ABCDEF);
    /// let m = 0x3456789ABCDEF012u64;
    ///
    /// assert_eq!(n.rotate_left(12), m);
    /// ```
    fn rotate_left(self, n: T) -> Self;

    /// Shifts the bits to the left by a specified amount, `n`,
    /// wrapping the truncated bits to the end of the resulting integer.
    ///
    /// Please note this isn't the same operation as `>>`!
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// Please note that this example is shared between integer types.
    /// Which explains why `u64` is used here.
    ///
    /// ```
    /// let n = u64x2::splat(0x0123456789ABCDEF);
    /// let m = 0x3456789ABCDEF012u64;
    ///
    /// assert_eq!(n.rotate_right(12), m);
    /// ```
    fn rotate_right(self, n: T) -> Self;
}

#[cfg(feature="simd_support")]
macro_rules! impl_simd_int_math {
    ($ty:ty, ($($rot_ty:ty,)+), $BITS:expr) => (
        $(impl SimdIntegerMath<$rot_ty> for $ty {
            #[inline(always)]
            fn rotate_left(self, n: $rot_ty) -> Self {
                // Protect against undefined behaviour for over-long bit shifts
                let n = n % $BITS;
                (self << n) | (self >> (($BITS - n) % $BITS))
            }

            #[inline(always)]
            fn rotate_right(self, n: $rot_ty) -> Self {
                // Protect against undefined behaviour for over-long bit shifts
                let n = n % $BITS;
                (self >> n) | (self << (($BITS - n) % $BITS))
            }
        })+
    );

    (vectors: ($($ty:ty,)+), $BITS:expr) => (
        $(impl_simd_int_math!($ty, (u32, $ty,), $BITS);)+
    )
}

#[cfg(feature="simd_support")]
impl_simd_int_math!(vectors: (u8x2, u8x4, u8x8, u8x16, u8x32, u8x64,), 8);
#[cfg(feature="simd_support")]
impl_simd_int_math!(vectors: (u16x2, u16x4, u16x8, u16x16, u16x32,), 16);
#[cfg(feature="simd_support")]
impl_simd_int_math!(vectors: (u32x2, u32x4, u32x8, u32x16,), 32);
#[cfg(feature="simd_support")]
impl_simd_int_math!(vectors: (u64x2, u64x4, u64x8,), 64);

/// Implements SIMD base 2 logarithm for use with the `zone` approximation of
/// SIMD implementations of `sample_single`.
#[cfg(feature="simd_support")]
pub trait Log {
    /// Returns the base 2 logarithm of the vector.
    fn log2(self) -> Self;
}

// adapted from https://graphics.stanford.edu/%7Eseander/bithacks.html#IntegerLogIEEE64Float
// Only works natively for `u32x` types.
//
// Casting `u16x`|`u8x` to `u32x` works but only for `2|4|8` number of lanes.
// It could work natively for smaller types if we knew how the `20` shift and
// `0x3FF` "bias" were chosen.
#[cfg(feature="simd_support")]
macro_rules! impl_log {
    ($ty:ty, $large:ident, $fty:ident, $scalar:ty, $l_scalar:ty, $f_scalar:ident, $bits:expr) => {
        impl Log for $ty {
            fn log2(self) -> Self {
                let power =
                    ((1 as $l_scalar) << (::core::$f_scalar::MANTISSA_DIGITS - 1)) as $f_scalar;
                let exponent: $l_scalar = unsafe { transmute(power) };

                let mut t = $fty::from_bits($large::from(self) | exponent);
                t -= power;
                let x = Self::from($large::from_bits(t) >> $bits);
                $bits as $scalar - 1 - ((x >> 20) - 0x3FF as $scalar)
            }
        }
    };
}

#[cfg(feature="simd_support")]
impl_log!(u32x2, u64x2, f64x2, u32, u64, f64, 32);
#[cfg(feature="simd_support")]
impl_log!(u32x4, u64x4, f64x4, u32, u64, f64, 32);
#[cfg(feature="simd_support")]
impl_log!(u32x8, u64x8, f64x8, u32, u64, f64, 32);

#[cfg(feature="simd_support")]
macro_rules! impl_log_w_u32 {
    ($( ( $ty:ident, $large:ident ), )+, $bits:expr) => (
        $(impl Log for $ty {
            #[inline(always)]
            fn log2(self) -> Self {
                $ty::from($large::from(self).log2()) - $bits
            }
        })+
    )
}

#[cfg(feature="simd_support")]
impl_log_w_u32! {
    (u16x2, u32x2),
    (u16x4, u32x4),
    (u16x8, u32x8),,
    16
}

#[cfg(feature="simd_support")]
impl_log_w_u32! {
    (u8x2, u32x2),
    (u8x4, u32x4),
    (u8x8, u32x8),,
    8
}

#[cfg(feature="simd_support")]
macro_rules! impl_log_unim {
    ($ty:ty) => {
        impl Log for $ty {
            fn log2(self) -> Self {
                unimplemented!()
            }
        }
    };
}

#[cfg(feature="simd_support")]
impl_log_unim!(u8x16);
#[cfg(feature="simd_support")]
impl_log_unim!(u8x32);
#[cfg(feature="simd_support")]
impl_log_unim!(u8x64);
#[cfg(feature="simd_support")]
impl_log_unim!(u16x16);
#[cfg(feature="simd_support")]
impl_log_unim!(u16x32);
#[cfg(feature="simd_support")]
impl_log_unim!(u32x16);
#[cfg(feature="simd_support")]
impl_log_unim!(u64x2);
#[cfg(feature="simd_support")]
impl_log_unim!(u64x4);
#[cfg(feature="simd_support")]
impl_log_unim!(u64x8);

#[cfg(feature="simd_support")]
macro_rules! impl_simd_math {
    ($fty:ident, $uty:ident, $uscalar:ty, $fscalar:ident) => (
        impl SimdMath for $fty {
            fn sin_cos(self) -> ($fty, $fty) {
                const SIGN_MASK: $uscalar = 1 << size_of::<$uscalar>() * 8 - 1;

                let mut x = self;
                /* extract the sign bit (upper one) */
                let mut sign_bit_sin = $uty::from_bits(x) & SIGN_MASK;
                /* take the absolute value */
                x = $fty::from_bits($uty::from_bits(x) & !SIGN_MASK);

                /* scale by 4/Pi */ // y= x * 4 / pi = x * (pi / 4)^-1
                let mut y = x * ::core::$fscalar::consts::FRAC_PI_4.recip();

                /* store the integer part of y in emm2 */
                let mut emm2 = $uty::from(y);

                /* j=(j+1) & (~1) (see the cephes sources) */
                emm2 += 1;
                emm2 &= !1;
                y = $fty::from(emm2);

                let mut emm4 = emm2;

                /* get the swap sign flag for the sine */
                let mut emm0 = emm2 & 4;
                emm0 <<= 29;
                let swap_sign_bit_sin = $fty::from_bits(emm0);

                /* get the polynom selection mask for the sine*/
                emm2 &= 2;
                emm2 = $uty::from_bits(emm2.eq($uty::default()));
                let poly_mask = $fty::from_bits(emm2);

                /* The magic pass: "Extended precision modular arithmetic"
                   x = ((x - y * DP1) - y * DP2) - y * DP3; */
                let mut xmm1 = $fty::splat(-0.785_156_25);
                let mut xmm2 = $fty::splat(-2.418_756_484_985_351_562_5e-4);
                let mut xmm3 = $fty::splat(-3.774_894_977_445_941_08e-8);
                xmm1 *= y;
                xmm2 *= y;
                xmm3 *= y;
                x += xmm1;
                x += xmm2;
                x += xmm3;

                emm4 -= 2;
                emm4 = !emm4 & 4;
                emm4 <<= 29;
                let sign_bit_cos = $fty::from_bits(emm4);

                sign_bit_sin ^= $uty::from_bits(swap_sign_bit_sin);


                /* Evaluate the first polynom  (0 <= x <= Pi/4) */
                let z = x * x;
                y = $fty::splat(2.443_315_711_809_948e-5);

                y *= z;
                y += -1.388_731_625_493_765e-3;
                y *= z;
                y += 4.166_664_568_298_827e-2;
                y *= z;
                y *= z;
                let tmp = z * 1.666_805_766_5e-1;
                y -= tmp;
                y += 1.0;

                /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

                let mut y2 = $fty::splat(-1.951_529_589_1e-4);
                y2 *= z;
                y2 += 8.332_160_873_6e-3;
                y2 *= z;
                y2 += -1.666_665_461_1e-1;
                y2 *= z;
                y2 *= x;
                y2 += x;

                /* select the correct result from the two polynoms */
                xmm3 = poly_mask;
                let ysin2 = $fty::from_bits($uty::from_bits(xmm3) & $uty::from_bits(y2));
                let ysin1 = $fty::from_bits(!$uty::from_bits(xmm3) & $uty::from_bits(y));
                y2 -= ysin2;
                y -= ysin1;

                xmm1 = ysin1 + ysin2;
                xmm2 = y + y2;

                /* update the sign */
                (
                    $fty::from_bits($uty::from_bits(xmm1) ^ $uty::from_bits(sign_bit_sin)),
                    $fty::from_bits($uty::from_bits(xmm2) ^ $uty::from_bits(sign_bit_cos))
                )
                // (self.sin(), self.cos())
            }

            fn tan(self) -> Self {
               let mut x = self;
               for i in 0..Self::lanes() {
                   x = x.replace(i, x.extract(i).tan());
               }
               x
            }

            fn floor(self) -> Self {
               $fty::from($uty::from(self))
            }

            fn ln(self) -> $fty {
                let mut x = self;

                let one = $fty::splat(1.0);

                let invalid_mask = x.le($fty::default());

                x = x.max($fty::from_bits($uty::splat(0x0080_0000)));

                let emm0 = ($uty::from_bits(x) >> 23) - 0x7f_u32;

                x = $fty::from_bits($uty::from_bits(x) & !0x7f80_0000);
                x = $fty::from_bits($uty::from_bits(x) | $uty::from_bits($fty::splat(0.5)));

                let mut e = $fty::from(emm0) + one;

                // must use intrinsic (available with the `stdsimd` crate tho)
                // let mask = $fty::from_bits(_mm_cmplt_ps(__m128::from_bits(x), __m128::from_bits($fty::splat(0.707106781186547524))));
                let mask = x.lt($fty::splat(0.707_106_781_186_547_524));
                let tmp = $uty::from_bits(x) & $uty::from(mask); //_mm_and_ps(x, mask);
                x -= one; // _mm_sub_ps(x, one);
                e -= $fty::from_bits($uty::from_bits(one) & $uty::from_bits(mask)); //_mm_sub_ps(e, _mm_and_ps(one, mask));
                x += $fty::from_bits(tmp); //_mm_add_ps(x, tmp);

                let z = x * x; //_mm_mul_ps(x, x);

                let mut y = $fty::splat(7.037_683_629_2e-2);
                y = y * x - 1.151_461_031_0e-1;
                y = y * x + 1.167_699_874_0e-1;
                y = y * x - 1.242_014_084_6e-1;
                y = y * x + 1.424_932_278_7e-1;
                y = y * x - 1.666_805_766_5e-1;
                y = y * x + 2.000_071_476_5e-1;
                y = y * x - 2.499_999_399_3e-1;
                y = y * x + 3.333_333_117_4e-1;
                y *= x * z;

                y += e * -2.121_944_40e-4;

                y -= z * 0.5;

                x += y;
                x += e * 0.693_359_375;
                $fty::from_bits($uty::from_bits(x) | $uty::from_bits(invalid_mask)) // negative arg will be NAN
            }

            // should compile down to a single instruction
            fn sqrt(self) -> Self {
               /*let mut x = self;
               for i in 0..Self::lanes() {
                   x = x.replace(i, x.extract(i).sqrt());
               }
               x*/
               self.sqrt()
            }

            fn exp(self) -> Self {
                let mut x = self;

                let one = $fty::splat(1.0);

                x = x.min($fty::splat(88.376_262_664_794_9));
                x = x.max($fty::splat(-88.376_262_664_794_9));

                /* express exp(x) as exp(g + n*log(2)) */
                let mut fx = x * 1.442_695_040_888_963_41;
                fx += 0.5;

                /* how to perform a floorf with SSE: just below */
                //imm0 = _mm256_cvttps_epi32(fx);
                //tmp  = _mm256_cvtepi32_ps(imm0);

                let tmp = $fty::from($uty::from(fx));

                /* if greater, substract 1 */
                //v8sf mask = _mm256_cmpgt_ps(tmp, fx);
                let mut mask = $uty::from_bits(tmp.gt(fx));
                mask &= $uty::from_bits(one);
                fx = tmp - $fty::from_bits(mask);

                let tmp = fx * 0.693_359_375;
                let mut z = fx * -2.121_944_40e-4;
                x -= tmp;
                x -= z;

                z = x * x;

                let mut y = $fty::splat(1.987_569_150_0e-4);
                y *= x;
                y += 1.398_199_950_7e-3;
                y *= x;
                y += 8.333_451_907_3e-3;
                y *= x;
                y += 4.166_579_589_4e-2;
                y *= x;
                y += 1.666_666_545_9e-1;
                y *= x;
                y += 5.000_000_120_1e-1;
                y *= z;
                y += x;
                y += one;

                /* build 2^n */
                let mut imm0 = $uty::from(fx);
                // another two AVX2 instructions
                imm0 += 0x7f;
                imm0 <<= 23;
                let pow2n = $fty::from(imm0);
                y * pow2n
            }

            fn powf(self, n: Self) -> Self {
                let mut powf = $fty::default();
                for i in 0..$fty::lanes() {
                    let n = n.extract(i);
                    powf = powf.replace(i, self.extract(i).powf(n));
                }
                powf
            }

            fn log_gamma(self) -> Self {
                // precalculated 6 coefficients for the first 6 terms of the series
                let coefficients = [
                    76.18009172947146,
                    -86.50532032941677,
                    24.01409824083091,
                    -1.231739572450155,
                    0.1208650973866179e-2,
                    -0.5395239384953e-5,
                ];

                // (x+0.5)*ln(x+g+0.5)-(x+g+0.5)
                let tmp = self + 5.5;
                let log = (self + 0.5) * tmp.ln() - tmp;

                // the first few terms of the series for Ag(x)
                let mut a = $fty::splat(1.000000000190015);
                let mut denom = self;
                for &coeff in &coefficients {
                    denom += 1.0;
                    a += coeff / denom;
                }

                // get everything together
                // a is Ag(x)
                // 2.5066... is sqrt(2pi)
                log + (2.5066282746310005 * a / self).ln()
            }
        }
    )
}

#[cfg(feature="simd_support")]
impl_simd_math!(f32x2, u32x2, u32, f32);
/*#[cfg(feature="simd_support")]
impl_simd_math!(f32x4, u32x4, u32, f32);
#[cfg(feature="simd_support")]
impl_simd_math!(f32x8, u32x8, u32, f32);*/
#[cfg(feature="simd_support")]
impl_simd_math!(f32x16, u32x16, u32, f32);

// We implement the math using SIMD intrinsics for better speed.
// TODO: implement AVX2 & SVML versions with conditional compilation
#[cfg(feature="simd_support")]
impl SimdMath for f32x4 {
    fn ln(self) -> Self {
        let one = __m128::from_bits(Self::splat(1.0));

        let mut x = __m128::from_bits(self);

        unsafe {
            let invalid_mask = _mm_cmple_ps(x, __m128::from_bits(u32x4::default()));

            x = _mm_max_ps(
                x,
                __m128::from_bits(Self::splat(::core::f32::MIN_POSITIVE)),
            );

            let mut emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);

            x = _mm_and_ps(x, __m128::from_bits(u32x4::splat(!0x7f800000)));
            x = _mm_or_ps(x, __m128::from_bits(Self::splat(0.5)));

            emm0 = _mm_sub_epi32(emm0, __m128i::from_bits(u32x4::splat(0x7f)));
            let mut e = _mm_cvtepi32_ps(emm0);

            e = _mm_add_ps(e, one);

            let mask = _mm_cmplt_ps(x, __m128::from_bits(Self::splat(0.707106781186547524)));
            let mut tmp = _mm_and_ps(x, mask);
            x = _mm_sub_ps(x, one);
            e = _mm_sub_ps(e, _mm_and_ps(one, mask));
            x = _mm_add_ps(x, tmp);

            let z = _mm_mul_ps(x, x);

            let mut y = __m128::from_bits(Self::splat(7.0376836292e-2));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(Self::splat(-1.1514610310e-1)));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(Self::splat(1.1676998740e-1)));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(Self::splat(-1.2420140846e-1)));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(Self::splat(1.4249322787e-1)));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(Self::splat(-1.6668057665e-1)));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(Self::splat(2.0000714765e-1)));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(Self::splat(-2.4999993993e-1)));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(Self::splat(3.3333331174e-1)));
            y = _mm_mul_ps(y, x);

            y = _mm_mul_ps(y, z);

            tmp = _mm_mul_ps(e, __m128::from_bits(Self::splat(-2.12194440e-4)));
            y = _mm_add_ps(y, tmp);

            tmp = _mm_mul_ps(z, __m128::from_bits(Self::splat(0.5)));
            y = _mm_sub_ps(y, tmp);

            tmp = _mm_mul_ps(e, __m128::from_bits(Self::splat(0.693359375)));
            x = _mm_add_ps(x, y);
            x = _mm_add_ps(x, tmp);
            x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN
            Self::from_bits(x)
        }
    }

    fn sin_cos(self) -> (Self, Self) {
        let mut x = __m128::from_bits(self);

        unsafe {
            /* extract the sign bit (upper one) */
            let mut sign_bit_sin = _mm_and_ps(x, __m128::from_bits(u32x4::splat(0x80000000)));
            /* take the absolute value */
            x = _mm_and_ps(x, __m128::from_bits(u32x4::splat(!0x80000000)));

            /* scale by 4/Pi */
            // y= x * 4 / pi = x * (pi / 4)^-1
            let mut y = _mm_mul_ps(
                x,
                __m128::from_bits(Self::splat(::core::f32::consts::FRAC_PI_4.recip())),
            );

            /* store the integer part of y in emm2 */
            let mut emm2 = _mm_cvttps_epi32(y);

            /* j=(j+1) & (~1) (see the cephes sources) */
            emm2 = _mm_add_epi32(emm2, __m128i::from_bits(u32x4::splat(1)));
            emm2 = _mm_and_si128(emm2, __m128i::from_bits(u32x4::splat(!1)));
            y = _mm_cvtepi32_ps(emm2);

            let mut emm4 = emm2;

            /* get the swap sign flag for the sine */
            let mut emm0 = _mm_and_si128(emm2, __m128i::from_bits(u32x4::splat(4)));
            emm0 = _mm_slli_epi32(emm0, 29);
            let swap_sign_bit_sin = _mm_castsi128_ps(emm0);

            /* get the polynom selection mask for the sine*/
            emm2 = _mm_and_si128(emm2, __m128i::from_bits(u32x4::splat(2)));
            emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
            let poly_mask = _mm_castsi128_ps(emm2);

            /* The magic pass: "Extended precision modular arithmetic"
               x = ((x - y * DP1) - y * DP2) - y * DP3; */
            let mut xmm1 = __m128::from_bits(Self::splat(-0.78515625));
            let mut xmm2 = __m128::from_bits(Self::splat(-2.4187564849853515625e-4));
            let mut xmm3 = __m128::from_bits(Self::splat(-3.77489497744594108e-8));
            xmm1 = _mm_mul_ps(y, xmm1);
            xmm2 = _mm_mul_ps(y, xmm2);
            xmm3 = _mm_mul_ps(y, xmm3);
            x = _mm_add_ps(x, xmm1);
            x = _mm_add_ps(x, xmm2);
            x = _mm_add_ps(x, xmm3);

            emm4 = _mm_sub_epi32(emm4, __m128i::from_bits(u32x4::splat(2)));
            emm4 = _mm_andnot_si128(emm4, __m128i::from_bits(u32x4::splat(4)));
            emm4 = _mm_slli_epi32(emm4, 29);
            let sign_bit_cos = _mm_castsi128_ps(emm4);

            sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);

            /* Evaluate the first polynom  (0 <= x <= Pi/4) */
            let z = _mm_mul_ps(x, x);
            y = __m128::from_bits(Self::splat(2.443315711809948e-5));

            y = _mm_mul_ps(y, z);
            y = _mm_add_ps(y, __m128::from_bits(Self::splat(-1.388731625493765e-3)));
            y = _mm_mul_ps(y, z);
            y = _mm_add_ps(y, __m128::from_bits(Self::splat(4.166664568298827e-2)));
            y = _mm_mul_ps(y, z);
            y = _mm_mul_ps(y, z);
            let tmp = _mm_mul_ps(z, __m128::from_bits(Self::splat(1.6668057665e-1)));
            y = _mm_sub_ps(y, tmp);
            y = _mm_add_ps(y, __m128::from_bits(Self::splat(1.0)));

            /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

            let mut y2 = __m128::from_bits(Self::splat(-1.9515295891e-4));
            y2 = _mm_mul_ps(y2, z);
            y2 = _mm_add_ps(y2, __m128::from_bits(Self::splat(8.3321608736e-3)));
            y2 = _mm_mul_ps(y2, z);
            y2 = _mm_add_ps(y2, __m128::from_bits(Self::splat(-1.6666654611e-1)));
            y2 = _mm_mul_ps(y2, z);
            y2 = _mm_mul_ps(y2, x);
            y2 = _mm_add_ps(y2, x);

            /* select the correct result from the two polynoms */
            xmm3 = poly_mask;
            let ysin2 = _mm_and_ps(xmm3, y2);
            let ysin1 = _mm_andnot_ps(xmm3, y);
            y2 = _mm_sub_ps(y2, ysin2);
            y = _mm_sub_ps(y, ysin1);

            xmm1 = _mm_add_ps(ysin1, ysin2);
            xmm2 = _mm_add_ps(y, y2);

            /* update the sign */
            (
                Self::from_bits(_mm_xor_ps(xmm1, sign_bit_sin)),
                Self::from_bits(_mm_xor_ps(xmm2, sign_bit_cos)),
            )
        }
        // (self.sin(), self.cos())
    }

    fn tan(self) -> Self {
        let mut x = self;
        for i in 0..Self::lanes() {
            x = x.replace(i, x.extract(i).tan());
        }
        x
    }

    fn floor(self) -> Self {
        Self::from(u32x4::from(self))
    }

    fn sqrt(self) -> Self {
        /*let mut x = self;
        for i in 0..Self::lanes() {
            x = x.replace(i, x.extract(i).sqrt());
        }
        x*/
        self.sqrt()
    }

    fn exp(self) -> Self {
        let mut x = self;
        for i in 0..Self::lanes() {
            x = x.replace(i, x.extract(i).exp());
        }
        x
    }

    fn powf(self, n: Self) -> Self {
        let mut powf = Self::default();
        for i in 0..Self::lanes() {
            let n = n.extract(i);
            powf = powf.replace(i, self.extract(i).powf(n));
        }
        powf
    }

    fn log_gamma(self) -> Self {
        // precalculated 6 coefficients for the first 6 terms of the series
        let coefficients = [
            76.18009172947146,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.1208650973866179e-2,
            -0.5395239384953e-5,
        ];

        // (x+0.5)*ln(x+g+0.5)-(x+g+0.5)
        let tmp = self + 5.5;
        let log = (self + 0.5) * tmp.ln() - tmp;

        // the first few terms of the series for Ag(x)
        let mut a = Self::splat(1.000000000190015);
        let mut denom = self;
        for &coeff in &coefficients {
            denom += 1.0;
            a += coeff / denom;
        }

        // get everything together
        // a is Ag(x)
        // 2.5066... is sqrt(2pi)
        log + (2.5066282746310005 * a / self).ln()
    }
}

#[cfg(feature="simd_support")]
impl SimdMath for f32x8 {
    fn sin_cos(self) -> (Self, Self) {
        let minus_cephes_dp1 = __m256::from_bits(f32x8::splat(-0.78515625));
        let minus_cephes_dp2 = __m256::from_bits(f32x8::splat(-2.4187564849853515625e-4));
        let minus_cephes_dp3 = __m256::from_bits(f32x8::splat(-3.77489497744594108e-8));

        let sincof_p0 = __m256::from_bits(f32x8::splat(-1.9515295891e-4));
        let sincof_p1 = __m256::from_bits(f32x8::splat(8.3321608736e-3));
        let sincof_p2 = __m256::from_bits(f32x8::splat(-1.6666654611e-1));
        let coscof_p0 = __m256::from_bits(f32x8::splat(2.443315711809948e-5));
        let coscof_p1 = __m256::from_bits(f32x8::splat(-1.388731625493765e-3));
        let coscof_p2 = __m256::from_bits(f32x8::splat(4.166664568298827e-2));

        let mut xmm1: __m256; // = _mm256_setzero_ps();
        let mut xmm2: __m256; // = _mm256_setzero_ps();
        let mut xmm3: __m256; // = _mm256_setzero_ps();
        let mut sign_bit_sin: __m256;
        let mut y: __m256;
        // let v8si: __m256;
        let imm0: __m256i;
        let mut imm2: __m256i;
        let imm4: __m256i;

        // #ifndef __AVX2__
        let mut imm0_1: __m128i;
        let mut imm0_2: __m128i;
        let mut imm2_1: __m128i;
        let mut imm2_2: __m128i;
        let mut imm4_1: __m128i;
        let mut imm4_2: __m128i;
        // #endif

        let mut x = __m256::from_bits(self);

        sign_bit_sin = x;

        unsafe {
            /* take the absolute value */
            x = _mm256_and_ps(x, __m256::from_bits(u32x8::splat(!0x80000000)));
            /* extract the sign bit (upper one) */
            sign_bit_sin = _mm256_and_ps(sign_bit_sin, __m256::from_bits(u32x8::splat(0x80000000)));

            /* scale by 4/Pi */
            y = _mm256_mul_ps(
                x,
                __m256::from_bits(f32x8::splat(::core::f32::consts::FRAC_PI_4.recip())),
            );

            /*#ifdef __AVX2__
                /* store the integer part of y in imm2 */
                imm2 = _mm256_cvttps_epi32(y);

                /* j=(j+1) & (~1) (see the cephes sources) */
                imm2 = _mm256_add_epi32(imm2, *(v8si*)_pi32_256_1);
                imm2 = _mm256_and_si128(imm2, *(v8si*)_pi32_256_inv1);

                y = _mm256_cvtepi32_ps(imm2);
                imm4 = imm2;

                /* get the swap sign flag for the sine */
                imm0 = _mm256_and_si128(imm2, *(v8si*)_pi32_256_4);
                imm0 = _mm256_slli_epi32(imm0, 29);
                //v8sf swap_sign_bit_sin = _mm256_castsi256_ps(imm0);

                /* get the polynom selection mask for the sine*/
                imm2 = _mm256_and_si128(imm2, *(v8si*)_pi32_256_2);
                imm2 = _mm256_cmpeq_epi32(imm2, *(v8si*)_pi32_256_0);
                //v8sf poly_mask = _mm256_castsi256_ps(imm2);
            #else*/
            /* we use SSE2 routines to perform the integer ops */
            // COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y), imm2_1, imm2_2);
            let (a, b): (__m128i, __m128i) = transmute(_mm256_cvttps_epi32(y));
            imm2_1 = a;
            imm2_2 = b;

            imm2_1 = _mm_add_epi32(imm2_1, __m128i::from_bits(i32x4::splat(1)));
            imm2_2 = _mm_add_epi32(imm2_2, __m128i::from_bits(i32x4::splat(1)));

            imm2_1 = _mm_and_si128(imm2_1, __m128i::from_bits(i32x4::splat(!1)));
            imm2_2 = _mm_and_si128(imm2_2, __m128i::from_bits(i32x4::splat(!1)));

            // COPY_XMM_TO_IMM(imm2_1,imm2_2,imm2);
            imm2 = transmute((imm2_1, imm2_2));
            y = _mm256_cvtepi32_ps(imm2);

            imm4_1 = imm2_1;
            imm4_2 = imm2_2;

            imm0_1 = _mm_and_si128(imm2_1, __m128i::from_bits(i32x4::splat(4)));
            imm0_2 = _mm_and_si128(imm2_2, __m128i::from_bits(i32x4::splat(4)));

            imm0_1 = _mm_slli_epi32(imm0_1, 29);
            imm0_2 = _mm_slli_epi32(imm0_2, 29);

            // COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);
            imm0 = transmute((imm0_1, imm0_2));

            imm2_1 = _mm_and_si128(imm2_1, __m128i::from_bits(i32x4::splat(2)));
            imm2_2 = _mm_and_si128(imm2_2, __m128i::from_bits(i32x4::splat(2)));

            imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
            imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

            // COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
            imm2 = transmute((imm2_1, imm2_2));
            // #endif
            let swap_sign_bit_sin: __m256 = _mm256_castsi256_ps(imm0);
            let poly_mask: __m256 = _mm256_castsi256_ps(imm2);

            /* The magic pass: "Extended precision modular arithmetic"
             x = ((x - y * DP1) - y * DP2) - y * DP3; */
            xmm1 = minus_cephes_dp1;
            xmm2 = minus_cephes_dp2;
            xmm3 = minus_cephes_dp3;
            xmm1 = _mm256_mul_ps(y, xmm1);
            xmm2 = _mm256_mul_ps(y, xmm2);
            xmm3 = _mm256_mul_ps(y, xmm3);
            x = _mm256_add_ps(x, xmm1);
            x = _mm256_add_ps(x, xmm2);
            x = _mm256_add_ps(x, xmm3);

            /*#ifdef __AVX2__
            imm4 = _mm256_sub_epi32(imm4, *(v8si*)_pi32_256_2);
            imm4 = _mm256_andnot_si128(imm4, *(v8si*)_pi32_256_4);
            imm4 = _mm256_slli_epi32(imm4, 29);
            #else*/
            imm4_1 = _mm_sub_epi32(imm4_1, __m128i::from_bits(i32x4::splat(2)));
            imm4_2 = _mm_sub_epi32(imm4_2, __m128i::from_bits(i32x4::splat(2)));

            imm4_1 = _mm_andnot_si128(imm4_1, __m128i::from_bits(i32x4::splat(4)));
            imm4_2 = _mm_andnot_si128(imm4_2, __m128i::from_bits(i32x4::splat(4)));

            imm4_1 = _mm_slli_epi32(imm4_1, 29);
            imm4_2 = _mm_slli_epi32(imm4_2, 29);

            // COPY_XMM_TO_IMM(imm4_1, imm4_2, imm4);
            imm4 = transmute((imm4_1, imm4_2));
            // #endif

            let sign_bit_cos: __m256 = _mm256_castsi256_ps(imm4);

            sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

            /* Evaluate the first polynom  (0 <= x <= Pi/4) */
            let z: __m256 = _mm256_mul_ps(x, x);
            y = coscof_p0;

            y = _mm256_mul_ps(y, z);
            y = _mm256_add_ps(y, coscof_p1);
            y = _mm256_mul_ps(y, z);
            y = _mm256_add_ps(y, coscof_p2);
            y = _mm256_mul_ps(y, z);
            y = _mm256_mul_ps(y, z);
            let tmp: __m256 = _mm256_mul_ps(z, __m256::from_bits(f32x8::splat(0.5)));
            y = _mm256_sub_ps(y, tmp);
            y = _mm256_add_ps(y, __m256::from_bits(f32x8::splat(1.0)));

            /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

            let mut y2: __m256 = sincof_p0;
            y2 = _mm256_mul_ps(y2, z);
            y2 = _mm256_add_ps(y2, sincof_p1);
            y2 = _mm256_mul_ps(y2, z);
            y2 = _mm256_add_ps(y2, sincof_p2);
            y2 = _mm256_mul_ps(y2, z);
            y2 = _mm256_mul_ps(y2, x);
            y2 = _mm256_add_ps(y2, x);

            /* select the correct result from the two polynoms */
            xmm3 = poly_mask;
            let ysin2: __m256 = _mm256_and_ps(xmm3, y2);
            let ysin1: __m256 = _mm256_andnot_ps(xmm3, y);
            y2 = _mm256_sub_ps(y2, ysin2);
            y = _mm256_sub_ps(y, ysin1);

            xmm1 = _mm256_add_ps(ysin1, ysin2);
            xmm2 = _mm256_add_ps(y, y2);

            /* update the sign */
            (
                f32x8::from_bits(_mm256_xor_ps(xmm1, sign_bit_sin)),
                f32x8::from_bits(_mm256_xor_ps(xmm2, sign_bit_cos)),
            )
        }
        // (self.sin(), self.cos())
    }

    fn tan(self) -> Self {
        let mut x = self;
        for i in 0..Self::lanes() {
            x = x.replace(i, x.extract(i).tan());
        }
        x
    }

    fn floor(self) -> Self {
        Self::from(u32x8::from(self))
    }

    fn ln(self) -> Self {
        let mut x = __m256::from_bits(self);
        let mut imm0: __m256i;
        let one: __m256 = __m256::from_bits(Self::splat(1.0));
        unsafe {
            //__m256 invalid_mask = _mm256_cmple_ps(x, _mm256_setzero_ps());
            // __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);
            let invalid_mask: __m256 = _mm256_cmp_ps(x, _mm256_setzero_ps(), 2);

            /* cut off denormalized stuff */
            x = _mm256_max_ps(x, __m256::from_bits(u32x8::splat(0x00800000)));

            // can be done with AVX2
          // imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
          /*let temp = _mm256_castps_si256(x);
          let (mut t1, mut t2): (__m128i, __m128i) = transmute(temp);

          t1 = _mm_srli_epi32(t1, 23);
          t2 = _mm_srli_epi32(t2, 23);

          imm0 = transmute((t1, t2));*/
            imm0 = __m256i::from_bits(u32x8::from_bits(_mm256_castps_si256(x)) >> 23);

            /* keep only the fractional part */
            x = _mm256_and_ps(x, __m256::from_bits(u32x8::splat(!0x7f800000)));
            x = _mm256_or_ps(x, __m256::from_bits(Self::splat(0.5)));

            // this is again another AVX2 instruction
          // imm0 = _mm256_sub_epi32(imm0, __m256i::from_bits(u32x8::splat(0x7f)));
          /*let (mut t1, mut t2): (__m128i, __m128i) = transmute(imm0);

          t1 = _mm_sub_epi32(t1, __m128i::from_bits(u32x4::splat(0x7f)));
          t2 = _mm_sub_epi32(t2, __m128i::from_bits(u32x4::splat(0x7f)));

          imm0 = transmute((t1, t2));*/
            imm0 = __m256i::from_bits(u32x8::from_bits(imm0) - u32x8::splat(0x7f));

            let mut e: __m256 = _mm256_cvtepi32_ps(imm0);

            e = _mm256_add_ps(e, one);

            /* part2:
         if( x < SQRTHF ) {
           e -= 1;
           x = x + x - 1.0;
         } else { x = x - 1.0; }
      */
            //__m256 mask = _mm256_cmplt_ps(x, *(__m256*)_ps256_cephes_SQRTHF);
            let mask: __m256 =
                _mm256_cmp_ps(x, __m256::from_bits(Self::splat(0.707106781186547524)), 1);
            let mut tmp: __m256 = _mm256_and_ps(x, mask);
            x = _mm256_sub_ps(x, one);
            e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
            x = _mm256_add_ps(x, tmp);

            let z: __m256 = _mm256_mul_ps(x, x);

            let mut y: __m256 = __m256::from_bits(Self::splat(7.0376836292E-2));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(Self::splat(-1.1514610310E-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(Self::splat(1.1676998740E-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(Self::splat(-1.2420140846E-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(Self::splat(1.4249322787E-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(Self::splat(-1.6668057665E-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(Self::splat(2.0000714765E-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(Self::splat(-2.4999993993E-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(Self::splat(3.3333331174E-1)));
            y = _mm256_mul_ps(y, x);

            y = _mm256_mul_ps(y, z);

            tmp = _mm256_mul_ps(e, __m256::from_bits(Self::splat(-2.12194440e-4)));
            y = _mm256_add_ps(y, tmp);

            tmp = _mm256_mul_ps(z, __m256::from_bits(Self::splat(0.5)));
            y = _mm256_sub_ps(y, tmp);

            tmp = _mm256_mul_ps(e, __m256::from_bits(Self::splat(0.693359375)));
            x = _mm256_add_ps(x, y);
            x = _mm256_add_ps(x, tmp);
            x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
            Self::from_bits(x)
        }
    }

    fn sqrt(self) -> Self {
        /*let mut x = self;
        for i in 0..Self::lanes() {
            x = x.replace(i, x.extract(i).sqrt());
        }
        x*/
        self.sqrt()
    }

    fn exp(self) -> Self {
        let mut x = __m256::from_bits(self);
        let one: __m256 = __m256::from_bits(Self::splat(1.0));

        unsafe {
            x = _mm256_min_ps(x, __m256::from_bits(Self::splat(88.3762626647949)));
            x = _mm256_max_ps(x, __m256::from_bits(Self::splat(-88.3762626647949)));

            /* express exp(x) as exp(g + n*log(2)) */
            let mut fx = _mm256_mul_ps(x, __m256::from_bits(Self::splat(1.44269504088896341)));
            fx = _mm256_add_ps(fx, __m256::from_bits(Self::splat(0.5)));

            /* how to perform a floorf with SSE: just below */
            let imm0 = _mm256_cvttps_epi32(fx);
            let tmp = _mm256_cvtepi32_ps(imm0);

            // let tmp = _mm256_floor_ps(fx);

            /* if greater, substract 1 */
            //v8sf mask = _mm256_cmpgt_ps(tmp, fx);
            let mut mask: __m256 = _mm256_cmp_ps(tmp, fx, 14);
            mask = _mm256_and_ps(mask, one);
            fx = _mm256_sub_ps(tmp, mask);

            let tmp = _mm256_mul_ps(fx, __m256::from_bits(Self::splat(0.693359375)));
            let mut z: __m256 = _mm256_mul_ps(fx, __m256::from_bits(Self::splat(-2.12194440e-4)));
            x = _mm256_sub_ps(x, tmp);
            x = _mm256_sub_ps(x, z);

            z = _mm256_mul_ps(x, x);

            let mut y: __m256 = __m256::from_bits(Self::splat(1.9875691500e-4));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(Self::splat(1.3981999507e-3)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(Self::splat(8.3334519073e-3)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(Self::splat(4.1665795894e-2)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(Self::splat(1.6666665459e-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(Self::splat(5.0000001201e-1)));
            y = _mm256_mul_ps(y, z);
            y = _mm256_add_ps(y, x);
            y = _mm256_add_ps(y, one);

            /* build 2^n */
            let mut imm0 = _mm256_cvttps_epi32(fx);

            // another two AVX2 instructions
            /*imm0 = _mm256_add_epi32(imm0, __m256i::from_bits(u32x8::splat(0x7f)));
            imm0 = _mm256_slli_epi32(imm0, 23);*/
            // impl `(x + 0x7f) << 32` without avx2...
                // ...with intrinsics
                /*let (mut a, mut b): (__m128i, __m128i) = transmute(imm0);
                a = _mm_add_epi32(a, __m128i::from_bits(u32x4::splat(0x7f)));
                b = _mm_add_epi32(b, __m128i::from_bits(u32x4::splat(0x7f)));
                a = _mm_slli_epi32(a, 23);
                b = _mm_slli_epi32(b, 23);
                imm0 = transmute((a, b));*/
            // ...or with abstraction
            imm0 = __m256i::from_bits(u32x8::from_bits(imm0) + u32x8::splat(0x7f));
            imm0 = __m256i::from_bits(u32x8::from_bits(imm0) << 23);

            let pow2n: __m256 = _mm256_castsi256_ps(imm0);
            y = _mm256_mul_ps(y, pow2n);
            Self::from_bits(y)
        }
    }

    fn powf(self, n: Self) -> Self {
        let mut powf = Self::default();
        for i in 0..Self::lanes() {
            let n = n.extract(i);
            powf = powf.replace(i, self.extract(i).powf(n));
        }
        powf
    }

    fn log_gamma(self) -> Self {
        // precalculated 6 coefficients for the first 6 terms of the series
        let coefficients = [
            76.18009172947146,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.1208650973866179e-2,
            -0.5395239384953e-5,
        ];

        // (x+0.5)*ln(x+g+0.5)-(x+g+0.5)
        let tmp = self + 5.5;
        let log = (self + 0.5) * tmp.ln() - tmp;

        // the first few terms of the series for Ag(x)
        let mut a = Self::splat(1.000000000190015);
        let mut denom = self;
        for &coeff in &coefficients {
            denom += 1.0;
            a += coeff / denom;
        }

        // get everything together
        // a is Ag(x)
        // 2.5066... is sqrt(2pi)
        log + (2.5066282746310005 * a / self).ln()
    }
}

#[cfg(feature="simd_support")]
macro_rules! impl_simd_math_f64 {
    ($fty:ident, $uty:ident, $uscalar:ty, $fscalar:ident, $intrinsic:ident, $round:ident) => {
        impl SimdMath for $fty {
            fn sin_cos(self) -> ($fty, $fty) {
                /*let mut sin = $fty::default();
                let mut cos = $fty::default();
                for i in 0..$fty::lanes() {
                    let (_sin, _cos) = self.extract(i).sin_cos();
                    sin = sin.replace(i, _sin);
                    cos = cos.replace(i, _cos);
                }
                (sin, cos)*/
                (self.sin(), self.cos())
            }

            fn tan(self) -> Self {
                let mut x = self;

                let z = x * ::core::f64::consts::FRAC_PI_2.recip();
                let y = unsafe { $fty::from_bits($round($intrinsic::from_bits(z), 0)) };
                let m = ($uty::from(z) & 1).eq($uty::splat(1));

                #[cfg(target_feature = "fma")]
                {
                    let mut x_vec = Self::default();
                    for i in 0..Self::lanes() {
                        let mut x = x.extract(i);
                        unsafe {
                            x = fmaf64(y, -2.0 * 0.78539816290140151978, x);
                            x = fmaf64(y, -2.0 * 4.9604678871439933374e-10, x);
                            x = fmaf64(y, -2.0 * 1.1258708853173288931e-18, x);
                            x = fmaf64(y, -2.0 * 1.7607799325916000908e-27, x);
                        }
                        x_vec = x_vec.replace(i, p);
                    }

                    x = m.select($fty::splat(-0.0), x_vec);

                    let x2 = x * x;

                    // force the compiler to use FMA intructions
                    let mut p_vec = Self::default();
                    for i in 0..Self::lanes() {
                        let x2 = x2.extract(i);

                        let mut p;
                        unsafe {
                            p = fmaf64(
                                1.01419718511083373224408e-5,
                                x2,
                                -2.59519791585924697698614e-5,
                            );
                            p = fmaf64(p, x2, 5.23388081915899855325186e-5);
                            p = fmaf64(p, x2, -3.05033014433946488225616e-5);
                            p = fmaf64(p, x2, 7.14707504084242744267497e-5);
                            p = fmaf64(p, x2, 8.09674518280159187045078e-5);
                            p = fmaf64(p, x2, 0.000244884931879331847054404);
                            p = fmaf64(p, x2, 0.000588505168743587154904506);
                            p = fmaf64(p, x2, 0.00145612788922812427978848);
                            p = fmaf64(p, x2, 0.00359208743836906619142924);
                            p = fmaf64(p, x2, 0.00886323944362401618113356);
                            p = fmaf64(p, x2, 0.0218694882853846389592078);
                            p = fmaf64(p, x2, 0.0539682539781298417636002);
                            p = fmaf64(p, x2, 0.133333333333125941821962);
                            p = fmaf64(p, x2, 0.333333333333334980164153);
                        }

                        p_vec = p_vec.replace(i, p);
                    }

                    let p = x2 * p_vec * x + x;
                    return m.select(p, 1.0 / p);
                }

                x += y * (-2.0 * 0.78539816290140151978);
                x += y * (-2.0 * 4.9604678871439933374e-10);
                x += y * (-2.0 * 1.1258708853173288931e-18);
                x += y * (-2.0 * 1.7607799325916000908e-27);

                x = m.select($fty::splat(-0.0), x);

                let x2 = x * x;

                let mut p;
                p = 1.01419718511083373224408e-5 * x2 + -2.59519791585924697698614e-5;
                p = p * x2 + 5.23388081915899855325186e-5;
                p = p * x2 + -3.05033014433946488225616e-5;
                p = p * x2 + 7.14707504084242744267497e-5;
                p = p * x2 + 8.09674518280159187045078e-5;
                p = p * x2 + 0.000244884931879331847054404;
                p = p * x2 + 0.000588505168743587154904506;
                p = p * x2 + 0.00145612788922812427978848;
                p = p * x2 + 0.00359208743836906619142924;
                p = p * x2 + 0.00886323944362401618113356;
                p = p * x2 + 0.0218694882853846389592078;
                p = p * x2 + 0.0539682539781298417636002;
                p = p * x2 + 0.133333333333125941821962;
                p = p * x2 + 0.333333333333334980164153;
                p = x2 * p * x + x;

                m.select(p, 1.0 / p)
            }

            fn floor(self) -> Self {
                $fty::from($uty::from(self))
            }

            fn ln(self) -> $fty {
                let mut ln = $fty::default();
                for i in 0..$fty::lanes() {
                    ln = ln.replace(i, self.extract(i).ln());
                }
                ln
            }

            // should compile down to a single instruction
            fn sqrt(self) -> Self {
                /*let mut x = self;
                for i in 0..Self::lanes() {
                    x = x.replace(i, x.extract(i).sqrt());
                }
                x*/
                self.sqrt()
            }

            fn exp(self) -> Self {
                let mut exp = $fty::default();
                for i in 0..$fty::lanes() {
                    exp = exp.replace(i, self.extract(i).exp());
                }
                exp
            }

            fn powf(self, n: Self) -> Self {
                let mut powf = $fty::default();
                for i in 0..$fty::lanes() {
                    let n = n.extract(i);
                    powf = powf.replace(i, self.extract(i).powf(n));
                }
                powf
            }

            fn log_gamma(self) -> Self {
                // precalculated 6 coefficients for the first 6 terms of the series
                let coefficients = [
                    76.18009172947146,
                    -86.50532032941677,
                    24.01409824083091,
                    -1.231739572450155,
                    0.1208650973866179e-2,
                    -0.5395239384953e-5,
                ];

                // (x+0.5)*ln(x+g+0.5)-(x+g+0.5)
                let tmp = self + 5.5;
                let log = (self + 0.5) * tmp.ln() - tmp;

                // the first few terms of the series for Ag(x)
                let mut a = Self::splat(1.000000000190015);
                let mut denom = self;
                for &coeff in &coefficients {
                    denom += 1.0;
                    a += coeff / denom;
                }

                // get everything together
                // a is Ag(x)
                // 2.5066... is sqrt(2pi)
                log + (2.5066282746310005 * a / self).ln()
            }
        }
    };
}

#[cfg(feature="simd_support")]
impl_simd_math_f64!(f64x2, u64x2, u64, f64, __m128d, _mm_round_pd);
#[cfg(feature="simd_support")]
impl_simd_math_f64!(f64x4, u64x4, u64, f64, __m256d, _mm256_round_pd);
/*#[cfg(feature="simd_support")]
impl_simd_math_f64!(f64x8, u64x8, u64, f64, __m512d, _mm512_round_pd);*/

#[cfg(feature="simd_support")]
impl SimdMath for f64x8 {
    fn sin_cos(self) -> (Self, Self) {
        /*let mut sin = Self::default();
        let mut cos = Self::default();
        for i in 0..Self::lanes() {
            let (_sin, _cos) = self.extract(i).sin_cos();
            sin = sin.replace(i, _sin);
            cos = cos.replace(i, _cos);
        }
        (sin, cos)*/
        (self.sin(), self.cos())
    }

    fn tan(self) -> Self {
        let mut x = self;

        let z = x * ::core::f64::consts::FRAC_PI_2.recip();
        let ceil = |x: Self| -(-x).floor();
        let y = (ceil(2.0 * z) / 2.0).floor();
        let m = (u64x8::from(z) & 1).eq(u64x8::splat(1));

        #[cfg(target_feature = "fma")]
        {
            // We have no FMA intrinsics, so we force the compiler to do it
            // with the scalar version.
            let mut x_vec = Self::default();
            for i in 0..Self::lanes() {
                let mut x = x.extract(i);
                unsafe {
                    x = fmaf64(y, -2.0 * 0.78539816290140151978, x);
                    x = fmaf64(y, -2.0 * 4.9604678871439933374e-10, x);
                    x = fmaf64(y, -2.0 * 1.1258708853173288931e-18, x);
                    x = fmaf64(y, -2.0 * 1.7607799325916000908e-27, x);
                }
                x_vec = x_vec.replace(i, p);
            }

            x = m.select(Self::splat(-0.0), x_vec);

            let x2 = x * x;

            // force the compiler to use FMA intructions
            let mut p_vec = Self::default();
            for i in 0..Self::lanes() {
                let x2 = x2.extract(i);

                let mut p;
                unsafe {
                    p = fmaf64(
                        1.01419718511083373224408e-5,
                        x2,
                        -2.59519791585924697698614e-5,
                    );
                    p = fmaf64(p, x2, 5.23388081915899855325186e-5);
                    p = fmaf64(p, x2, -3.05033014433946488225616e-5);
                    p = fmaf64(p, x2, 7.14707504084242744267497e-5);
                    p = fmaf64(p, x2, 8.09674518280159187045078e-5);
                    p = fmaf64(p, x2, 0.000244884931879331847054404);
                    p = fmaf64(p, x2, 0.000588505168743587154904506);
                    p = fmaf64(p, x2, 0.00145612788922812427978848);
                    p = fmaf64(p, x2, 0.00359208743836906619142924);
                    p = fmaf64(p, x2, 0.00886323944362401618113356);
                    p = fmaf64(p, x2, 0.0218694882853846389592078);
                    p = fmaf64(p, x2, 0.0539682539781298417636002);
                    p = fmaf64(p, x2, 0.133333333333125941821962);
                    p = fmaf64(p, x2, 0.333333333333334980164153);
                }

                p_vec = p_vec.replace(i, p);
            }

            let p = x2 * p_vec * x + x;
            return m.select(p, 1.0 / p);
        }

        x += y * (-2.0 * 0.78539816290140151978);
        x += y * (-2.0 * 4.9604678871439933374e-10);
        x += y * (-2.0 * 1.1258708853173288931e-18);
        x += y * (-2.0 * 1.7607799325916000908e-27);

        x = m.select(Self::splat(-0.0), x);

        let x2 = x * x;

        let mut p;
        p = 1.01419718511083373224408e-5 * x2 + -2.59519791585924697698614e-5;
        p = p * x2 + 5.23388081915899855325186e-5;
        p = p * x2 + -3.05033014433946488225616e-5;
        p = p * x2 + 7.14707504084242744267497e-5;
        p = p * x2 + 8.09674518280159187045078e-5;
        p = p * x2 + 0.000244884931879331847054404;
        p = p * x2 + 0.000588505168743587154904506;
        p = p * x2 + 0.00145612788922812427978848;
        p = p * x2 + 0.00359208743836906619142924;
        p = p * x2 + 0.00886323944362401618113356;
        p = p * x2 + 0.0218694882853846389592078;
        p = p * x2 + 0.0539682539781298417636002;
        p = p * x2 + 0.133333333333125941821962;
        p = p * x2 + 0.333333333333334980164153;
        p = x2 * p * x + x;

        m.select(p, 1.0 / p)
    }

    fn floor(self) -> Self {
        Self::from(u64x8::from(self))
    }

    fn ln(self) -> Self {
        let mut ln = Self::default();
        for i in 0..Self::lanes() {
            ln = ln.replace(i, self.extract(i).ln());
        }
        ln
    }

    // should compile down to a single instruction
    fn sqrt(self) -> Self {
        /*let mut x = self;
        for i in 0..Self::lanes() {
            x = x.replace(i, x.extract(i).sqrt());
        }
        x*/
        self.sqrt()
    }

    fn exp(self) -> Self {
        let mut exp = Self::default();
        for i in 0..Self::lanes() {
            exp = exp.replace(i, self.extract(i).exp());
        }
        exp
    }

    fn powf(self, n: Self) -> Self {
        let mut powf = Self::default();
        for i in 0..Self::lanes() {
            let n = n.extract(i);
            powf = powf.replace(i, self.extract(i).powf(n));
        }
        powf
    }

    fn log_gamma(self) -> Self {
        // precalculated 6 coefficients for the first 6 terms of the series
        let coefficients = [
            76.18009172947146,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.1208650973866179e-2,
            -0.5395239384953e-5,
        ];

        // (x+0.5)*ln(x+g+0.5)-(x+g+0.5)
        let tmp = self + 5.5;
        let log = (self + 0.5) * tmp.ln() - tmp;

        // the first few terms of the series for Ag(x)
        let mut a = Self::splat(1.000000000190015);
        let mut denom = self;
        for &coeff in &coefficients {
            denom += 1.0;
            a += coeff / denom;
        }

        // get everything together
        // a is Ag(x)
        // 2.5066... is sqrt(2pi)
        log + (2.5066282746310005 * a / self).ln()
    }
}

#[cfg(all(test, feature="simd_support"))]
mod tests {
    use super::*;
    use prng::Sfc32x4Rng;
    use SeedableRng;

    const TEST_N: usize = 1 << 10;

    macro_rules! make_log_test {
        ($test_name:ident, $ty:ident) => {
            #[test]
            fn $test_name() {
                let mut rng = Sfc32x4Rng::from_rng(&mut ::thread_rng()).unwrap();
                for _ in 0..TEST_N {
                    let num = rng.gen::<$ty>();
                    let actual = num.ln();

                    for i in 0..$ty::lanes() {
                        let expected = num.extract(i).ln();
                        let actual = actual.extract(i);
                        assert!((expected - actual) < 1e-6, "\n{:?}\n{:?}", expected, actual);
                    }
                }
            }
        };
    }

    make_log_test!(log_f32x2, f32x2);
    make_log_test!(log_f32x4, f32x4);
    make_log_test!(log_f32x8, f32x8);
    make_log_test!(log_f32x16, f32x16);

    macro_rules! make_sincos_test {
        ($test_name:ident, $ty:ident) => {
            #[test]
            fn $test_name() {
                let mut rng = Sfc32x4Rng::from_rng(&mut ::thread_rng()).unwrap();
                for _ in 0..TEST_N {
                    let num = rng.gen::<$ty>();
                    let (actual_sin, actual_cos) = num.sin_cos();

                    for i in 0..$ty::lanes() {
                        let (expected_sin, expected_cos) = num.extract(i).sin_cos();

                        let actual_sin = actual_sin.extract(i);
                        let actual_cos = actual_cos.extract(i);

                        assert!(
                            (expected_sin - actual_sin) < 5e-7,
                            "\n{:?}\n{:?}",
                            expected_sin,
                            actual_sin
                        );
                        assert!(
                            (expected_cos - actual_cos) < 5e-7,
                            "\n{:?}\n{:?}",
                            expected_cos,
                            actual_cos
                        );
                    }
                }
            }
        };
    }

    make_sincos_test!(sincos_f32x2, f32x2);
    make_sincos_test!(sincos_f32x4, f32x4);
    make_sincos_test!(sincos_f32x8, f32x8);
    make_sincos_test!(sincos_f32x16, f32x16);

    macro_rules! make_exp_test {
        ($test_name:ident, $ty:ident) => {
            #[test]
            fn $test_name() {
                let mut rng = Sfc32x4Rng::from_rng(&mut ::thread_rng()).unwrap();
                for _ in 0..TEST_N {
                    let num = rng.gen::<$ty>();
                    let actual = num.exp();

                    for i in 0..$ty::lanes() {
                        let expected = num.extract(i).exp();
                        let actual = actual.extract(i);

                        assert!((expected - actual) < 5e-7, "\n{:?}\n{:?}", expected, actual);
                    }
                }
            }
        };
    }

    make_exp_test!(exp_f32x2, f32x2);
    make_exp_test!(exp_f32x4, f32x4);
    make_exp_test!(exp_f32x8, f32x8);
    make_exp_test!(exp_f32x16, f32x16);
}
