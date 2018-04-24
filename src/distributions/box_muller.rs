//! The Box-Muller transform and derived distributions, for use with SIMD PRNGs.
//!
//! <https://en.wikipedia.org/wiki/Box-Muller_transform>

#[cfg(feature="simd_support")]
use core::simd::*;
#[cfg(feature="simd_support")]
use core::arch::x86_64::*;
#[cfg(feature="simd_support")]
use core::mem::*;
use core::f64::consts::PI as PI_64;
use core::f32::consts::PI as PI_32;

use Rng;

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
pub struct BoxMuller<T> {
    flag: bool,
    z1: T,
    mean: T,
    std_dev: T,
}

impl<T: Default + PartialOrd> BoxMuller<T> {
    /// Construct a new `BoxMuller` normal distribution with the given mean
    /// and standard deviation.
    ///
    /// # Panics
    ///
    /// Panics if `std_dev < 0.0`.
    pub fn new(mean: T, std_dev: T) -> Self {
        assert!(std_dev >= T::default(), "BoxMuller::new called with `std_dev` < 0");
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
    #[inline(always)]
    fn box_muller<R: Rng>(rng: &mut R) -> (T, T);

    /// Generate a random value of `T`, using `rng` as the source of randomness.
    #[inline(always)]
    fn sample<R: Rng>(&mut self, rng: &mut R) -> T;
}

impl BoxMullerCore<f64> for BoxMuller<f64> {
    fn box_muller<R: Rng>(rng: &mut R) -> (f64, f64) {
        const TWO_PI: f64 = PI_64 * 2.0;

        let (u0, u1): (f64, f64) = rng.gen();

        let radius = (-2.0 * u0.ln()).sqrt();
        let (sin_theta, cos_theta) = (TWO_PI * u1).sin_cos();

        let z0 = radius * sin_theta;
        let z1 = radius * cos_theta;
        (z0, z1)
    }

    fn sample<R: Rng>(&mut self, rng: &mut R) -> f64 {
        self.flag = !self.flag;

        if !self.flag {
            return self.z1 * self.std_dev + self.mean;
        }

        let (z0, z1) = Self::box_muller(rng);
        self.z1 = z1;
        z0 * self.std_dev + self.mean
    }
}

impl BoxMullerCore<f32> for BoxMuller<f32> {
    fn box_muller<R: Rng>(rng: &mut R) -> (f32, f32) {
        const TWO_PI: f32 = PI_32 * 2.0;

        let (u0, u1): (f32, f32) = rng.gen();

        let radius = (-2.0 * u0.ln()).sqrt();
        let (sin_theta, cos_theta) = (TWO_PI * u1).sin_cos();

        let z0 = radius * sin_theta;
        let z1 = radius * cos_theta;
        (z0, z1)
    }

    fn sample<R: Rng>(&mut self, rng: &mut R) -> f32 {
        self.flag = !self.flag;

        if !self.flag {
            return self.z1 * self.std_dev + self.mean;
        }

        let (z0, z1) = Self::box_muller(rng);
        self.z1 = z1;
        z0 * self.std_dev + self.mean
    }
}

#[cfg(feature="simd_support")]
macro_rules! impl_box_muller {
    ($pi:expr, $(($vector:ident, $uty:ident)),+) => (
        $(impl BoxMullerCore<$vector> for BoxMuller<$vector> {
            fn box_muller<R: Rng>(rng: &mut R) -> ($vector, $vector) {
                const TWO_PI: $vector = $vector::splat(2.0 * $pi);

                let radius = ($vector::splat(-2.0) * rng.gen::<$vector>().ln()).sqrt();
                let (sin_theta, cos_theta) = (TWO_PI * rng.gen::<$vector>()).sincos();

                (radius * sin_theta, radius * cos_theta)
            }

            fn sample<R: Rng>(&mut self, rng: &mut R) -> $vector {
                self.flag = !self.flag;

                if !self.flag {
                    return self.z1 * self.std_dev + self.mean;
                }

                let (z0, z1) = Self::box_muller(rng);
                self.z1 = z1;
                z0 * self.std_dev + self.mean
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
/*impl_box_muller!(
    PI_64,
    (f64x2, u64x2),
    (f64x4, u64x4),
    (f64x8, u64x8)
);*/

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
    pub fn sample<R: Rng>(&mut self, rng: &mut R) -> T {
        self.box_muller.sample(rng).exp()
    }
}

// TODO: add explicit standard normal distr?

/// SIMD math functions not included in `feature(stdsimd)`.
#[cfg(feature="simd_support")]
pub trait SimdMath
where
    Self: Sized,
{
    /// Returns the natural logarithm of each lane of the vector.
    #[inline(always)]
    fn ln(&self) -> Self;

    /// Simultaneously computes the sine and cosine of the vector. Returns
    /// (sin, cos).
    #[inline(always)]
    fn sincos(&self) -> (Self, Self);

    /// Returns the square root of each lane of the vector.
    /// It should compile down to a single instruction.
    #[inline(always)]
    fn sqrt(&self) -> Self;

    /// Returns `e^(self)`, (the exponential function).
    #[inline(always)]
    fn exp(&self) -> Self;
}

#[cfg(feature="simd_support")]
macro_rules! impl_simd_math {
    ($fty:ident, $uty:ident, $uscalar:ty, $fscalar:ident) => (
        impl SimdMath for $fty {
            fn sincos(&self) -> ($fty, $fty) {
                const SIGN_MASK: $uscalar = 1 << size_of::<$uscalar>() * 8 - 1;

                let mut x = *self;
                /* extract the sign bit (upper one) */
                let mut sign_bit_sin = $uty::from_bits(x) & $uty::splat(SIGN_MASK);
                /* take the absolute value */
                x = $fty::from_bits($uty::from_bits(x) & $uty::splat(!SIGN_MASK));

                /* scale by 4/Pi */ // y= x * 4 / pi = x * (pi / 4)^-1
                let mut y = x * $fty::splat(::core::$fscalar::consts::FRAC_PI_4.recip());

                /* store the integer part of y in emm2 */
                let mut emm2 = $uty::from(y);

                /* j=(j+1) & (~1) (see the cephes sources) */
                emm2 += $uty::splat(1);
                emm2 &= $uty::splat(!1);
                y = $fty::from(emm2);

                let mut emm4 = emm2;

                /* get the swap sign flag for the sine */
                let mut emm0 = emm2 & $uty::splat(4);
                emm0 <<= 29;
                let swap_sign_bit_sin = $fty::from_bits(emm0);

                /* get the polynom selection mask for the sine*/
                emm2 &= $uty::splat(2);
                emm2 = $uty::from_bits(emm2.eq($uty::default()));
                let poly_mask = $fty::from_bits(emm2);

                /* The magic pass: "Extended precision modular arithmetic"
                   x = ((x - y * DP1) - y * DP2) - y * DP3; */
                let mut xmm1 = $fty::splat(-0.78515625);
                let mut xmm2 = $fty::splat(-2.4187564849853515625e-4);
                let mut xmm3 = $fty::splat(-3.77489497744594108e-8);
                xmm1 *= y;
                xmm2 *= y;
                xmm3 *= y;
                x += xmm1;
                x += xmm2;
                x += xmm3;

                emm4 -= $uty::splat(2);
                emm4 = !emm4 & $uty::splat(4);
                emm4 <<= 29;
                let sign_bit_cos = $fty::from_bits(emm4);

                sign_bit_sin ^= $uty::from_bits(swap_sign_bit_sin);


                /* Evaluate the first polynom  (0 <= x <= Pi/4) */
                let z = x * x;
                y = $fty::splat(2.443315711809948e-5);

                y *= z;
                y += $fty::splat(-1.388731625493765e-3);
                y *= z;
                y += $fty::splat(4.166664568298827e-2);
                y *= z;
                y *= z;
                let tmp = z * $fty::splat(1.6668057665e-1);
                y -= tmp;
                y += $fty::splat(1.0);

                /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

                let mut y2 = $fty::splat(-1.9515295891e-4);
                y2 *= z;
                y2 += $fty::splat(8.3321608736e-3);
                y2 *= z;
                y2 += $fty::splat(-1.6666654611e-1);
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
            }

            fn ln(&self) -> $fty {
                let mut x = *self;

                let one = $fty::splat(1.0);

                // must use intrinsic (available with the `stdsimd` crate tho)
                // let invalid_mask = _mm_cmple_ps(__m128::from_bits(x), __m128::from_bits($fty::default()));
                let invalid_mask = x.le($fty::default());

                // must use intrinsic (available with the `stdsimd` crate tho https://github.com/rust-lang-nursery/stdsimd/pull/418)
                // x = $fty::from_bits(_mm_max_ps(__m128::from_bits(x), __m128::from_bits($uty::splat(0x00800000))));
                x = x.max($fty::from_bits($uty::splat(0x00800000)));

                let emm0 = ($uty::from_bits(x) >> 23) - $uty::splat(0x7f);

                x = $fty::from_bits($uty::from_bits(x) & $uty::splat(!0x7f800000));
                x = $fty::from_bits($uty::from_bits(x) | $uty::from_bits($fty::splat(0.5)));

                let mut e = $fty::from(emm0) + one;

                // must use intrinsic (available with the `stdsimd` crate tho)
                // let mask = $fty::from_bits(_mm_cmplt_ps(__m128::from_bits(x), __m128::from_bits($fty::splat(0.707106781186547524))));
                let mask = x.lt($fty::splat(0.707106781186547524));
                let tmp = $uty::from_bits(x) & $uty::from(mask); //_mm_and_ps(x, mask);
                x -= one; // _mm_sub_ps(x, one);
                e -= $fty::from_bits($uty::from_bits(one) & $uty::from_bits(mask)); //_mm_sub_ps(e, _mm_and_ps(one, mask));
                x += $fty::from_bits(tmp); //_mm_add_ps(x, tmp);

                let z = x * x; //_mm_mul_ps(x, x);

                let mut y = $fty::splat(7.0376836292e-2);
                y = y * x + $fty::splat(-1.1514610310e-1);
                y = y * x + $fty::splat(1.1676998740e-1);
                y = y * x + $fty::splat(-1.2420140846e-1);
                y = y * x + $fty::splat(1.4249322787e-1);
                y = y * x + $fty::splat(-1.6668057665e-1);
                y = y * x + $fty::splat(2.0000714765e-1);
                y = y * x + $fty::splat(-2.4999993993e-1);
                y = y * x + $fty::splat(3.3333331174e-1);
                y *= x * z;

                y += e * $fty::splat(-2.12194440e-4);

                y -= z * $fty::splat(0.5);

                x += y;
                x += e * $fty::splat(0.693359375);
                $fty::from_bits($uty::from_bits(x) | $uty::from_bits(invalid_mask)) // negative arg will be NAN
            }

            // should compile down to a single instruction
            fn sqrt(&self) -> Self {
               let mut x = *self;
               for i in 0..Self::lanes() {
                   x = x.replace(i, x.extract(i).sqrt());
               }
               x
            }

            fn exp(&self) -> Self {
                let mut x = *self;

                let one = $fty::splat(1.0);

                x = x.min($fty::splat(88.3762626647949));
                x = x.max($fty::splat(-88.3762626647949));

                /* express exp(x) as exp(g + n*log(2)) */
                let mut fx = x * $fty::splat(1.44269504088896341);
                fx += $fty::splat(0.5);

                /* how to perform a floorf with SSE: just below */
                //imm0 = _mm256_cvttps_epi32(fx);
                //tmp  = _mm256_cvtepi32_ps(imm0);

                let tmp = $fty::from($uty::from(fx));

                /* if greater, substract 1 */
                //v8sf mask = _mm256_cmpgt_ps(tmp, fx);
                let mut mask = $uty::from_bits(tmp.gt(fx));
                mask &= $uty::from_bits(one);
                fx = tmp - $fty::from_bits(mask);

                let tmp = fx * $fty::splat(0.693359375);
                let mut z = fx * $fty::splat(-2.12194440e-4);
                x -= tmp;
                x -= z;

                z = x * x;

                let mut y = $fty::splat(1.9875691500e-4);
                y *= x;
                y += $fty::splat(1.3981999507e-3);
                y *= x;
                y += $fty::splat(8.3334519073e-3);
                y *= x;
                y += $fty::splat(4.1665795894e-2);
                y *= x;
                y += $fty::splat(1.6666665459e-1);
                y *= x;
                y += $fty::splat(5.0000001201e-1);
                y *= z;
                y += x;
                y += one;

                /* build 2^n */
                let mut imm0 = $uty::from(fx);
                // another two AVX2 instructions
                imm0 += $uty::splat(0x7f);
                imm0 <<= 23;
                let pow2n = $fty::from(imm0);
                y * pow2n
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
    fn ln(&self) -> f32x4 {
        let one = __m128::from_bits(f32x4::splat(1.0));

        let mut x = __m128::from_bits(*self);

        unsafe {
            let invalid_mask = _mm_cmple_ps(x, __m128::from_bits(u32x4::default()));

            x = _mm_max_ps(
                x,
                __m128::from_bits(f32x4::splat(::core::f32::MIN_POSITIVE)),
            );

            let mut emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);

            x = _mm_and_ps(x, __m128::from_bits(u32x4::splat(!0x7f800000)));
            x = _mm_or_ps(x, __m128::from_bits(f32x4::splat(0.5)));

            emm0 = _mm_sub_epi32(emm0, __m128i::from_bits(u32x4::splat(0x7f)));
            let mut e = _mm_cvtepi32_ps(emm0);

            e = _mm_add_ps(e, one);

            let mask = _mm_cmplt_ps(x, __m128::from_bits(f32x4::splat(0.707106781186547524)));
            let mut tmp = _mm_and_ps(x, mask);
            x = _mm_sub_ps(x, one);
            e = _mm_sub_ps(e, _mm_and_ps(one, mask));
            x = _mm_add_ps(x, tmp);

            let z = _mm_mul_ps(x, x);

            let mut y = __m128::from_bits(f32x4::splat(7.0376836292e-2));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(f32x4::splat(-1.1514610310e-1)));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(f32x4::splat(1.1676998740e-1)));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(f32x4::splat(-1.2420140846e-1)));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(f32x4::splat(1.4249322787e-1)));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(f32x4::splat(-1.6668057665e-1)));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(f32x4::splat(2.0000714765e-1)));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(f32x4::splat(-2.4999993993e-1)));
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, __m128::from_bits(f32x4::splat(3.3333331174e-1)));
            y = _mm_mul_ps(y, x);

            y = _mm_mul_ps(y, z);

            tmp = _mm_mul_ps(e, __m128::from_bits(f32x4::splat(-2.12194440e-4)));
            y = _mm_add_ps(y, tmp);

            tmp = _mm_mul_ps(z, __m128::from_bits(f32x4::splat(0.5)));
            y = _mm_sub_ps(y, tmp);

            tmp = _mm_mul_ps(e, __m128::from_bits(f32x4::splat(0.693359375)));
            x = _mm_add_ps(x, y);
            x = _mm_add_ps(x, tmp);
            x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN
            f32x4::from_bits(x)
        }
    }

    fn sincos(&self) -> (f32x4, f32x4) {
        let mut x = __m128::from_bits(*self);

        unsafe {
            /* extract the sign bit (upper one) */
            let mut sign_bit_sin = _mm_and_ps(x, __m128::from_bits(u32x4::splat(0x80000000)));
            /* take the absolute value */
            x = _mm_and_ps(x, __m128::from_bits(u32x4::splat(!0x80000000)));

            /* scale by 4/Pi */
            // y= x * 4 / pi = x * (pi / 4)^-1
            let mut y = _mm_mul_ps(
                x,
                __m128::from_bits(f32x4::splat(::core::f32::consts::FRAC_PI_4.recip())),
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
            let mut xmm1 = __m128::from_bits(f32x4::splat(-0.78515625));
            let mut xmm2 = __m128::from_bits(f32x4::splat(-2.4187564849853515625e-4));
            let mut xmm3 = __m128::from_bits(f32x4::splat(-3.77489497744594108e-8));
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
            y = __m128::from_bits(f32x4::splat(2.443315711809948e-5));

            y = _mm_mul_ps(y, z);
            y = _mm_add_ps(y, __m128::from_bits(f32x4::splat(-1.388731625493765e-3)));
            y = _mm_mul_ps(y, z);
            y = _mm_add_ps(y, __m128::from_bits(f32x4::splat(4.166664568298827e-2)));
            y = _mm_mul_ps(y, z);
            y = _mm_mul_ps(y, z);
            let tmp = _mm_mul_ps(z, __m128::from_bits(f32x4::splat(1.6668057665e-1)));
            y = _mm_sub_ps(y, tmp);
            y = _mm_add_ps(y, __m128::from_bits(f32x4::splat(1.0)));

            /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

            let mut y2 = __m128::from_bits(f32x4::splat(-1.9515295891e-4));
            y2 = _mm_mul_ps(y2, z);
            y2 = _mm_add_ps(y2, __m128::from_bits(f32x4::splat(8.3321608736e-3)));
            y2 = _mm_mul_ps(y2, z);
            y2 = _mm_add_ps(y2, __m128::from_bits(f32x4::splat(-1.6666654611e-1)));
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
                f32x4::from_bits(_mm_xor_ps(xmm1, sign_bit_sin)),
                f32x4::from_bits(_mm_xor_ps(xmm2, sign_bit_cos)),
            )
        }
    }

    fn sqrt(&self) -> Self {
        let mut x = *self;
        for i in 0..Self::lanes() {
            x = x.replace(i, x.extract(i).sqrt());
        }
        x
    }

    fn exp(&self) -> Self {
        let mut x = *self;
        for i in 0..Self::lanes() {
            x = x.replace(i, x.extract(i).exp());
        }
        x
    }
}

#[cfg(feature="simd_support")]
impl SimdMath for f32x8 {
    fn sincos(&self) -> (Self, Self) {
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

        let mut x = __m256::from_bits(*self);

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
    }

    fn ln(&self) -> Self {
        let mut x = __m256::from_bits(*self);
        let mut imm0: __m256i;
        let one: __m256 = __m256::from_bits(f32x8::splat(1.0));
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
            x = _mm256_or_ps(x, __m256::from_bits(f32x8::splat(0.5)));

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
                _mm256_cmp_ps(x, __m256::from_bits(f32x8::splat(0.707106781186547524)), 1);
            let mut tmp: __m256 = _mm256_and_ps(x, mask);
            x = _mm256_sub_ps(x, one);
            e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
            x = _mm256_add_ps(x, tmp);

            let z: __m256 = _mm256_mul_ps(x, x);

            let mut y: __m256 = __m256::from_bits(f32x8::splat(7.0376836292E-2));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(f32x8::splat(-1.1514610310E-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(f32x8::splat(1.1676998740E-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(f32x8::splat(-1.2420140846E-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(f32x8::splat(1.4249322787E-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(f32x8::splat(-1.6668057665E-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(f32x8::splat(2.0000714765E-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(f32x8::splat(-2.4999993993E-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(f32x8::splat(3.3333331174E-1)));
            y = _mm256_mul_ps(y, x);

            y = _mm256_mul_ps(y, z);

            tmp = _mm256_mul_ps(e, __m256::from_bits(f32x8::splat(-2.12194440e-4)));
            y = _mm256_add_ps(y, tmp);

            tmp = _mm256_mul_ps(z, __m256::from_bits(f32x8::splat(0.5)));
            y = _mm256_sub_ps(y, tmp);

            tmp = _mm256_mul_ps(e, __m256::from_bits(f32x8::splat(0.693359375)));
            x = _mm256_add_ps(x, y);
            x = _mm256_add_ps(x, tmp);
            x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
            return f32x8::from_bits(x);
        }
    }

    fn sqrt(&self) -> Self {
        let mut x = *self;
        for i in 0..Self::lanes() {
            x = x.replace(i, x.extract(i).sqrt());
        }
        x
    }

    fn exp(&self) -> Self {
        let mut x = __m256::from_bits(*self);
        let one: __m256 = __m256::from_bits(f32x8::splat(1.0));

        unsafe {
            x = _mm256_min_ps(x, __m256::from_bits(f32x8::splat(88.3762626647949)));
            x = _mm256_max_ps(x, __m256::from_bits(f32x8::splat(-88.3762626647949)));

            /* express exp(x) as exp(g + n*log(2)) */
            let mut fx = _mm256_mul_ps(x, __m256::from_bits(f32x8::splat(1.44269504088896341)));
            fx = _mm256_add_ps(fx, __m256::from_bits(f32x8::splat(0.5)));

            /* how to perform a floorf with SSE: just below */
            let imm0 = _mm256_cvttps_epi32(fx);
            let tmp = _mm256_cvtepi32_ps(imm0);

            // let tmp = _mm256_floor_ps(fx);

            /* if greater, substract 1 */
            //v8sf mask = _mm256_cmpgt_ps(tmp, fx);
            let mut mask: __m256 = _mm256_cmp_ps(tmp, fx, 14);
            mask = _mm256_and_ps(mask, one);
            fx = _mm256_sub_ps(tmp, mask);

            let tmp = _mm256_mul_ps(fx, __m256::from_bits(f32x8::splat(0.693359375)));
            let mut z: __m256 = _mm256_mul_ps(fx, __m256::from_bits(f32x8::splat(-2.12194440e-4)));
            x = _mm256_sub_ps(x, tmp);
            x = _mm256_sub_ps(x, z);

            z = _mm256_mul_ps(x,x);

            let mut y: __m256 = __m256::from_bits(f32x8::splat(1.9875691500e-4));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(f32x8::splat(1.3981999507e-3)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(f32x8::splat(8.3334519073e-3)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(f32x8::splat(4.1665795894e-2)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(f32x8::splat(1.6666665459e-1)));
            y = _mm256_mul_ps(y, x);
            y = _mm256_add_ps(y, __m256::from_bits(f32x8::splat(5.0000001201e-1)));
            y = _mm256_mul_ps(y, z);
            y = _mm256_add_ps(y, x);
            y = _mm256_add_ps(y, one);

            /* build 2^n */
            let mut imm0 = _mm256_cvttps_epi32(fx);
            // another two AVX2 instructions
            /*imm0 = _mm256_add_epi32(imm0, __m256i::from_bits(u32x8::splat(0x7f)));
            imm0 = _mm256_slli_epi32(imm0, 23);*/
                /*let (mut a, mut b): (__m128i, __m128i) = transmute(imm0);
                a = _mm_add_epi32(a, __m128i::from_bits(u32x4::splat(0x7f)));
                b = _mm_add_epi32(b, __m128i::from_bits(u32x4::splat(0x7f)));
                a = _mm_slli_epi32(a, 23);
                b = _mm_slli_epi32(b, 23);
                imm0 = transmute((a, b));*/
                imm0 = __m256i::from_bits(u32x8::from_bits(imm0) + u32x8::splat(0x7f));
                imm0 = __m256i::from_bits(u32x8::from_bits(imm0) << 23);
            let pow2n: __m256 = _mm256_castsi256_ps(imm0);
            y = _mm256_mul_ps(y, pow2n);
            return f32x8::from_bits(y);
        }
    }
}

#[cfg(all(test, feature="simd_support"))]
mod tests {
    use super::*;
    use SeedableRng;
    use prng::Sfc32x4Rng;

    const BENCH_N: usize = 1 << 10;
    const TEST_N: usize = 1 << 15;

    macro_rules! make_log_test {
        ($test_name:ident, $ty:ident) => (
            #[test]
            fn $test_name() {
                let mut rng = Sfc32x4Rng::from_rng(&mut ::thread_rng()).unwrap();
                for _ in 0..TEST_N {
                    let num = rng.gen::<$ty>();
                    let actual = num.ln();

                    for i in 0..$ty::lanes() {
                        let expected = num.extract(i).ln();
                        let actual = actual.extract(i);
                        assert!(
                            (expected - actual) < 1e-6,
                            "\n{:?}\n{:?}", expected, actual
                        );
                    }
                }
            }
        )
    }

    make_log_test!(log_f32x2, f32x2);
    make_log_test!(log_f32x4, f32x4);
    make_log_test!(log_f32x8, f32x8);
    make_log_test!(log_f32x16, f32x16);

    macro_rules! make_sincos_test {
        ($test_name:ident, $ty:ident) => (
            #[test]
            fn $test_name() {
                let mut rng = Sfc32x4Rng::from_rng(&mut ::thread_rng()).unwrap();
                for _ in 0..TEST_N {
                    let num = rng.gen::<$ty>();
                    let (actual_sin, actual_cos) = num.sincos();

                    for i in 0..$ty::lanes() {
                        let (expected_sin, expected_cos) = num.extract(i).sin_cos();

                        let actual_sin = actual_sin.extract(i);
                        let actual_cos = actual_cos.extract(i);

                        assert!(
                            (expected_sin - actual_sin) < 5e-7,
                            "\n{:?}\n{:?}", expected_sin, actual_sin
                        );
                        assert!(
                            (expected_cos - actual_cos) < 5e-7,
                            "\n{:?}\n{:?}", expected_cos, actual_cos
                        );
                    }
                }
            }
        )
    }

    make_sincos_test!(sincos_f32x2, f32x2);
    make_sincos_test!(sincos_f32x4, f32x4);
    make_sincos_test!(sincos_f32x8, f32x8);
    make_sincos_test!(sincos_f32x16, f32x16);

    macro_rules! make_exp_test {
        ($test_name:ident, $ty:ident) => (
            #[test]
            fn $test_name() {
                let mut rng = Sfc32x4Rng::from_rng(&mut ::thread_rng()).unwrap();
                for _ in 0..TEST_N {
                    let num = rng.gen::<$ty>();
                    let actual = num.exp();

                    for i in 0..$ty::lanes() {
                        let expected = num.extract(i).exp();
                        let actual = actual.extract(i);

                        assert!(
                            (expected - actual) < 5e-7,
                            "\n{:?}\n{:?}", expected, actual
                        );
                    }
                }
            }
        )
    }

    make_exp_test!(exp_f32x2, f32x2);
    make_exp_test!(exp_f32x4, f32x4);
    make_exp_test!(exp_f32x8, f32x8);
    make_exp_test!(exp_f32x16, f32x16);
}
