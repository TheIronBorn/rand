//! The central limit theorem and derived distributions.

use core::marker::PhantomData;
use core::simd::*;

use distributions::Distribution;
use Rng;

///
pub trait CentralLimit<T> {
    ///
    fn new(mean: T, std_dev: T) -> Self;
}

/// The normal distribution `N(mean, std_dev**2)`.
///
/// This uses the central limit theorem. It is well suited to an SIMD
/// implementation, even on older hardware.
///
/// # Example
///
/// ```
/// use rand::distributions::{CentralLimit, Distribution};
///
/// // mean 2, standard deviation 3
/// let normal = CentralLimit::new(2.0, 3.0);
/// let v = normal.sample(&mut rand::thread_rng());
/// println!("{} is from a N(2, 9) distribution", v)
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CentralLimitVector<T> {
    mean: T,
    std_dev: T,
}

macro_rules! impl_clt_vector {
    ($ty:ident, $scalar:ty, $num:expr) => {
        impl CentralLimit<$ty> for CentralLimitVector<$ty> {
            /// Construct a new `CentralLimitVector` distribution with the given mean and
            /// standard deviation.
            ///
            /// # Panics
            ///
            /// Panics if `std_dev < 0`.
            #[inline]
            fn new(mean: $ty, std_dev: $ty) -> Self {
                assert!(
                    std_dev.ge($ty::splat(0.0)).all(),
                    "CentralLimitVector::new called with `std_dev` < 0"
                );
                Self { mean, std_dev }
            }
        }

        impl Distribution<$ty> for CentralLimitVector<$ty> {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $ty {
                // Irwin–Hall mean and std_dev (square root of variance)
                // https://en.wikipedia.org/wiki/Irwin–Hall_distribution
                const IH_MEAN: $scalar = $num as $scalar / 2.0;
                let ih_std_dev_inv = 1.0 / ($num as $scalar / 12.0).sqrt();

                // get Irwin–Hall distribution
                let mut sum = $ty::default();
                for _ in 0..$num {
                    sum += rng.gen::<$ty>();
                }
                // adjust Irwin–Hall distribution to specified normal distribution
                // TODO: ensure optimized when mean and std_dev are SIMD vectors
                // NOTE: variable names here might be misleading
                // NOTE: this is fast when mean and std_dev are compile-time constant,
                // slower than other math when not. We prioritize the constant
                // case here.
                let std_dev = self.std_dev * ih_std_dev_inv;
                let mean = self.mean - IH_MEAN * std_dev;
                mean + std_dev * sum
            }
        }
    };
}

// TODO: tune for better number of samples?
impl_clt_vector! { f32x2, f32, 4 }
impl_clt_vector! { f32x4, f32, 4 }
impl_clt_vector! { f32x8, f32, 4 }
impl_clt_vector! { f32x16, f32, 4 }
impl_clt_vector! { f64x2, f64, 4 }
impl_clt_vector! { f64x4, f64, 4 }
impl_clt_vector! { f64x8, f64, 4 }

/// The normal distribution `N(mean, std_dev**2)`.
///
/// This uses the central limit theorem. It is well suited to an SIMD
/// implementation, even on older hardware.
///
/// # Example
///
/// ```
/// use rand::distributions::{CentralLimit, Distribution};
///
/// // mean 2, standard deviation 3
/// let normal = CentralLimit::new(2.0, 3.0);
/// let v = normal.sample(&mut rand::thread_rng());
/// println!("{} is from a N(2, 9) distribution", v)
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CentralLimitScalar<T, V> {
    mean: T,
    std_dev: T,
    phantom: PhantomData<V>,
}

macro_rules! impl_clt_scalar {
    ($ty:ident, $scalar:ty) => {
        impl CentralLimit<$scalar> for CentralLimitScalar<$scalar, $ty> {
            /// Construct a new `CentralLimitScalar` distribution with the given mean and
            /// standard deviation.
            ///
            /// # Panics
            ///
            /// Panics if `std_dev < 0`.
            #[inline]
            fn new(mean: $scalar, std_dev: $scalar) -> Self {
                assert!(
                    std_dev >= 0.0,
                    "CentralLimitScalar::new called with `std_dev` < 0"
                );
                Self {
                    mean,
                    std_dev,
                    phantom: PhantomData,
                }
            }
        }

        impl Distribution<$scalar> for CentralLimitScalar<$scalar, $ty> {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $scalar {
                // Irwin–Hall mean and std_dev (square root of variance)
                // https://en.wikipedia.org/wiki/Irwin–Hall_distribution
                const IH_MEAN: $scalar = $ty::lanes() as $scalar / 2.0;
                let ih_std_dev_inv = 1.0 / ($ty::lanes() as $scalar / 12.0).sqrt();

                // get Irwin–Hall distribution
                let sum = rng.gen::<$ty>().sum();

                // adjust Irwin–Hall distribution to specified normal distribution
                let std_dev = self.std_dev * ih_std_dev_inv;
                let mean = self.mean - IH_MEAN * std_dev;
                mean + std_dev * sum
            }
        }
    };
}

impl_clt_scalar! { f32x2, f32 }
impl_clt_scalar! { f32x4, f32 }
impl_clt_scalar! { f32x8, f32 }
impl_clt_scalar! { f32x16, f32 }
impl_clt_scalar! { f64x2, f64 }
impl_clt_scalar! { f64x4, f64 }
impl_clt_scalar! { f64x8, f64 }

#[cfg(test)]
mod tests {
    use super::*;
    use core::simd::*;

    #[test]
    fn test_clt_vector() {
        let norm = CentralLimitVector::new(f64x2::splat(10.0), f64x2::splat(10.0));
        let mut rng = ::test::rng(210);
        for _ in 0..1000 {
            rng.sample(norm);
        }
    }
    #[test]
    fn test_clt_scalar() {
        let norm = CentralLimitScalar::<f64, f64x2>::new(10.0, 10.0);
        let mut rng = ::test::rng(210);
        for _ in 0..1000 {
            rng.sample(norm);
        }
    }
    #[test]
    #[should_panic]
    fn test_clt_invalid_sd() {
        CentralLimitScalar::<f64, f64x2>::new(10.0, -1.0);
    }
}
