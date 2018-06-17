//! The central limit theorem and derived distributions.

use core::marker::PhantomData;
use stdsimd::simd::*;

use Rng;
use distributions::Distribution;

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
pub struct CentralLimit<T> {
    mean: f64,
    std_dev: f64,
    phantom: PhantomData<T>,
}

impl<T> CentralLimit<T> {
    /// Construct a new `CentralLimit` distribution with the given mean and
    /// standard deviation.
    ///
    /// # Panics
    ///
    /// Panics if `std_dev < 0`.
    #[inline]
    // TODO: implement for vectors/f32
    pub fn new(mean: f64, std_dev: f64) -> Self {
        assert!(std_dev >= 0.0, "CentralLimit::new called with `std_dev` < 0");
        Self {
            mean,
            std_dev,
            phantom: PhantomData,
        }
    }
}

macro_rules! impl_simd {
    ($ty:ident, $scalar:ty, $num:expr) => (
        impl Distribution<$ty> for CentralLimit<$ty> {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $ty {
                // Irwin–Hall mean and std_dev (sqrt variance)
                // https://en.wikipedia.org/wiki/Irwin–Hall_distribution
                const IH_MEAN: $scalar = $num as $scalar / 2.0;
                // TODO: verify eval at compile
                // TODO: benchmark `sqrt` vs `sqrte`
                let ih_std_dev_inv = 1.0 / ($num as $scalar / 12.0).sqrt();

                // get Irwin–Hall distr
                let mut sum = $ty::default();
                for _ in 0..$num {
                    sum += rng.gen::<$ty>();
                }
                // adjust Irwin–Hall distr to normal distr
                // TODO: look into optimizing/combining the two distribution
                // adjustments
                let n = (sum - IH_MEAN) * ih_std_dev_inv;

                self.mean + self.std_dev * n
            }
        }
    )
}

// TODO: tune for better number of samples?
/*impl_simd! { f32x2, f32, 4 }
impl_simd! { f32x4, f32, 4 }
impl_simd! { f32x8, f32, 4 }
impl_simd! { f32x16, f32, 4 }*/
impl_simd! { f64x2, f64, 4 }
impl_simd! { f64x4, f64, 4 }
impl_simd! { f64x8, f64, 4 }

macro_rules! impl_simd_to_scalar {
    ($ty:ident, $scalar:ty) => (
        impl Distribution<$scalar> for CentralLimit<$ty> {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $scalar {
                const IH_MEAN: $scalar = $ty::lanes / 2.0;
                let ih_std_dev_inv = 1.0 / ($ty::lanes / 12.0).sqrt();

                let n = (rng.gen::<$ty>().sum() - IH_MEAN) * ih_std_dev_inv;
                self.mean + self.std_dev * n
            }
        }
    )
}

/*impl_simd_to_scalar! { f32x2, f32 }
impl_simd_to_scalar! { f32x4, f32 }
impl_simd_to_scalar! { f32x8, f32 }
impl_simd_to_scalar! { f32x16, f32 }*/
impl_simd_to_scalar! { f64x2, f64 }
impl_simd_to_scalar! { f64x4, f64 }
impl_simd_to_scalar! { f64x8, f64 }

#[cfg(test)]
mod tests {
    use stdsimd::simd::*;
    use super::{Rng, CentralLimit};

    #[test]
    fn test_clt_vector() {
        let norm = CentralLimit::<f64x2>::new(10.0, 10.0);
        let mut rng = ::test::rng(210);
        for _ in 0..1000 {
            let _: f64x2 = rng.sample(norm);
        }
    }
    #[test]
    fn test_clt_scalar() {
        let norm = CentralLimit::<f64x2>::new(10.0, 10.0);
        let mut rng = ::test::rng(210);
        for _ in 0..1000 {
            let _: f64 = rng.sample(norm);
        }
    }
    #[test]
    #[should_panic]
    fn test_clt_invalid_sd() {
        CentralLimit::<f32x2>::new(10.0, -1.0);
    }
}
