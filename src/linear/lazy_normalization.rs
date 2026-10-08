//! Borrowed normalized designs for the interior-point method of Kim et al.
//!
//! See <https://web.stanford.edu/~boyd/papers/l1_ls.html>.

use std::convert::Infallible;
use std::marker::PhantomData;

use lazymatrix::{
    DotSlice, ElemDivAssign, LazyMatrix, MatTransposeVecInto, MatVecInto, MatrixErrorType,
    MatrixShape, MatrixWrite, Scalar, ScaledSubSlice, SubScalarAssign, SumEntries, VectorView,
    WeightedGramInto, WeightedGramKernel,
};

use crate::linalg::basic::arrays::{Array1, Array2};
use crate::numbers::floatnum::FloatNumber;

/// Keep the design borrowed while coefficient matrices retain the caller's storage type.
pub(super) struct LazyDesign<'a, T: FloatNumber, X: Array2<T>> {
    normalized: LazyMatrix<MatrixAdapter<'a, T, X>, T>,
    augmentation: Option<(T, T)>,
}

impl<'a, T: FloatNumber, X: Array2<T>> LazyDesign<'a, T, X> {
    pub(super) fn new(x: &'a X, centers: Option<Vec<T>>, scales: Option<Vec<T>>) -> Self {
        Self {
            normalized: LazyMatrix::from_parts(MatrixAdapter(x, PhantomData), centers, scales),
            augmentation: None,
        }
    }

    pub(super) fn with_l2(mut self, penalty: T) -> Self {
        let gamma = T::one() / (T::one() + penalty).sqrt();
        self.augmentation = Some((gamma, gamma * penalty.sqrt()));
        self
    }

    pub(super) fn shape(&self) -> (usize, usize) {
        let n = self.normalized.nrows();
        let p = self.normalized.ncols();
        (n + if self.augmentation.is_some() { p } else { 0 }, p)
    }

    pub(super) fn centers(&self) -> &[T] {
        self.normalized.centers().unwrap()
    }

    pub(super) fn scales(&self) -> &[T] {
        self.normalized.scales().unwrap()
    }

    pub(super) fn gamma(&self) -> T {
        self.augmentation.map_or(T::one(), |(gamma, _)| gamma)
    }

    pub(super) fn matvec(&self, x: &Vec<T>) -> Vec<T> {
        let mut out = Vec::with_capacity(self.shape().0);
        out.resize(self.normalized.nrows(), T::zero());
        self.normalized.matvec_into(x, &mut out).unwrap();
        if let Some((gamma, padding)) = self.augmentation {
            for value in &mut out {
                *value *= gamma;
            }
            out.extend(x.iter().map(|&value| padding * value));
        }
        out
    }

    pub(super) fn transpose_matvec(&self, x: &[T]) -> Vec<T> {
        check_vector_length(x.len(), self.shape().0);
        let n = self.normalized.nrows();
        let mut out = Vec::zeros(self.normalized.ncols());
        self.normalized
            .mat_transpose_vec_into(&VectorRef(&x[..n]), &mut out)
            .unwrap();
        if let Some((gamma, padding)) = self.augmentation {
            for (value, &tail) in out.iter_mut().zip(&x[n..]) {
                *value = gamma * *value + padding * tail;
            }
        }
        out
    }

    pub(super) fn gram(&self) -> X {
        let p = self.normalized.ncols();
        let mut out = X::zeros(p, p);
        self.normalized
            .weighted_gram_into(
                &UnitWeights(self.normalized.nrows()),
                &mut MatrixOutput(&mut out, PhantomData),
            )
            .unwrap();
        if let Some((gamma, padding)) = self.augmentation {
            out.mul_scalar_mut(gamma * gamma);
            for j in 0..p {
                out.add_element_mut((j, j), padding * padding);
            }
        }
        out
    }
}

struct MatrixAdapter<'a, T: FloatNumber, X: Array2<T>>(&'a X, PhantomData<T>);

impl<T: FloatNumber, X: Array2<T>> MatrixShape for MatrixAdapter<'_, T, X> {
    fn nrows(&self) -> usize {
        self.0.shape().0
    }
    fn ncols(&self) -> usize {
        self.0.shape().1
    }
}

impl<T: FloatNumber, X: Array2<T>> MatrixErrorType for MatrixAdapter<'_, T, X> {
    type Error = Infallible;
}

impl<T: FloatNumber, X: Array2<T>> MatVecInto<Vec<T>> for MatrixAdapter<'_, T, X> {
    fn matvec_into(&self, x: &Vec<T>, out: &mut Vec<T>) -> Result<(), Infallible> {
        self.matvec_normalized_into::<T>(x, None, None, out)
    }

    fn matvec_normalized_into<F: Scalar>(
        &self,
        x: &Vec<T>,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Vec<T>,
    ) -> Result<(), Infallible>
    where
        Vec<T>: ElemDivAssign<F> + DotSlice<F> + SubScalarAssign<F>,
    {
        check_vector_length(x.len(), self.ncols());
        check_vector_length(out.len(), self.nrows());
        out.fill(T::zero());
        // Column traversal reuses each view while preserving each row's summation order.
        for (j, &v) in x.iter().enumerate() {
            let center = centers.map_or(T::zero(), |c| T::from(c[j]).unwrap());
            let scale = scales.map_or(T::one(), |s| T::from(s[j]).unwrap());
            let column = self.0.get_col(j);
            for (&raw, result) in column.iterator(0).zip(out.iter_mut()) {
                // Center before accumulation to retain small variations at large offsets.
                *result += ((raw - center) / scale) * v;
            }
        }
        Ok(())
    }
}

struct VectorRef<'a, T>(&'a [T]);

impl<T: FloatNumber> SumEntries<T> for VectorRef<'_, T> {
    fn sum_entries(&self) -> T {
        self.0.iter().copied().sum()
    }
}

impl<'v, T: FloatNumber, X: Array2<T>> MatTransposeVecInto<VectorRef<'v, T>, Vec<T>>
    for MatrixAdapter<'_, T, X>
{
    fn mat_transpose_vec_into(
        &self,
        x: &VectorRef<'v, T>,
        out: &mut Vec<T>,
    ) -> Result<(), Infallible> {
        self.mat_transpose_vec_normalized_into::<T>(x, None, None, out)
    }

    fn mat_transpose_vec_normalized_into<F: Scalar>(
        &self,
        x: &VectorRef<'v, T>,
        centers: Option<&[F]>,
        scales: Option<&[F]>,
        out: &mut Vec<T>,
    ) -> Result<(), Infallible>
    where
        VectorRef<'v, T>: SumEntries<F>,
        Vec<T>: ScaledSubSlice<F> + ElemDivAssign<F>,
    {
        check_vector_length(x.0.len(), self.nrows());
        check_vector_length(out.len(), self.ncols());
        for (j, result) in out.iter_mut().enumerate() {
            let center = centers.map_or(T::zero(), |c| T::from(c[j]).unwrap());
            let scale = scales.map_or(T::one(), |s| T::from(s[j]).unwrap());
            let column = self.0.get_col(j);
            *result = column
                .iterator(0)
                .zip(x.0)
                .fold(T::zero(), |sum, (&raw, &v)| {
                    sum + ((raw - center) / scale) * v
                });
        }
        Ok(())
    }
}

impl<T: FloatNumber, X: Array2<T>> WeightedGramKernel<T> for MatrixAdapter<'_, T, X> {
    fn weighted_gram_normalized_into<W, O>(
        &self,
        weights: &W,
        centers: Option<&[T]>,
        scales: Option<&[T]>,
        out: &mut O,
    ) -> Result<(), Infallible>
    where
        W: VectorView<T> + ?Sized,
        O: MatrixWrite<T> + ?Sized,
    {
        // Panels reuse normalized values across column pairs without storing the design.
        const ROWS: usize = 256;
        const COLS: usize = 32;
        let (n, p) = self.0.shape();
        let rows = n.min(ROWS);
        let cols = p.min(COLS);
        let mut left = Vec::zeros(rows * cols);
        let mut right = Vec::zeros(rows * cols);
        let mut sums = Vec::zeros(cols * cols);
        for j in (0..p).step_by(COLS) {
            let nj = (p - j).min(COLS);
            for k in (0..=j).step_by(COLS) {
                let nk = (p - k).min(COLS);
                sums.fill(T::zero());
                for row in (0..n).step_by(ROWS) {
                    let nr = (n - row).min(ROWS);
                    self.normalized_panel(row, nr, j, nj, centers, scales, &mut left);
                    if j == k {
                        right[..nr * nk].copy_from_slice(&left[..nr * nj]);
                    } else {
                        self.normalized_panel(row, nr, k, nk, centers, scales, &mut right);
                    }
                    for column in left[..nr * nj].chunks_exact_mut(nr) {
                        for (i, value) in column.iter_mut().enumerate() {
                            *value = weights.get(row + i) * *value;
                        }
                    }
                    for a in 0..nj {
                        let left_column = &left[a * nr..(a + 1) * nr];
                        let width = if j == k { a + 1 } else { nk };
                        for b in 0..width {
                            let right_column = &right[b * nr..(b + 1) * nr];
                            let sum = &mut sums[a * cols + b];
                            // Carry the accumulator across panels to retain row order.
                            for (&x, &y) in left_column.iter().zip(right_column) {
                                *sum += x * y;
                            }
                        }
                    }
                }
                for a in 0..nj {
                    let width = if j == k { a + 1 } else { nk };
                    for b in 0..width {
                        let value = sums[a * cols + b];
                        out.set(j + a, k + b, value);
                        if j + a != k + b {
                            out.set(k + b, j + a, value);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl<T: FloatNumber, X: Array2<T>> MatrixAdapter<'_, T, X> {
    fn normalized_panel(
        &self,
        row: usize,
        nrows: usize,
        column: usize,
        ncols: usize,
        centers: Option<&[T]>,
        scales: Option<&[T]>,
        out: &mut [T],
    ) {
        for j in 0..ncols {
            let center = centers.map_or(T::zero(), |c| c[column + j]);
            let scale = scales.map_or(T::one(), |s| s[column + j]);
            let view = self.0.slice(row..row + nrows, column + j..column + j + 1);
            for (value, &raw) in out[j * nrows..(j + 1) * nrows]
                .iter_mut()
                .zip(view.iterator(0))
            {
                *value = (raw - center) / scale;
            }
        }
    }
}

struct UnitWeights(usize);

impl<T: FloatNumber> VectorView<T> for UnitWeights {
    fn len(&self) -> usize {
        self.0
    }
    fn get(&self, index: usize) -> T {
        if index >= self.0 {
            std::panic::panic_any("weight index is out of bounds");
        }
        T::one()
    }
}

fn check_vector_length(actual: usize, expected: usize) {
    if actual != expected {
        std::panic::panic_any("matrix and vector dimensions do not agree");
    }
}

struct MatrixOutput<'a, T: FloatNumber, X: Array2<T>>(&'a mut X, PhantomData<T>);

impl<T: FloatNumber, X: Array2<T>> MatrixShape for MatrixOutput<'_, T, X> {
    fn nrows(&self) -> usize {
        self.0.shape().0
    }
    fn ncols(&self) -> usize {
        self.0.shape().1
    }
}

impl<T: FloatNumber, X: Array2<T>> MatrixWrite<T> for MatrixOutput<'_, T, X> {
    fn set(&mut self, row: usize, column: usize, value: T) {
        self.0.set((row, column), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::arrays::{Array, Array1, Array2, MutArray, MutArrayView2};
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::linear::lasso_optimizer::InteriorPointOptimizer;
    use crate::numbers::floatnum::FloatNumber;

    fn check_products<T: FloatNumber, X: Array2<T>>(x: X, tolerance: f64) {
        let value = |v| T::from_f64(v).unwrap();
        let original = x.clone();
        for centered in [false, true] {
            for scaled in [false, true] {
                let centers = centered.then(|| vec![value(1.0), value(2.0)]);
                let scales = scaled.then(|| vec![value(2.0), value(4.0)]);
                let mut eager = x.clone();
                eager.scale_mut(
                    centers.as_deref().unwrap_or(&[T::zero(); 2]),
                    scales.as_deref().unwrap_or(&[T::one(); 2]),
                    0,
                );
                let design = LazyDesign::new(&x, centers, scales);
                let coefficients = vec![value(2.0), value(-1.0)];
                let response = vec![value(1.0), value(-2.0), value(3.0)];
                assert_close(
                    &design.matvec(&coefficients),
                    &coefficients.xa(true, &eager),
                    tolerance,
                );
                assert_close(
                    &design.transpose_matvec(&response),
                    &response.xa(false, &eager),
                    tolerance,
                );
                let gram = design.gram();
                assert_close(
                    &gram.iterator(0).copied().collect::<Vec<_>>(),
                    &eager
                        .ab(true, &eager, false)
                        .iterator(0)
                        .copied()
                        .collect::<Vec<_>>(),
                    tolerance,
                );
                let left = design
                    .matvec(&coefficients)
                    .iter()
                    .zip(&response)
                    .map(|(&a, &b)| a * b)
                    .sum::<T>();
                let right = coefficients
                    .iter()
                    .zip(design.transpose_matvec(&response))
                    .map(|(&a, b)| a * b)
                    .sum::<T>();
                assert!((left - right).abs().to_f64().unwrap() <= tolerance);
            }
        }
        assert!(x.iterator(0).zip(original.iterator(0)).all(|(a, b)| a == b));
    }

    fn assert_close<T: FloatNumber>(actual: &[T], expected: &[T], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &b) in actual.iter().zip(expected) {
            assert!((a - b).abs().to_f64().unwrap() <= tolerance, "{a} != {b}");
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn products_match_eager_for_both_dense_layouts_and_precisions() {
        for column_major in [false, true] {
            let data = if column_major {
                vec![1., 2., 4., 4., 6., 10.]
            } else {
                vec![1., 4., 2., 6., 4., 10.]
            };
            check_products(
                DenseMatrix::new(3, 2, data.clone(), column_major).unwrap(),
                1e-12,
            );
            check_products(
                DenseMatrix::new(3, 2, data.iter().map(|&v| v as f32).collect(), column_major)
                    .unwrap(),
                1e-5,
            );
        }
    }

    #[cfg(feature = "ndarray-bindings")]
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn products_match_eager_for_ndarray() {
        use ndarray::ShapeBuilder;
        for column_major in [false, true] {
            let shape = (3, 2).set_f(column_major);
            let data = if column_major {
                vec![1., 2., 4., 4., 6., 10.]
            } else {
                vec![1., 4., 2., 6., 4., 10.]
            };
            check_products(
                ndarray::Array2::from_shape_vec(shape, data.clone()).unwrap(),
                1e-12,
            );
            check_products(
                ndarray::Array2::from_shape_vec(shape, data.iter().map(|&v| v as f32).collect())
                    .unwrap(),
                1e-5,
            );
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn centered_products_preserve_small_variation_at_large_offsets() {
        let x = DenseMatrix::from_2d_array(&[
            &[1e12 + 1., 1e12 + 4.],
            &[1e12 + 2., 1e12 + 6.],
            &[1e12 + 4., 1e12 + 10.],
        ])
        .unwrap();
        let design = LazyDesign::new(&x, Some(vec![1e12 + 1., 1e12 + 2.]), Some(vec![2., 4.]));
        assert_eq!(design.matvec(&vec![2., -1.]), vec![-0.5, 0., 1.]);
        assert_eq!(design.transpose_matvec(&vec![1., -2., 3.]), vec![3.5, 4.5]);
        let gram = design.gram();
        assert_eq!(*gram.get((0, 0)), 2.5);
        assert_eq!(*gram.get((0, 1)), 3.5);
        assert_eq!(*gram.get((1, 1)), 5.25);
    }

    fn check_panel_gram<T: FloatNumber, X: Array2<T>>() {
        for (n, p) in [
            (0, 3),
            (3, 0),
            (1, 1),
            (255, 31),
            (256, 32),
            (257, 33),
            (513, 65),
        ] {
            for axis in [0, 1] {
                let x = X::from_iterator(
                    (0..n * p).map(|index| {
                        let (i, j) = if axis == 0 {
                            (index / p, index % p)
                        } else {
                            (index % n, index / n)
                        };
                        T::from_f64(100_000.0 + 2.0 * j as f64 + ((i * 7 + j) % 23) as f64 / 8.0)
                            .unwrap()
                    }),
                    n,
                    p,
                    axis,
                );
                let weights: Vec<T> = (0..n)
                    .map(|i| T::from_f64([1.0, 0.0, -0.5, 2.0][i % 4]).unwrap())
                    .collect();
                for centered in [false, true] {
                    for scaled in [false, true] {
                        let centers: Vec<T> = (0..p)
                            .map(|j| {
                                T::from_f64(if centered {
                                    100_000.0 + 2.0 * j as f64
                                } else {
                                    0.0
                                })
                                .unwrap()
                            })
                            .collect();
                        let scales: Vec<T> = (0..p)
                            .map(|j| {
                                T::from_f64(if scaled { [3.0, 0.5, -2.0][j % 3] } else { 1.0 })
                                    .unwrap()
                            })
                            .collect();
                        let mut eager = x.clone();
                        eager.scale_mut(&centers, &scales, 0);
                        let design = LazyDesign::new(
                            &x,
                            centered.then_some(centers),
                            scaled.then_some(scales),
                        );
                        let mut actual = X::fill(p, p, T::nan());
                        design
                            .normalized
                            .weighted_gram_into(
                                &weights,
                                &mut MatrixOutput(&mut actual, PhantomData),
                            )
                            .unwrap();
                        for j in 0..p {
                            for k in 0..=j {
                                let expected = (0..n).fold(T::zero(), |sum, i| {
                                    sum + weights[i] * *eager.get((i, j)) * *eager.get((i, k))
                                });
                                assert_eq!(
                                    *actual.get((j, k)),
                                    expected,
                                    "shape=({n},{p}), axis={axis}, pair=({j},{k})"
                                );
                                assert_eq!(*actual.get((k, j)), expected);
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn panel_gram_preserves_eager_accumulation_across_boundaries() {
        check_panel_gram::<f64, DenseMatrix<f64>>();
        check_panel_gram::<f32, DenseMatrix<f32>>();
        #[cfg(feature = "ndarray-bindings")]
        {
            check_panel_gram::<f64, ndarray::Array2<f64>>();
            check_panel_gram::<f32, ndarray::Array2<f32>>();
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn panel_gram_preserves_nonfinite_products() {
        let x = DenseMatrix::from_2d_array(&[
            &[0.0, f64::INFINITY, 2.0],
            &[1.0, 3.0, f64::NAN],
            &[-2.0, 0.0, 4.0],
        ])
        .unwrap();
        let weights = vec![0.0, 1.0, f64::INFINITY];
        let design = LazyDesign::new(&x, None, Some(vec![1.0, -2.0, f64::INFINITY]));
        let mut eager = x.clone();
        eager.scale_mut(&[0.0; 3], &[1.0, -2.0, f64::INFINITY], 0);
        let mut actual = DenseMatrix::zeros(3, 3);
        design
            .normalized
            .weighted_gram_into(&weights, &mut MatrixOutput(&mut actual, PhantomData))
            .unwrap();
        for j in 0..3 {
            for k in 0..=j {
                let expected = (0..3).fold(0.0, |sum, i| {
                    sum + weights[i] * *eager.get((i, j)) * *eager.get((i, k))
                });
                for index in [(j, k), (k, j)] {
                    if expected.is_nan() {
                        assert!(actual.get(index).is_nan());
                    } else {
                        assert_eq!(*actual.get(index), expected);
                    }
                }
            }
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn implicit_augmentation_matches_materialized_products_and_fit() {
        let x = DenseMatrix::from_2d_array(&[&[1., 4.], &[2., 6.], &[4., 10.]]).unwrap();
        for penalty in [0.0_f64, 0.7, 100.0] {
            let gamma = 1.0 / (1.0 + penalty).sqrt();
            let padding = gamma * penalty.sqrt();
            let design =
                LazyDesign::new(&x, Some(vec![1., 2.]), Some(vec![2., 4.])).with_l2(penalty);
            let mut eager = DenseMatrix::zeros(5, 2);
            for j in 0..2 {
                for i in 0..3 {
                    eager.set((i, j), gamma * (x.get((i, j)) - [1., 2.][j]) / [2., 4.][j]);
                }
                eager.set((3 + j, j), padding);
            }
            let w = vec![2., -1.];
            let y = vec![7., 9., 13., 0., 0.];
            assert_close(&design.matvec(&w), &w.xa(true, &eager), 1e-12);
            assert_close(&design.transpose_matvec(&y), &y.xa(false, &eager), 1e-12);
            assert_close(
                &design.gram().iterator(0).copied().collect::<Vec<_>>(),
                &eager
                    .ab(true, &eager, false)
                    .iterator(0)
                    .copied()
                    .collect::<Vec<_>>(),
                1e-12,
            );
            let mut lazy_opt = InteriorPointOptimizer::new_lazy(&design);
            let mut eager_opt = InteriorPointOptimizer::new(&eager, 2);
            let actual = lazy_opt
                .optimize_lazy(&design, &y, 0.2 * gamma, 1000, 1e-8, true)
                .unwrap();
            let expected = eager_opt
                .optimize(&eager, &y, 0.2 * gamma, 1000, 1e-8, true)
                .unwrap();
            assert_close(&actual, &expected, 1e-6);
        }
    }
}
