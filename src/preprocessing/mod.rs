use crate::linalg::Matrix;
use crate::math::num::RealNumber;

pub(crate) fn binarize<T: RealNumber, M: Matrix<T>>(x: &M, threshold: T) -> M {
    let (nrows, ncols) = x.shape();
    let mut output = M::zeros(nrows, ncols);

    for row in 0..nrows {
        for col in 0..ncols {
            if x.get(row, col) > threshold {
                output.set(row, col, T::one());
            } else {
                output.set(row, col, T::zero());
            }
        }
    }

    output
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::linalg::naive::dense_matrix::DenseMatrix;

    #[test]
    fn binarize_1() {
        let x = DenseMatrix::from_2d_array(&[
            &[-1., 2.],
            &[-2., 0.],
            &[5., -2.],
            &[1., 1.],
            &[-3., 1.],
            &[3., 0.],
        ]);
        let expected = DenseMatrix::from_2d_array(&[
            &[0., 1.],
            &[0., 0.],
            &[1., 0.],
            &[1., 1.],
            &[0., 1.],
            &[1., 0.],
        ]);
        assert_eq!(binarize(&x, 0.), expected);
    }
}
