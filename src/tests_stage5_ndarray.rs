//! Stage 5: ndarray-bindings parity tests.
//!
//! Ensures `ArrayBase<OwnedRepr<T>, Ix2>` (ndarray) behaves identically to
//! `DenseMatrix<T>` for all operations covered in `linalg/basic/`.
//!
//! Gated on `#[cfg(feature = "ndarray-bindings")]`.
//! Tracking issue: #396 / #391.

#[cfg(all(test, feature = "ndarray-bindings"))]
mod ndarray_parity {
    use crate::linalg::basic::arrays::{Array, Array2, ArrayView1, ArrayView2};
    use crate::linalg::basic::matrix::DenseMatrix;
    use ndarray::{arr2, Array2 as NdArray2};

    fn assert_close_f64(a: f64, b: f64, tol: f64, label: &str) {
        assert!((a - b).abs() < tol, "{label}: expected {b}, got {a}");
    }

    // ── shape ─────────────────────────────────────────────────────────────────

    #[test]
    fn ndarray_shape_parity() {
        let nd: NdArray2<f64> = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let dm = DenseMatrix::from_ndarray2(&nd);
        assert_eq!(Array::shape(&nd), Array::shape(&dm));
    }

    // ── get ───────────────────────────────────────────────────────────────────

    #[test]
    fn ndarray_get_parity() {
        let nd: NdArray2<i32> = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let dm = DenseMatrix::from_ndarray2(&nd);
        for r in 0..2 {
            for c in 0..3 {
                assert_eq!(
                    Array::get(&nd, (r, c)),
                    Array::get(&dm, (r, c)),
                    "get({r},{c}) mismatch"
                );
            }
        }
    }

    // ── is_empty ─────────────────────────────────────────────────────────────

    #[test]
    fn ndarray_is_empty_parity() {
        let empty_nd = NdArray2::<f64>::zeros((0, 0));
        let empty_dm = DenseMatrix::from_ndarray2(&empty_nd);
        assert_eq!(Array::is_empty(&empty_nd), Array::is_empty(&empty_dm));

        let nonempty_nd: NdArray2<f64> = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let nonempty_dm = DenseMatrix::from_ndarray2(&nonempty_nd);
        assert_eq!(Array::is_empty(&nonempty_nd), Array::is_empty(&nonempty_dm));
    }

    // ── iterator axis=0 (row-major) ───────────────────────────────────────────

    #[test]
    fn ndarray_iterator_axis0_parity() {
        let nd: NdArray2<i32> = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let dm = DenseMatrix::from_ndarray2(&nd);
        let nd_vals: Vec<i32> = nd.iterator(0).copied().collect();
        let dm_vals: Vec<i32> = dm.iterator(0).copied().collect();
        assert_eq!(nd_vals, dm_vals, "axis=0 iterator mismatch");
    }

    // ── iterator axis=1 (column-major) ────────────────────────────────────────

    #[test]
    fn ndarray_iterator_axis1_parity() {
        let nd: NdArray2<i32> = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let dm = DenseMatrix::from_ndarray2(&nd);
        let nd_vals: Vec<i32> = nd.iterator(1).copied().collect();
        let dm_vals: Vec<i32> = dm.iterator(1).copied().collect();
        assert_eq!(nd_vals, dm_vals, "axis=1 iterator mismatch");
    }

    // ── get_row ───────────────────────────────────────────────────────────────

    #[test]
    fn ndarray_get_row_parity() {
        let nd: NdArray2<i32> = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let dm = DenseMatrix::from_ndarray2(&nd);
        for r in 0..3 {
            let nd_row = Array2::get_row(&nd, r);
            let dm_row = Array2::get_row(&dm, r);
            assert_eq!(nd_row.shape(), dm_row.shape());
            for c in 0..3 {
                assert_eq!(nd_row.get(c), dm_row.get(c), "row {r} col {c}");
            }
        }
    }

    // ── get_col ───────────────────────────────────────────────────────────────

    #[test]
    fn ndarray_get_col_parity() {
        let nd: NdArray2<i32> = arr2(&[[1, 2], [3, 4], [5, 6]]);
        let dm = DenseMatrix::from_ndarray2(&nd);
        for c in 0..2 {
            let nd_col = Array2::get_col(&nd, c);
            let dm_col = Array2::get_col(&dm, c);
            assert_eq!(nd_col.shape(), dm_col.shape());
            for r in 0..3 {
                assert_eq!(nd_col.get(r), dm_col.get(r), "col {c} row {r}");
            }
        }
    }

    // ── slice ─────────────────────────────────────────────────────────────────

    #[test]
    fn ndarray_slice_parity() {
        let nd: NdArray2<i32> = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let dm = DenseMatrix::from_ndarray2(&nd);
        let nd_slice = Array2::slice(&nd, 1..3, 0..2);
        let dm_slice = Array2::slice(&dm, 1..3, 0..2);
        assert_eq!(nd_slice.shape(), dm_slice.shape());
        for r in 0..2 {
            for c in 0..2 {
                assert_eq!(nd_slice.get((r, c)), dm_slice.get((r, c)), "slice ({r},{c})");
            }
        }
    }

    // ── fill ─────────────────────────────────────────────────────────────────

    #[test]
    fn ndarray_fill_parity() {
        let nd = <NdArray2<f64> as Array2<f64>>::fill(3, 4, 7.0);
        let dm = <DenseMatrix<f64> as Array2<f64>>::fill(3, 4, 7.0);
        assert_eq!(Array::shape(&nd), Array::shape(&dm));
        assert_eq!(nd.iterator(0).copied().collect::<Vec<_>>(),
                   dm.iterator(0).copied().collect::<Vec<_>>());
    }

    // ── transpose ─────────────────────────────────────────────────────────────

    #[test]
    fn ndarray_transpose_parity() {
        let nd: NdArray2<i32> = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let dm = DenseMatrix::from_ndarray2(&nd);
        let nd_t = Array2::transpose(&nd);
        let dm_t = Array2::transpose(&dm);
        assert_eq!(Array::shape(&nd_t), Array::shape(&dm_t));
        let nd_vals: Vec<i32> = nd_t.iterator(0).copied().collect();
        let dm_vals: Vec<i32> = dm_t.iterator(0).copied().collect();
        assert_eq!(nd_vals, dm_vals, "transpose row-major values mismatch");
    }

    // ── from_ndarray2 round-trip ──────────────────────────────────────────────

    #[test]
    fn from_ndarray2_round_trip_values() {
        let original: NdArray2<f64> = arr2(&[
            [1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6],
            [7.7, 8.8, 9.9],
        ]);
        let dm = DenseMatrix::from_ndarray2(&original);
        for r in 0..3 {
            for c in 0..3 {
                assert_close_f64(
                    *Array::get(&dm, (r, c)),
                    *original.get((r, c)).unwrap(),
                    1e-10,
                    &format!("round-trip ({r},{c})"),
                );
            }
        }
    }

    // ── Fortran-order (transposed layout) ────────────────────────────────────

    #[test]
    fn from_ndarray2_fortran_order_correct() {
        let c_order: NdArray2<i32> = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let f_order = c_order.t().to_owned(); // shape (3,2), Fortran layout
        let dm = DenseMatrix::from_ndarray2(&f_order);
        // from_ndarray2 must always read logical order, so element (0,0) == 1
        assert_eq!(*Array::get(&dm, (0, 0)), *f_order.get((0, 0)).unwrap());
        assert_eq!(*Array::get(&dm, (2, 1)), *f_order.get((2, 1)).unwrap());
    }
}
