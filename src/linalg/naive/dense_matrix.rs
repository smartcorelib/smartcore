extern crate num;
use std::fmt;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;

use serde::de::{Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

use crate::linalg::evd::EVDDecomposableMatrix;
use crate::linalg::lu::LUDecomposableMatrix;
use crate::linalg::qr::QRDecomposableMatrix;
use crate::linalg::svd::SVDDecomposableMatrix;
use crate::linalg::Matrix;
pub use crate::linalg::{BaseMatrix, BaseVector};
use crate::math::num::RealNumber;

impl<T: RealNumber> BaseVector<T> for Vec<T> {
    fn get(&self, i: usize) -> T {
        self[i]
    }
    fn set(&mut self, i: usize, x: T) {
        self[i] = x
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn to_vec(&self) -> Vec<T> {
        let v = self.clone();
        v
    }

    fn zeros(len: usize) -> Self {
        vec![T::zero(); len]
    }

    fn ones(len: usize) -> Self {
        vec![T::one(); len]
    }

    fn fill(len: usize, value: T) -> Self {
        vec![value; len]
    }
}

/// Column-major, dense matrix. See [Simple Dense Matrix](../index.html).
#[derive(Debug, Clone)]
pub struct DenseMatrix<T: RealNumber> {
    ncols: usize,
    nrows: usize,
    values: Vec<T>,
}

/// Column-major, dense matrix. See [Simple Dense Matrix](../index.html).
#[derive(Debug)]
pub struct DenseMatrixIterator<'a, T: RealNumber> {
    cur_c: usize,
    cur_r: usize,
    max_c: usize,
    max_r: usize,
    m: &'a DenseMatrix<T>,
}

impl<T: RealNumber> fmt::Display for DenseMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for r in 0..self.nrows {
            rows.push(
                self.get_row_as_vec(r)
                    .iter()
                    .map(|x| (x.to_f64().unwrap() * 1e4).round() / 1e4)
                    .collect(),
            );
        }
        write!(f, "{:?}", rows)
    }
}

impl<T: RealNumber> DenseMatrix<T> {
    /// Create new instance of `DenseMatrix` without copying data.
    /// `values` should be in column-major order.
    pub fn new(nrows: usize, ncols: usize, values: Vec<T>) -> Self {
        DenseMatrix {
            ncols: ncols,
            nrows: nrows,
            values: values,
        }
    }

    /// New instance of `DenseMatrix` from 2d array.
    pub fn from_2d_array(values: &[&[T]]) -> Self {
        DenseMatrix::from_2d_vec(&values.into_iter().map(|row| Vec::from(*row)).collect())
    }

    /// New instance of `DenseMatrix` from 2d vector.
    pub fn from_2d_vec(values: &Vec<Vec<T>>) -> Self {
        let nrows = values.len();
        let ncols = values
            .first()
            .unwrap_or_else(|| panic!("Cannot create 2d matrix from an empty vector"))
            .len();
        let mut m = DenseMatrix {
            ncols: ncols,
            nrows: nrows,
            values: vec![T::zero(); ncols * nrows],
        };
        for row in 0..nrows {
            for col in 0..ncols {
                m.set(row, col, values[row][col]);
            }
        }
        m
    }

    /// Creates new matrix from an array.
    /// * `nrows` - number of rows in new matrix.
    /// * `ncols` - number of columns in new matrix.
    /// * `values` - values to initialize the matrix.
    pub fn from_array(nrows: usize, ncols: usize, values: &[T]) -> Self {
        DenseMatrix::from_vec(nrows, ncols, &Vec::from(values))
    }

    /// Creates new matrix from a vector.
    /// * `nrows` - number of rows in new matrix.
    /// * `ncols` - number of columns in new matrix.
    /// * `values` - values to initialize the matrix.
    pub fn from_vec(nrows: usize, ncols: usize, values: &Vec<T>) -> DenseMatrix<T> {
        let mut m = DenseMatrix {
            ncols: ncols,
            nrows: nrows,
            values: vec![T::zero(); ncols * nrows],
        };
        for row in 0..nrows {
            for col in 0..ncols {
                m.set(row, col, values[col + row * ncols]);
            }
        }
        m
    }

    /// Creates new row vector (_1xN_ matrix) from an array.     
    /// * `values` - values to initialize the matrix.
    pub fn row_vector_from_array(values: &[T]) -> Self {
        DenseMatrix::row_vector_from_vec(Vec::from(values))
    }

    /// Creates new row vector (_1xN_ matrix) from a vector.
    /// * `values` - values to initialize the matrix.
    pub fn row_vector_from_vec(values: Vec<T>) -> Self {
        DenseMatrix {
            ncols: values.len(),
            nrows: 1,
            values: values,
        }
    }

    /// Creates new column vector (_1xN_ matrix) from an array.     
    /// * `values` - values to initialize the matrix.
    pub fn column_vector_from_array(values: &[T]) -> Self {
        DenseMatrix::column_vector_from_vec(Vec::from(values))
    }

    /// Creates new column vector (_1xN_ matrix) from a vector.     
    /// * `values` - values to initialize the matrix.
    pub fn column_vector_from_vec(values: Vec<T>) -> Self {
        DenseMatrix {
            ncols: 1,
            nrows: values.len(),
            values: values,
        }
    }

    /// Creates new column vector (_1xN_ matrix) from a vector.     
    /// * `values` - values to initialize the matrix.
    pub fn iter<'a>(&'a self) -> DenseMatrixIterator<'a, T> {
        DenseMatrixIterator {
            cur_c: 0,
            cur_r: 0,
            max_c: self.ncols,
            max_r: self.nrows,
            m: &self,
        }
    }
}

impl<'a, T: RealNumber> Iterator for DenseMatrixIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.cur_r * self.max_c + self.cur_c >= self.max_c * self.max_r {
            None
        } else {
            let v = self.m.get(self.cur_r, self.cur_c);
            self.cur_c += 1;
            if self.cur_c >= self.max_c {
                self.cur_c = 0;
                self.cur_r += 1;
            }
            Some(v)
        }
    }
}

impl<'de, T: RealNumber + fmt::Debug + Deserialize<'de>> Deserialize<'de> for DenseMatrix<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            NRows,
            NCols,
            Values,
        }

        struct DenseMatrixVisitor<T: RealNumber + fmt::Debug> {
            t: PhantomData<T>,
        }

        impl<'a, T: RealNumber + fmt::Debug + Deserialize<'a>> Visitor<'a> for DenseMatrixVisitor<T> {
            type Value = DenseMatrix<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct DenseMatrix")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<DenseMatrix<T>, V::Error>
            where
                V: SeqAccess<'a>,
            {
                let nrows = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let ncols = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let values = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                Ok(DenseMatrix::new(nrows, ncols, values))
            }

            fn visit_map<V>(self, mut map: V) -> Result<DenseMatrix<T>, V::Error>
            where
                V: MapAccess<'a>,
            {
                let mut nrows = None;
                let mut ncols = None;
                let mut values = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::NRows => {
                            if nrows.is_some() {
                                return Err(serde::de::Error::duplicate_field("nrows"));
                            }
                            nrows = Some(map.next_value()?);
                        }
                        Field::NCols => {
                            if ncols.is_some() {
                                return Err(serde::de::Error::duplicate_field("ncols"));
                            }
                            ncols = Some(map.next_value()?);
                        }
                        Field::Values => {
                            if values.is_some() {
                                return Err(serde::de::Error::duplicate_field("values"));
                            }
                            values = Some(map.next_value()?);
                        }
                    }
                }
                let nrows = nrows.ok_or_else(|| serde::de::Error::missing_field("nrows"))?;
                let ncols = ncols.ok_or_else(|| serde::de::Error::missing_field("ncols"))?;
                let values = values.ok_or_else(|| serde::de::Error::missing_field("values"))?;
                Ok(DenseMatrix::new(nrows, ncols, values))
            }
        }

        const FIELDS: &'static [&'static str] = &["nrows", "ncols", "values"];
        deserializer.deserialize_struct(
            "DenseMatrix",
            FIELDS,
            DenseMatrixVisitor { t: PhantomData },
        )
    }
}

impl<T: RealNumber + fmt::Debug + Serialize> Serialize for DenseMatrix<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let (nrows, ncols) = self.shape();
        let mut state = serializer.serialize_struct("DenseMatrix", 3)?;
        state.serialize_field("nrows", &nrows)?;
        state.serialize_field("ncols", &ncols)?;
        state.serialize_field("values", &self.values)?;
        state.end()
    }
}

impl<T: RealNumber> SVDDecomposableMatrix<T> for DenseMatrix<T> {}

impl<T: RealNumber> EVDDecomposableMatrix<T> for DenseMatrix<T> {}

impl<T: RealNumber> QRDecomposableMatrix<T> for DenseMatrix<T> {}

impl<T: RealNumber> LUDecomposableMatrix<T> for DenseMatrix<T> {}

impl<T: RealNumber> Matrix<T> for DenseMatrix<T> {}

impl<T: RealNumber> PartialEq for DenseMatrix<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            return false;
        }

        let len = self.values.len();
        let other_len = other.values.len();

        if len != other_len {
            return false;
        }

        for i in 0..len {
            if (self.values[i] - other.values[i]).abs() > T::epsilon() {
                return false;
            }
        }

        true
    }
}

impl<T: RealNumber> Into<Vec<T>> for DenseMatrix<T> {
    fn into(self) -> Vec<T> {
        self.values
    }
}

impl<T: RealNumber> BaseMatrix<T> for DenseMatrix<T> {
    type RowVector = Vec<T>;

    fn from_row_vector(vec: Self::RowVector) -> Self {
        DenseMatrix::new(1, vec.len(), vec)
    }

    fn to_row_vector(self) -> Self::RowVector {
        let mut v = vec![T::zero(); self.nrows * self.ncols];

        for r in 0..self.nrows {
            for c in 0..self.ncols {
                v[r * self.ncols + c] = self.get(r, c);
            }
        }

        v
    }

    fn get(&self, row: usize, col: usize) -> T {
        if row >= self.nrows || col >= self.ncols {
            panic!(
                "Invalid index ({},{}) for {}x{} matrix",
                row, col, self.nrows, self.ncols
            );
        }
        self.values[col * self.nrows + row]
    }

    fn get_row_as_vec(&self, row: usize) -> Vec<T> {
        let mut result = vec![T::zero(); self.ncols];
        for c in 0..self.ncols {
            result[c] = self.get(row, c);
        }
        result
    }

    fn copy_row_as_vec(&self, row: usize, result: &mut Vec<T>) {
        for c in 0..self.ncols {
            result[c] = self.get(row, c);
        }
    }

    fn get_col_as_vec(&self, col: usize) -> Vec<T> {
        let mut result = vec![T::zero(); self.nrows];
        for r in 0..self.nrows {
            result[r] = self.get(r, col);
        }
        result
    }

    fn copy_col_as_vec(&self, col: usize, result: &mut Vec<T>) {
        for r in 0..self.nrows {
            result[r] = self.get(r, col);
        }
    }

    fn set(&mut self, row: usize, col: usize, x: T) {
        self.values[col * self.nrows + row] = x;
    }

    fn zeros(nrows: usize, ncols: usize) -> Self {
        DenseMatrix::fill(nrows, ncols, T::zero())
    }

    fn ones(nrows: usize, ncols: usize) -> Self {
        DenseMatrix::fill(nrows, ncols, T::one())
    }

    fn eye(size: usize) -> Self {
        let mut matrix = Self::zeros(size, size);

        for i in 0..size {
            matrix.set(i, i, T::one());
        }

        return matrix;
    }

    fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    fn v_stack(&self, other: &Self) -> Self {
        if self.ncols != other.ncols {
            panic!("Number of columns in both matrices should be equal");
        }
        let mut result = Self::zeros(self.nrows + other.nrows, self.ncols);
        for c in 0..self.ncols {
            for r in 0..self.nrows + other.nrows {
                if r < self.nrows {
                    result.set(r, c, self.get(r, c));
                } else {
                    result.set(r, c, other.get(r - self.nrows, c));
                }
            }
        }
        result
    }

    fn h_stack(&self, other: &Self) -> Self {
        if self.nrows != other.nrows {
            panic!("Number of rows in both matrices should be equal");
        }
        let mut result = Self::zeros(self.nrows, self.ncols + other.ncols);
        for r in 0..self.nrows {
            for c in 0..self.ncols + other.ncols {
                if c < self.ncols {
                    result.set(r, c, self.get(r, c));
                } else {
                    result.set(r, c, other.get(r, c - self.ncols));
                }
            }
        }
        result
    }

    fn matmul(&self, other: &Self) -> Self {
        if self.ncols != other.nrows {
            panic!("Number of rows of A should equal number of columns of B");
        }
        let inner_d = self.ncols;
        let mut result = Self::zeros(self.nrows, other.ncols);

        for r in 0..self.nrows {
            for c in 0..other.ncols {
                let mut s = T::zero();
                for i in 0..inner_d {
                    s = s + self.get(r, i) * other.get(i, c);
                }
                result.set(r, c, s);
            }
        }

        result
    }

    fn dot(&self, other: &Self) -> T {
        if self.nrows != 1 && other.nrows != 1 {
            panic!("A and B should both be 1-dimentional vectors.");
        }
        if self.nrows * self.ncols != other.nrows * other.ncols {
            panic!("A and B should have the same size");
        }

        let mut result = T::zero();
        for i in 0..(self.nrows * self.ncols) {
            result = result + self.values[i] * other.values[i];
        }

        result
    }

    fn slice(&self, rows: Range<usize>, cols: Range<usize>) -> Self {
        let ncols = cols.len();
        let nrows = rows.len();

        let mut m = DenseMatrix::new(nrows, ncols, vec![T::zero(); nrows * ncols]);

        for r in rows.start..rows.end {
            for c in cols.start..cols.end {
                m.set(r - rows.start, c - cols.start, self.get(r, c));
            }
        }

        m
    }

    fn approximate_eq(&self, other: &Self, error: T) -> bool {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            return false;
        }

        for c in 0..self.ncols {
            for r in 0..self.nrows {
                if (self.get(r, c) - other.get(r, c)).abs() > error {
                    return false;
                }
            }
        }

        true
    }

    fn fill(nrows: usize, ncols: usize, value: T) -> Self {
        DenseMatrix::new(nrows, ncols, vec![value; ncols * nrows])
    }

    fn add_mut(&mut self, other: &Self) -> &Self {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            panic!("A and B should have the same shape");
        }
        for c in 0..self.ncols {
            for r in 0..self.nrows {
                self.add_element_mut(r, c, other.get(r, c));
            }
        }

        self
    }

    fn sub_mut(&mut self, other: &Self) -> &Self {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            panic!("A and B should have the same shape");
        }
        for c in 0..self.ncols {
            for r in 0..self.nrows {
                self.sub_element_mut(r, c, other.get(r, c));
            }
        }

        self
    }

    fn mul_mut(&mut self, other: &Self) -> &Self {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            panic!("A and B should have the same shape");
        }
        for c in 0..self.ncols {
            for r in 0..self.nrows {
                self.mul_element_mut(r, c, other.get(r, c));
            }
        }

        self
    }

    fn div_mut(&mut self, other: &Self) -> &Self {
        if self.ncols != other.ncols || self.nrows != other.nrows {
            panic!("A and B should have the same shape");
        }
        for c in 0..self.ncols {
            for r in 0..self.nrows {
                self.div_element_mut(r, c, other.get(r, c));
            }
        }

        self
    }

    fn div_element_mut(&mut self, row: usize, col: usize, x: T) {
        self.values[col * self.nrows + row] = self.values[col * self.nrows + row] / x;
    }

    fn mul_element_mut(&mut self, row: usize, col: usize, x: T) {
        self.values[col * self.nrows + row] = self.values[col * self.nrows + row] * x;
    }

    fn add_element_mut(&mut self, row: usize, col: usize, x: T) {
        self.values[col * self.nrows + row] = self.values[col * self.nrows + row] + x
    }

    fn sub_element_mut(&mut self, row: usize, col: usize, x: T) {
        self.values[col * self.nrows + row] = self.values[col * self.nrows + row] - x;
    }

    fn transpose(&self) -> Self {
        let mut m = DenseMatrix {
            ncols: self.nrows,
            nrows: self.ncols,
            values: vec![T::zero(); self.ncols * self.nrows],
        };
        for c in 0..self.ncols {
            for r in 0..self.nrows {
                m.set(c, r, self.get(r, c));
            }
        }
        m
    }

    fn rand(nrows: usize, ncols: usize) -> Self {
        let values: Vec<T> = (0..nrows * ncols).map(|_| T::rand()).collect();
        DenseMatrix {
            ncols: ncols,
            nrows: nrows,
            values: values,
        }
    }

    fn norm2(&self) -> T {
        let mut norm = T::zero();

        for xi in self.values.iter() {
            norm = norm + *xi * *xi;
        }

        norm.sqrt()
    }

    fn norm(&self, p: T) -> T {
        if p.is_infinite() && p.is_sign_positive() {
            self.values
                .iter()
                .map(|x| x.abs())
                .fold(T::neg_infinity(), |a, b| a.max(b))
        } else if p.is_infinite() && p.is_sign_negative() {
            self.values
                .iter()
                .map(|x| x.abs())
                .fold(T::infinity(), |a, b| a.min(b))
        } else {
            let mut norm = T::zero();

            for xi in self.values.iter() {
                norm = norm + xi.abs().powf(p);
            }

            norm.powf(T::one() / p)
        }
    }

    fn column_mean(&self) -> Vec<T> {
        let mut mean = vec![T::zero(); self.ncols];

        for r in 0..self.nrows {
            for c in 0..self.ncols {
                mean[c] = mean[c] + self.get(r, c);
            }
        }

        for i in 0..mean.len() {
            mean[i] = mean[i] / T::from(self.nrows).unwrap();
        }

        mean
    }

    fn add_scalar_mut(&mut self, scalar: T) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i] + scalar;
        }
        self
    }

    fn sub_scalar_mut(&mut self, scalar: T) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i] - scalar;
        }
        self
    }

    fn mul_scalar_mut(&mut self, scalar: T) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i] * scalar;
        }
        self
    }

    fn div_scalar_mut(&mut self, scalar: T) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i] / scalar;
        }
        self
    }

    fn negative_mut(&mut self) {
        for i in 0..self.values.len() {
            self.values[i] = -self.values[i];
        }
    }

    fn reshape(&self, nrows: usize, ncols: usize) -> Self {
        if self.nrows * self.ncols != nrows * ncols {
            panic!(
                "Can't reshape {}x{} matrix into {}x{}.",
                self.nrows, self.ncols, nrows, ncols
            );
        }
        let mut dst = DenseMatrix::zeros(nrows, ncols);
        let mut dst_r = 0;
        let mut dst_c = 0;
        for r in 0..self.nrows {
            for c in 0..self.ncols {
                dst.set(dst_r, dst_c, self.get(r, c));
                if dst_c + 1 >= ncols {
                    dst_c = 0;
                    dst_r += 1;
                } else {
                    dst_c += 1;
                }
            }
        }
        dst
    }

    fn copy_from(&mut self, other: &Self) {
        if self.nrows != other.nrows || self.ncols != other.ncols {
            panic!(
                "Can't copy {}x{} matrix into {}x{}.",
                self.nrows, self.ncols, other.nrows, other.ncols
            );
        }

        for i in 0..self.values.len() {
            self.values[i] = other.values[i];
        }
    }

    fn abs_mut(&mut self) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i].abs();
        }
        self
    }

    fn max_diff(&self, other: &Self) -> T {
        let mut max_diff = T::zero();
        for i in 0..self.values.len() {
            max_diff = max_diff.max((self.values[i] - other.values[i]).abs());
        }
        max_diff
    }

    fn sum(&self) -> T {
        let mut sum = T::zero();
        for i in 0..self.values.len() {
            sum = sum + self.values[i];
        }
        sum
    }

    fn max(&self) -> T {
        let mut max = T::neg_infinity();
        for i in 0..self.values.len() {
            max = T::max(max, self.values[i]);
        }
        max
    }

    fn min(&self) -> T {
        let mut min = T::infinity();
        for i in 0..self.values.len() {
            min = T::min(min, self.values[i]);
        }
        min
    }

    fn softmax_mut(&mut self) {
        let max = self
            .values
            .iter()
            .map(|x| x.abs())
            .fold(T::neg_infinity(), |a, b| a.max(b));
        let mut z = T::zero();
        for r in 0..self.nrows {
            for c in 0..self.ncols {
                let p = (self.get(r, c) - max).exp();
                self.set(r, c, p);
                z = z + p;
            }
        }
        for r in 0..self.nrows {
            for c in 0..self.ncols {
                self.set(r, c, self.get(r, c) / z);
            }
        }
    }

    fn pow_mut(&mut self, p: T) -> &Self {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i].powf(p);
        }
        self
    }

    fn argmax(&self) -> Vec<usize> {
        let mut res = vec![0usize; self.nrows];

        for r in 0..self.nrows {
            let mut max = T::neg_infinity();
            let mut max_pos = 0usize;
            for c in 0..self.ncols {
                let v = self.get(r, c);
                if max < v {
                    max = v;
                    max_pos = c;
                }
            }
            res[r] = max_pos;
        }

        res
    }

    fn unique(&self) -> Vec<T> {
        let mut result = self.values.clone();
        result.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result.dedup();
        result
    }

    fn cov(&self) -> Self {
        let (m, n) = self.shape();

        let mu = self.column_mean();

        let mut cov = Self::zeros(n, n);

        for k in 0..m {
            for i in 0..n {
                for j in 0..=i {
                    cov.add_element_mut(i, j, (self.get(k, i) - mu[i]) * (self.get(k, j) - mu[j]));
                }
            }
        }

        let m_t = T::from(m - 1).unwrap();

        for i in 0..n {
            for j in 0..=i {
                cov.div_element_mut(i, j, m_t);
                cov.set(j, i, cov.get(i, j));
            }
        }

        cov
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_array() {
        let vec = [1., 2., 3., 4., 5., 6.];
        assert_eq!(
            DenseMatrix::from_array(3, 2, &vec),
            DenseMatrix::new(3, 2, vec![1., 3., 5., 2., 4., 6.])
        );
        assert_eq!(
            DenseMatrix::from_array(2, 3, &vec),
            DenseMatrix::new(2, 3, vec![1., 4., 2., 5., 3., 6.])
        );
    }

    #[test]
    fn row_column_vec_from_array() {
        let vec = vec![1., 2., 3., 4., 5., 6.];
        assert_eq!(
            DenseMatrix::row_vector_from_array(&vec),
            DenseMatrix::new(1, 6, vec![1., 2., 3., 4., 5., 6.])
        );
        assert_eq!(
            DenseMatrix::column_vector_from_array(&vec),
            DenseMatrix::new(6, 1, vec![1., 2., 3., 4., 5., 6.])
        );
    }

    #[test]
    fn from_to_row_vec() {
        let vec = vec![1., 2., 3.];
        assert_eq!(
            DenseMatrix::from_row_vector(vec.clone()),
            DenseMatrix::new(1, 3, vec![1., 2., 3.])
        );
        assert_eq!(
            DenseMatrix::from_row_vector(vec.clone()).to_row_vector(),
            vec![1., 2., 3.]
        );
    }

    #[test]
    fn iter() {
        let vec = vec![1., 2., 3., 4., 5., 6.];
        let m = DenseMatrix::from_array(3, 2, &vec);
        assert_eq!(vec, m.iter().collect::<Vec<f32>>());
    }

    #[test]
    fn v_stack() {
        let a = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]);
        let b = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.]]);
        let expected = DenseMatrix::from_2d_array(&[
            &[1., 2., 3.],
            &[4., 5., 6.],
            &[7., 8., 9.],
            &[1., 2., 3.],
            &[4., 5., 6.],
        ]);
        let result = a.v_stack(&b);
        assert_eq!(result, expected);
    }

    #[test]
    fn h_stack() {
        let a = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]);
        let b = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.]]);
        let expected = DenseMatrix::from_2d_array(&[
            &[1., 2., 3., 1., 2.],
            &[4., 5., 6., 3., 4.],
            &[7., 8., 9., 5., 6.],
        ]);
        let result = a.h_stack(&b);
        assert_eq!(result, expected);
    }

    #[test]
    fn matmul() {
        let a = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.]]);
        let b = DenseMatrix::from_2d_array(&[&[1., 2.], &[3., 4.], &[5., 6.]]);
        let expected = DenseMatrix::from_2d_array(&[&[22., 28.], &[49., 64.]]);
        let result = a.matmul(&b);
        assert_eq!(result, expected);
    }

    #[test]
    fn dot() {
        let a = DenseMatrix::from_array(1, 3, &[1., 2., 3.]);
        let b = DenseMatrix::from_array(1, 3, &[4., 5., 6.]);
        assert_eq!(a.dot(&b), 32.);
    }

    #[test]
    fn slice() {
        let m = DenseMatrix::from_2d_array(&[
            &[1., 2., 3., 1., 2.],
            &[4., 5., 6., 3., 4.],
            &[7., 8., 9., 5., 6.],
        ]);
        let expected = DenseMatrix::from_2d_array(&[&[2., 3.], &[5., 6.]]);
        let result = m.slice(0..2, 1..3);
        assert_eq!(result, expected);
    }

    #[test]
    fn approximate_eq() {
        let m = DenseMatrix::from_2d_array(&[&[2., 3.], &[5., 6.]]);
        let m_eq = DenseMatrix::from_2d_array(&[&[2.5, 3.0], &[5., 5.5]]);
        let m_neq = DenseMatrix::from_2d_array(&[&[3.0, 3.0], &[5., 6.5]]);
        assert!(m.approximate_eq(&m_eq, 0.5));
        assert!(!m.approximate_eq(&m_neq, 0.5));
    }

    #[test]
    fn rand() {
        let m: DenseMatrix<f64> = DenseMatrix::rand(3, 3);
        for c in 0..3 {
            for r in 0..3 {
                assert!(m.get(r, c) != 0f64);
            }
        }
    }

    #[test]
    fn transpose() {
        let m = DenseMatrix::from_2d_array(&[&[1.0, 3.0], &[2.0, 4.0]]);
        let expected = DenseMatrix::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let m_transposed = m.transpose();
        for c in 0..2 {
            for r in 0..2 {
                assert!(m_transposed.get(r, c) == expected.get(r, c));
            }
        }
    }

    #[test]
    fn reshape() {
        let m_orig = DenseMatrix::row_vector_from_array(&[1., 2., 3., 4., 5., 6.]);
        let m_2_by_3 = m_orig.reshape(2, 3);
        let m_result = m_2_by_3.reshape(1, 6);
        assert_eq!(m_2_by_3.shape(), (2, 3));
        assert_eq!(m_2_by_3.get(1, 1), 5.);
        assert_eq!(m_result.get(0, 1), 2.);
        assert_eq!(m_result.get(0, 3), 4.);
    }

    #[test]
    fn norm() {
        let v = DenseMatrix::row_vector_from_array(&[3., -2., 6.]);
        assert_eq!(v.norm(1.), 11.);
        assert_eq!(v.norm(2.), 7.);
        assert_eq!(v.norm(std::f64::INFINITY), 6.);
        assert_eq!(v.norm(std::f64::NEG_INFINITY), 2.);
    }

    #[test]
    fn softmax_mut() {
        let mut prob: DenseMatrix<f64> = DenseMatrix::row_vector_from_array(&[1., 2., 3.]);
        prob.softmax_mut();
        assert!((prob.get(0, 0) - 0.09).abs() < 0.01);
        assert!((prob.get(0, 1) - 0.24).abs() < 0.01);
        assert!((prob.get(0, 2) - 0.66).abs() < 0.01);
    }

    #[test]
    fn col_mean() {
        let a = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.], &[7., 8., 9.]]);
        let res = a.column_mean();
        assert_eq!(res, vec![4., 5., 6.]);
    }

    #[test]
    fn min_max_sum() {
        let a = DenseMatrix::from_2d_array(&[&[1., 2., 3.], &[4., 5., 6.]]);
        assert_eq!(21., a.sum());
        assert_eq!(1., a.min());
        assert_eq!(6., a.max());
    }

    #[test]
    fn eye() {
        let a = DenseMatrix::from_2d_array(&[&[1., 0., 0.], &[0., 1., 0.], &[0., 0., 1.]]);
        let res = DenseMatrix::eye(3);
        assert_eq!(res, a);
    }

    #[test]
    fn to_from_json() {
        let a = DenseMatrix::from_2d_array(&[&[0.9, 0.4, 0.7], &[0.4, 0.5, 0.3], &[0.7, 0.3, 0.8]]);
        let deserialized_a: DenseMatrix<f64> =
            serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
        assert_eq!(a, deserialized_a);
    }

    #[test]
    fn to_from_bincode() {
        let a = DenseMatrix::from_2d_array(&[&[0.9, 0.4, 0.7], &[0.4, 0.5, 0.3], &[0.7, 0.3, 0.8]]);
        let deserialized_a: DenseMatrix<f64> =
            bincode::deserialize(&bincode::serialize(&a).unwrap()).unwrap();
        assert_eq!(a, deserialized_a);
    }

    #[test]
    fn to_string() {
        let a = DenseMatrix::from_2d_array(&[&[0.9, 0.4, 0.7], &[0.4, 0.5, 0.3], &[0.7, 0.3, 0.8]]);
        assert_eq!(
            format!("{}", a),
            "[[0.9, 0.4, 0.7], [0.4, 0.5, 0.3], [0.7, 0.3, 0.8]]"
        );
    }

    #[test]
    fn cov() {
        let a = DenseMatrix::from_2d_array(&[
            &[64.0, 580.0, 29.0],
            &[66.0, 570.0, 33.0],
            &[68.0, 590.0, 37.0],
            &[69.0, 660.0, 46.0],
            &[73.0, 600.0, 55.0],
        ]);
        let expected = DenseMatrix::from_2d_array(&[
            &[11.5, 50.0, 34.75],
            &[50.0, 1250.0, 205.0],
            &[34.75, 205.0, 110.0],
        ]);
        assert_eq!(a.cov(), expected);
    }
}
