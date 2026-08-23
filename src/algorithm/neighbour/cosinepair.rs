///
/// ### CosinePair: Data-structure for the dynamic closest-pair problem.
///
/// The structure keeps, for every row, the cosine-distance closest neighbour
/// found by an exact symmetric half-scan. Construction costs Theta(n^2) dot
/// products; `top_k` does not make it sub-quadratic.
///
/// Reference:
///  Eppstein, David: Fast hierarchical clustering and other applications of
///  dynamic closest pairs. Journal of Experimental Algorithmics 5 (2000) 1.
///
/// Example:
/// ```
/// use smartcore::metrics::distance::PairwiseDistance;
/// use smartcore::linalg::basic::matrix::DenseMatrix;
/// use smartcore::algorithm::neighbour::cosinepair::CosinePair;
/// let x = DenseMatrix::<f64>::from_2d_array(&[
///     &[5.1, 3.5, 1.4, 0.2],
///     &[4.9, 3.0, 1.4, 0.2],
///     &[4.7, 3.2, 1.3, 0.2],
///     &[4.6, 3.1, 1.5, 0.2],
///     &[5.0, 3.6, 1.4, 0.2],
///     &[5.4, 3.9, 1.7, 0.4],
/// ]).unwrap();
/// let cosinepair = CosinePair::new(&x);
/// let closest_pair: PairwiseDistance<f64> = cosinepair.unwrap().closest_pair();
/// ```
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use ordered_float::{FloatCore, OrderedFloat};

use std::collections::{BinaryHeap, HashMap};

use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::{Array2, ArrayView1};
use crate::metrics::distance::PairwiseDistance;
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;

/// Parameters for CosinePair construction
#[derive(Debug, Clone)]
pub struct CosinePairParameters {
    /// Maximum number of neighbours returned by
    /// [`CosinePair::query_row_top_k`] (default: all points). The build stays
    /// an exact Theta(n^2) scan regardless of this value.
    pub top_k: Option<usize>,
    /// When `true`, queries score only `top_k` evenly strided candidate rows
    /// instead of every row, so results are approximate. When `false`
    /// (default), queries are exact.
    pub approximate: bool,
}

#[expect(clippy::derivable_impls)]
impl Default for CosinePairParameters {
    fn default() -> Self {
        Self {
            top_k: None,
            approximate: false,
        }
    }
}

///
/// Inspired by Python implementation:
/// <https://github.com/carsonfarmer/fastpair/blob/b8b4d3000ab6f795a878936667eee1b557bf353d/fastpair/base.py>
/// MIT License (MIT) Copyright (c) 2016 Carson Farmer
///
/// affinity used is Cosine as it is the most used
///
/// Construction performs a symmetric half-scan over all unordered row pairs,
/// so it costs Theta(n^2) dot products over zero-copy row views, with the
/// Euclidean norm of every row precomputed once in O(n * d). The `top_k`
/// parameter bounds the number of neighbours kept per row by
/// [`CosinePair::query_row_top_k`]; it does not make the construction
/// sub-quadratic.
///
#[derive(Debug, Clone)]
pub struct CosinePair<'a, T: RealNumber + FloatNumber, M: Array2<T>> {
    /// initial matrix
    pub samples: &'a M,
    /// closest pair hashmap (connectivity matrix for closest pairs)
    pub distances: HashMap<usize, PairwiseDistance<T>>,
    /// conga line used to keep track of the closest pair
    pub neighbours: Vec<usize>,
    /// Euclidean norm (L2) of each row, computed once during construction
    row_norms: Vec<f64>,
    /// parameters used during construction
    pub parameters: CosinePairParameters,
}

impl<'a, T: RealNumber + FloatNumber + FloatCore, M: Array2<T>> CosinePair<'a, T, M> {
    /// Constructor with default parameters (backward compatibility)
    pub fn new(m: &'a M) -> Result<Self, Failed> {
        Self::with_parameters(m, CosinePairParameters::default())
    }

    /// Constructor that caps the number of neighbours returned by
    /// [`CosinePair::query_row_top_k`] at `top_k`. Queries stay exact; set
    /// `approximate` through [`CosinePair::with_parameters`] to score only
    /// strided candidates.
    pub fn with_top_k(m: &'a M, top_k: usize) -> Result<Self, Failed> {
        Self::with_parameters(
            m,
            CosinePairParameters {
                top_k: Some(top_k),
                approximate: false,
            },
        )
    }

    /// Constructor with full parameter control
    pub fn with_parameters(m: &'a M, parameters: CosinePairParameters) -> Result<Self, Failed> {
        if m.shape().0 < 2 {
            return Err(Failed::because(
                FailedError::FindFailed,
                "min number of rows should be 2",
            ));
        }

        let row_norms = (0..m.shape().0).map(|i| m.get_row(i).norm2()).collect();

        let mut init = Self {
            samples: m,
            distances: HashMap::with_capacity(m.shape().0),
            neighbours: Vec::with_capacity(m.shape().0),
            row_norms,
            parameters,
        };
        init.init();
        Ok(init)
    }

    /// Helper function to create ordered float wrapper
    fn ordered_float(value: T) -> OrderedFloat<T> {
        OrderedFloat(value)
    }

    /// Helper function to extract value from ordered float wrapper
    fn extract_float(ordered: OrderedFloat<T>) -> T {
        ordered.into_inner()
    }

    /// Cosine distance between two rows seen as zero-copy views, reusing the
    /// norms precomputed at construction time. Mirrors
    /// `Cosine::new().distance(...)`: a zero-magnitude row yields the
    /// sentinel distance `1 - f64::MIN`.
    fn cosine_distance_with_norms(
        row_i: &dyn ArrayView1<T>,
        norm_i: f64,
        row_j: &dyn ArrayView1<T>,
        norm_j: f64,
    ) -> T {
        let similarity = if norm_i == 0.0 || norm_j == 0.0 {
            f64::MIN
        } else {
            row_i.dot(row_j).to_f64().unwrap() / (norm_i * norm_j)
        };
        T::from(1.0 - similarity).unwrap()
    }

    /// Cosine distance between two rows of the sample matrix
    fn row_distance(&self, i: usize, j: usize) -> T {
        let row_i = self.samples.get_row(i);
        let row_j = self.samples.get_row(j);
        Self::cosine_distance_with_norms(
            row_i.as_ref(),
            self.row_norms[i],
            row_j.as_ref(),
            self.row_norms[j],
        )
    }

    /// Exact closest-neighbour search per row.
    ///
    /// Cosine distance is symmetric, so each unordered pair `(i, j)` with
    /// `i < j` is evaluated once and updates the running best candidate of
    /// both rows. This halves the Theta(n^2) distance evaluations and avoids
    /// all per-pair allocations by operating on row views.
    fn init(&mut self) {
        let len = self.samples.shape().0;

        let mut distances = HashMap::with_capacity(len);
        let mut neighbours = Vec::with_capacity(len);

        neighbours.extend(0..len);

        // best[i] = Some((distance, neighbour index)) of the closest row to i
        // found so far; `None` until the first candidate arrives
        let mut best: Vec<Option<(OrderedFloat<T>, usize)>> = vec![None; len];

        for i in 0..len {
            for j in (i + 1)..len {
                let distance = Self::ordered_float(self.row_distance(i, j));
                if best[i].is_none_or(|(d, _)| distance < d) {
                    best[i] = Some((distance, j));
                }
                if best[j].is_none_or(|(d, _)| distance < d) {
                    best[j] = Some((distance, i));
                }
            }
        }

        for (i, best_of_i) in best.iter().enumerate() {
            let (distance, neighbour) = best_of_i.expect("every row has at least one neighbour");
            distances.insert(
                i,
                PairwiseDistance {
                    node: i,
                    neighbour: Some(neighbour),
                    distance: Some(Self::extract_float(distance)),
                },
            );
        }

        self.distances = distances;
        self.neighbours = neighbours;
    }

    /// Query the `k` nearest neighbours of a dataset row by cosine distance.
    ///
    /// When `parameters.approximate` is `false` (the default), every row is
    /// scored through a zero-copy view and the returned neighbours are exact.
    /// When `approximate` is `true` and `top_k` is set, only `top_k` evenly
    /// strided candidate rows are scored (`step = n / top_k`), so the result
    /// is approximate. The number of returned neighbours is capped at `top_k`.
    pub fn query_row_top_k(
        &self,
        query_row_index: usize,
        k: usize,
    ) -> Result<Vec<(T, usize)>, Failed> {
        if query_row_index >= self.samples.shape().0 {
            return Err(Failed::because(
                FailedError::FindFailed,
                "Query row index out of bounds",
            ));
        }

        if k == 0 {
            return Ok(Vec::new());
        }

        let n = self.samples.shape().0;
        let max_candidates = self.parameters.top_k.unwrap_or(n);
        let actual_k: usize = k.min(max_candidates);

        // Max-heap of the `actual_k` closest candidates seen so far: the
        // greatest entry is evicted first, so the heap keeps the nearest rows
        let mut heap = BinaryHeap::with_capacity(actual_k + 1);

        let query_row = self.samples.get_row(query_row_index);
        let query_norm = self.row_norms[query_row_index];

        let score_candidate = |heap: &mut BinaryHeap<(OrderedFloat<T>, usize)>, index: usize| {
            let row = self.samples.get_row(index);
            let distance = Self::cosine_distance_with_norms(
                query_row.as_ref(),
                query_norm,
                row.as_ref(),
                self.row_norms[index],
            );
            heap.push((Self::ordered_float(distance), index));
            if heap.len() > actual_k {
                heap.pop();
            }
        };

        match (self.parameters.approximate, self.parameters.top_k) {
            (true, Some(top_k)) => {
                let step = (n / top_k).max(1);
                for candidate in (0..n)
                    .step_by(step)
                    .filter(|&i| i != query_row_index)
                    .take(top_k)
                {
                    score_candidate(&mut heap, candidate);
                }
            }
            _ => {
                for candidate in (0..n).filter(|&i| i != query_row_index) {
                    score_candidate(&mut heap, candidate);
                }
            }
        }

        // Convert heap to a vector sorted by ascending distance, ties broken
        // by ascending index
        let mut neighbors: Vec<(T, usize)> = heap
            .into_iter()
            .map(|(distance, index)| (Self::extract_float(distance), index))
            .collect();
        neighbors.sort_by(|a, b| {
            Self::ordered_float(a.0)
                .cmp(&Self::ordered_float(b.0))
                .then(a.1.cmp(&b.1))
        });

        Ok(neighbors)
    }

    /// Query k nearest neighbors for a row that's already in the dataset
    pub fn query_row(&self, query_row_index: usize, k: usize) -> Result<Vec<(T, usize)>, Failed> {
        if query_row_index >= self.samples.shape().0 {
            return Err(Failed::because(
                FailedError::FindFailed,
                "Query row index out of bounds",
            ));
        }

        if k == 0 {
            return Ok(Vec::new());
        }

        // Get distances to all other points
        let mut distances = self.distances_from(query_row_index);

        // Sort by distance (ascending)
        distances.sort_by(|a, b| {
            a.distance
                .unwrap()
                .partial_cmp(&b.distance.unwrap())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top k neighbors and convert to (distance, index) format
        let neighbors: Vec<(T, usize)> = distances
            .into_iter()
            .take(k)
            .map(|pd| (pd.distance.unwrap(), pd.neighbour.unwrap()))
            .collect();

        Ok(neighbors)
    }

    /// Query k nearest neighbors for an external query vector
    pub fn query(&self, query_vector: &Vec<T>, k: usize) -> Result<Vec<(T, usize)>, Failed> {
        if query_vector.len() != self.samples.shape().1 {
            return Err(Failed::because(
                FailedError::FindFailed,
                "Query vector dimension mismatch",
            ));
        }

        if k == 0 {
            return Ok(Vec::new());
        }

        // Compute distances from query vector to all points in the dataset
        // through zero-copy row views, reusing the precomputed row norms
        let query_norm = query_vector.norm2();
        let mut distances = Vec::<PairwiseDistance<T>>::with_capacity(self.samples.shape().0);

        for i in 0..self.samples.shape().0 {
            let dataset_point = self.samples.get_row(i);

            distances.push(PairwiseDistance {
                node: i, // This represents the dataset point index
                neighbour: Some(i),
                distance: Some(Self::cosine_distance_with_norms(
                    query_vector,
                    query_norm,
                    dataset_point.as_ref(),
                    self.row_norms[i],
                )),
            });
        }

        // Sort by distance (ascending)
        distances.sort_by(|a, b| {
            a.distance
                .unwrap()
                .partial_cmp(&b.distance.unwrap())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top k neighbors and convert to (distance, index) format
        let neighbors: Vec<(T, usize)> = distances
            .into_iter()
            .take(k)
            .map(|pd| (pd.distance.unwrap(), pd.node))
            .collect();

        Ok(neighbors)
    }

    /// Optimized version that reuses the existing distances_from method
    /// This is more efficient for queries that are points already in the dataset
    pub fn query_optimized(
        &self,
        query_row_index: usize,
        k: usize,
    ) -> Result<Vec<(T, usize)>, Failed> {
        // Reuse existing method and sort the results
        self.query_row(query_row_index, k)
    }

    /// Find closest pair by scanning list of nearest neighbors.
    #[allow(dead_code)]
    pub fn closest_pair(&self) -> PairwiseDistance<T> {
        let mut a = self.neighbours[0]; // Start with first point
        let mut d = self.distances[&a].distance;
        for p in self.neighbours.iter() {
            if self.distances[p].distance < d {
                a = *p; // Update `a` and distance `d`
                d = self.distances[p].distance;
            }
        }
        let b = self.distances[&a].neighbour;
        PairwiseDistance {
            node: a,
            neighbour: b,
            distance: d,
        }
    }

    ///
    /// Return order dissimilarities from closest to furthest
    ///
    #[allow(dead_code)]
    pub fn ordered_pairs(&self) -> std::vec::IntoIter<&PairwiseDistance<T>> {
        // improvement: implement this to return `impl Iterator<Item = &PairwiseDistance<T>>`
        // need to implement trait `Iterator` for `Vec<&PairwiseDistance<T>>`
        let mut distances = self
            .distances
            .values()
            .collect::<Vec<&PairwiseDistance<T>>>();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        distances.into_iter()
    }

    //
    // Compute distances from input to all other points in data-structure.
    // input is the row index of the sample matrix
    //
    #[allow(dead_code)]
    fn distances_from(&self, index_row: usize) -> Vec<PairwiseDistance<T>> {
        let mut distances = Vec::<PairwiseDistance<T>>::with_capacity(self.samples.shape().0);
        let query_row = self.samples.get_row(index_row);
        let query_norm = self.row_norms[index_row];
        for other in self.neighbours.iter() {
            if index_row != *other {
                let row = self.samples.get_row(*other);
                distances.push(PairwiseDistance {
                    node: index_row,
                    neighbour: Some(*other),
                    distance: Some(Self::cosine_distance_with_norms(
                        query_row.as_ref(),
                        query_norm,
                        row.as_ref(),
                        self.row_norms[*other],
                    )),
                })
            }
        }
        distances
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::arrays::Array1;
    use crate::linalg::basic::{arrays::Array, matrix::DenseMatrix};
    use crate::metrics::distance::Distance;
    use crate::metrics::distance::cosine::Cosine;
    use approx::{assert_relative_eq, relative_eq};

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_pair_initialization() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
        ])
        .unwrap();

        let cosine_pair = CosinePair::new(&x);

        assert!(cosine_pair.is_ok());
        let cp = cosine_pair.unwrap();

        assert_eq!(cp.samples.shape().0, 6);
        assert_eq!(cp.distances.len(), 6);
        assert_eq!(cp.neighbours.len(), 6);
        assert!(!cp.distances.is_empty());
        assert!(!cp.neighbours.is_empty());
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_pair_minimum_rows_error() {
        // Test with only one row - should fail
        let x = DenseMatrix::<f64>::from_2d_array(&[&[5.1, 3.5, 1.4, 0.2]]).unwrap();

        let result = CosinePair::new(&x);
        assert!(result.is_err());

        if let Err(e) = result {
            let expected_error =
                Failed::because(FailedError::FindFailed, "min number of rows should be 2");
            assert_eq!(e, expected_error);
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_pair_closest_pair() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 1.0],
            &[2.0, 2.0], // This should be closest to [1.0, 1.0] with cosine distance
        ])
        .unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();
        let closest_pair = cosine_pair.closest_pair();

        // Verify structure
        assert!(closest_pair.distance.is_some());
        assert!(closest_pair.neighbour.is_some());

        // The closest pair should have the smallest cosine distance
        let distance = closest_pair.distance.unwrap();
        assert!(distance >= 0.0 && distance <= 2.0); // Cosine distance range
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_pair_identical_vectors() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0, 3.0], // Identical vector
            &[4.0, 5.0, 6.0],
        ])
        .unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();
        let closest_pair = cosine_pair.closest_pair();

        // Distance between identical vectors should be 0
        let distance = closest_pair.distance.unwrap();
        assert!((distance - 0.0).abs() < 1e-8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_pair_orthogonal_vectors() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[1.0, 0.0],
            &[0.0, 1.0], // Orthogonal to first
            &[2.0, 3.0],
        ])
        .unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();

        // Check that orthogonal vectors have cosine distance of 1.0
        let distances_from_first = cosine_pair.distances_from(0);
        let orthogonal_distance = distances_from_first
            .iter()
            .find(|pd| pd.neighbour == Some(1))
            .unwrap()
            .distance
            .unwrap();

        assert!((orthogonal_distance - 1.0).abs() < 1e-8);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_pair_ordered_pairs() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[1.0, 2.0],
            &[2.0, 1.0],
            &[3.0, 4.0],
            &[4.0, 3.0],
        ])
        .unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();
        let ordered_pairs: Vec<_> = cosine_pair.ordered_pairs().collect();

        assert_eq!(ordered_pairs.len(), 4);

        // Check that pairs are ordered by distance (ascending)
        for i in 1..ordered_pairs.len() {
            let prev_distance = ordered_pairs[i - 1].distance.unwrap();
            let curr_distance = ordered_pairs[i].distance.unwrap();
            assert!(prev_distance <= curr_distance);
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_pair_query_row() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[1.0, 1.0, 0.0],
            &[0.0, 1.0, 1.0],
        ])
        .unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();

        // Query k=2 nearest neighbors for row 0
        let neighbors = cosine_pair.query_row(0, 2).unwrap();

        assert_eq!(neighbors.len(), 2);

        // Check that distances are in ascending order
        assert!(neighbors[0].0 <= neighbors[1].0);

        // All distances should be valid cosine distances (0 to 2)
        for (distance, _) in &neighbors {
            assert!(*distance >= 0.0 && *distance <= 2.0);
        }
    }

    #[test]
    fn cosine_pair_query_row_bounds_error() {
        let x = DenseMatrix::<f64>::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();

        // Query with out-of-bounds row index
        let result = cosine_pair.query_row(5, 1);
        assert!(result.is_err());

        if let Err(e) = result {
            let expected_error =
                Failed::because(FailedError::FindFailed, "Query row index out of bounds");
            assert_eq!(e, expected_error);
        }
    }

    #[test]
    fn cosine_pair_query_row_k_zero() {
        let x =
            DenseMatrix::<f64>::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]).unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();
        let neighbors = cosine_pair.query_row(0, 0).unwrap();

        assert_eq!(neighbors.len(), 0);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_pair_query_external_vector() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[1.0, 1.0, 0.0],
        ])
        .unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();

        // Query with external vector
        let query_vector = vec![1.0, 0.5, 0.0];
        let neighbors = cosine_pair.query(&query_vector, 2).unwrap();

        assert_eq!(neighbors.len(), 2);

        // Verify distances are valid and ordered
        assert!(neighbors[0].0 <= neighbors[1].0);
        for (distance, index) in &neighbors {
            assert!(*distance >= 0.0 && *distance <= 2.0);
            assert!(*index < x.shape().0);
        }
    }

    #[test]
    fn cosine_pair_query_dimension_mismatch() {
        let x = DenseMatrix::<f64>::from_2d_array(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]).unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();

        // Query with mismatched dimensions
        let query_vector = vec![1.0, 2.0]; // Only 2 dimensions, but data has 3
        let result = cosine_pair.query(&query_vector, 1);

        assert!(result.is_err());
        if let Err(e) = result {
            let expected_error =
                Failed::because(FailedError::FindFailed, "Query vector dimension mismatch");
            assert_eq!(e, expected_error);
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_pair_query_k_zero_external() {
        let x = DenseMatrix::<f64>::from_2d_array(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();
        let query_vector = vec![1.0, 1.0];
        let neighbors = cosine_pair.query(&query_vector, 0).unwrap();

        assert_eq!(neighbors.len(), 0);
    }

    #[test]
    fn cosine_pair_large_dataset() {
        // Test with larger dataset (similar to Iris)
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
            &[4.6, 3.4, 1.4, 0.3],
            &[5.0, 3.4, 1.5, 0.2],
            &[4.4, 2.9, 1.4, 0.2],
            &[4.9, 3.1, 1.5, 0.1],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.9, 3.1, 4.9, 1.5],
            &[5.5, 2.3, 4.0, 1.3],
            &[6.5, 2.8, 4.6, 1.5],
        ])
        .unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();

        assert_eq!(cosine_pair.samples.shape().0, 15);
        assert_eq!(cosine_pair.distances.len(), 15);
        assert_eq!(cosine_pair.neighbours.len(), 15);

        // Test closest pair computation
        let closest_pair = cosine_pair.closest_pair();
        assert!(closest_pair.distance.is_some());
        assert!(closest_pair.neighbour.is_some());

        let distance = closest_pair.distance.unwrap();
        assert!(distance >= 0.0 && distance <= 2.0);
    }

    #[test]
    fn query_row_top_k_top_k_limiting() {
        // Test that query_row_top_k respects top_k parameter and returns correct results
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[1.0, 0.0, 0.0], // Point 0
            &[0.0, 1.0, 0.0], // Point 1 - orthogonal to point 0
            &[0.0, 0.0, 1.0], // Point 2 - orthogonal to point 0
            &[1.0, 1.0, 0.0], // Point 3 - closer to point 0 than points 1,2
            &[0.5, 0.0, 0.0], // Point 4 - very close to point 0 (parallel)
            &[2.0, 0.0, 0.0], // Point 5 - very close to point 0 (parallel)
            &[0.0, 1.0, 1.0], // Point 6 - far from point 0
            &[3.0, 3.0, 3.0], // Point 7 - moderately close to point 0
        ])
        .unwrap();

        // Create CosinePair with top_k=4 to limit candidates
        let cosine_pair = CosinePair::with_top_k(&x, 4).unwrap();

        // Query for 3 nearest neighbors to point 0
        let neighbors = cosine_pair.query_row_top_k(0, 3).unwrap();

        // Should return exactly 3 neighbors
        assert_eq!(neighbors.len(), 3);

        // Verify that distances are in ascending order
        for i in 1..neighbors.len() {
            assert!(
                neighbors[i - 1].0 <= neighbors[i].0,
                "Distances should be in ascending order: {} <= {}",
                neighbors[i - 1].0,
                neighbors[i].0
            );
        }

        // All distances should be valid cosine distances (0 to 2)
        for (distance, index) in &neighbors {
            assert!(
                *distance >= 0.0 && *distance <= 2.0,
                "Cosine distance {} should be between 0 and 2",
                distance
            );
            assert!(
                *index < x.shape().0,
                "Neighbor index {} should be less than dataset size {}",
                index,
                x.shape().0
            );
            assert!(
                *index != 0,
                "Neighbor index should not include query point itself"
            );
        }

        // The closest neighbor should be either point 4 or 5 (parallel vectors)
        // These should have cosine distance ≈ 0
        let closest_distance = neighbors[0].0;
        assert!(
            closest_distance < 0.01,
            "Closest parallel vector should have distance close to 0, got {}",
            closest_distance
        );

        // Verify that we get different results with different top_k values
        let cosine_pair_full = CosinePair::new(&x).unwrap();
        let neighbors_full = cosine_pair_full.query_row(0, 3).unwrap();

        // Results should be the same or very close since we're asking for top 3
        // but the algorithm might find different candidates due to top_k limiting
        assert_eq!(neighbors.len(), neighbors_full.len());

        // The closest neighbor should be the same in both cases
        let closest_idx_fast = neighbors[0].1;
        let closest_idx_full = neighbors_full[0].1;
        let closest_dist_fast = neighbors[0].0;
        let closest_dist_full = neighbors_full[0].0;

        // Either we get the same closest neighbor, or distances are very close
        if closest_idx_fast == closest_idx_full {
            assert!(relative_eq!(
                closest_dist_fast,
                closest_dist_full,
                epsilon = 1e-10
            ));
        } else {
            // Different neighbors, but distances should be very close (parallel vectors)
            assert!(relative_eq!(
                closest_dist_fast,
                closest_dist_full,
                epsilon = 1e-6
            ));
        }
    }

    #[test]
    fn query_row_top_k_performance_vs_accuracy() {
        // Test that query_row_top_k provides reasonable performance/accuracy tradeoff
        // and handles edge cases properly
        let large_dataset = DenseMatrix::<f32>::from_2d_array(&[
            &[1.0f32, 2.0, 3.0, 4.0],     // Point 0 - query point
            &[1.1f32, 2.1, 3.1, 4.1],     // Point 1 - very close to 0
            &[1.05f32, 2.05, 3.05, 4.05], // Point 2 - very close to 0
            &[2.0f32, 4.0, 6.0, 8.0],     // Point 3 - parallel to 0 (2x scaling)
            &[0.5f32, 1.0, 1.5, 2.0],     // Point 4 - parallel to 0 (0.5x scaling)
            &[-1.0f32, -2.0, -3.0, -4.0], // Point 5 - opposite to 0
            &[4.0f32, 3.0, 2.0, 1.0],     // Point 6 - different direction
            &[0.0f32, 0.0, 0.0, 0.1],     // Point 7 - mostly orthogonal
            &[10.0f32, 20.0, 30.0, 40.0], // Point 8 - parallel but far
            &[1.0f32, 0.0, 0.0, 0.0],     // Point 9 - partially similar
            &[0.0f32, 2.0, 0.0, 0.0],     // Point 10 - partially similar
            &[0.0f32, 0.0, 3.0, 0.0],     // Point 11 - partially similar
        ])
        .unwrap();

        // Test with aggressive top_k limiting (only consider 5 out of 11 other points)
        let cosine_pair_limited = CosinePair::with_top_k(&large_dataset, 5).unwrap();

        // Query for 4 nearest neighbors
        let neighbors_limited = cosine_pair_limited.query_row_top_k(0, 4).unwrap();

        // Should return exactly 4 neighbors
        assert_eq!(neighbors_limited.len(), 4);

        // Test error handling - out of bounds query
        let result_oob = cosine_pair_limited.query_row_top_k(15, 2);
        assert!(result_oob.is_err());
        if let Err(e) = result_oob {
            assert_eq!(
                e,
                Failed::because(FailedError::FindFailed, "Query row index out of bounds")
            );
        }

        // Test k=0 case
        let neighbors_zero = cosine_pair_limited.query_row_top_k(0, 0).unwrap();
        assert_eq!(neighbors_zero.len(), 0);

        // Test k > available candidates
        let neighbors_large_k = cosine_pair_limited.query_row_top_k(0, 20).unwrap();
        assert!(neighbors_large_k.len() <= 11); // At most 11 other points

        // Verify ordering is correct
        for i in 1..neighbors_limited.len() {
            assert!(
                neighbors_limited[i - 1].0 <= neighbors_limited[i].0,
                "Distance ordering violation at position {}: {} > {}",
                i,
                neighbors_limited[i - 1].0,
                neighbors_limited[i].0
            );
        }

        // The closest neighbors should be the parallel vectors (points 1, 2, 3, 4)
        // since they have the smallest cosine distances
        let closest_distance = neighbors_limited[0].0;
        assert!(
            closest_distance < 0.1,
            "Closest neighbor should be nearly parallel, distance: {}",
            closest_distance
        );

        // Compare with full algorithm for accuracy assessment
        let cosine_pair_full = CosinePair::new(&large_dataset).unwrap();
        let neighbors_full = cosine_pair_full.query_row(0, 4).unwrap();

        // The fast version might not find the exact same neighbors due to sampling,
        // but the closest neighbor's distance should be very similar
        let dist_diff = (neighbors_limited[0].0 - neighbors_full[0].0).abs();
        assert!(
            dist_diff < 0.01,
            "Fast and full algorithms should give similar closest distances. Diff: {}",
            dist_diff
        );

        // Verify that all returned indices are valid and unique
        let mut indices: Vec<usize> = neighbors_limited.iter().map(|(_, idx)| *idx).collect();
        indices.sort();
        indices.dedup();
        assert_eq!(
            indices.len(),
            neighbors_limited.len(),
            "All neighbor indices should be unique"
        );

        for &idx in &indices {
            assert!(
                idx < large_dataset.shape().0,
                "Neighbor index {} should be valid",
                idx
            );
            assert!(idx != 0, "Neighbor should not include query point itself");
        }

        // Test with f32 precision to ensure type compatibility
        for (distance, _) in &neighbors_limited {
            assert!(!distance.is_nan(), "Distance should not be NaN");
            assert!(distance.is_finite(), "Distance should be finite");
            assert!(*distance >= 0.0, "Distance should be non-negative");
        }
    }

    #[test]
    fn cosine_pair_float_precision() {
        // Test with f32 precision
        let x = DenseMatrix::<f32>::from_2d_array(&[
            &[1.0f32, 2.0, 3.0],
            &[4.0f32, 5.0, 6.0],
            &[7.0f32, 8.0, 9.0],
        ])
        .unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();
        let closest_pair = cosine_pair.closest_pair();

        assert!(closest_pair.distance.is_some());
        let distance = closest_pair.distance.unwrap();
        assert!(distance >= 0.0 && distance <= 2.0);

        // Test querying
        let neighbors = cosine_pair.query_row(0, 2).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].1, 1);
        assert_relative_eq!(neighbors[0].0, 0.025368154);
        assert_eq!(neighbors[1].1, 2);
        assert_relative_eq!(neighbors[1].0, 0.040588055);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_pair_distances_from() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[1.0, 1.0],
            &[2.0, 0.0],
        ])
        .unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();
        let distances = cosine_pair.distances_from(0);

        // Should have 3 distances (excluding self)
        assert_eq!(distances.len(), 3);

        // All should be from node 0
        for pd in &distances {
            assert_eq!(pd.node, 0);
            assert!(pd.neighbour.is_some());
            assert!(pd.distance.is_some());
            assert!(pd.neighbour.unwrap() != 0); // Should not include self
        }
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn cosine_pair_consistency_check() {
        // Verify that different query methods return consistent results
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
            &[2.0, 3.0, 4.0],
        ])
        .unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();

        // Query row 0 using internal method
        let neighbors_internal = cosine_pair.query_row(0, 2).unwrap();

        // Query row 0 using optimized method (should be same)
        let neighbors_optimized = cosine_pair.query_optimized(0, 2).unwrap();

        assert_eq!(neighbors_internal.len(), neighbors_optimized.len());
        for i in 0..neighbors_internal.len() {
            let (dist1, idx1) = neighbors_internal[i];
            let (dist2, idx2) = neighbors_optimized[i];
            assert!((dist1 - dist2).abs() < 1e-10);
            assert_eq!(idx1, idx2);
        }
    }

    // Brute force algorithm for testing/comparison
    fn closest_pair_brute_force(
        cosine_pair: &CosinePair<'_, f64, DenseMatrix<f64>>,
    ) -> PairwiseDistance<f64> {
        use itertools::Itertools;

        let m = cosine_pair.samples.shape().0;
        let mut closest_pair = PairwiseDistance {
            node: 0,
            neighbour: None,
            distance: Some(f64::MAX),
        };

        for pair in (0..m).combinations(2) {
            let d = Cosine::new().distance(
                &Vec::from_iterator(
                    cosine_pair.samples.get_row(pair[0]).iterator(0).copied(),
                    cosine_pair.samples.shape().1,
                ),
                &Vec::from_iterator(
                    cosine_pair.samples.get_row(pair[1]).iterator(0).copied(),
                    cosine_pair.samples.shape().1,
                ),
            );

            if d < closest_pair.distance.unwrap() {
                closest_pair.node = pair[0];
                closest_pair.neighbour = Some(pair[1]);
                closest_pair.distance = Some(d);
            }
        }

        closest_pair
    }

    #[test]
    fn cosine_pair_vs_brute_force() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
            &[1.1, 2.1, 3.1], // Close to first point
        ])
        .unwrap();

        let cosine_pair = CosinePair::new(&x).unwrap();
        let cp_result = cosine_pair.closest_pair();
        let brute_result = closest_pair_brute_force(&cosine_pair);

        // Results should be identical or very close
        assert!((cp_result.distance.unwrap() - brute_result.distance.unwrap()).abs() < 1e-10);
    }

    // Dataset with a known angular geometry: row 0 points along the x-axis,
    // the odd rows 1, 3, 5, 7 are near-parallel to row 0 (its true nearest
    // neighbours), and the even rows 2, 4, 6 are orthogonal to row 0.
    // Exact top-3 neighbours of row 0 are rows 5, 3, 1 in this order.
    fn mixed_direction_rows() -> DenseMatrix<f64> {
        DenseMatrix::<f64>::from_2d_array(&[
            &[1.0, 0.0, 0.0],   // 0: query row, +x direction
            &[0.9, 0.1, 0.0],   // 1: near-parallel to row 0
            &[0.0, 1.0, 0.0],   // 2: orthogonal to row 0
            &[0.95, 0.05, 0.0], // 3: near-parallel to row 0
            &[0.0, 0.0, 1.0],   // 4: orthogonal to row 0
            &[0.99, 0.01, 0.0], // 5: closest row to row 0
            &[0.0, 1.0, 1.0],   // 6: orthogonal to row 0
            &[0.8, 0.2, 0.0],   // 7: near-parallel to row 0
        ])
        .unwrap()
    }

    fn cosine_distance_between_rows(m: &DenseMatrix<f64>, i: usize, j: usize) -> f64 {
        Cosine::new().distance(
            &Vec::from_iterator(m.get_row(i).iterator(0).copied(), m.shape().1),
            &Vec::from_iterator(m.get_row(j).iterator(0).copied(), m.shape().1),
        )
    }

    // Brute-force oracle: exact top-k neighbours of a row, ties broken by index
    fn brute_force_top_k(m: &DenseMatrix<f64>, row: usize, k: usize) -> Vec<(f64, usize)> {
        let mut scored: Vec<(f64, usize)> = (0..m.shape().0)
            .filter(|&j| j != row)
            .map(|j| (cosine_distance_between_rows(m, row, j), j))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        scored.truncate(k);
        scored
    }

    #[test]
    fn query_row_top_k_is_exact_when_approximate_is_false_with_top_k() {
        let x = mixed_direction_rows();
        let cosine_pair = CosinePair::with_top_k(&x, 4).unwrap();

        let neighbors = cosine_pair.query_row_top_k(0, 3).unwrap();

        let expected = brute_force_top_k(&x, 0, 3);
        assert_eq!(neighbors.len(), expected.len());
        for (got, want) in neighbors.iter().zip(expected.iter()) {
            assert_eq!(got.1, want.1);
            assert_relative_eq!(got.0, want.0, epsilon = 1e-12);
        }
        // The true nearest neighbours are the near-parallel rows 5, 3, 1.
        let indices: Vec<usize> = neighbors.iter().map(|&(_, i)| i).collect();
        assert_eq!(indices, vec![5, 3, 1]);
    }

    #[test]
    fn query_row_top_k_is_exact_when_approximate_is_false_with_parameters() {
        let x = mixed_direction_rows();
        let cosine_pair = CosinePair::with_parameters(
            &x,
            CosinePairParameters {
                top_k: Some(3),
                approximate: false,
            },
        )
        .unwrap();

        let neighbors = cosine_pair.query_row_top_k(0, 3).unwrap();

        let expected = brute_force_top_k(&x, 0, 3);
        assert_eq!(neighbors.len(), expected.len());
        for (got, want) in neighbors.iter().zip(expected.iter()) {
            assert_eq!(got.1, want.1);
            assert_relative_eq!(got.0, want.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn query_row_top_k_exact_matches_query_row_from_new() {
        let x = mixed_direction_rows();
        let limited = CosinePair::with_top_k(&x, 4).unwrap();
        let full = CosinePair::new(&x).unwrap();

        // Rows 2, 4, 6 have several neighbours tied at cosine distance 1.0,
        // where the relative order of equal distances is not specified.
        for row in [0usize, 1, 3, 5, 7] {
            let fast = limited.query_row_top_k(row, 3).unwrap();
            let exact = full.query_row(row, 3).unwrap();
            assert_eq!(fast.len(), exact.len(), "row {}", row);
            for (got, want) in fast.iter().zip(exact.iter()) {
                assert_eq!(got.1, want.1, "row {}", row);
                assert_relative_eq!(got.0, want.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn with_top_k_build_keeps_true_closest_neighbour_per_row() {
        let x = mixed_direction_rows();
        let cosine_pair = CosinePair::with_top_k(&x, 4).unwrap();

        for i in 0..x.shape().0 {
            let expected = brute_force_top_k(&x, i, 1)[0];
            let stored = cosine_pair.distances[&i];
            assert_eq!(stored.neighbour, Some(expected.1), "row {}", i);
            assert_relative_eq!(stored.distance.unwrap(), expected.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn with_top_k_full_parity_with_new() {
        let x = mixed_direction_rows();
        let n = x.shape().0;
        let limited = CosinePair::with_top_k(&x, n - 1).unwrap();
        let full = CosinePair::new(&x).unwrap();

        for i in 0..n {
            let a = limited.distances[&i];
            let b = full.distances[&i];
            assert_eq!(a.neighbour, b.neighbour, "row {}", i);
            assert_relative_eq!(a.distance.unwrap(), b.distance.unwrap(), epsilon = 1e-12);
        }

        for row in [0usize, 1, 3, 5, 7] {
            let fast = limited.query_row_top_k(row, 3).unwrap();
            let exact = full.query_row(row, 3).unwrap();
            assert_eq!(fast.len(), exact.len(), "row {}", row);
            for (got, want) in fast.iter().zip(exact.iter()) {
                assert_eq!(got.1, want.1, "row {}", row);
                assert_relative_eq!(got.0, want.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn build_closest_neighbour_matches_brute_force_per_row() {
        let x = mixed_direction_rows();
        let cosine_pair = CosinePair::new(&x).unwrap();

        for i in 0..x.shape().0 {
            let expected = brute_force_top_k(&x, i, 1)[0];
            let stored = cosine_pair.distances[&i];
            assert_eq!(stored.neighbour, Some(expected.1), "row {}", i);
            assert_relative_eq!(stored.distance.unwrap(), expected.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn query_row_top_k_samples_strided_candidates_when_approximate_is_true() {
        let x = mixed_direction_rows();
        let cosine_pair = CosinePair::with_parameters(
            &x,
            CosinePairParameters {
                top_k: Some(4),
                approximate: true,
            },
        )
        .unwrap();

        let neighbors = cosine_pair.query_row_top_k(0, 3).unwrap();

        assert_eq!(neighbors.len(), 3);
        // Sampled candidates with step = 8 / 4 = 2 are the even rows 2, 4, 6,
        // all orthogonal to row 0, so every sampled distance is 1.0 and the
        // approximate result differs from the exact neighbours 5, 3, 1.
        let indices: Vec<usize> = neighbors.iter().map(|&(_, i)| i).collect();
        let mut sorted_indices = indices.clone();
        sorted_indices.sort_unstable();
        assert_eq!(sorted_indices, vec![2, 4, 6]);
        for (distance, index) in &neighbors {
            assert_relative_eq!(*distance, 1.0, epsilon = 1e-12);
            assert_ne!(*index, 0);
        }
        for i in 1..neighbors.len() {
            assert!(neighbors[i - 1].0 <= neighbors[i].0);
        }
    }
}
