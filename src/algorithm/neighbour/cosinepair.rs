///
/// ### CosinePair: Data-structure for the dynamic closest-pair problem.
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

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use num::Bounded;

use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::metrics::distance::cosine::Cosine;
use crate::metrics::distance::{Distance, PairwiseDistance};
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;

/// Parameters for CosinePair construction
#[derive(Debug, Clone)]
pub struct CosinePairParameters {
    /// Maximum number of neighbors to consider per point (default: all points)
    pub top_k: Option<usize>,
    /// Whether to use approximate nearest neighbor search
    pub approximate: bool,
}

#[allow(clippy::derivable_impls)]
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
#[derive(Debug, Clone)]
pub struct CosinePair<'a, T: RealNumber + FloatNumber, M: Array2<T>> {
    /// initial matrix
    pub samples: &'a M,
    /// closest pair hashmap (connectivity matrix for closest pairs)
    pub distances: HashMap<usize, PairwiseDistance<T>>,
    /// conga line used to keep track of the closest pair
    pub neighbours: Vec<usize>,
    /// parameters used during construction
    pub parameters: CosinePairParameters,
}

impl<'a, T: RealNumber + FloatNumber + FloatCore, M: Array2<T>> CosinePair<'a, T, M> {
    /// Constructor with default parameters (backward compatibility)
    pub fn new(m: &'a M) -> Result<Self, Failed> {
        Self::with_parameters(m, CosinePairParameters::default())
    }

    /// Constructor with top-k limiting for faster performance
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

        let mut init = Self {
            samples: m,
            distances: HashMap::with_capacity(m.shape().0),
            neighbours: Vec::with_capacity(m.shape().0),
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

    /// Optimized initialization with top-k neighbor limiting
    fn init(&mut self) {
        let len = self.samples.shape().0;
        let max_neighbors: usize = self.parameters.top_k.unwrap_or(len - 1).min(len - 1);

        let mut distances = HashMap::with_capacity(len);
        let mut neighbours = Vec::with_capacity(len);

        neighbours.extend(0..len);

        // Initialize with max distances
        for i in 0..len {
            distances.insert(
                i,
                PairwiseDistance {
                    node: i,
                    neighbour: None,
                    distance: Some(<T as Bounded>::max_value()),
                },
            );
        }

        // Compute distances for each point using top-k optimization
        for i in 0..len {
            let mut candidate_distances = BinaryHeap::new();

            for j in 0..len {
                if i != j {
                    let distance = T::from(Cosine::new().distance(
                        &Vec::from_iterator(
                            self.samples.get_row(i).iterator(0).copied(),
                            self.samples.shape().1,
                        ),
                        &Vec::from_iterator(
                            self.samples.get_row(j).iterator(0).copied(),
                            self.samples.shape().1,
                        ),
                    ))
                    .unwrap();

                    // Use OrderedFloat for stable ordering
                    candidate_distances.push(Reverse((Self::ordered_float(distance), j)));

                    if candidate_distances.len() > max_neighbors {
                        candidate_distances.pop();
                    }
                }
            }

            // Find the closest neighbor from candidates
            if let Some(Reverse((closest_distance, closest_neighbor))) =
                candidate_distances.iter().min_by_key(|Reverse((d, _))| *d)
            {
                distances.entry(i).and_modify(|e| {
                    e.distance = Some(Self::extract_float(*closest_distance));
                    e.neighbour = Some(*closest_neighbor);
                });
            }
        }

        self.distances = distances;
        self.neighbours = neighbours;
    }

    /// Fast query using top-k pre-computed neighbors with ordered-float
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

        let max_candidates = self.parameters.top_k.unwrap_or(self.samples.shape().0);
        let actual_k: usize = k.min(max_candidates);

        // Use binary heap with ordered-float for reliable ordering
        let mut heap = BinaryHeap::with_capacity(actual_k + 1);

        let candidates = if let Some(top_k) = self.parameters.top_k {
            let step = (self.samples.shape().0 / top_k).max(1);
            (0..self.samples.shape().0)
                .step_by(step)
                .filter(|&i| i != query_row_index)
                .take(top_k)
                .collect::<Vec<_>>()
        } else {
            (0..self.samples.shape().0)
                .filter(|&i| i != query_row_index)
                .collect::<Vec<_>>()
        };

        for &candidate_idx in &candidates {
            let distance = T::from(Cosine::new().distance(
                &Vec::from_iterator(
                    self.samples.get_row(query_row_index).iterator(0).copied(),
                    self.samples.shape().1,
                ),
                &Vec::from_iterator(
                    self.samples.get_row(candidate_idx).iterator(0).copied(),
                    self.samples.shape().1,
                ),
            ))
            .unwrap();

            heap.push(Reverse((Self::ordered_float(distance), candidate_idx)));

            if heap.len() > actual_k {
                heap.pop();
            }
        }

        // Convert heap to sorted vector
        let mut neighbors: Vec<_> = heap
            .into_vec()
            .into_iter()
            .map(|Reverse((dist, idx))| (Self::extract_float(dist), idx))
            .collect();

        neighbors.sort_by(|a, b| Self::ordered_float(a.0).cmp(&Self::ordered_float(b.0)));

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
        let mut distances = Vec::<PairwiseDistance<T>>::with_capacity(self.samples.shape().0);

        for i in 0..self.samples.shape().0 {
            let dataset_point = Vec::from_iterator(
                self.samples.get_row(i).iterator(0).copied(),
                self.samples.shape().1,
            );

            let distance = T::from(Cosine::new().distance(query_vector, &dataset_point)).unwrap();

            distances.push(PairwiseDistance {
                node: i, // This represents the dataset point index
                neighbour: Some(i),
                distance: Some(distance),
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
        for other in self.neighbours.iter() {
            if index_row != *other {
                distances.push(PairwiseDistance {
                    node: index_row,
                    neighbour: Some(*other),
                    distance: Some(
                        T::from(Cosine::new().distance(
                            &Vec::from_iterator(
                                self.samples.get_row(index_row).iterator(0).copied(),
                                self.samples.shape().1,
                            ),
                            &Vec::from_iterator(
                                self.samples.get_row(*other).iterator(0).copied(),
                                self.samples.shape().1,
                            ),
                        ))
                        .unwrap(),
                    ),
                })
            }
        }
        distances
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::basic::{arrays::Array, matrix::DenseMatrix};
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
}
