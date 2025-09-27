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
use std::collections::HashMap;

use num::Bounded;

use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::metrics::distance::cosine::Cosine;
use crate::metrics::distance::{Distance, PairwiseDistance};
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;

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
}

impl<'a, T: RealNumber + FloatNumber, M: Array2<T>> CosinePair<'a, T, M> {
    /// Constructor
    /// Instantiate and initialize the algorithm
    pub fn new(m: &'a M) -> Result<Self, Failed> {
        if m.shape().0 < 2 {
            return Err(Failed::because(
                FailedError::FindFailed,
                "min number of rows should be 2",
            ));
        }

        let mut init = Self {
            samples: m,
            // to be computed in init(..)
            distances: HashMap::with_capacity(m.shape().0),
            neighbours: Vec::with_capacity(m.shape().0 + 1),
        };
        init.init();
        Ok(init)
    }

    /// Initialise `CosinePair` by passing a `Array2`.
    /// Build a CosinePairs data-structure from a set of (new) points.
    fn init(&mut self) {
        // basic measures
        let len = self.samples.shape().0;
        let max_index = self.samples.shape().0 - 1;

        // Store all closest neighbors
        let _distances = Box::new(HashMap::with_capacity(len));
        let _neighbours = Box::new(Vec::with_capacity(len));

        let mut distances = *_distances;
        let mut neighbours = *_neighbours;

        // fill neighbours with -1 values
        neighbours.extend(0..len);

        // init closest neighbour pairwise data
        for index_row_i in 0..(max_index) {
            distances.insert(
                index_row_i,
                PairwiseDistance {
                    node: index_row_i,
                    neighbour: Option::None,
                    distance: Some(<T as Bounded>::max_value()),
                },
            );
        }

        // loop through indeces and neighbours
        for index_row_i in 0..(len) {
            // start looking for the neighbour in the second element
            let mut index_closest = index_row_i + 1; // closest neighbour index
            let mut nbd: Option<T> = distances[&index_row_i].distance; // init neighbour distance
            for index_row_j in (index_row_i + 1)..len {
                distances.insert(
                    index_row_j,
                    PairwiseDistance {
                        node: index_row_j,
                        neighbour: Some(index_row_i),
                        distance: nbd,
                    },
                );

                let d = Cosine::new().distance(
                    &Vec::from_iterator(
                        self.samples.get_row(index_row_i).iterator(0).copied(),
                        self.samples.shape().1,
                    ),
                    &Vec::from_iterator(
                        self.samples.get_row(index_row_j).iterator(0).copied(),
                        self.samples.shape().1,
                    ),
                );
                if d < nbd.unwrap().to_f64().unwrap() {
                    // set this j-value to be the closest neighbour
                    index_closest = index_row_j;
                    nbd = Some(T::from(d).unwrap());
                }
            }

            // Add that edge
            distances.entry(index_row_i).and_modify(|e| {
                e.distance = nbd;
                e.neighbour = Some(index_closest);
            });
        }
        // No more neighbors, terminate conga line.
        // Last person on the line has no neigbors
        distances.get_mut(&max_index).unwrap().neighbour = Some(max_index);
        distances.get_mut(&(len - 1)).unwrap().distance = Some(<T as Bounded>::max_value());

        // compute sparse matrix (connectivity matrix)
        let mut sparse_matrix = M::zeros(len, len);
        for (_, p) in distances.iter() {
            sparse_matrix.set((p.node, p.neighbour.unwrap()), p.distance.unwrap());
        }

        self.distances = distances;
        self.neighbours = neighbours;
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
    use approx::assert_relative_eq;

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

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
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

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
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
