///
/// ### FastPair: Data-structure for the dynamic closest-pair problem.
///
/// Reference:
///  Eppstein, David: Fast hierarchical clustering and other applications of
///  dynamic closest pairs. Journal of Experimental Algorithmics 5 (2000) 1.
///
/// Example:
/// ```
/// use smartcore::metrics::distance::PairwiseDistance;
/// use smartcore::linalg::basic::matrix::DenseMatrix;
/// use smartcore::algorithm::neighbour::fastpair::FastPair;
/// let x = DenseMatrix::<f64>::from_2d_array(&[
///     &[5.1, 3.5, 1.4, 0.2],
///     &[4.9, 3.0, 1.4, 0.2],
///     &[4.7, 3.2, 1.3, 0.2],
///     &[4.6, 3.1, 1.5, 0.2],
///     &[5.0, 3.6, 1.4, 0.2],
///     &[5.4, 3.9, 1.7, 0.4],
/// ]).unwrap();
/// let fastpair = FastPair::new(&x);
/// let closest_pair: PairwiseDistance<f64> = fastpair.unwrap().closest_pair();
/// ```
/// <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
/// <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
use std::collections::HashMap;

use num::Bounded;

use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::metrics::distance::euclidian::Euclidian;
use crate::metrics::distance::PairwiseDistance;
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;

///
/// Inspired by Python implementation:
/// <https://github.com/carsonfarmer/fastpair/blob/b8b4d3000ab6f795a878936667eee1b557bf353d/fastpair/base.py>
/// MIT License (MIT) Copyright (c) 2016 Carson Farmer
///
/// affinity used is Euclidean so to allow linkage with single, ward, complete and average
///
#[derive(Debug, Clone)]
pub struct FastPair<'a, T: RealNumber + FloatNumber, M: Array2<T>> {
    /// initial matrix
    samples: &'a M,
    /// closest pair hashmap (connectivity matrix for closest pairs)
    pub distances: HashMap<usize, PairwiseDistance<T>>,
    /// conga line used to keep track of the closest pair
    pub neighbours: Vec<usize>,
}

impl<'a, T: RealNumber + FloatNumber, M: Array2<T>> FastPair<'a, T, M> {
    ///
    /// Constructor
    /// Instantiate and inizialise the algorithm
    ///
    pub fn new(m: &'a M) -> Result<Self, Failed> {
        if m.shape().0 < 3 {
            return Err(Failed::because(
                FailedError::FindFailed,
                "min number of rows should be 3",
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

    ///
    /// Initialise `FastPair` by passing a `Array2`.
    /// Build a FastPairs data-structure from a set of (new) points.
    ///
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

                let d = Euclidian::squared_distance(
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

    ///
    /// Find closest pair by scanning list of nearest neighbors.
    ///
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
                        T::from(Euclidian::squared_distance(
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
mod tests_fastpair {

    use super::*;
    use crate::linalg::basic::{arrays::Array, matrix::DenseMatrix};

    ///
    /// Brute force algorithm, used only for comparison and testing
    ///
    pub fn closest_pair_brute(fastpair: &FastPair<f64, DenseMatrix<f64>>) -> PairwiseDistance<f64> {
        use itertools::Itertools;
        let m = fastpair.samples.shape().0;

        let mut closest_pair = PairwiseDistance {
            node: 0,
            neighbour: Option::None,
            distance: Some(f64::max_value()),
        };
        for pair in (0..m).combinations(2) {
            let d = Euclidian::squared_distance(
                &Vec::from_iterator(
                    fastpair.samples.get_row(pair[0]).iterator(0).copied(),
                    fastpair.samples.shape().1,
                ),
                &Vec::from_iterator(
                    fastpair.samples.get_row(pair[1]).iterator(0).copied(),
                    fastpair.samples.shape().1,
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
    fn fastpair_init() {
        let x: DenseMatrix<f64> = DenseMatrix::rand(10, 4);
        let _fastpair = FastPair::new(&x);
        assert!(_fastpair.is_ok());

        let fastpair = _fastpair.unwrap();

        let distances = fastpair.distances;
        let neighbours = fastpair.neighbours;

        assert!(!distances.is_empty());
        assert!(!neighbours.is_empty());

        assert_eq!(10, neighbours.len());
        assert_eq!(10, distances.len());
    }

    #[test]
    fn dataset_has_at_least_three_points() {
        // Create a dataset which consists of only two points:
        // A(0.0, 0.0) and B(1.0, 1.0).
        let dataset = DenseMatrix::<f64>::from_2d_array(&[&[0.0, 0.0], &[1.0, 1.0]]).unwrap();

        // We expect an error when we run `FastPair` on this dataset,
        // becuase `FastPair` currently only works on a minimum of 3
        // points.
        let fastpair = FastPair::new(&dataset);
        assert!(fastpair.is_err());

        if let Err(e) = fastpair {
            let expected_error =
                Failed::because(FailedError::FindFailed, "min number of rows should be 3");
            assert_eq!(e, expected_error)
        }
    }

    #[test]
    fn one_dimensional_dataset_minimal() {
        let dataset = DenseMatrix::<f64>::from_2d_array(&[&[0.0], &[2.0], &[9.0]]).unwrap();

        let result = FastPair::new(&dataset);
        assert!(result.is_ok());

        let fastpair = result.unwrap();
        let closest_pair = fastpair.closest_pair();
        let expected_closest_pair = PairwiseDistance {
            node: 0,
            neighbour: Some(1),
            distance: Some(4.0),
        };
        assert_eq!(closest_pair, expected_closest_pair);

        let closest_pair_brute = closest_pair_brute(&fastpair);
        assert_eq!(closest_pair_brute, expected_closest_pair);
    }

    #[test]
    fn one_dimensional_dataset_2() {
        let dataset =
            DenseMatrix::<f64>::from_2d_array(&[&[27.0], &[0.0], &[9.0], &[2.0]]).unwrap();

        let result = FastPair::new(&dataset);
        assert!(result.is_ok());

        let fastpair = result.unwrap();
        let closest_pair = fastpair.closest_pair();
        let expected_closest_pair = PairwiseDistance {
            node: 1,
            neighbour: Some(3),
            distance: Some(4.0),
        };
        assert_eq!(closest_pair, closest_pair_brute(&fastpair));
        assert_eq!(closest_pair, expected_closest_pair);
    }

    #[test]
    fn fastpair_new() {
        // compute
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
        let fastpair = FastPair::new(&x);
        assert!(fastpair.is_ok());

        // unwrap results
        let result = fastpair.unwrap();

        // list of minimal pairwise dissimilarities
        let dissimilarities = vec![
            (
                1,
                PairwiseDistance {
                    node: 1,
                    neighbour: Some(9),
                    distance: Some(0.030000000000000037),
                },
            ),
            (
                10,
                PairwiseDistance {
                    node: 10,
                    neighbour: Some(12),
                    distance: Some(0.07000000000000003),
                },
            ),
            (
                11,
                PairwiseDistance {
                    node: 11,
                    neighbour: Some(14),
                    distance: Some(0.18000000000000013),
                },
            ),
            (
                12,
                PairwiseDistance {
                    node: 12,
                    neighbour: Some(14),
                    distance: Some(0.34000000000000086),
                },
            ),
            (
                13,
                PairwiseDistance {
                    node: 13,
                    neighbour: Some(14),
                    distance: Some(1.6499999999999997),
                },
            ),
            (
                14,
                PairwiseDistance {
                    node: 14,
                    neighbour: Some(14),
                    distance: Some(f64::MAX),
                },
            ),
            (
                6,
                PairwiseDistance {
                    node: 6,
                    neighbour: Some(7),
                    distance: Some(0.18000000000000027),
                },
            ),
            (
                0,
                PairwiseDistance {
                    node: 0,
                    neighbour: Some(4),
                    distance: Some(0.01999999999999995),
                },
            ),
            (
                8,
                PairwiseDistance {
                    node: 8,
                    neighbour: Some(9),
                    distance: Some(0.3100000000000001),
                },
            ),
            (
                2,
                PairwiseDistance {
                    node: 2,
                    neighbour: Some(3),
                    distance: Some(0.0600000000000001),
                },
            ),
            (
                3,
                PairwiseDistance {
                    node: 3,
                    neighbour: Some(8),
                    distance: Some(0.08999999999999982),
                },
            ),
            (
                7,
                PairwiseDistance {
                    node: 7,
                    neighbour: Some(9),
                    distance: Some(0.10999999999999982),
                },
            ),
            (
                9,
                PairwiseDistance {
                    node: 9,
                    neighbour: Some(13),
                    distance: Some(8.69),
                },
            ),
            (
                4,
                PairwiseDistance {
                    node: 4,
                    neighbour: Some(7),
                    distance: Some(0.050000000000000086),
                },
            ),
            (
                5,
                PairwiseDistance {
                    node: 5,
                    neighbour: Some(7),
                    distance: Some(0.4900000000000002),
                },
            ),
        ];

        let expected: HashMap<_, _> = dissimilarities.into_iter().collect();

        for i in 0..(x.shape().0 - 1) {
            let input_neighbour: usize = expected.get(&i).unwrap().neighbour.unwrap();
            let distance = Euclidian::squared_distance(
                &Vec::from_iterator(
                    result.samples.get_row(i).iterator(0).copied(),
                    result.samples.shape().1,
                ),
                &Vec::from_iterator(
                    result.samples.get_row(input_neighbour).iterator(0).copied(),
                    result.samples.shape().1,
                ),
            );

            assert_eq!(i, expected.get(&i).unwrap().node);
            assert_eq!(
                input_neighbour,
                expected.get(&i).unwrap().neighbour.unwrap()
            );
            assert_eq!(distance, expected.get(&i).unwrap().distance.unwrap());
        }
    }

    #[test]
    fn fastpair_closest_pair() {
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
        // compute
        let fastpair = FastPair::new(&x);
        assert!(fastpair.is_ok());

        let dissimilarity = fastpair.unwrap().closest_pair();
        let closest = PairwiseDistance {
            node: 0,
            neighbour: Some(4),
            distance: Some(0.01999999999999995),
        };

        assert_eq!(closest, dissimilarity);
    }

    #[test]
    fn fastpair_closest_pair_random_matrix() {
        let x = DenseMatrix::<f64>::rand(200, 25);
        // compute
        let fastpair = FastPair::new(&x);
        assert!(fastpair.is_ok());

        let result = fastpair.unwrap();

        let dissimilarity1 = result.closest_pair();
        let dissimilarity2 = closest_pair_brute(&result);

        assert_eq!(dissimilarity1, dissimilarity2);
    }

    #[test]
    fn fastpair_distances() {
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
        // compute
        let fastpair = FastPair::new(&x);
        assert!(fastpair.is_ok());

        let dissimilarities = fastpair.unwrap().distances_from(0);

        let mut min_dissimilarity = PairwiseDistance {
            node: 0,
            neighbour: Option::None,
            distance: Some(f64::MAX),
        };
        for p in dissimilarities.iter() {
            if p.distance.unwrap() < min_dissimilarity.distance.unwrap() {
                min_dissimilarity = *p
            }
        }

        let closest = PairwiseDistance {
            node: 0,
            neighbour: Some(4),
            distance: Some(0.01999999999999995),
        };

        assert_eq!(closest, min_dissimilarity);
    }

    #[test]
    fn fastpair_ordered_pairs() {
        let x = DenseMatrix::<f64>::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
            &[4.9, 3.1, 1.5, 0.1],
            &[7.0, 3.2, 4.7, 1.4],
            &[6.4, 3.2, 4.5, 1.5],
            &[6.9, 3.1, 4.9, 1.5],
            &[5.5, 2.3, 4.0, 1.3],
            &[6.5, 2.8, 4.6, 1.5],
            &[4.6, 3.4, 1.4, 0.3],
            &[5.0, 3.4, 1.5, 0.2],
            &[4.4, 2.9, 1.4, 0.2],
        ]);
        let fastpair = FastPair::new(&x).unwrap();

        let ordered = fastpair.ordered_pairs();

        let mut previous: f64 = -1.0;
        for p in ordered {
            if previous == -1.0 {
                previous = p.distance.unwrap();
            } else {
                let current = p.distance.unwrap();
                assert!(current >= previous);
                previous = current;
            }
        }
    }
}
