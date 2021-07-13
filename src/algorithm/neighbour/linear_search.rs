//! # Brute Force Linear Search
//!
//! see [KNN algorithms](../index.html)
//! ```
//! use smartcore::algorithm::neighbour::linear_search::*;
//! use smartcore::metrics::distance::Distance;
//!
//! #[derive(Clone)]
//! struct SimpleDistance {} // Our distance function
//!
//! impl Distance<i32> for SimpleDistance {
//!   fn distance(&self, a: &i32, b: &i32) -> f64 { // simple simmetrical scalar distance
//!     (a - b).abs() as f64
//!   }
//! }
//!
//! let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9]; // data points
//!
//! let knn = LinearKNNSearch::new(data, SimpleDistance {}).unwrap();
//!
//! knn.find(&5, 3); // find 3 knn points from 5
//!
//! ```

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, PartialOrd};

use crate::algorithm::sort::heap_select::HeapSelection;
use crate::error::{Failed, FailedError};
use crate::metrics::distance::Distance;

/// Implements Linear Search algorithm, see [KNN algorithms](../index.html)
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct LinearKNNSearch<T, D: Distance<T>> {
    distance: D,
    data: Vec<T>,
}

impl<T, D: Distance<T>> LinearKNNSearch<T, D> {
    /// Initializes algorithm.
    /// * `data` - vector of data points to search for.
    /// * `distance` - distance metric to use for searching. This function should extend [`Distance`](../../../math/distance/index.html) interface.
    pub fn new(data: Vec<T>, distance: D) -> Result<LinearKNNSearch<T, D>, Failed> {
        Ok(LinearKNNSearch { data, distance })
    }

    /// Find k nearest neighbors
    /// * `from` - look for k nearest points to `from`
    /// * `k` - the number of nearest neighbors to return
    pub fn find(&self, from: &T, k: usize) -> Result<Vec<(usize, f64, &T)>, Failed> {
        if k < 1 || k > self.data.len() {
            return Err(Failed::because(
                FailedError::FindFailed,
                "k should be >= 1 and <= length(data)",
            ));
        }

        let mut heap = HeapSelection::<KNNPoint>::with_capacity(k);

        for _ in 0..k {
            heap.add(KNNPoint {
                distance: std::f64::INFINITY,
                index: None,
            });
        }

        for i in 0..self.data.len() {
            let d = self.distance.distance(&from, &self.data[i]);
            let datum = heap.peek_mut();
            if d < datum.distance {
                datum.distance = d;
                datum.index = Some(i);
                heap.heapify();
            }
        }

        Ok(heap
            .get()
            .into_iter()
            .flat_map(|x| x.index.map(|i| (i, x.distance, &self.data[i])))
            .collect())
    }

    /// Find all nearest neighbors within radius `radius` from `p`
    /// * `p` - look for k nearest points to `p`
    /// * `radius` - radius of the search
    pub fn find_radius(&self, from: &T, radius: f64) -> Result<Vec<(usize, f64, &T)>, Failed> {
        if radius <= 0f64 {
            return Err(Failed::because(
                FailedError::FindFailed,
                "radius should be > 0",
            ));
        }

        let mut neighbors: Vec<(usize, f64, &T)> = Vec::new();

        for i in 0..self.data.len() {
            let d = self.distance.distance(&from, &self.data[i]);

            if d <= radius {
                neighbors.push((i, d, &self.data[i]));
            }
        }

        Ok(neighbors)
    }
}

#[derive(Debug)]
struct KNNPoint {
    distance: f64,
    index: Option<usize>,
}

impl PartialOrd for KNNPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl PartialEq for KNNPoint {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for KNNPoint {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::distance::Distances;

    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[derive(Debug, Clone)]
    struct SimpleDistance {}

    impl Distance<i32> for SimpleDistance {
        fn distance(&self, a: &i32, b: &i32) -> f64 {
            (a - b).abs() as f64
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn knn_find() {
        let data1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let algorithm1 = LinearKNNSearch::new(data1, SimpleDistance {}).unwrap();

        let mut found_idxs1: Vec<usize> = algorithm1
            .find(&2, 3)
            .unwrap()
            .iter()
            .map(|v| v.0)
            .collect();
        found_idxs1.sort_unstable();

        assert_eq!(vec!(0, 1, 2), found_idxs1);

        let mut found_idxs1: Vec<i32> = algorithm1
            .find_radius(&5, 3.0)
            .unwrap()
            .iter()
            .map(|v| *v.2)
            .collect();
        found_idxs1.sort_unstable();

        assert_eq!(vec!(2, 3, 4, 5, 6, 7, 8), found_idxs1);

        let data2 = vec![
            vec![1., 1.],
            vec![2., 2.],
            vec![3., 3.],
            vec![4., 4.],
            vec![5., 5.],
        ];

        let algorithm2 = LinearKNNSearch::new(data2, Distances::euclidian()).unwrap();

        let mut found_idxs2: Vec<usize> = algorithm2
            .find(&vec![3., 3.], 3)
            .unwrap()
            .iter()
            .map(|v| v.0)
            .collect();
        found_idxs2.sort_unstable();

        assert_eq!(vec!(1, 2, 3), found_idxs2);
    }
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn knn_point_eq() {
        let point1 = KNNPoint {
            distance: 10.,
            index: Some(0),
        };

        let point2 = KNNPoint {
            distance: 100.,
            index: Some(1),
        };

        let point3 = KNNPoint {
            distance: 10.,
            index: Some(2),
        };

        let point_inf = KNNPoint {
            distance: std::f64::INFINITY,
            index: Some(3),
        };

        assert!(point2 > point1);
        assert_eq!(point3, point1);
        assert_ne!(point3, point2);
        assert!(point_inf > point3 && point_inf > point2 && point_inf > point1);
    }
}
