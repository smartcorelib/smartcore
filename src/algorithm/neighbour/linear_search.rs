use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, PartialOrd};
use std::marker::PhantomData;

use crate::algorithm::sort::heap_select::HeapSelect;
use crate::math::distance::Distance;
use crate::math::num::FloatExt;

#[derive(Serialize, Deserialize, Debug)]
pub struct LinearKNNSearch<T, F: FloatExt, D: Distance<T, F>> {
    distance: D,
    data: Vec<T>,
    f: PhantomData<F>,
}

impl<T, F: FloatExt, D: Distance<T, F>> LinearKNNSearch<T, F, D> {
    pub fn new(data: Vec<T>, distance: D) -> LinearKNNSearch<T, F, D> {
        LinearKNNSearch {
            data: data,
            distance: distance,
            f: PhantomData,
        }
    }

    pub fn find(&self, from: &T, k: usize) -> Vec<(usize, F)> {
        if k < 1 || k > self.data.len() {
            panic!("k should be >= 1 and <= length(data)");
        }

        let mut heap = HeapSelect::<KNNPoint<F>>::with_capacity(k);

        for _ in 0..k {
            heap.add(KNNPoint {
                distance: F::infinity(),
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

        heap.sort();

        heap.get()
            .into_iter()
            .flat_map(|x| x.index.map(|i| (i, x.distance)))
            .collect()
    }
}

#[derive(Debug)]
struct KNNPoint<F: FloatExt> {
    distance: F,
    index: Option<usize>,
}

impl<F: FloatExt> PartialOrd for KNNPoint<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl<F: FloatExt> PartialEq for KNNPoint<F> {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<F: FloatExt> Eq for KNNPoint<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::distance::Distances;

    struct SimpleDistance {}

    impl Distance<i32, f64> for SimpleDistance {
        fn distance(&self, a: &i32, b: &i32) -> f64 {
            (a - b).abs() as f64
        }
    }

    #[test]
    fn knn_find() {
        let data1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let algorithm1 = LinearKNNSearch::new(data1, SimpleDistance {});

        let found_idxs1: Vec<usize> = algorithm1.find(&2, 3).iter().map(|v| v.0).collect();

        assert_eq!(vec!(1, 2, 0), found_idxs1);

        let data2 = vec![
            vec![1., 1.],
            vec![2., 2.],
            vec![3., 3.],
            vec![4., 4.],
            vec![5., 5.],
        ];

        let algorithm2 = LinearKNNSearch::new(data2, Distances::euclidian());

        let found_idxs2: Vec<usize> = algorithm2
            .find(&vec![3., 3.], 3)
            .iter()
            .map(|v| v.0)
            .collect();

        assert_eq!(vec!(2, 3, 1), found_idxs2);
    }

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
