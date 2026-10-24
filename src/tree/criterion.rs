//! Split criteria for decision trees.
//!
//! A criterion keeps sufficient statistics for a set of samples. With these statistics,
//! the tree finds the best split and the value of a leaf.
//!
//! References:
//! * ["Classification and regression trees", Breiman L, Friedman J H, Olshen R A, and Stone C J, 1984](https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman)
//! * scikit-learn, `sklearn/tree/_criterion.pyx` (proxy improvement)

use crate::linalg::basic::arrays::ArrayView1;
use crate::numbers::basenum::Number;
use std::fmt::Debug;

/// Impurity measure that a tree uses to select splits.
pub(crate) trait Criterion {
    /// Sufficient statistics for a set of samples.
    type Stats: Clone;
    /// Target value of one sample.
    type Target: Copy + PartialEq;
    /// Value of a leaf.
    type Output: Clone + Debug;

    /// Statistics for an empty set. Use only this function to make new statistics.
    fn empty_stats(&self) -> Self::Stats;

    /// Set `stats` to the empty set. Keeps the allocation.
    fn reset(&self, stats: &mut Self::Stats);

    /// Add one sample with weight `mass` to `stats`.
    fn add(&self, stats: &mut Self::Stats, target: Self::Target, mass: f64);

    /// Set `out = parent - part`. `part` must be a subset of `parent`.
    fn difference(&self, parent: &Self::Stats, part: &Self::Stats, out: &mut Self::Stats);

    /// Larger is better. Only comparable between splits of the same parent (proxy, as in sklearn).
    fn proxy_improvement(&self, true_side: &Self::Stats, false_side: &Self::Stats) -> f64;

    /// Weighted decrease in impurity from this split:
    /// `mass(parent) * imp(parent) - mass(true) * imp(true) - mass(false) * imp(false)`.
    /// Not normalized by the total mass. Compute it one time for each split.
    fn impurity_improvement(
        &self,
        parent: &Self::Stats,
        true_side: &Self::Stats,
        false_side: &Self::Stats,
    ) -> f64;

    /// Value of a leaf that contains the samples of `stats`.
    fn leaf_value(&self, stats: &Self::Stats) -> Self::Output;

    /// Convert a target value of the training data to the target of this criterion.
    fn target<T: Number>(&self, y: T) -> Self::Target;
}

/// Value in a leaf of a tree.
pub(crate) trait LeafOutput: Clone + Debug {
    fn approximate_eq(&self, other: &Self) -> bool;
}

impl LeafOutput for f64 {
    fn approximate_eq(&self, other: &f64) -> bool {
        (self - other).abs() < f64::EPSILON
    }
}

impl LeafOutput for usize {
    fn approximate_eq(&self, other: &usize) -> bool {
        self == other
    }
}

/// Statistics for the weighted mean of the targets.
#[derive(Clone, Debug, Default)]
pub(crate) struct MeanStats {
    sum: f64,  // \sum w_i y_i
    mass: f64, // \sum w_i
}

/// Mean squared error criterion for regression. The leaf value is the weighted mean.
#[derive(Debug)]
pub(crate) struct MSECriterion;

impl Criterion for MSECriterion {
    type Stats = MeanStats;
    type Target = f64;
    type Output = f64;

    fn empty_stats(&self) -> Self::Stats {
        MeanStats {
            sum: 0.0,
            mass: 0.0,
        }
    }

    fn reset(&self, stats: &mut Self::Stats) {
        stats.sum = 0.0;
        stats.mass = 0.0;
    }

    #[inline]
    fn add(&self, stats: &mut Self::Stats, target: Self::Target, mass: f64) {
        stats.mass += mass;
        stats.sum += target * mass;
    }

    #[inline]
    fn difference(&self, parent: &Self::Stats, part: &Self::Stats, out: &mut Self::Stats) {
        out.mass = parent.mass - part.mass;
        out.sum = parent.sum - part.sum;
    }

    #[inline]
    fn proxy_improvement(&self, true_side: &Self::Stats, false_side: &Self::Stats) -> f64 {
        if true_side.mass <= 0.0 || false_side.mass <= 0.0 {
            return 0.0;
        }
        let d = true_side.sum * false_side.mass - false_side.sum * true_side.mass;
        (d * d) / (true_side.mass * false_side.mass)
    }

    fn impurity_improvement(
        &self,
        parent: &Self::Stats,
        true_side: &Self::Stats,
        false_side: &Self::Stats,
    ) -> f64 {
        if true_side.mass <= 0.0 || false_side.mass <= 0.0 {
            return 0.0;
        }
        let d = true_side.sum * false_side.mass - false_side.sum * true_side.mass;
        (d * d) / (true_side.mass * false_side.mass * parent.mass)
    }

    fn leaf_value(&self, stats: &Self::Stats) -> Self::Output {
        if stats.mass > 0.0 {
            stats.sum / stats.mass
        } else {
            0.0
        }
    }

    fn target<T: Number>(&self, y: T) -> Self::Target {
        y.to_f64().unwrap()
    }
}

#[derive(Clone, Debug, Default)]
/// Weighted count for each class.
pub(crate) struct ClassStats {
    class_counts: Vec<f64>,
    mass: f64,
}

impl ClassStats {
    fn empty_stats(num_classes: usize) -> Self {
        ClassStats {
            class_counts: vec![0.0; num_classes],
            mass: 0.0,
        }
    }

    fn reset(&mut self) {
        self.class_counts.fill(0.0);
        self.mass = 0.0;
    }

    fn add(&mut self, target: usize, mass: f64) {
        self.class_counts[target] += mass;
        self.mass += mass;
    }

    fn difference(&self, part: &ClassStats, out: &mut ClassStats) {
        for ((diff, p), c) in out
            .class_counts
            .iter_mut()
            .zip(self.class_counts.iter())
            .zip(part.class_counts.iter())
        {
            *diff = p - c;
        }
        out.mass = self.mass - part.mass;
    }
}

pub(crate) struct GiniCriterion {
    num_classes: usize,
}

impl GiniCriterion {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }

    // One term of the proxy improvement: sum_k count_k^2 / mass
    fn term(stats: &ClassStats) -> f64 {
        if stats.mass > 0.0 {
            stats.class_counts.iter().map(|c| c * c).sum::<f64>() / stats.mass
        } else {
            0.0
        }
    }
}

impl Criterion for GiniCriterion {
    type Stats = ClassStats;
    type Target = usize;
    type Output = usize;

    fn empty_stats(&self) -> Self::Stats {
        ClassStats::empty_stats(self.num_classes)
    }

    fn reset(&self, stats: &mut Self::Stats) {
        stats.reset();
    }

    #[inline]
    fn add(&self, stats: &mut Self::Stats, target: Self::Target, mass: f64) {
        debug_assert!(target < self.num_classes);
        stats.add(target, mass);
    }

    #[inline]
    fn difference(&self, parent: &Self::Stats, part: &Self::Stats, out: &mut Self::Stats) {
        debug_assert_eq!(parent.class_counts.len(), part.class_counts.len());
        debug_assert_eq!(parent.class_counts.len(), out.class_counts.len());
        parent.difference(part, out);
    }

    #[inline]
    fn proxy_improvement(&self, true_side: &Self::Stats, false_side: &Self::Stats) -> f64 {
        Self::term(true_side) + Self::term(false_side)
    }

    fn impurity_improvement(
        &self,
        parent: &Self::Stats,
        true_side: &Self::Stats,
        false_side: &Self::Stats,
    ) -> f64 {
        if true_side.mass <= 0.0 || false_side.mass <= 0.0 {
            return 0.0;
        }
        let s: f64 = true_side
            .class_counts
            .iter()
            .zip(false_side.class_counts.iter())
            .map(|(t, f)| {
                let d = t * false_side.mass - f * true_side.mass;
                d * d
            })
            .sum();
        s / (true_side.mass * false_side.mass * parent.mass)
    }

    fn leaf_value(&self, stats: &Self::Stats) -> Self::Output {
        stats.class_counts.argmax()
    }

    fn target<T: Number>(&self, y: T) -> Self::Target {
        y.to_usize().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn mse_stats(samples: &[(f64, f64)]) -> MeanStats {
        let criterion = MSECriterion;
        let mut stats = criterion.empty_stats();
        for &(y, w) in samples {
            criterion.add(&mut stats, y, w);
        }
        stats
    }

    // Sum of squared errors around the mean
    fn sse(ys: &[f64]) -> f64 {
        let mean = ys.iter().sum::<f64>() / ys.len() as f64;
        ys.iter().map(|y| (y - mean) * (y - mean)).sum()
    }

    #[test]
    fn mse_empty_stats() {
        let criterion = MSECriterion;
        let empty = criterion.empty_stats();
        assert_eq!(empty.sum, 0.0);
        assert_eq!(empty.mass, 0.0);
        assert_eq!(criterion.leaf_value(&empty), 0.0);
        assert_eq!(criterion.proxy_improvement(&empty, &empty), 0.0);
    }

    #[test]
    fn mse_leaf_value_is_weighted_mean() {
        let criterion = MSECriterion;
        let stats = mse_stats(&[(1.0, 1.0), (3.0, 1.0), (5.0, 2.0)]);
        assert_eq!(stats.mass, 4.0);
        assert_eq!(stats.sum, 14.0);
        assert!((criterion.leaf_value(&stats) - 3.5).abs() < EPS);
    }

    #[test]
    fn mse_reset_clears_stats() {
        let criterion = MSECriterion;
        let mut stats = mse_stats(&[(1.0, 1.0), (3.0, 2.0)]);
        criterion.reset(&mut stats);
        assert_eq!(stats.sum, 0.0);
        assert_eq!(stats.mass, 0.0);
    }

    #[test]
    fn mse_difference_gives_complement() {
        let criterion = MSECriterion;
        let parent = mse_stats(&[(1.0, 1.0), (2.0, 2.0), (10.0, 1.0), (11.0, 3.0)]);
        let part = mse_stats(&[(1.0, 1.0), (2.0, 2.0)]);
        let expected = mse_stats(&[(10.0, 1.0), (11.0, 3.0)]);

        let mut out = criterion.empty_stats();
        criterion.difference(&parent, &part, &mut out);
        assert!((out.sum - expected.sum).abs() < EPS);
        assert!((out.mass - expected.mass).abs() < EPS);
    }

    #[test]
    fn mse_proxy_is_sse_reduction_times_parent_mass() {
        let criterion = MSECriterion;
        let ys = [1.0, 2.0, 10.0, 11.0];
        let parent = mse_stats(&[(1.0, 1.0), (2.0, 1.0), (10.0, 1.0), (11.0, 1.0)]);
        let left = mse_stats(&[(1.0, 1.0), (2.0, 1.0)]);
        let right = mse_stats(&[(10.0, 1.0), (11.0, 1.0)]);

        let proxy = criterion.proxy_improvement(&left, &right);
        assert!((proxy - 324.0).abs() < EPS);

        let expected = sse(&ys) - sse(&ys[..2]) - sse(&ys[2..]);
        assert!((proxy / parent.mass - expected).abs() < EPS);
        assert!((proxy / parent.mass - 81.0).abs() < EPS);
    }

    #[test]
    fn mse_proxy_ranks_better_split_higher() {
        let criterion = MSECriterion;
        let good = criterion.proxy_improvement(
            &mse_stats(&[(1.0, 1.0), (2.0, 1.0)]),
            &mse_stats(&[(10.0, 1.0), (11.0, 1.0)]),
        );
        let bad = criterion.proxy_improvement(
            &mse_stats(&[(1.0, 1.0)]),
            &mse_stats(&[(2.0, 1.0), (10.0, 1.0), (11.0, 1.0)]),
        );
        assert!(good > bad);
    }

    // Weighted sum of squared errors around the weighted mean
    fn weighted_sse(samples: &[(f64, f64)]) -> f64 {
        let mass: f64 = samples.iter().map(|&(_, w)| w).sum();
        if mass == 0.0 {
            return 0.0;
        }
        let mean = samples.iter().map(|&(y, w)| w * y).sum::<f64>() / mass;
        samples
            .iter()
            .map(|&(y, w)| w * (y - mean) * (y - mean))
            .sum()
    }

    #[test]
    fn mse_improvement_known_answer() {
        let criterion = MSECriterion;
        let parent = mse_stats(&[(1.0, 1.0), (2.0, 1.0), (10.0, 1.0), (11.0, 1.0)]);
        let left = mse_stats(&[(1.0, 1.0), (2.0, 1.0)]);
        let right = mse_stats(&[(10.0, 1.0), (11.0, 1.0)]);

        // SSE(parent) = 82, SSE(left) = SSE(right) = 0.5
        let improvement = criterion.impurity_improvement(&parent, &left, &right);
        assert!((improvement - 81.0).abs() < EPS);
    }

    #[test]
    fn mse_improvement_matches_weighted_sse_reduction() {
        let criterion = MSECriterion;
        let left_samples = [(1.5, 0.5), (2.0, 2.0), (4.0, 1.25)];
        let right_samples = [(7.0, 3.0), (9.5, 0.75)];
        let all_samples: Vec<(f64, f64)> = left_samples
            .iter()
            .chain(right_samples.iter())
            .copied()
            .collect();

        let improvement = criterion.impurity_improvement(
            &mse_stats(&all_samples),
            &mse_stats(&left_samples),
            &mse_stats(&right_samples),
        );
        let expected =
            weighted_sse(&all_samples) - weighted_sse(&left_samples) - weighted_sse(&right_samples);
        assert!((improvement - expected).abs() < 1e-9);
    }

    #[test]
    fn mse_improvement_weight_equals_repeated_rows() {
        let criterion = MSECriterion;
        let weighted = criterion.impurity_improvement(
            &mse_stats(&[(1.0, 2.0), (3.0, 1.0), (8.0, 3.0)]),
            &mse_stats(&[(1.0, 2.0), (3.0, 1.0)]),
            &mse_stats(&[(8.0, 3.0)]),
        );
        let repeated = criterion.impurity_improvement(
            &mse_stats(&[
                (1.0, 1.0),
                (1.0, 1.0),
                (3.0, 1.0),
                (8.0, 1.0),
                (8.0, 1.0),
                (8.0, 1.0),
            ]),
            &mse_stats(&[(1.0, 1.0), (1.0, 1.0), (3.0, 1.0)]),
            &mse_stats(&[(8.0, 1.0), (8.0, 1.0), (8.0, 1.0)]),
        );
        assert!((weighted - repeated).abs() < EPS);
    }

    #[test]
    fn mse_improvement_pure_children_removes_all_error() {
        let criterion = MSECriterion;
        let samples = [(2.0, 1.0), (2.0, 3.0), (6.0, 2.0)];
        let improvement = criterion.impurity_improvement(
            &mse_stats(&samples),
            &mse_stats(&samples[..2]),
            &mse_stats(&samples[2..]),
        );
        assert!((improvement - weighted_sse(&samples)).abs() < EPS);
    }

    #[test]
    fn mse_improvement_is_zero_without_information() {
        let criterion = MSECriterion;
        // Both children have the same mean as the parent
        let parent = mse_stats(&[(1.0, 1.0), (3.0, 1.0), (1.0, 1.0), (3.0, 1.0)]);
        let left = mse_stats(&[(1.0, 1.0), (3.0, 1.0)]);
        let right = mse_stats(&[(1.0, 1.0), (3.0, 1.0)]);
        assert!(criterion.impurity_improvement(&parent, &left, &right).abs() < EPS);

        // An empty side does not change the impurity
        let empty = criterion.empty_stats();
        assert!(
            criterion
                .impurity_improvement(&parent, &parent, &empty)
                .abs()
                < EPS
        );
        assert!(
            criterion
                .impurity_improvement(&parent, &empty, &parent)
                .abs()
                < EPS
        );
    }

    #[test]
    fn mse_improvement_is_additive_over_the_tree() {
        // The sum of the improvements of all splits is SSE(root) - sum of SSE(leaves)
        let criterion = MSECriterion;
        let samples = [(1.0, 1.0), (2.0, 2.0), (5.0, 1.0), (9.0, 0.5), (12.0, 1.5)];
        let root = mse_stats(&samples);
        let left = mse_stats(&samples[..3]);
        let right = mse_stats(&samples[3..]);
        let left_left = mse_stats(&samples[..2]);
        let left_right = mse_stats(&samples[2..3]);

        let total = criterion.impurity_improvement(&root, &left, &right)
            + criterion.impurity_improvement(&left, &left_left, &left_right);
        let expected = weighted_sse(&samples)
            - weighted_sse(&samples[..2])
            - weighted_sse(&samples[2..3])
            - weighted_sse(&samples[3..]);
        assert!((total - expected).abs() < 1e-9);
    }

    #[test]
    fn mse_proxy_is_improvement_times_parent_mass() {
        // For all splits of one parent, the ranking must be the same as for the proxy
        let criterion = MSECriterion;
        let samples = [(1.0, 1.0), (4.0, 2.0), (2.0, 0.5), (8.0, 1.0), (3.0, 1.5)];
        let parent = mse_stats(&samples);
        let mut right = criterion.empty_stats();
        for k in 0..=samples.len() {
            let left = mse_stats(&samples[..k]);
            criterion.difference(&parent, &left, &mut right);
            let improvement = criterion.impurity_improvement(&parent, &left, &right);
            let proxy = criterion.proxy_improvement(&left, &right);
            assert!(improvement >= 0.0, "split {k}: {improvement}");
            assert!(
                (proxy - improvement * parent.mass).abs() < 1e-9,
                "split {k}: {improvement} {proxy}"
            );
        }
    }

    #[test]
    fn mse_improvement_is_invariant_to_target_offset() {
        // The SSE does not change when all targets get the same offset.
        // A large offset must not change the improvement (cancellation in sum^2 / mass).
        let criterion = MSECriterion;
        for offset in [0.0, 1e6, 1e8, 1e9, 1e10] {
            let y = |v: f64| (offset + v, 1.0);
            let parent = mse_stats(&[y(1.0), y(2.0), y(10.0), y(11.0)]);
            let left = mse_stats(&[y(1.0), y(2.0)]);
            let right = mse_stats(&[y(10.0), y(11.0)]);
            let improvement = criterion.impurity_improvement(&parent, &left, &right);
            assert!(
                (improvement - 81.0).abs() < 1e-6,
                "offset {offset}: improvement {improvement}, expected 81"
            );
        }
    }

    #[test]
    fn mse_target_converts_numbers_to_f64() {
        let criterion = MSECriterion;
        assert_eq!(criterion.target(3.25_f64), 3.25);
        assert_eq!(criterion.target(-1.5_f32), -1.5);
        assert_eq!(criterion.target(-7_i32), -7.0);
        assert_eq!(criterion.target(200_u8), 200.0);
        assert_eq!(criterion.target(42_usize), 42.0);
        // Largest integer that f64 represents exactly
        assert_eq!(
            criterion.target(9_007_199_254_740_992_i64),
            9_007_199_254_740_992.0
        );
    }

    #[test]
    fn f64_approximate_eq() {
        assert!(1.5.approximate_eq(&1.5));
        assert!(0.0.approximate_eq(&-0.0));
        // Difference smaller than EPSILON
        assert!(0.0.approximate_eq(&(f64::EPSILON / 2.0)));
        // Difference of exactly EPSILON is not "approximately equal" (strict comparison)
        assert!(!0.0.approximate_eq(&f64::EPSILON));
        assert!(!1.0.approximate_eq(&1.1));
        assert!(!f64::NAN.approximate_eq(&f64::NAN));
    }

    fn class_stats(num_classes: usize, samples: &[(usize, f64)]) -> ClassStats {
        let criterion = GiniCriterion { num_classes };
        let mut stats = criterion.empty_stats();
        for &(class, w) in samples {
            criterion.add(&mut stats, class, w);
        }
        stats
    }

    // Weighted Gini impurity: \sum_c n_c * (1 - \sum_k p_k^2)
    fn weighted_gini(sides: &[&ClassStats]) -> f64 {
        sides
            .iter()
            .filter(|s| s.mass > 0.0)
            .map(|s| {
                let sum_p2: f64 = s
                    .class_counts
                    .iter()
                    .map(|c| (c / s.mass) * (c / s.mass))
                    .sum();
                s.mass * (1.0 - sum_p2)
            })
            .sum()
    }

    #[test]
    fn gini_empty_stats() {
        let criterion = GiniCriterion { num_classes: 3 };
        let empty = criterion.empty_stats();
        assert_eq!(empty.class_counts, vec![0.0; 3]);
        assert_eq!(empty.mass, 0.0);
        assert_eq!(criterion.leaf_value(&empty), 0);
        assert_eq!(criterion.proxy_improvement(&empty, &empty), 0.0);
    }

    #[test]
    fn gini_add_with_weights() {
        let stats = class_stats(3, &[(2, 2.5), (0, 1.0)]);
        assert_eq!(stats.class_counts, vec![1.0, 0.0, 2.5]);
        assert_eq!(stats.mass, 3.5);
    }

    #[test]
    fn gini_term_known_answer() {
        let criterion = GiniCriterion { num_classes: 2 };
        let stats = class_stats(2, &[(0, 7.0), (1, 3.0)]);
        let empty = criterion.empty_stats();
        // (7^2 + 3^2) / 10
        assert!((criterion.proxy_improvement(&stats, &empty) - 5.8).abs() < EPS);
        assert!((criterion.proxy_improvement(&empty, &stats) - 5.8).abs() < EPS);
    }

    #[test]
    fn gini_proxy_equals_n_minus_weighted_gini() {
        let criterion = GiniCriterion { num_classes: 2 };
        let left = class_stats(2, &[(0, 3.0), (1, 1.0)]);
        let right = class_stats(2, &[(0, 1.0), (1, 3.0)]);
        let n = left.mass + right.mass;

        let impurity = weighted_gini(&[&left, &right]);
        assert!((impurity - 3.0).abs() < EPS);

        let proxy = criterion.proxy_improvement(&left, &right);
        assert!((proxy - (n - impurity)).abs() < EPS);
        assert!((proxy - 5.0).abs() < EPS);
    }

    #[test]
    fn gini_difference_gives_complement() {
        let criterion = GiniCriterion { num_classes: 2 };
        let parent = class_stats(2, &[(0, 4.0), (1, 4.0)]);
        let part = class_stats(2, &[(0, 3.0), (1, 1.0)]);

        let mut out = criterion.empty_stats();
        criterion.difference(&parent, &part, &mut out);
        assert_eq!(out.class_counts, vec![1.0, 3.0]);
        assert_eq!(out.mass, 4.0);
    }

    #[test]
    fn gini_reset_clears_stats() {
        let criterion = GiniCriterion { num_classes: 3 };
        let mut stats = class_stats(3, &[(0, 1.0), (2, 2.0)]);
        criterion.reset(&mut stats);
        assert_eq!(stats.class_counts, vec![0.0; 3]);
        assert_eq!(stats.mass, 0.0);
    }

    #[test]
    fn gini_pure_split_beats_mixed() {
        let criterion = GiniCriterion { num_classes: 2 };
        let pure =
            criterion.proxy_improvement(&class_stats(2, &[(0, 4.0)]), &class_stats(2, &[(1, 4.0)]));
        let mixed = criterion.proxy_improvement(
            &class_stats(2, &[(0, 3.0), (1, 1.0)]),
            &class_stats(2, &[(0, 1.0), (1, 3.0)]),
        );
        assert!((pure - 8.0).abs() < EPS);
        assert!(pure > mixed);
    }

    #[test]
    fn gini_leaf_value() {
        let criterion = GiniCriterion { num_classes: 3 };
        let stats = class_stats(3, &[(0, 1.0), (1, 5.0), (2, 2.0)]);
        assert_eq!(criterion.leaf_value(&stats), 1);

        // A tie gives the first class, same as `which_max`
        let tie = class_stats(3, &[(0, 3.0), (1, 3.0)]);
        assert_eq!(criterion.leaf_value(&tie), 0);
    }

    #[test]
    fn gini_improvement_known_answer() {
        let criterion = GiniCriterion { num_classes: 2 };
        let parent = class_stats(2, &[(0, 4.0), (1, 4.0)]);
        let left = class_stats(2, &[(0, 3.0), (1, 1.0)]);
        let right = class_stats(2, &[(0, 1.0), (1, 3.0)]);

        // 8 * gini(parent) = 4, weighted gini of children = 3
        let improvement = criterion.impurity_improvement(&parent, &left, &right);
        assert!((improvement - 1.0).abs() < EPS);
    }

    #[test]
    fn gini_improvement_matches_weighted_gini_reduction() {
        let criterion = GiniCriterion { num_classes: 3 };
        let left_samples = [(0, 1.5), (1, 0.25), (2, 2.0), (0, 0.75)];
        let right_samples = [(1, 3.0), (2, 0.5), (0, 1.0)];
        let all_samples: Vec<(usize, f64)> = left_samples
            .iter()
            .chain(right_samples.iter())
            .copied()
            .collect();

        let parent = class_stats(3, &all_samples);
        let left = class_stats(3, &left_samples);
        let right = class_stats(3, &right_samples);

        let improvement = criterion.impurity_improvement(&parent, &left, &right);
        let expected = weighted_gini(&[&parent]) - weighted_gini(&[&left, &right]);
        assert!((improvement - expected).abs() < EPS);
    }

    #[test]
    fn gini_improvement_pure_children_removes_all_impurity() {
        let criterion = GiniCriterion { num_classes: 3 };
        let parent = class_stats(3, &[(0, 2.0), (1, 1.0), (2, 3.0)]);
        let left = class_stats(3, &[(0, 2.0), (1, 1.0)]);
        let right = class_stats(3, &[(2, 3.0)]);

        let improvement = criterion.impurity_improvement(&parent, &left, &right);
        let expected = weighted_gini(&[&parent]) - weighted_gini(&[&left]);
        assert!((improvement - expected).abs() < EPS);

        // A split into two pure children removes all impurity of the parent
        let parent = class_stats(2, &[(0, 4.0), (1, 4.0)]);
        let improvement = criterion.impurity_improvement(
            &parent,
            &class_stats(2, &[(0, 4.0)]),
            &class_stats(2, &[(1, 4.0)]),
        );
        assert!((improvement - 4.0).abs() < EPS);
    }

    #[test]
    fn gini_improvement_is_zero_without_information() {
        let criterion = GiniCriterion { num_classes: 2 };
        // Both children have the same class proportions as the parent
        let parent = class_stats(2, &[(0, 6.0), (1, 3.0)]);
        let left = class_stats(2, &[(0, 2.0), (1, 1.0)]);
        let right = class_stats(2, &[(0, 4.0), (1, 2.0)]);
        assert!(criterion.impurity_improvement(&parent, &left, &right).abs() < EPS);

        // An empty side does not change the impurity
        let empty = criterion.empty_stats();
        assert!(
            criterion
                .impurity_improvement(&parent, &parent, &empty)
                .abs()
                < EPS
        );
        assert!(
            criterion
                .impurity_improvement(&parent, &empty, &parent)
                .abs()
                < EPS
        );
    }

    #[test]
    fn gini_improvement_differs_from_proxy_by_a_constant() {
        // For all splits of one parent, the ranking must be the same as for the proxy
        let criterion = GiniCriterion { num_classes: 3 };
        let samples = [(0, 1.0), (2, 2.0), (1, 0.5), (0, 1.0), (2, 1.5), (1, 3.0)];
        let parent = class_stats(3, &samples);
        let mut right = criterion.empty_stats();
        let mut offsets = Vec::new();
        for k in 0..=samples.len() {
            let left = class_stats(3, &samples[..k]);
            criterion.difference(&parent, &left, &mut right);
            let improvement = criterion.impurity_improvement(&parent, &left, &right);
            assert!(improvement > -1e-9, "split {k}: {improvement}");
            offsets.push(criterion.proxy_improvement(&left, &right) - improvement);
        }
        for offset in &offsets {
            assert!((offset - offsets[0]).abs() < 1e-9);
        }
    }
}
