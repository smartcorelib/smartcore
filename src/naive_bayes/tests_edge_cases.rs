//! Stage 3: edge-case & known-answer parity tests for src/naive_bayes.
//!
//! Covers: GaussianNB, BernoulliNB, CategoricalNB, MultinomialNB.
//! Tracking issue: #394 / #391.

#[cfg(test)]
mod naive_bayes_edge_cases {
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::naive_bayes::gaussian::{GaussianNB, GaussianNBParameters};
    use crate::naive_bayes::bernoulli::{BernoulliNB, BernoulliNBParameters};
    use crate::naive_bayes::categorical::{CategoricalNB, CategoricalNBParameters};
    use crate::naive_bayes::multinomial::{MultinomialNB, MultinomialNBParameters};

    // ── GaussianNB ────────────────────────────────────────────────────────────

    /// Clearly separated Gaussian blobs: must classify correctly.
    #[test]
    fn gaussian_nb_separated_blobs() {
        let x = DenseMatrix::from_2d_array(&[
            &[-5.0_f64, 0.0],
            &[-4.5, 0.0],
            &[4.5,  0.0],
            &[5.0,  0.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 0, 1, 1];
        let model = GaussianNB::fit(&x, &y, Default::default()).unwrap();
        assert_eq!(model.predict(&x).unwrap(), y);
    }

    /// Single-class input: all predictions equal that class.
    #[test]
    fn gaussian_nb_single_class_input() {
        let x = DenseMatrix::from_2d_array(&[
            &[1.0_f64, 2.0],
            &[1.1, 2.1],
            &[0.9, 1.9],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 0, 0];
        let model = GaussianNB::fit(&x, &y, Default::default()).unwrap();
        let preds = model.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 0));
    }

    /// Prior override: when class=1 prior is 0.99, ambiguous point should be class 1.
    #[test]
    fn gaussian_nb_prior_override() {
        // x=0 is equidistant from class 0 (mean=-1) and class 1 (mean=1).
        let x_train = DenseMatrix::from_2d_array(&[
            &[-1.0_f64], &[-1.0], &[1.0], &[1.0],
        ]).unwrap();
        let y_train: Vec<u32> = vec![0, 0, 1, 1];
        let x_test = DenseMatrix::from_2d_array(&[&[0.0_f64]]).unwrap();

        // Strongly biased prior toward class 1.
        let model = GaussianNB::fit(
            &x_train, &y_train,
            GaussianNBParameters::default().with_priors(vec![0.01, 0.99]),
        ).unwrap();
        let pred = model.predict(&x_test).unwrap();
        assert_eq!(pred[0], 1, "strong prior should push ambiguous point to class 1");
    }

    // ── BernoulliNB ───────────────────────────────────────────────────────────

    /// Binary features, clean separation: 100% accuracy.
    #[test]
    fn bernoulli_nb_clean_separation() {
        let x = DenseMatrix::from_2d_array(&[
            &[1_f64, 0.0, 0.0],
            &[1.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[0.0, 1.0, 1.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 0, 1, 1];
        let model = BernoulliNB::fit(&x, &y, Default::default()).unwrap();
        assert_eq!(model.predict(&x).unwrap(), y);
    }

    /// Laplace smoothing (alpha): fit must not panic and predictions are valid.
    #[test]
    fn bernoulli_nb_laplace_smoothing() {
        let x = DenseMatrix::from_2d_array(&[
            &[1_f64, 0.0],
            &[0.0, 1.0],
            &[1.0, 1.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 1, 0];
        let model = BernoulliNB::fit(&x, &y, BernoulliNBParameters::default().with_alpha(1.0)).unwrap();
        assert!(model.predict(&x).is_ok());
    }

    // ── CategoricalNB ─────────────────────────────────────────────────────────

    /// Single-category per feature: should predict without panic.
    #[test]
    fn categorical_nb_single_category() {
        let x = DenseMatrix::from_2d_array(&[
            &[0_f64, 0.0],
            &[0.0, 0.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 0];
        let model = CategoricalNB::fit(&x, &y, Default::default()).unwrap();
        let preds = model.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p == 0));
    }

    /// Multi-class categorical: predictions within valid class range.
    #[test]
    fn categorical_nb_multiclass() {
        let x = DenseMatrix::from_2d_array(&[
            &[0_f64, 1.0],
            &[1.0, 0.0],
            &[2.0, 2.0],
            &[0.0, 1.0],
            &[1.0, 0.0],
            &[2.0, 2.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 1, 2, 0, 1, 2];
        let model = CategoricalNB::fit(&x, &y, Default::default()).unwrap();
        let preds = model.predict(&x).unwrap();
        assert!(preds.iter().all(|&p| p <= 2));
    }

    // ── MultinomialNB ─────────────────────────────────────────────────────────

    /// Word-count bag-of-words toy: must fit and predict.
    #[test]
    fn multinomial_nb_word_counts() {
        let x = DenseMatrix::from_2d_array(&[
            &[3_f64, 0.0, 0.0],
            &[2.0, 0.0, 0.0],
            &[0.0, 0.0, 2.0],
            &[0.0, 0.0, 3.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 0, 1, 1];
        let model = MultinomialNB::fit(&x, &y, Default::default()).unwrap();
        assert_eq!(model.predict(&x).unwrap(), y);
    }

    /// Zero-count smoothing: alpha=1 prevents log(0) panic.
    #[test]
    fn multinomial_nb_zero_count_smoothing() {
        // Class 0 never sees feature 2 → without smoothing log(0) would occur.
        let x = DenseMatrix::from_2d_array(&[
            &[2_f64, 0.0, 0.0],
            &[0.0, 2.0, 0.0],
            &[0.0, 0.0, 2.0],
        ]).unwrap();
        let y: Vec<u32> = vec![0, 1, 2];
        let result = MultinomialNB::fit(&x, &y, MultinomialNBParameters::default().with_alpha(1.0));
        assert!(result.is_ok());
        let x_test = DenseMatrix::from_2d_array(&[&[0_f64, 0.0, 1.0]]).unwrap();
        assert!(result.unwrap().predict(&x_test).is_ok());
    }
}
