use std::collections::HashMap;
use std::marker::PhantomData;

use crate::api::Predictor;
use crate::error::Failed;
use crate::linalg::basic::arrays::{Array1, Array2};
use crate::metrics::accuracy;

// -----------------------------------------------------------------------------
// Some basic structures
// -----------------------------------------------------------------------------

/// Strategy for aggregating votes from ensemble members.
///
/// Determines how individual model predictions are combined into
/// a final ensemble prediction.
///
/// # Usage
/// ```ignore
/// // Uniform: each model gets 1 vote
/// let ens = Ensemble::with_strategy(VotingStrategy::Uniform);
///
/// // Weighted: assign confidence scores to models
/// let mut ens = Ensemble::with_strategy(VotingStrategy::Weighted);
/// ens.add_with_params(None, model, Some(2.0), None, vec![])?;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VotingStrategy {
    /// Simple majority voting. Each member contributes 1 vote per prediction.
    /// The `weight` field in members is ignored.
    #[default]
    Uniform,
    /// Weighted voting. Each member's vote is multiplied by its `weight`.
    /// Final score for a class = sum of (weight * vote) across members.
    ///
    /// # Constraints
    /// * All members must have `weight: Some(f64)` when using this strategy.
    /// * Weights must be finite and non-negative (enforced at insertion).
    Weighted,
}

/// Summary information about the ensemble configuration.
///
/// Returned by [`Ensemble::get_ensemble_info`]. Use this to inspect
/// the current state of the ensemble without accessing internal fields.
///
/// # Example
/// ```ignore
/// let ensemble = Ensemble::<DenseMatrix<f64>, Vec<i32>>::new();
/// let info = ensemble.get_ensemble_info();
/// assert_eq!(info.total_members, 0);
/// ```
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq)]
pub struct EnsembleInfo {
    pub strategy: VotingStrategy,
    pub total_members: usize,
    pub enabled_members: usize,
    pub uses_weighted_voting: bool,
}

// -----------------------------------------------------------------------------
// Ensemble Member
// -----------------------------------------------------------------------------

/// Container for a model and its metadata within an ensemble.
///
/// This struct wraps a predictive model along with voting weight,
/// description, enabled state, and tags. It is managed internally
/// by [`Ensemble`] and not intended for direct construction.
///
/// # Type Parameters
/// * `X` - Input feature type (must implement `Array2<f64>`)
/// * `Y` - Label type (must implement `Array1<i32> + Clone`)
struct EnsembleMember<X, Y> {
    /// The underlying predictive model, boxed as a trait object.
    pub model: Box<dyn Predictor<X, Y>>,

    /// Optional weight for voting. Used only if strategy is `Weighted`.
    pub weight: Option<f64>,

    /// Optional human-readable description for documentation/debugging.
    pub description: Option<String>,

    /// Whether the model is enabled for inference. Disabled models
    /// are skipped during prediction but retained in the ensemble.
    pub is_enabled: bool,

    /// Tags for grouping/filtering models. Empty by default.
    /// Reserved for future extensibility.
    pub tags: Vec<String>,
}

impl<X, Y> EnsembleMember<X, Y> {
    // TODO We'll use it later, maybe someone
    /// Check if this member has a specific tag.
    #[allow(dead_code)]
    fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

impl<X, Y> std::fmt::Debug for EnsembleMember<X, Y>
where
    X: Array2<f64>,
    Y: Array1<i32> + Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnsembleMember")
            .field("weight", &self.weight)
            .field("description", &self.description)
            .field("is_enabled", &self.is_enabled)
            .field("tags", &self.tags)
            .field("model", &"<dyn Predictor>") // Dummy placeholder
            .finish()
    }
}

// -----------------------------------------------------------------------------
// Ensemble Structure
// -----------------------------------------------------------------------------

/// A voting ensemble for classification models.
///
/// Aggregates predictions from multiple `Predictor` instances using
/// hard voting (majority/weighted) via score aggregation.
/// # Type Parameters
/// * `X` - Input data type (e.g., `Array2<f64>` for feature vectors)
/// * `Y` - Label type (e.g., `Array1<i32>` for class labels)
///
/// # Constraints
/// * All models must predict the same label type (`i32`).
/// * Input `x` to `predict` methods should represent a single sample.
pub struct Ensemble<X, Y>
where
    X: Array2<f64>,
    Y: Array1<i32> + Clone,
{
    /// Registered ensemble members.
    members: HashMap<String, EnsembleMember<X, Y>>,

    /// Active voting strategy.
    strategy: VotingStrategy,

    /// Counter for auto-generated names.
    counter: usize,

    _phantom: PhantomData<(X, Y)>,
}

// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

impl<X, Y> Ensemble<X, Y>
where
    X: Array2<f64>,
    Y: Array1<i32> + Clone,
{
    /// Creates a new empty ensemble with `Uniform` voting strategy.
    ///
    /// # Type Inference
    /// Rust can usually infer `X` and `Y` from the first model you add:
    /// ```ignore
    /// let mut ens = Ensemble::new();
    /// ens.add(knn_model)?; // X, Y inferred from knn_model
    /// ```
    ///
    /// If inference fails, specify types explicitly:
    /// ```ignore
    /// let mut ens: Ensemble<DenseMatrix<f64>, Vec<i32>> = Ensemble::new();
    /// ```
    pub fn new() -> Self {
        Self {
            members: HashMap::new(),
            strategy: VotingStrategy::default(),
            counter: 0,
            _phantom: PhantomData,
        }
    }

    /// Creates a new ensemble with a specific voting strategy.
    ///
    /// # Arguments
    /// * `strategy` - `Uniform` for simple majority, `Weighted` for confidence-based voting.
    ///
    /// # Example
    /// ```ignore
    /// let mut ens = Ensemble::with_strategy(VotingStrategy::Weighted);
    /// // Now you must provide weights when adding models
    /// ens.add_with_params(None, model, Some(1.0), None, vec![])?;
    /// ```
    pub fn with_strategy(strategy: VotingStrategy) -> Self {
        Self {
            strategy,
            ..Self::new()
        }
    }

    // -------------------------------------------------------------------------
    // Add
    // -------------------------------------------------------------------------

    /// Convenience method: adds a model with auto-generated name and default metadata.
    ///
    /// Uses `Uniform` voting weight (1.0). For advanced options, use `add_with_params`.
    ///
    /// # Returns
    /// The auto-generated name of the added member, or error if addition failed.
    ///
    /// # Example
    /// ```ignore
    /// let mut ensemble = Ensemble::new();
    /// let model_name = ensemble.add(knn_model)?;
    /// println!("Added: {}", model_name);
    /// ```
    pub fn add<P>(&mut self, model: P) -> Result<String, Failed>
    where
        P: Predictor<X, Y> + 'static,
    {
        self.add_with_params(None, model, None, None, vec![])
    }

    /// Convenience method: adds a model with a custom name and default metadata.
    ///
    /// Equivalent to `add_with_params(Some(name), model, None, None, vec![])`.
    ///
    /// # Arguments
    /// * `name` - Unique identifier for the model. Must not already exist in the ensemble.
    /// * `model` - Any type implementing `Predictor<X, Y>`.
    ///
    /// # Returns
    /// * `Ok(String)` — The name of the added member (same as input `name`).
    /// * `Err(Failed)` — If name already exists or addition failed.
    ///
    /// # Example
    /// ```ignore
    /// let mut ensemble = Ensemble::new();
    /// ensemble.add_named("knn_k3".into(), knn_model)?;
    /// ensemble.add_named("rf_depth10".into(), rf_model)?;
    /// assert!(ensemble.names().contains(&"knn_k3".to_string()));
    /// ```
    pub fn add_named<P>(&mut self, name: String, model: P) -> Result<String, Failed>
    where
        P: Predictor<X, Y> + 'static,
    {
        self.add_with_params(Some(name), model, None, None, vec![])
    }

    /// Adds a model to the ensemble with optional metadata.
    ///
    /// # Arguments
    /// * `name` - Optional unique identifier. Auto-generated if None.
    /// * `model` - Any type implementing Predictor<X, Y>.
    /// * `weight` - Required if strategy is Weighted.
    ///
    /// # Example
    /// ```ignore
    /// ensemble.add_with_params(
    ///     Some("rf_v1".into()),
    ///     random_forest_model,
    ///     Some(0.8),
    ///     Some("RF with depth=10".into()),
    ///     vec!["tree".into()]
    /// )?;
    /// ```
    pub fn add_with_params<P>(
        &mut self,
        name: Option<String>,
        model: P,
        weight: Option<f64>,
        description: Option<String>,
        tags: Vec<String>,
    ) -> Result<String, Failed>
    where
        P: Predictor<X, Y> + 'static,
    {
        let final_name = name.unwrap_or_else(|| self.generate_auto_name());

        if self.members.contains_key(&final_name) {
            return Err(Failed::input("Duplicate member name"));
        }

        if matches!(self.strategy, VotingStrategy::Weighted) {
            match weight {
                Some(w) if w.is_finite() && w >= 0.0 => {}
                _ => return Err(Failed::input("Invalid weight")),
            }
        }

        let is_enabled = true;

        self.members.insert(
            final_name.clone(),
            EnsembleMember {
                model: Box::new(model),
                weight,
                description,
                is_enabled,
                tags,
            },
        );

        Ok(final_name)
    }

    // -------------------------------------------------------------------------
    // Metadata Management
    // -------------------------------------------------------------------------

    // Get

    /// Returns the total number of registered members.
    ///
    /// Includes both enabled and disabled models.
    /// Use [`enabled()`](Self::enabled) to count only active models.
    ///
    /// # Example
    /// ```ignore
    /// assert_eq!(ensemble.len(), 0);
    /// ensemble.add(model)?;
    /// assert_eq!(ensemble.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns names of all registered members (enabled and disabled).
    ///
    /// Order is arbitrary (HashMap iteration order).
    ///
    /// # Example
    /// ```ignore
    /// let names = ensemble.names();
    /// assert!(names.contains(&"my_model".to_string()));
    /// ```
    pub fn names(&self) -> Vec<String> {
        self.members.keys().cloned().collect()
    }

    /// Returns `true` if the ensemble has no registered members.
    ///
    /// # Example
    /// ```ignore
    /// let ens = Ensemble::<_, _>::new();
    /// assert!(ens.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Returns the current voting strategy.
    ///
    /// # Example
    /// ```ignore
    /// let ens = Ensemble::with_strategy(VotingStrategy::Weighted);
    /// assert_eq!(ens.strategy(), VotingStrategy::Weighted);
    /// ```
    pub fn strategy(&self) -> VotingStrategy {
        self.strategy
    }

    /// Returns the voting weight for a specific member.
    ///
    /// # Returns
    /// * `Some(weight)` if strategy is `Weighted` and member exists
    /// * `None` if strategy is `Uniform` OR member not found
    ///
    /// # Example
    /// ```ignore
    /// let w = ensemble.weight("my_model");
    /// if let Some(weight) = w {
    ///     println!("Weight: {}", weight);
    /// }
    /// ```
    pub fn weight(&self, name: &str) -> Option<f64> {
        if !self.members.contains_key(name) {
            return None;
        }

        match self.strategy {
            VotingStrategy::Uniform => None,
            VotingStrategy::Weighted => self.members.get(name).and_then(|member| member.weight),
        }
    }

    /// Returns summary information about the ensemble configuration.
    ///
    /// Does not include per-model hyperparameters (use [`get_model_metadata`](Self::get_model_metadata) for that).
    ///
    /// # Example
    /// ```ignore
    /// let info = ensemble.get_ensemble_info();
    /// println!("Strategy: {:?}", info.strategy);
    /// println!("Active models: {}/{}", info.enabled_members, info.total_members);
    /// ```
    pub fn get_ensemble_info(&self) -> EnsembleInfo {
        EnsembleInfo {
            strategy: self.strategy,
            total_members: self.members.len(),
            enabled_members: self.enabled().len(),
            uses_weighted_voting: matches!(self.strategy, VotingStrategy::Weighted),
        }
    }

    // Set

    /// Updates the voting weight of an existing member.
    ///
    /// # Arguments
    /// * `name` - Name of the member to update
    /// * `weight` - New weight value
    ///
    /// # Constraints
    /// * Member must exist
    /// * If strategy is `Weighted`, weight must be finite and non-negative
    ///
    /// # Errors
    /// * `Failed::input` if member not found or weight invalid
    ///
    /// # Example
    /// ```ignore
    /// ensemble.set_weight("strong_model", 2.0)?;
    /// ```
    pub fn set_weight(&mut self, name: &str, weight: f64) -> Result<(), Failed> {
        if matches!(self.strategy, VotingStrategy::Weighted) {
            if !weight.is_finite() || weight < 0.0 {
                return Err(Failed::input("Weight must be finite and non-negative"));
            }
        }
        let member = self
            .members
            .get_mut(name)
            .ok_or_else(|| Failed::input(&format!("Member '{}' not found", name)))?;
        member.weight = Some(weight);
        Ok(())
    }

    /// Updates the human-readable description of a member.
    ///
    /// Useful for documentation, debugging, or UI display.
    ///
    /// # Arguments
    /// * `name` - Name of the member
    /// * `desc` - New description string
    ///
    /// # Example
    /// ```ignore
    /// ensemble.set_description("rf_v1", "Random Forest, depth=10, trained on Q1 data")?;
    /// ```
    pub fn set_description(&mut self, name: &str, desc: String) -> Result<(), Failed> {
        let member = self
            .members
            .get_mut(name)
            .ok_or_else(|| Failed::input(&format!("Member '{}' not found", name)))?;
        member.description = Some(desc);
        Ok(())
    }

    /// Changes the voting strategy for the ensemble.
    ///
    /// # Arguments
    /// * `strategy` - New strategy (`Uniform` or `Weighted`)
    ///
    /// # Constraints
    /// * If switching to `Weighted`, all members must already have a weight set
    ///
    /// # Errors
    /// * `Failed::input` if switching to `Weighted` and any member lacks a weight
    ///
    /// # Example
    /// ```ignore
    /// // Start with Uniform
    /// let mut ens = Ensemble::new();
    /// ens.add(model1)?;
    /// ens.add(model2)?;
    ///
    /// // Switch to Weighted — must set weights first!
    /// ens.set_weight("model_0", 1.0)?;
    /// ens.set_weight("model_1", 2.0)?;
    /// ens.set_voting_strategy(VotingStrategy::Weighted)?;
    /// ```
    pub fn set_voting_strategy(&mut self, strategy: VotingStrategy) -> Result<(), Failed> {
        if matches!(strategy, VotingStrategy::Weighted) {
            for member in self.members.values() {
                if member.weight.is_none() {
                    return Err(Failed::input(
                        "All members must have weights when using weighted strategy",
                    ));
                }
            }
            self.strategy = strategy;

            return Ok(());
        }

        if matches!(strategy, VotingStrategy::Uniform) {
            self.strategy = strategy;
            return Ok(());
        }

        Err(Failed::input("Invalid voting strategy"))
    }

    // -------------------------------------------------------------------------
    // Model Management
    // -------------------------------------------------------------------------

    /// Temporarily excludes a member from prediction without removing it.
    ///
    /// Disabled models are skipped during `predict()` and `score()`,
    /// but remain in the ensemble and can be re-enabled later.
    ///
    /// # Arguments
    /// * `name` - Name of the member to disable
    ///
    /// # Errors
    /// * `Failed::input` if member not found or already disabled
    ///
    /// # Example
    /// ```ignore
    /// ensemble.disable("underperforming_model")?;
    /// let preds = ensemble.predict(&x)?; // model excluded from voting
    /// ```
    pub fn disable(&mut self, name: &str) -> Result<(), Failed> {
        if let Some(member) = self.members.get_mut(name) {
            if member.is_enabled {
                member.is_enabled = false;
                return Ok(());
            }
            return Err(Failed::input("Model is already disabled"));
        }
        Err(Failed::input("Model not found"))
    }

    /// Re-includes a previously disabled member in prediction.
    ///
    /// # Arguments
    /// * `name` - Name of the member to enable
    ///
    /// # Errors
    /// * `Failed::input` if member not found or already enabled
    ///
    /// # Example
    /// ```ignore
    /// ensemble.enable("previously_disabled_model")?;
    /// ```
    pub fn enable(&mut self, name: &str) -> Result<(), Failed> {
        if let Some(member) = self.members.get_mut(name) {
            if !member.is_enabled {
                member.is_enabled = true;
                return Ok(());
            }
            return Err(Failed::input("Model is already enabled"));
        }
        Err(Failed::input("Model not found"))
    }

    /// Returns names of all currently enabled members.
    ///
    /// Useful for debugging, logging, or selective inspection.
    ///
    /// # Example
    /// ```ignore
    /// let active = ensemble.enabled();
    /// println!("Active models: {:?}", active);
    /// ```
    pub fn enabled(&self) -> Vec<String> {
        self.members
            .iter()
            .filter(|(_, member)| member.is_enabled)
            .map(|(name, _)| name.clone())
            .collect()
    }

    // -------------------------------------------------------------------------
    // Predictions, regular and distributed, and a simple scoring
    // -------------------------------------------------------------------------

    /// Predicts class labels for input samples using ensemble voting.
    ///
    /// All enabled members receive the **same** input `x` and contribute
    /// votes according to the active [`VotingStrategy`].
    ///
    /// # Arguments
    /// * `x` - Input feature matrix (samples × features)
    ///
    /// # Returns
    /// * `Ok(Y)` - Vector of predicted class labels
    /// * `Err(Failed)` - If ensemble is empty or prediction fails
    ///
    /// # Example
    /// ```ignore
    /// let predictions = ensemble.predict(&x_test)?;
    /// let acc = accuracy(&y_test, &predictions);
    /// ```
    ///
    /// # See Also
    /// * [`predict_using_names()`](Self::predict_using_names) — for feature-slicing scenarios
    /// * [`score()`](Self::score) — for quick accuracy evaluation
    pub fn predict(&self, x: &X) -> Result<Y, Failed> {
        if self.members.is_empty() {
            return Err(Failed::predict("Empty ensemble"));
        }

        let enabled: Vec<String> = self.enabled();

        let mut all_preds: Vec<Y> = Vec::with_capacity(enabled.len());

        for model_name in &enabled {
            let model = self.members.get(model_name).unwrap(); // TODO unwrap is safe here, but maybe "if let" will just look better
            let pred = model.model.predict(x)?;
            all_preds.push(pred);
        }

        let n_samples = all_preds[0].shape();
        let mut result = Y::zeros(n_samples);

        for i in 0..n_samples {
            let mut scores: HashMap<i32, f64> = HashMap::new();

            for (model_name, preds) in enabled.iter().zip(all_preds.iter()) {
                let class = *preds.get(i);

                let vote = match (self.strategy, self.weight(model_name)) {
                    (VotingStrategy::Uniform, _) => 1.0,
                    (VotingStrategy::Weighted, Some(w)) => w,
                    (VotingStrategy::Weighted, None) => {
                        return Err(Failed::predict("Missing weight"))
                    }
                };

                *scores.entry(class).or_insert(0.0) += vote;
            }

            // argmax
            let (best_class, _) = scores
                .into_iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .ok_or_else(|| Failed::predict("No votes"))?;

            result.set(i, best_class);
        }

        Ok(result)
    }

    /// Computes accuracy score on given data.
    ///
    /// Equivalent to sklearn `accuracy(y, self.predict(x))`.
    pub fn score(&self, x: &X, y: &Y) -> Result<f64, Failed>
    where
        Y: Array1<i32>,
    {
        let preds = self.predict(x)?;
        Ok(accuracy(y, &preds))
    }

    /// Predicts using per-model input subsets (feature slicing).
    ///
    /// Each ensemble member receives its own input from the `inputs` map,
    /// keyed by member name. Useful when models are trained on different
    /// feature subsets or representations.
    ///
    /// # Arguments
    /// * `inputs` - HashMap mapping member names to their specific input matrices
    ///
    /// # Returns
    /// * `Ok(Y)` - Aggregated predictions via voting
    /// * `Err(Failed)` - If any required input is missing or prediction fails
    ///
    /// # Example
    /// ```ignore
    /// // Models trained on different feature slices
    /// let mut inputs = HashMap::new();
    /// inputs.insert("slice_A".into(), x_slice_a);
    /// inputs.insert("slice_B".into(), x_slice_b);
    ///
    /// let predictions = ensemble.predict_using_names(&inputs)?;
    /// ```
    ///
    /// # Constraints
    /// * Every enabled member must have a corresponding entry in `inputs`
    /// * All input matrices must have the same number of samples
    pub fn predict_using_names(&self, inputs: &HashMap<String, X>) -> Result<Y, Failed> {
        if self.members.is_empty() {
            return Err(Failed::predict("Empty ensemble"));
        }

        let mut all_preds: Vec<(String, Y)> = Vec::new();

        for (name, member) in &self.members {
            if !member.is_enabled {
                continue;
            }

            let x = inputs
                .get(name)
                .ok_or_else(|| Failed::input("Missing X set in input"))?;

            let pred = member.model.predict(x)?;
            all_preds.push((name.clone(), pred));
        }

        let n_samples = all_preds[0].1.shape();
        let mut result = Y::zeros(n_samples);

        for i in 0..n_samples {
            let mut scores: HashMap<i32, f64> = HashMap::new();

            for (name, preds) in &all_preds {
                let member = &self.members[name];
                let class = *preds.get(i);

                let vote = match (self.strategy, member.weight) {
                    (VotingStrategy::Uniform, _) => 1.0,
                    (VotingStrategy::Weighted, Some(w)) => w,
                    (VotingStrategy::Weighted, None) => {
                        return Err(Failed::predict("Missing weight"))
                    }
                };

                *scores.entry(class).or_insert(0.0) += vote;
            }

            let (best_class, _) = scores
                .into_iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .ok_or_else(|| Failed::predict("No votes"))?;

            result.set(i, best_class);
        }

        Ok(result)
    }

    // -------------------------------------------------------------------------
    // Utilities
    // -------------------------------------------------------------------------

    fn generate_auto_name(&mut self) -> String {
        let name = format!("model_{}", self.counter);
        self.counter += 1;
        name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    };
    use crate::linalg::basic::matrix::DenseMatrix;
    use crate::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};

    fn dummy_data_2class() -> (DenseMatrix<f64>, Vec<i32>) {
        // 6 samples, 2 features, balanced classes (3 vs 3)
        let x = DenseMatrix::from_2d_vec(&vec![
            vec![1.0, 1.0], // Class 0
            vec![1.5, 1.2], // Class 0
            vec![2.0, 1.5], // Class 0
            vec![4.0, 4.0], // Class 1
            vec![4.5, 4.2], // Class 1
            vec![5.0, 4.5], // Class 1
        ])
        .unwrap();
        let y = vec![0, 0, 0, 1, 1, 1];
        (x, y)
    }

    // Apply wasm_bindgen_test to all tests in this module
    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]

    // Test 1: Simple add() with auto-generated names
    #[test]
    fn test_add_simple_knn_models() {
        let (x, y) = dummy_data_2class();
        let mut ensemble = Ensemble::<DenseMatrix<f64>, Vec<i32>>::new();

        let knn1 =
            KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(3)).unwrap();
        let knn2 =
            KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(5)).unwrap();

        let name1 = ensemble.add(knn1).unwrap();
        let name2 = ensemble.add(knn2).unwrap();

        assert_eq!(ensemble.len(), 2);
        assert_eq!(name1, "model_0");
        assert_eq!(name2, "model_1");

        let names = ensemble.names();
        assert!(names.contains(&"model_0".to_string()));
        assert!(names.contains(&"model_1".to_string()));
    }

    // Test 2: Heterogeneous ensemble via add() — KNN + RF
    #[test]
    fn test_add_heterogeneous_models() {
        let (x, y) = dummy_data_2class();
        let mut ensemble = Ensemble::<DenseMatrix<f64>, Vec<i32>>::new();

        let knn = KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(3)).unwrap();
        let rf = RandomForestClassifier::fit(
            &x,
            &y,
            RandomForestClassifierParameters::default().with_n_trees(5),
        )
        .unwrap();

        let name_knn = ensemble.add(knn).unwrap();
        let name_rf = ensemble.add(rf).unwrap();

        assert_eq!(ensemble.len(), 2);
        let names = ensemble.names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&name_knn));
        assert!(names.contains(&name_rf));
    }

    // Test 3: add_named() with custom names
    #[test]
    fn test_add_named_custom_names() {
        let (x, y) = dummy_data_2class();
        let mut ensemble = Ensemble::<DenseMatrix<f64>, Vec<i32>>::new();

        let knn = KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(3)).unwrap();
        let rf = RandomForestClassifier::fit(
            &x,
            &y,
            RandomForestClassifierParameters::default().with_n_trees(5),
        )
        .unwrap();

        let name1 = ensemble.add_named("my_knn".into(), knn).unwrap();
        let name2 = ensemble.add_named("my_rf".into(), rf).unwrap();

        assert_eq!(name1, "my_knn");
        assert_eq!(name2, "my_rf");
        assert_eq!(ensemble.len(), 2);

        let names = ensemble.names();
        assert!(names.contains(&"my_knn".to_string()));
        assert!(names.contains(&"my_rf".to_string()));
    }

    // Test 4: Error on duplicate name via add_named()
    #[test]
    fn test_error_duplicate_name() {
        let (x, y) = dummy_data_2class();
        let mut ensemble = Ensemble::<DenseMatrix<f64>, Vec<i32>>::new();

        let knn1 =
            KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(2)).unwrap();
        let knn2 =
            KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(2)).unwrap();

        ensemble.add_named("same_name".into(), knn1).unwrap();

        let result = ensemble.add_named("same_name".into(), knn2);
        assert!(result.is_err());
    }

    // Test 5: Weighted voting with explicit weights
    #[test]
    fn test_weighted_voting_with_weights() {
        let (x, y) = dummy_data_2class();
        let mut ensemble =
            Ensemble::<DenseMatrix<f64>, Vec<i32>>::with_strategy(VotingStrategy::Weighted);

        let knn_weak =
            KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(3)).unwrap();
        let knn_strong =
            KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(5)).unwrap();

        // Add with different weights
        ensemble
            .add_with_params(Some("weak".into()), knn_weak, Some(0.5), None, vec![])
            .unwrap();
        ensemble
            .add_with_params(Some("strong".into()), knn_strong, Some(2.0), None, vec![])
            .unwrap();

        // Predict should work without error
        let preds = ensemble.predict(&x).unwrap();
        assert_eq!(preds.len(), y.len());

        // Score should be in valid range
        let score = ensemble.score(&x, &y).unwrap();
        assert!((0.0..=1.0).contains(&score));
    }

    // Test 6: Error when adding without weight in Weighted mode
    #[test]
    fn test_error_missing_weight_in_weighted_mode() {
        let (x, y) = dummy_data_2class();
        let mut ensemble =
            Ensemble::<DenseMatrix<f64>, Vec<i32>>::with_strategy(VotingStrategy::Weighted);

        let knn = KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(2)).unwrap();

        // add() does not provide weight → should fail in Weighted mode
        let result = ensemble.add(knn);
        assert!(result.is_err());
    }

    // Test 7: predict_using_names() with feature slicing
    #[test]
    fn test_predict_using_names_feature_slicing() {
        let (_, y) = dummy_data_2class(); // y имеет длину 6
        let mut ensemble = Ensemble::<DenseMatrix<f64>, Vec<i32>>::new();

        // Model A: trained on feature 0 only (must have 6 rows to match y!)
        let x_a_train = DenseMatrix::from_2d_vec(&vec![
            vec![1.0], // sample 0, feat 0
            vec![1.5], // sample 1, feat 0
            vec![2.0], // sample 2, feat 0
            vec![4.0], // sample 3, feat 0
            vec![4.5], // sample 4, feat 0
            vec![5.0], // sample 5, feat 0
        ])
        .unwrap();
        let knn_a =
            KNNClassifier::fit(&x_a_train, &y, KNNClassifierParameters::default().with_k(3))
                .unwrap();

        // Model B: trained on feature 1 only (must have 6 rows to match y!)
        let x_b_train = DenseMatrix::from_2d_vec(&vec![
            vec![1.0], // sample 0, feat 1
            vec![1.2], // sample 1, feat 1
            vec![1.5], // sample 2, feat 1
            vec![4.0], // sample 3, feat 1
            vec![4.2], // sample 4, feat 1
            vec![4.5], // sample 5, feat 1
        ])
        .unwrap();
        let knn_b =
            KNNClassifier::fit(&x_b_train, &y, KNNClassifierParameters::default().with_k(3))
                .unwrap();

        ensemble.add_named("model_A".into(), knn_a).unwrap();
        ensemble.add_named("model_B".into(), knn_b).unwrap();

        // Prepare per-model inputs for prediction (2 test samples)
        let mut inputs = HashMap::new();
        let x_a_test = DenseMatrix::from_2d_vec(&vec![vec![1.8], vec![4.3]]).unwrap();
        let x_b_test = DenseMatrix::from_2d_vec(&vec![vec![1.6], vec![4.4]]).unwrap();
        inputs.insert("model_A".into(), x_a_test);
        inputs.insert("model_B".into(), x_b_test);

        // Predict with per-model inputs
        let preds = ensemble.predict_using_names(&inputs).unwrap();
        assert_eq!(preds.len(), 2); // 2 test samples
    }

    // Test 8: enable/disable affects prediction
    #[test]
    fn test_enable_disable_affects_prediction() {
        let (x_train, y_train) = dummy_data_2class();
        let (x_test, y_test) = dummy_data_2class();

        let mut ensemble = Ensemble::<DenseMatrix<f64>, Vec<i32>>::new();

        let knn1 = KNNClassifier::fit(
            &x_train,
            &y_train,
            KNNClassifierParameters::default().with_k(3),
        )
        .unwrap();
        let knn2 = KNNClassifier::fit(
            &x_train,
            &y_train,
            KNNClassifierParameters::default().with_k(5),
        )
        .unwrap();

        ensemble.add_named("k1".into(), knn1).unwrap();
        ensemble.add_named("k3".into(), knn2).unwrap();

        // Score with both models
        let score_both = ensemble.score(&x_test, &y_test).unwrap();

        // Disable one model
        ensemble.disable("k3").unwrap();
        let score_one = ensemble.score(&x_test, &y_test).unwrap();

        // Both scores should be valid; they may differ
        assert!((0.0..=1.0).contains(&score_both));
        assert!((0.0..=1.0).contains(&score_one));

        // Re-enable and verify count
        ensemble.enable("k3").unwrap();
        assert_eq!(ensemble.enabled().len(), 2);
    }

    // Test 9: get_ensemble_info() reflects state
    #[test]
    fn test_get_ensemble_info() {
        let mut ensemble =
            Ensemble::<DenseMatrix<f64>, Vec<i32>>::with_strategy(VotingStrategy::Weighted);

        let info = ensemble.get_ensemble_info();
        assert_eq!(info.strategy, VotingStrategy::Weighted);
        assert_eq!(info.total_members, 0);
        assert_eq!(info.enabled_members, 0);
        assert!(info.uses_weighted_voting);

        // Add a model
        let (x, y) = dummy_data_2class();
        let knn = KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(2)).unwrap();
        ensemble
            .add_with_params(None, knn, Some(1.0), None, vec![])
            .unwrap();

        let info2 = ensemble.get_ensemble_info();
        assert_eq!(info2.total_members, 1);
        assert_eq!(info2.enabled_members, 1);
    }

    // Test 10: All supported classifiers (smoke test) — 3 models
    #[test]
    fn test_add_all_classifier_types() {
        use crate::tree::decision_tree_classifier::DecisionTreeClassifier;

        let (x, y) = dummy_data_2class();
        let mut ensemble = Ensemble::<DenseMatrix<f64>, Vec<i32>>::new();

        // 1. KNN
        let knn = KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(3)).unwrap();
        ensemble.add_named("knn_k3".into(), knn).unwrap();

        // 2. Random Forest
        let rf = RandomForestClassifier::fit(
            &x,
            &y,
            RandomForestClassifierParameters::default().with_n_trees(3),
        )
        .unwrap();
        ensemble.add_named("rf_3trees".into(), rf).unwrap();

        // 3. Decision Tree
        let dt = DecisionTreeClassifier::fit(&x, &y, Default::default()).unwrap();
        ensemble.add_named("decision_tree".into(), dt).unwrap();

        // All 3 models are active
        assert_eq!(ensemble.len(), 3);
        assert_eq!(ensemble.enabled().len(), 3);

        // Name check
        let names = ensemble.names();
        assert!(names.contains(&"knn_k3".to_string()));
        assert!(names.contains(&"rf_3trees".to_string()));
        assert!(names.contains(&"decision_tree".to_string()));

        // The most beautiful thing - predict() on 3 different models
        let preds = ensemble.predict(&x).unwrap();
        assert_eq!(preds.len(), y.len());

        // Score must be valid
        let score = ensemble.score(&x, &y).unwrap();
        assert!((0.0..=1.0).contains(&score));
    }

    // Test 10a: add_with_params() with custom names
    #[test]
    fn test_add_with_custom_names() {
        let (x, y) = dummy_data_2class();
        let mut ensemble = Ensemble::<DenseMatrix<f64>, Vec<i32>>::new();

        let knn = KNNClassifier::fit(&x, &y, KNNClassifierParameters::default().with_k(2)).unwrap();
        let rf = RandomForestClassifier::fit(
            &x,
            &y,
            RandomForestClassifierParameters::default().with_n_trees(5),
        )
        .unwrap();

        let name1 = ensemble
            .add_with_params(
                Some("my_knn".into()),
                knn,
                None,
                Some("k=1".into()),
                vec!["fast".into()],
            )
            .unwrap();
        let name2 = ensemble
            .add_with_params(
                Some("my_rf".into()),
                rf,
                None,
                Some("5 trees".into()),
                vec!["tree".into()],
            )
            .unwrap();

        assert_eq!(name1, "my_knn");
        assert_eq!(name2, "my_rf");
        assert_eq!(ensemble.len(), 2);
        // Check metadata
        // TODO To be implemented one day
    }

    // Test 10b: score() is still valid after adding a model
    #[test]
    fn test_score_with_increasing_models() {
        let (x_train, y_train) = dummy_data_2class();
        let (x_test, y_test) = dummy_data_2class();

        let mut ensemble = Ensemble::<DenseMatrix<f64>, Vec<i32>>::new();

        // Add first model
        let knn1 = KNNClassifier::fit(
            &x_train,
            &y_train,
            KNNClassifierParameters::default().with_k(2),
        )
        .unwrap();
        ensemble.add(knn1).unwrap();
        let score1 = ensemble.score(&x_test, &y_test).unwrap();
        assert!((0.0..=1.0).contains(&score1));

        // Add second model
        let knn2 = KNNClassifier::fit(
            &x_train,
            &y_train,
            KNNClassifierParameters::default().with_k(3),
        )
        .unwrap();
        ensemble.add(knn2).unwrap();
        let score2 = ensemble.score(&x_test, &y_test).unwrap();

        // Score may go up or down — just ensure it's valid
        assert!((0.0..=1.0).contains(&score2));
    }

    // Test 10c: score() is still valid after disabling a model
    #[test]
    fn test_score_after_disable() {
        let (x_train, y_train) = dummy_data_2class();
        let (x_test, y_test) = dummy_data_2class();

        let mut ensemble =
            Ensemble::<DenseMatrix<f64>, Vec<i32>>::with_strategy(VotingStrategy::Uniform);

        let knn1 = KNNClassifier::fit(
            &x_train,
            &y_train,
            KNNClassifierParameters::default().with_k(2),
        )
        .unwrap();
        let knn2 = KNNClassifier::fit(
            &x_train,
            &y_train,
            KNNClassifierParameters::default().with_k(3),
        )
        .unwrap();

        ensemble
            .add_with_params(Some("k1".into()), knn1, None, None, vec![])
            .unwrap();
        ensemble
            .add_with_params(Some("k3".into()), knn2, None, None, vec![])
            .unwrap();

        let score_before = ensemble.score(&x_test, &y_test).unwrap();

        ensemble.disable("k3").unwrap();
        let score_after = ensemble.score(&x_test, &y_test).unwrap();

        // Scores can differ; just ensure both are valid
        assert!((0.0..=1.0).contains(&score_before));
        assert!((0.0..=1.0).contains(&score_after));
    }
}
