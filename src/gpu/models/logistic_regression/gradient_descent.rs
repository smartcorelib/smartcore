
use std::mem::size_of_val;
use std::convert::{TryFrom, TryInto};
use crate::linear::logistic_regression::{LogisticRegressionParameters, LogisticRegressionSolverName};
use crate::gpu::{GpuModule, GpuAlgorithm, GpuLayout, GpuMatrix, GpuParams};
use crate::error::GpuError;

pub struct BinaryClassifier {
    pub is_f32: bool,
    pub learning_rate: f32
}

pub struct MultiClassifier {
    pub is_f32: bool,
    pub learning_rate: f32
}

impl GpuModule for BinaryClassifier {
    fn get_params(&self, matrix: &GpuMatrix, num_classes: usize) -> Result<GpuParams, GpuError> {

        if num_classes > 2 {
            return Err(GpuError::Generic("Trying to run binary classification with more than two classes.  Uh oh, something went wrong somewhere!".to_string()));
        }

        Ok(GpuParams::new(GpuAlgorithm::LogisticRegressionGradientDescentBinaryClassification, GpuLayout::Supervised, &matrix, num_classes as u32, 3, "main", self.learning_rate)) 
    }

    fn get_params_buffer_data(&self, params: &GpuParams) -> Vec<u32> {
        vec![params.step, params.num_features, params.num_samples, params.learning_rate.to_bits()]
    }

    fn get_wgsl_code(&self, matrix: &GpuMatrix, params: &GpuParams) -> String {

        let mut code = r#"
@group(0) @binding(0)
var<storage, read> input: array<f32>;
@group(0) @binding(1)
var<storage, read> targets: array<f32>;        
@group(0) @binding(2)
var<storage, read_write> weights: array<f32>;
@group(0) @binding(3)
var<storage, read_write> output: array<f32>;
@group(0) @binding(4)
var<uniform> params: Params;

struct Params {
    step: u32,           // 0=predictions, 1=gradients, 2=update
    num_features: u32,
    num_samples: u32,
    learning_rate: f32,
}

// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(~workgroup_size~)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    // Get index
    let idx = global_id.x;
    if (idx >= params.num_features) { return; }

    switch params.step {
        case 0u: {

            // Predictions
            var prediction = 0.0;
            for (var j = 0u; j < params.num_features; j++) {
                let x_val = input[idx * params.num_features + j];
                let w_val = weights[j];
                prediction += x_val * w_val;
            }

            // Apply sigmoid and store prediction
            output[idx] = sigmoid(prediction);
        }
        case 1u: {

            // Gradient calculation
            var gradient = 0.0;
            for (var i = 0u; i < params.num_samples; i++) {
                let x_val = input[i * params.num_features + idx];
                let y_pred = output[i];  // predictions from step 0
                let y_true = targets[i];
                let error = y_pred - y_true;
                gradient += x_val * error;
            }

            // Average the gradient
            gradient = gradient / f32(params.num_samples);
            output[idx] = gradient;
        }
        case 2u: {

            // Update weights
            let gradient = output[idx];  // gradients from step 1
            weights[idx] = weights[idx] - params.learning_rate * gradient;
        }
        default: {}
        }
    }
        "#;

        // Replace variables
        let workgroup_size = format!("{}", matrix.get_workgroup_size());
        //code = code.replace("~workgroup_size~", &workgroup_size);
        code.to_string()
    }

}

impl GpuModule for MultiClassifier {
    fn get_params(&self, matrix: &GpuMatrix, num_classes: usize) -> Result<GpuParams, GpuError> {

        if num_classes < 3 {
            return Err(GpuError::Generic("Trying to run multi classification with less than three classes.  Uh oh, something went wrong somewhere!".to_string()));
        }

        Ok(GpuParams::new(GpuAlgorithm::LogisticRegressionGradientDescentMultiClassification, GpuLayout::Supervised, &matrix, num_classes as u32, 3, "main", self.learning_rate))
    }

    fn get_params_buffer_data(&self, params: &GpuParams) -> Vec<u32> {
        vec![params.step, params.num_features, params.num_samples, params.num_classes, params.learning_rate.to_bits()]
    }

    fn get_wgsl_code(&self, matrix: &GpuMatrix, params: &GpuParams) -> String {

        let mut code = format!(r#"
    @group(0) @binding(0)
    var<storage, read> input: array<f32>;           // X matrix (flattened)
    @group(0) @binding(1)
    var<storage, read> targets: array<f32>;         // y vector (target values)
    @group(0) @binding(2)
    var<storage, read_write> weights: array<f32>;   // w matrix (weights, size: num_classes * num_features)
    @group(0) @binding(3)
    var<storage, read_write> output: array<f32>;    // temporary storage for predictions/gradients
    @group(0) @binding(4)
    var<uniform> params: Params;

    struct Params {{
        step: u32,           // 0=predictions, 1=gradients, 2=update
        num_features: u32,   // number of columns in X
        num_samples: u32,    // number of rows in X
        learning_rate: f32,  // alpha for weight updates
        num_classes: u32,    // number of classes
    }}

    // Softmax activation function for multi-class
    fn softmax(values: ptr<function, array<f32, {}>>, length: u32, idx: u32) -> f32 {{
        // Find max value for numerical stability
        var max_val: f32 = -3.402823e+38;
        for (var i = 0u; i < length; i++) {{
            if (*values)[i] > max_val) {{
                max_val = (*values)[i];
            }}
        }}
        
        // Compute sum of exponentials
        var sum: f32 = 0.0;
        for (var i = 0u; i < length; i++) {{
            sum += exp((*values)[i] - max_val);
        }}
        
        // Return softmax for the specified index
        return exp((*values)[idx] - max_val) / sum;
    }}

    @compute @workgroup_size(~workgroup_size~)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
        let idx = global_id.x;
        
        switch params.step {{
            case 0u: {{
                // Step 0: Calculate predictions
                let sample_idx = idx;
                if (sample_idx >= params.num_samples) {{ return; }}
                
                // Local array for class scores
                var class_scores: array<f32, {}>;
                
                // Calculate score for each class
                for (var c = 0u; c < params.num_classes; c++) {{
                    var score = 0.0;
                    for (var j = 0u; j < params.num_features; j++) {{
                        let x_val = input[sample_idx * params.num_features + j];
                        let w_val = weights[c * params.num_features + j];  // Simple indexing!
                        score += x_val * w_val;
                    }}
                    class_scores[c] = score;
                }}
                
                // Apply softmax and store predictions
                for (var c = 0u; c < params.num_classes; c++) {{
                    let prob = softmax(&class_scores, params.num_classes, c);
                    output[sample_idx * params.num_classes + c] = prob;
                }}
            }}
            case 1u: {{
                // Step 1: Calculate gradients
                let weight_idx = idx;
                if (weight_idx >= params.num_classes * params.num_features) {{ return; }}
                
                let class_idx = weight_idx / params.num_features;
                let feature_idx = weight_idx % params.num_features;
                
                var gradient = 0.0;
                for (var i = 0u; i < params.num_samples; i++) {{
                    let x_val = input[i * params.num_features + feature_idx];
                    let y_pred = output[i * params.num_classes + class_idx];  // predictions from step 0
                    
                    // One-hot encoding: target is 1.0 for true class, 0.0 for others
                    var y_true = 0.0;
                    if (targets[i] == f32(class_idx)) {{
                        y_true = 1.0;
                    }}
                    
                    let error = y_pred - y_true;
                    gradient += x_val * error;
                }}
                
                // Average the gradient
                gradient = gradient / f32(params.num_samples);
                output[weight_idx] = gradient;
            }}
            case 2u: {{
                // Step 2: Update weights
                let weight_idx = idx;
                if (weight_idx >= params.num_classes * params.num_features) {{ return; }}
                
                let gradient = output[weight_idx];  // gradients from step 1
                weights[weight_idx] = weights[weight_idx] - params.learning_rate * gradient;
            }}
            default: {{}}
        }}
    }}
            "#, params.num_classes, params.num_classes);
        
        // Replace variables
        let workgroup_size = format!("{}", matrix.get_workgroup_size());
        code = code.replace("~workgroup_size~", &workgroup_size);
        code.to_string()
    }

}


impl TryFrom<(&LogisticRegressionParameters<f32>, usize)> for Box<dyn GpuModule> {
    type Error = GpuError;

    fn try_from(value: (&LogisticRegressionParameters<f32>, usize)) -> Result<Self, Self::Error> {
        let (params, num_classes) = value;
        if params.solver == LogisticRegressionSolverName::GradientDescent {
            let learning_rate = params.alpha as f32;

            if num_classes > 2 {
                return Ok(Box::new(MultiClassifier { is_f32: true, learning_rate }));
            } else {
                return Ok(Box::new(BinaryClassifier { is_f32: true, learning_rate }));
            }
        }

        Err(GpuError::WorkerConversion)
    }
}


