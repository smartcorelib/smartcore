//! Compare complete fits with and without the `lazy-normalization` feature.
//!
//! Run the same arguments in separate processes to compare peak resident memory:
//! `cargo run --release --example lazy_normalization -- lasso 8192 32 5`
//! `cargo run --release --example lazy_normalization --features lazy-normalization -- lasso 8192 32 5`
//! Replace `lasso` with `elastic-net` for the augmented design comparison.
//! Input construction is excluded from the reported fit time. Process memory
//! measurements include the input and the fitting workspace.
//! For peak memory, build first, then run the executable directly under GNU
//! `time -v`; measuring `cargo run` also includes Cargo's memory use.

use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::elastic_net::{ElasticNet, ElasticNetParameters};
use smartcore::linear::lasso::{Lasso, LassoParameters};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model = args.get(1).map_or("lasso", String::as_str);
    let n = args.get(2).map_or(Ok(8192), |v| v.parse::<usize>())?;
    let p = args.get(3).map_or(Ok(32), |v| v.parse::<usize>())?;
    let repeats = args.get(4).map_or(Ok(5), |v| v.parse::<usize>())?;
    if n < p || p == 0 || repeats == 0 {
        return Err("Require n >= p > 0 and repeats > 0".into());
    }
    let mut state = 42_u64;
    let mut values = Vec::with_capacity(n * p);
    let mut y = vec![3.0; n];
    for j in 0..p {
        for response in &mut y {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let value = (state >> 11) as f64 / (1_u64 << 53) as f64 * 4.0 + j as f64;
            values.push(value);
            if j < 4 {
                *response += value * (j as f64 + 1.0) / 4.0;
            }
        }
    }
    let x = DenseMatrix::new(n, p, values, true)?;
    let lasso_parameters = LassoParameters::default().with_alpha(0.05);
    let elastic_net_parameters = ElasticNetParameters::default().with_alpha(0.05);
    #[cfg(feature = "lazy-normalization")]
    let lasso_parameters = lasso_parameters.with_lazy_normalization(true);
    #[cfg(feature = "lazy-normalization")]
    let elastic_net_parameters = elastic_net_parameters.with_lazy_normalization(true);
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..repeats {
        let coefficients = match model {
            "lasso" => Lasso::fit(black_box(&x), black_box(&y), lasso_parameters.clone())?
                .coefficients()
                .clone(),
            "elastic-net" => {
                ElasticNet::fit(black_box(&x), black_box(&y), elastic_net_parameters.clone())?
                    .coefficients()
                    .clone()
            }
            _ => return Err("Model must be lasso or elastic-net".into()),
        };
        checksum += black_box(coefficients.iterator(0).copied().sum::<f64>());
    }
    let elapsed = start.elapsed().as_secs_f64() / repeats as f64;
    let mode = if cfg!(feature = "lazy-normalization") {
        "lazy"
    } else {
        "eager"
    };
    println!("{mode},{model},{n},{p},{repeats},{elapsed:.6},{checksum:.12}");
    Ok(())
}
