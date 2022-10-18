<p align="center">
  <a href="https://smartcorelib.org">
    <img src="smartcore.svg" width="450" alt="SmartCore">    
  </a>  
</p>
<p align = "center">
    <strong>
        <a href="https://smartcorelib.org">User guide</a> | <a href="https://docs.rs/smartcore/">API</a> | <a href="https://github.com/smartcorelib/smartcore-examples">Examples</a>
    </strong>
</p>

-----

<p align = "center">
<b>The Most Advanced Machine Learning Library In Rust.</b>
</p>

-----

## Current status
* Current working branch is `development` (if you want something that you can test right away).
* Breaking changes are undergoing development at [`v0.5-wip`](https://github.com/smartcorelib/smartcore/tree/v0.5-wip#readme) (if you are a newcomer better to start from [this README](https://github.com/smartcorelib/smartcore/tree/v0.5-wip#readme) as this will be the next major release).

To start getting familiar with Smartcore v0.5, there is now available a [**Jupyter Notebook environment repository**](https://github.com/smartcorelib/smartcore-jupyter).

## Developers
Contributions welcome, please start from [CONTRIBUTING and other relevant files](.github/CONTRIBUTING.md).

### Walkthrough: traits system and basic structures

#### numbers
The library is founded on basic traits provided by `num-traits`. Basic traits are in `src/numbers`. These traits are used to define all the procedures in the library to make everything safer and provide constraints to what implementations can handle.

#### linalg
`numbers` are made at use in linear algebra structures in the **`src/linalg/basic`** module. These sub-modules define the traits used all over the code base. 

* *arrays*: In particular data structures like `Array`, `Array1` (1-dimensional), `Array2` (matrix, 2-D); plus their "views" traits. Views are used to provide no-footprint access to data, they have composed traits to allow writing (mutable traits: `MutArray`, `ArrayViewMut`, ...).
* *matrix*: This provides the main entrypoint to matrices operations and currently the only structure provided in the shape of `struct DenseMatrix`. A matrix can be instantiated and automatically make available all the traits in "arrays" (sparse matrices implementation will be provided).
* *vector*: Convenience traits are implemented for `std::Vec` to allow extensive reuse.

These are all traits and by definition they do not allow instantiation. For instantiable structures see implementation like `DenseMatrix` with relative constructor.

#### linalg/traits
The traits in `src/linalg/traits` are closely linked to Linear Algebra's theoretical framework. These traits are used to specify characteristics and constraints for types accepted by various algorithms. For example these allow to define if a matrix is `QRDecomposable` and/or `SVDDecomposable`. See docstring for referencese to theoretical framework.

As above these are all traits and by definition they do not allow instantiation. They are mostly used to provide constraints for implementations. For example, the implementation for Linear Regression requires the input data `X` to be in `smartcore`'s trait system `Array2<FloatNumber> + QRDecomposable<TX> + SVDDecomposable<TX>`, a 2-D matrix that is both QR and SVD decomposable; that is what the provided strucure `linalg::arrays::matrix::DenseMatrix` happens to be: `impl<T: FloatNumber> QRDecomposable<T> for DenseMatrix<T> {};impl<T: FloatNumber> SVDDecomposable<T> for DenseMatrix<T> {}`. 

#### metrics
Implementations for metrics (classification, regression, cluster, ...) and distance measure (Euclidean, Hamming, Manhattan, ...). For example: `Accuracy`, `F1`, `AUC`, `Precision`, `R2`. As everything else in the code base, these implementations reuse `numbers` and `linalg` traits and structures.

These are collected in structures like `pub struct ClassificationMetrics<T> {}` that implements `metrics::Metrics`, these are groups of functions (classification, regression, cluster, ...) that provide instantiation for the structures. Each of those instantiation can be passed around using the relative function, like `pub fn accuracy<T: Number + RealNumber + FloatNumber, V: ArrayView1<T>>(y_true: &V, y_pred: &V) -> T`. This provides a mechanism for metrics to be passed to higher interfaces like the `cross_validate`:
```rust
let results =
  cross_validate(
      BiasedEstimator::fit,  // custom estimator
      &x, &y,                // input data
      NoParameters {},       // extra parameters
      cv,                    // type of cross validator
      &accuracy              // **metrics function** <--------
  ).unwrap();
```


TODO: complete for all modules

