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

## Developers
Contributions welcome, please start from [CONTRIBUTING and other relevant files](.github/CONTRIBUTING.md).

### Basics

#### numbers
The library is founded on basic traits provided by `num-traits`. Basic traits are in `src/numbers`. These traits are used to define all the procedures in the library to make everything safer and provide constraints to what implementations can handle.

#### linalg
`numbers` are made at use in linear algebra structures in the **`src/linalg/basic`** module. These sub-modules define the traits used all over the code base. 

* *arrays*: In particular data structures like `Array`, `Array1` (1-dimensional), `Array2` (matrix, 2-D); plus their "views" traits. Views are used to provide no-footprint access to data, they have composed traits to allow writing (mutable traits: `MutArray`, `ArrayViewMut`, ...).
* *matrix*: This provides the main entrypoint to matrices operations and currently the only structure provided in the shape of `struct DenseMatrix`. A matrix can be instantiated and automatically make available all the traits in "arrays" (sparse matrices implementation will be provided).
* *vector*: Convenience traits are implemented for `std::Vec` to allow extensive reuse.

#### linalg/traits
The traits in `src/linalg/traits` are closely linked to Linear Algebra's theoretical framework. These traits are used to specify characteristics and constraints for types accepted by various algorithms. For example these allow to define if a matrix is `QRDecomposable` and/or `SVDDecomposable`. See docstring for referencese to theoretical framework.

#### metrics
Implementations for metrics (classification, regression, cluster, ...) and distance measure (Euclidean, Hamming, Manhattan, ...). For example: `Accuracy`, `F1`, `AUC`, `Precision`, `R2`. As everything else in the code base, these implementations reuse `numbers` and `linalg` traits and structures.


TODO: complete for all modules
