use ::rand::SeedableRng;
#[cfg(not(feature = "std"))]
pub(crate) use rand::rngs::SmallRng as RngImpl;
#[cfg(feature = "std")]
pub(crate) use rand::rngs::StdRng as RngImpl;

pub(crate) fn get_rng_impl(seed: Option<u64>) -> RngImpl {
    match seed {
        Some(seed) => RngImpl::seed_from_u64(seed),
        None => {
            cfg_if::cfg_if! {
                if #[cfg(feature = "std")] {
                    use rand::RngCore;
                    RngImpl::seed_from_u64(rand::thread_rng().next_u64())
                } else {
                    panic!("seed number needed for non-std build");
                }
            }
        }
    }
}
