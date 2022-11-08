#[cfg(not(feature = "std_rand"))]
pub use rand::rngs::SmallRng as RngImpl;
#[cfg(feature = "std_rand")]
pub use rand::rngs::StdRng as RngImpl;
use rand::SeedableRng;

/// Custom switch for random fuctions
pub fn get_rng_impl(seed: Option<u64>) -> RngImpl {
    match seed {
        Some(seed) => RngImpl::seed_from_u64(seed),
        None => {
            cfg_if::cfg_if! {
                if #[cfg(feature = "std_rand")] {
                    use rand::RngCore;
                    RngImpl::seed_from_u64(rand::thread_rng().next_u64())
                } else {
                    // no std_random feature build, use getrandom
                    let mut buf = [0u8; 64];
                    getrandom::getrandom(&mut buf).unwrap();
                    RngImpl::seed_from_u64(buf[0] as u64)
                }
            }
        }
    }
}
