#[cfg(not(feature = "std"))]
pub(crate) use rand::rngs::SmallRng as RngImpl;
#[cfg(not(feature = "std"))]
use getrandom;
#[cfg(feature = "std")]
pub(crate) use rand::rngs::StdRng as RngImpl;
use rand::SeedableRng;

pub(crate) fn get_rng_impl(seed: Option<u64>) -> RngImpl {
    match seed {
        Some(seed) => RngImpl::seed_from_u64(seed),
        None => {
            cfg_if::cfg_if! {
                if #[cfg(feature = "std")] {
                    use rand::RngCore;
                    RngImpl::seed_from_u64(rand::thread_rng().next_u64())
                } else {
                    // non-std build, use getrandom
                    let mut buf = [0u8; 64];
                    getrandom::getrandom(&mut buf).unwrap();
                    RngImpl::seed_from_u64(buf[0] as u64)
                }
            }
        }
    }
}
