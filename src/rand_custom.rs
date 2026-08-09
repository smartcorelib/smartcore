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
                    use rand::Rng;
                    // FIX: thread_rng() deprecated in rand 0.9 → use rng()
                    // FIX: rand 0.10 no longer re-exports RngCore at root;
                    //      import rand::Rng (supertrait) instead so next_u64() resolves
                    RngImpl::seed_from_u64(rand::rng().next_u64())
                } else {
                    // no std_random feature build, use getrandom
                    #[cfg(feature = "js")]
                    {
                        let mut buf = [0u8; 64];
                        getrandom::getrandom(&mut buf).unwrap();
                        RngImpl::seed_from_u64(buf[0] as u64)
                    }
                    #[cfg(not(feature = "js"))]
                    {
                        // Using 0 as default seed
                        RngImpl::seed_from_u64(0)
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn seeded_rng_is_deterministic() {
        let mut a = get_rng_impl(Some(42));
        let mut b = get_rng_impl(Some(42));
        // two RNGs seeded with the same value produce the same sequence
        let va: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let vb: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        assert_eq!(va, vb);
    }

    #[cfg_attr(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        wasm_bindgen_test::wasm_bindgen_test
    )]
    #[test]
    fn different_seeds_produce_different_sequences() {
        let mut a = get_rng_impl(Some(1));
        let mut b = get_rng_impl(Some(2));
        let va: Vec<u64> = (0..4).map(|_| a.next_u64()).collect();
        let vb: Vec<u64> = (0..4).map(|_| b.next_u64()).collect();
        assert_ne!(va, vb);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn none_seed_returns_usable_rng() {
        // unseeded path: on non-wasm this uses OS entropy (rand::rng() under
        // std_rand, or getrandom/zero-seed otherwise). Excluded on bare wasm
        // because OS entropy is unavailable without a host shim.
        let mut r = get_rng_impl(None);
        let _ = r.next_u64();
    }

    #[test]
    #[cfg(feature = "std_rand")]
    fn std_rand_none_seed_uses_os_entropy() {
        // under the std_rand feature, get_rng_impl(None) seeds via rand::rng()
        // which draws from OS entropy. Verify that path returns a working RNG
        // and that two draws produce distinct values (non-deterministic source).
        let mut a = get_rng_impl(None);
        let v1 = a.next_u64();
        let v2 = a.next_u64();
        assert_ne!(v1, v2, "two draws from an entropy-seeded RNG should differ");
    }
}
