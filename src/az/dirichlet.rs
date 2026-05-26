//! Symmetric Dirichlet sampling for AZ root noise.
//!
//! Used by self-play to inject randomness into the root priors so that
//! the search explores moves the policy network already considers weak.
//! Standard AZ recipe: at the root only, replace each prior `p_i` with
//! `(1 - eps) * p_i + eps * d_i` where `d ~ Dir(α, α, ..., α)`.
//!
//! For our typical α ≈ 0.2–0.3 (10 / branching), the shape parameter is
//! < 1, so Marsaglia-Tsang for Gamma(α, 1) is combined with the
//! standard "boost" transform — sample Gamma(α+1) and multiply by
//! U^(1/α). All RNG draws go through [`RandomPlayer`] so the stream is
//! reproducible with a fixed seed.

use std::f64::consts::PI;

use crate::agent::RandomPlayer;

#[inline]
fn next_unit(rng: &mut RandomPlayer) -> f64 {
    // Splitmix64 output → [0, 1). Avoid exact 0 to keep ln() finite below.
    ((rng.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)).max(1e-300)
        + 1.0e-300
}

#[inline]
fn standard_normal(rng: &mut RandomPlayer) -> f64 {
    // Box-Muller. Two uniforms in (0, 1); we only need one normal per call.
    let u1 = next_unit(rng);
    let u2 = next_unit(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Sample one Gamma(`alpha`, 1). Handles `alpha < 1` via the boost
/// transform; uses Marsaglia-Tsang for `alpha ≥ 1`.
pub fn gamma(alpha: f64, rng: &mut RandomPlayer) -> f64 {
    if alpha < 1.0 {
        // Boost: Gamma(α) ≡ Gamma(α+1) · U^(1/α).
        let g = gamma_marsaglia_tsang(alpha + 1.0, rng);
        let u = next_unit(rng);
        g * u.powf(1.0 / alpha)
    } else {
        gamma_marsaglia_tsang(alpha, rng)
    }
}

fn gamma_marsaglia_tsang(alpha: f64, rng: &mut RandomPlayer) -> f64 {
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = standard_normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = next_unit(rng);
        // Squeeze step (avoids one log) — not strictly necessary, but
        // skips ~95 % of the expensive comparisons.
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Sample one symmetric Dirichlet(α) of length `k`. Returns a vector
/// of `f32`s summing to ~1 (subject to f32 rounding).
pub fn symmetric_dirichlet(alpha: f64, k: usize, rng: &mut RandomPlayer) -> Vec<f32> {
    if k == 0 {
        return Vec::new();
    }
    let mut g: Vec<f64> = (0..k).map(|_| gamma(alpha, rng)).collect();
    let sum: f64 = g.iter().sum();
    if sum > 0.0 {
        for v in g.iter_mut() {
            *v /= sum;
        }
    } else {
        // Degenerate (all gammas underflowed to 0 — extremely unlikely
        // for α > 0). Fall back to uniform.
        let u = 1.0 / k as f64;
        for v in g.iter_mut() {
            *v = u;
        }
    }
    g.into_iter().map(|x| x as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirichlet_samples_sum_to_one_and_are_non_negative() {
        let mut rng = RandomPlayer::new(31415);
        for _ in 0..20 {
            for alpha in [0.1, 0.3, 1.0, 3.0] {
                let k = 50;
                let d = symmetric_dirichlet(alpha, k, &mut rng);
                assert_eq!(d.len(), k);
                let s: f32 = d.iter().sum();
                assert!(
                    (s - 1.0).abs() < 1e-3,
                    "dirichlet({alpha}, {k}) sum = {s}, expected ~1"
                );
                for &v in &d {
                    assert!(v >= 0.0, "dirichlet entry negative: {v}");
                }
            }
        }
    }

    #[test]
    fn gamma_sample_is_positive() {
        let mut rng = RandomPlayer::new(7);
        for alpha in [0.2_f64, 0.5, 1.0, 2.0, 5.0] {
            for _ in 0..50 {
                let g = gamma(alpha, &mut rng);
                assert!(g.is_finite() && g > 0.0, "Gamma({alpha}) returned {g}");
            }
        }
    }
}
