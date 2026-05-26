use adix::board::Board;
use adix::perft::{PerftTT, Hll14, perft, perft_search, perft_tt, unique_exact, unique_hll};

/// Numbers locked in at engine v0.1.0. If you change move generation and one
/// of these breaks, *that's the bug* — re-derive the counts from the rules
/// before updating, don't just patch the expected number.
const EXPECTED: &[(u32, u64)] = &[
    (0, 1),
    (1, 42),
    (2, 1764),
    (3, 82_110),
];

#[test]
fn perft_initial_position() {
    let board = Board::initial();
    for &(depth, expected) in EXPECTED {
        let got = perft(&board, depth);
        assert_eq!(got, expected, "perft({}) — expected {}, got {}", depth, expected, got);
    }
}

/// `perft_search` (no bulk count) must agree with the locked numbers too.
#[test]
fn perft_search_matches() {
    let board = Board::initial();
    for &(depth, expected) in EXPECTED {
        let got = perft_search(&board, depth);
        assert_eq!(got, expected, "perft_search({depth}) — expected {expected}, got {got}");
    }
}

/// `perft_tt` must agree with the locked numbers. Run with a small TT so
/// store collisions actually exercise the key check.
#[test]
fn perft_tt_matches() {
    let board = Board::initial();
    let mut tt = PerftTT::with_entries(1024);
    for &(depth, expected) in EXPECTED {
        tt.reset_stats();
        let got = perft_tt(&board, depth, &mut tt);
        assert_eq!(got, expected, "perft_tt({depth}) — expected {expected}, got {got}");
    }
}

/// Number of *distinct* positions (deduped by Zobrist) reachable in
/// exactly `n` plies. Always ≤ `perft(n)`; the gap is the count of
/// transpositions in the move tree at that depth.
const UNIQUE_EXPECTED: &[(u32, u64)] = &[
    (0, 1),
    (1, 41),
    (2, 1681),
    (3, 50223),
    (4, 1_459_274),
];

#[test]
fn unique_exact_matches() {
    let board = Board::initial();
    for &(depth, expected) in UNIQUE_EXPECTED {
        let got = unique_exact(&board, depth);
        assert_eq!(got, expected, "unique_exact({depth}) — expected {expected}, got {got}");
    }
}

/// HLL must stay within a few percent of the exact answer at the depths we
/// can cross-check. With m = 16384 the standard error is ~0.8 %; allow 2 %
/// to keep the test stable across reruns.
#[test]
fn unique_hll_close_to_exact() {
    let board = Board::initial();
    for &(depth, expected) in UNIQUE_EXPECTED {
        if expected < 100 {
            continue; // HLL is meaningless on tiny cardinalities
        }
        let got = unique_hll(&board, depth);
        let rel_err = (got as f64 - expected as f64).abs() / expected as f64;
        assert!(
            rel_err < 0.02,
            "unique_hll({depth}) — expected ~{expected}, got {got} (rel err {:.3})",
            rel_err
        );
    }
}

/// Sanity check on the HLL itself, independent of perft.
#[test]
fn hll_estimates_dense_set() {
    let mut hll = Hll14::new();
    let n = 100_000u64;
    for i in 0..n {
        // splitmix64 mixing so values are well-distributed.
        let mut x = i.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        hll.add(x);
    }
    let est = hll.estimate();
    let rel_err = (est as f64 - n as f64).abs() / n as f64;
    assert!(rel_err < 0.02, "HLL est {est} too far from {n} (rel err {:.3})", rel_err);
}
