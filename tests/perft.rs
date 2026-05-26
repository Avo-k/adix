use adix::board::Board;
use adix::perft::perft;

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
