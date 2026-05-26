use adix::board::Board;

/// The incremental Zobrist must match a recompute from scratch on the
/// initial position. Catches the empty-board key / starting layout.
#[test]
fn initial_zobrist_matches_scratch() {
    let b = Board::initial();
    assert_eq!(b.zobrist, b.zobrist_from_scratch());
}

/// `apply_legal` then `unmake` must restore the Zobrist exactly.
#[test]
fn apply_unmake_restores_zobrist() {
    let mut b = Board::initial();
    let z0 = b.zobrist;
    for mv in b.legal_moves() {
        let undo = b.apply_legal(mv);
        // Mid-move zobrist must equal a from-scratch recompute too.
        assert_eq!(
            b.zobrist,
            b.zobrist_from_scratch(),
            "mid-move zobrist drift after {mv:?}"
        );
        b.unmake(mv, undo);
        assert_eq!(b.zobrist, z0, "round-trip failed after {mv:?}");
    }
}

/// Deeper: walk a small 3-ply tree and check round-trip + from-scratch.
#[test]
fn deep_apply_unmake_keeps_zobrist_consistent() {
    fn walk(b: &mut Board, depth: u32) {
        if depth == 0 {
            return;
        }
        assert_eq!(
            b.zobrist,
            b.zobrist_from_scratch(),
            "from-scratch drift at depth {depth}"
        );
        let z_before = b.zobrist;
        for mv in b.legal_moves() {
            let undo = b.apply_legal(mv);
            walk(b, depth - 1);
            b.unmake(mv, undo);
            assert_eq!(b.zobrist, z_before, "round-trip drift after {mv:?}");
        }
    }
    let mut b = Board::initial();
    walk(&mut b, 3);
}
