//! Perft (performance test): count leaf nodes of the game tree at depth N.
//!
//! Used to validate move generation. Two engines that share the same rules
//! must agree on every perft number. Also a useful benchmark.
//!
//! A "node" is a board position; perft(0) = 1 (the current position counts
//! as one leaf at depth 0). Terminal positions short-circuit: once a side
//! has won or the game has drawn, no further plies are explored from that
//! line, so a terminal node at depth k still contributes 1 to perft(k).

use crate::board::Board;
use crate::moves::Move;

/// Count leaf nodes at exactly `depth` plies from `board`.
pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if board.outcome().is_some() {
        // Game over: the position itself is the only leaf in this subtree.
        return 1;
    }
    let moves = board.legal_moves();
    if moves.is_empty() {
        return 1;
    }
    let mut total = 0;
    for mv in moves {
        let mut child = board.clone();
        // legal_moves only returns moves apply() will accept, so unwrap is safe.
        child.apply(mv).expect("legal move must apply");
        total += perft(&child, depth - 1);
    }
    total
}

/// Like `perft`, but breaks the total down by top-level move.
pub fn perft_divide(board: &Board, depth: u32) -> Vec<(Move, u64)> {
    assert!(depth >= 1, "divide requires depth >= 1");
    let mut out = Vec::new();
    for mv in board.legal_moves() {
        let mut child = board.clone();
        child.apply(mv).expect("legal move must apply");
        out.push((mv, perft(&child, depth - 1)));
    }
    out
}
