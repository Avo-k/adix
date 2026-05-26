//! Perft (performance test): count leaf nodes of the game tree at depth N.
//!
//! Used to validate move generation. Two engines that share the same rules
//! must agree on every perft number. Also a useful benchmark.
//!
//! A "node" is a board position; perft(0) = 1 (the current position counts
//! as one leaf at depth 0). Terminal positions short-circuit: once a side
//! has won or the game has drawn, no further plies are explored from that
//! line, so a terminal node at depth k still contributes 1 to perft(k).

use std::collections::HashSet;

use crate::board::Board;
use crate::moves::Move;

/// Count leaf nodes at exactly `depth` plies from `board`.
pub fn perft(board: &Board, depth: u32) -> u64 {
    let mut b = board.clone();
    perft_in_place(&mut b, depth)
}

fn perft_in_place(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if board.outcome().is_some() {
        return 1;
    }
    let mut moves: Vec<Move> = Vec::with_capacity(64);
    board.legal_moves_into(&mut moves);
    if moves.is_empty() {
        return 1;
    }
    // Bulk-count: at depth 1, every legal move is a leaf. Skip apply.
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut total = 0;
    for mv in &moves {
        let undo = board.apply_legal(*mv);
        total += perft_in_place(board, depth - 1);
        board.unmake(*mv, undo);
    }
    total
}

/// Search-representative perft: like `perft`, but does not bulk-count at
/// depth 1 — it always applies and unmakes every move down to depth 0.
/// Useful as a benchmark of `apply_legal`/`unmake`/`legal_moves_into` per node,
/// which is the work an actual alpha-beta search will be doing.
pub fn perft_search(board: &Board, depth: u32) -> u64 {
    let mut b = board.clone();
    perft_search_in_place(&mut b, depth)
}

fn perft_search_in_place(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if board.outcome().is_some() {
        return 1;
    }
    let mut moves: Vec<Move> = Vec::with_capacity(64);
    board.legal_moves_into(&mut moves);
    if moves.is_empty() {
        return 1;
    }
    let mut total = 0;
    for mv in &moves {
        let undo = board.apply_legal(*mv);
        total += perft_search_in_place(board, depth - 1);
        board.unmake(*mv, undo);
    }
    total
}

/// Transposition table entry for perft caching. Stores the full Zobrist
/// key so collisions are detected and rejected — a stale slot with a
/// different key simply misses.
#[derive(Clone, Copy, Default)]
struct TTEntry {
    key: u64,
    depth: u32,
    count: u64,
}

/// Fixed-size always-replace transposition table indexed by `key & mask`.
/// Sized to a power of two so the modulus is a single AND.
pub struct PerftTT {
    entries: Box<[TTEntry]>,
    mask: usize,
    pub probes: u64,
    pub hits: u64,
    pub stores: u64,
}

impl PerftTT {
    /// Allocate a TT with at least `requested` slots, rounded up to the
    /// next power of two. Each slot is 24 bytes.
    pub fn with_entries(requested: usize) -> Self {
        let n = requested.max(1).next_power_of_two();
        Self {
            entries: vec![TTEntry::default(); n].into_boxed_slice(),
            mask: n - 1,
            probes: 0,
            hits: 0,
            stores: 0,
        }
    }
    /// Allocate a TT with budget ~`mb` megabytes.
    pub fn with_mb(mb: usize) -> Self {
        let bytes = mb.max(1) * 1024 * 1024;
        Self::with_entries(bytes / std::mem::size_of::<TTEntry>())
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    fn probe(&mut self, key: u64, depth: u32) -> Option<u64> {
        self.probes += 1;
        let e = &self.entries[(key as usize) & self.mask];
        if e.key == key && e.depth == depth {
            self.hits += 1;
            Some(e.count)
        } else {
            None
        }
    }
    #[inline]
    fn store(&mut self, key: u64, depth: u32, count: u64) {
        self.stores += 1;
        self.entries[(key as usize) & self.mask] = TTEntry { key, depth, count };
    }

    pub fn hit_rate(&self) -> f64 {
        if self.probes == 0 { 0.0 } else { self.hits as f64 / self.probes as f64 }
    }
    pub fn reset_stats(&mut self) {
        self.probes = 0;
        self.hits = 0;
        self.stores = 0;
    }
}

/// Like `perft`, but caches subtree counts in a transposition table.
/// Useful as a benchmark of how much ADIX positions actually transpose.
pub fn perft_tt(board: &Board, depth: u32, tt: &mut PerftTT) -> u64 {
    let mut b = board.clone();
    perft_tt_in_place(&mut b, depth, tt)
}

fn perft_tt_in_place(board: &mut Board, depth: u32, tt: &mut PerftTT) -> u64 {
    if depth == 0 {
        return 1;
    }
    if board.outcome().is_some() {
        return 1;
    }
    if depth >= 2 && let Some(c) = tt.probe(board.zobrist, depth) {
        return c;
    }
    let mut moves: Vec<Move> = Vec::with_capacity(64);
    board.legal_moves_into(&mut moves);
    if moves.is_empty() {
        return 1;
    }
    if depth == 1 {
        return moves.len() as u64;
    }
    let key = board.zobrist;
    let mut total = 0;
    for mv in &moves {
        let undo = board.apply_legal(*mv);
        total += perft_tt_in_place(board, depth - 1, tt);
        board.unmake(*mv, undo);
    }
    tt.store(key, depth, total);
    total
}

/// Count of *distinct* positions reachable in exactly `depth` plies from
/// `board`, deduplicated by Zobrist hash. Distinct from `perft`, which
/// counts paths in the move tree and thus double-counts transpositions.
///
/// Exact: stores every reached hash in a `HashSet`. Memory grows with the
/// number of unique positions (~16 bytes/entry with HashSet overhead) —
/// fine through depth 5, prohibitive beyond. For larger depths see
/// [`unique_hll`].
pub fn unique_exact(board: &Board, depth: u32) -> u64 {
    let mut b = board.clone();
    let mut set: HashSet<u64> = HashSet::new();
    unique_exact_in_place(&mut b, depth, &mut set);
    set.len() as u64
}

fn unique_exact_in_place(board: &mut Board, depth: u32, set: &mut HashSet<u64>) {
    if depth == 0 || board.outcome().is_some() {
        set.insert(board.zobrist);
        return;
    }
    let mut moves: Vec<Move> = Vec::with_capacity(64);
    board.legal_moves_into(&mut moves);
    if moves.is_empty() {
        set.insert(board.zobrist);
        return;
    }
    for mv in &moves {
        let undo = board.apply_legal(*mv);
        unique_exact_in_place(board, depth - 1, set);
        board.unmake(*mv, undo);
    }
}

/// HyperLogLog cardinality estimator (m = 2^14 registers, ~0.8% standard
/// error). Constant 16 KB memory. Used by [`unique_hll`] for depths where
/// the exact `HashSet` won't fit.
pub struct Hll14 {
    registers: [u8; Self::M],
}

impl Hll14 {
    const M: usize = 16384;
    const LOG_M: u32 = 14;

    pub fn new() -> Self {
        Self { registers: [0; Self::M] }
    }

    #[inline]
    pub fn add(&mut self, hash: u64) {
        let idx = (hash >> (64 - Self::LOG_M)) as usize;
        // Bound rho by appending a sentinel 1-bit: the remaining 50 bits
        // become a 64-bit word whose leading-zeros count is at most 50.
        let w = (hash << Self::LOG_M) | (1u64 << (Self::LOG_M - 1));
        let rho = w.leading_zeros() as u8 + 1;
        let r = &mut self.registers[idx];
        if rho > *r {
            *r = rho;
        }
    }

    pub fn estimate(&self) -> u64 {
        let m = Self::M as f64;
        let mut sum = 0.0;
        let mut zeros = 0u64;
        for &r in &self.registers {
            sum += 2f64.powi(-(r as i32));
            if r == 0 {
                zeros += 1;
            }
        }
        let alpha = 0.7213 / (1.0 + 1.079 / m);
        let raw = alpha * m * m / sum;
        // Linear-counting correction in the small-cardinality regime.
        if raw <= 2.5 * m && zeros > 0 {
            return (m * (m / zeros as f64).ln()) as u64;
        }
        raw as u64
    }
}

impl Default for Hll14 {
    fn default() -> Self {
        Self::new()
    }
}

/// Approximate count of distinct positions reachable in exactly `depth`
/// plies, via HyperLogLog over Zobrist hashes (~0.8% expected error,
/// constant 16 KB memory). Same semantics as [`unique_exact`].
pub fn unique_hll(board: &Board, depth: u32) -> u64 {
    let mut b = board.clone();
    let mut hll = Hll14::new();
    unique_hll_in_place(&mut b, depth, &mut hll);
    hll.estimate()
}

fn unique_hll_in_place(board: &mut Board, depth: u32, hll: &mut Hll14) {
    if depth == 0 || board.outcome().is_some() {
        hll.add(board.zobrist);
        return;
    }
    let mut moves: Vec<Move> = Vec::with_capacity(64);
    board.legal_moves_into(&mut moves);
    if moves.is_empty() {
        hll.add(board.zobrist);
        return;
    }
    for mv in &moves {
        let undo = board.apply_legal(*mv);
        unique_hll_in_place(board, depth - 1, hll);
        board.unmake(*mv, undo);
    }
}

/// Like `perft`, but breaks the total down by top-level move.
pub fn perft_divide(board: &Board, depth: u32) -> Vec<(Move, u64)> {
    assert!(depth >= 1, "divide requires depth >= 1");
    let mut b = board.clone();
    let moves = b.legal_moves();
    let mut out = Vec::with_capacity(moves.len());
    for mv in moves {
        let undo = b.apply_legal(mv);
        out.push((mv, perft_in_place(&mut b, depth - 1)));
        b.unmake(mv, undo);
    }
    out
}
