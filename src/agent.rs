//! Game-playing agents (a.k.a. *strategies* / players).
//!
//! Baseline only. Two players:
//! - [`RandomPlayer`] — picks a uniformly random legal move.
//! - [`AlphaBetaPlayer`] — fixed-depth negamax with a material-only
//!   evaluation. No move ordering, no quiescence, no TT, no iterative
//!   deepening. Deliberately minimal so future search work can be
//!   measured against it.
//!
//! Plus [`play_game`] to drive two players against each other.

use crate::board::{Board, Outcome};
use crate::eval::{self, full_eval};
use crate::moves::Move;
use crate::piece::{Color, Kind};
use crate::zobrist::splitmix64;

/// Anything that can pick a move given a board.
pub trait Player {
    /// Returns the chosen legal move, or `None` if the side to move has
    /// no legal moves (shouldn't happen in ADIX from a non-terminal
    /// position, but we don't enforce that).
    fn choose_move(&mut self, board: &Board) -> Option<Move>;
    fn name(&self) -> String;
}

// ---------------------------------------------------------------------------
// RandomPlayer
// ---------------------------------------------------------------------------

/// Uniform-random over legal moves. Seedable for reproducibility.
pub struct RandomPlayer {
    state: u64,
}

impl RandomPlayer {
    pub fn new(seed: u64) -> Self {
        // Splitmix is biased near 0 for trivial seeds, so step it once.
        Self { state: splitmix64(seed.wrapping_add(0xA5A5_5A5A_DEAD_BEEF)) }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = splitmix64(self.state);
        self.state
    }
}

impl Player for RandomPlayer {
    fn choose_move(&mut self, board: &Board) -> Option<Move> {
        let moves = board.legal_moves();
        if moves.is_empty() {
            return None;
        }
        let i = (self.next_u64() % moves.len() as u64) as usize;
        Some(moves[i])
    }
    fn name(&self) -> String {
        "random".to_string()
    }
}

// ---------------------------------------------------------------------------
// AlphaBetaPlayer
// ---------------------------------------------------------------------------

/// Mate score. Set well below `i32::MAX` so we can negate without overflow.
pub const MATE: i32 = 1_000_000;
const INF: i32 = MATE + 1;

/// Material-only evaluation from the side-to-move's perspective. Kept
/// for backwards-compat / quick comparisons; the alpha-beta engine
/// itself uses [`eval::full_eval`].
pub fn material_eval(board: &Board) -> i32 {
    let stm = board.side_to_move;
    eval::material(board, stm) - eval::material(board, stm.opp())
}

/// Search-TT entry. Bound flag tells us whether `score` is the true
/// minimax value (`Exact`), or just a bound usable to tighten α/β.
#[derive(Clone, Copy, Default)]
struct SearchTTEntry {
    /// Zobrist hash. `0` means "empty slot" (collisions with a real hash
    /// of 0 are astronomically unlikely; treat as a miss).
    key: u64,
    /// Search depth at which `score` was computed.
    depth: u16,
    bound: TTBound,
    score: i32,
    /// Best move from this position; tried first when the entry hits.
    best: Option<Move>,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
enum TTBound {
    #[default]
    Empty,
    Exact,
    Lower,
    Upper,
}

const SEARCH_TT_BITS: u32 = 17; // 2^17 = 131 072 slots × ~32 B ≈ 4 MB (fits in L3)
const SEARCH_TT_SIZE: usize = 1 << SEARCH_TT_BITS;
const SEARCH_TT_MASK: usize = SEARCH_TT_SIZE - 1;

/// Per-search transient state. Borrows the player's TT for the duration
/// of one `choose_move`. The TT outlives the context (it stays on the
/// player so consecutive turns can share entries).
struct SearchCtx<'a> {
    eval_fn: fn(&Board) -> i32,
    /// Total nodes visited by `negamax` (main search, including leaves).
    nodes: u64,
    /// Nodes visited inside `qsearch` (quiescence). Tracked separately so
    /// we can see how much extra work the capture-only extension is doing.
    qnodes: u64,
    /// Killer-move table indexed by `ply`. Two slots per ply: a quiet
    /// (non-capture) move that produced a β-cutoff is recorded, then
    /// tried first (after captures) at sibling positions at the same ply.
    killers: Vec<[Option<Move>; 2]>,
    /// Borrowed reference to the player's TT.
    tt: &'a mut [SearchTTEntry],
    /// TT-probe stats for the current search.
    tt_probes: u64,
    tt_hits: u64,
}

// ---------------------------------------------------------------------------
// Move ordering
// ---------------------------------------------------------------------------

/// Score `mv` so high-scoring moves are tried first. Reduces the size of
/// the α-β tree by maximising β-cutoffs near the front of each move list.
///
/// Order from high to low:
/// 1. **Captures** by MVV (most-valuable-victim): capturing a capitaine
///    scores ≫ capturing an equipier.
/// 2. **Killer moves** that recently produced cutoffs at this ply.
/// 3. **Everything else** (pivots, basculs, quiet deplacements).
#[inline]
fn score_move(board: &Board, mv: Move, killers: &[Option<Move>; 2]) -> i32 {
    if let Move::Deplacement { to, .. } = mv
        && let Some(target) = board.at(to)
    {
        // Capture. MVV: high bonus + victim value.
        let victim = match target.kind {
            Kind::Capitaine => 100_000,
            Kind::Equipier => 100,
        };
        return 1_000_000 + victim;
    }
    if killers[0] == Some(mv) {
        return 9_000;
    }
    if killers[1] == Some(mv) {
        return 8_000;
    }
    0
}

/// Sort `moves` in-place by `score_move` descending. Stable sort isn't
/// required; we use `sort_unstable_by_key` for speed.
#[inline]
fn order_moves(board: &Board, moves: &mut [Move], killers: &[Option<Move>; 2]) {
    moves.sort_unstable_by_key(|&m| std::cmp::Reverse(score_move(board, m, killers)));
}

// ---------------------------------------------------------------------------
// Quiescence search
// ---------------------------------------------------------------------------

/// Capture-only search invoked at the leaves of the main negamax search.
/// Resolves "horizon" tactics: at `depth==0` of the main search the
/// static eval can mis-rate a position with a pending capture, because
/// terms like `offensive_threats` and `mobility_differential` swing
/// wildly across a single capture. qsearch keeps applying captures until
/// neither side has one available, *then* falls back to the static eval.
///
/// Standard stand-pat framing: at every node the static eval is treated
/// as a lower bound (we always have the option *not* to capture). A
/// stand-pat ≥ β immediately cuts off; otherwise we use it to raise α
/// and try captures one by one in MVV order.
///
/// Termination: each ply removes at least one piece, and there are at
/// most 20 pieces on the board, so the recursion depth of qsearch is
/// bounded.
fn qsearch(
    board: &mut Board,
    mut alpha: i32,
    beta: i32,
    ply: u32,
    ctx: &mut SearchCtx,
) -> i32 {
    ctx.qnodes += 1;

    if let Some(o) = board.outcome() {
        return match o {
            Outcome::Win(winner) => {
                let side = board.side_to_move;
                if winner == side {
                    MATE - ply as i32
                } else {
                    -(MATE - ply as i32)
                }
            }
            Outcome::Draw => 0,
        };
    }

    // Stand-pat: if we just stop here, the eval is the score.
    let stand_pat = (ctx.eval_fn)(board);
    if stand_pat >= beta {
        return stand_pat;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    // Collect captures only — pivots and basculs are never captures, and
    // quiet deplacements onto an empty square don't change material.
    let mut all_moves: Vec<Move> = Vec::with_capacity(64);
    board.legal_moves_into(&mut all_moves);
    let mut captures: Vec<Move> = Vec::with_capacity(8);
    for mv in &all_moves {
        if let Move::Deplacement { to, .. } = mv
            && board.at(*to).is_some()
        {
            captures.push(*mv);
        }
    }
    if captures.is_empty() {
        return stand_pat;
    }
    // MVV ordering: try high-value captures first.
    captures.sort_unstable_by_key(|&m| std::cmp::Reverse(score_move(board, m, &[None, None])));

    let mut best = stand_pat;
    for mv in &captures {
        let undo = board.apply_legal(*mv);
        let score = -qsearch(board, -beta, -alpha, ply + 1, ctx);
        board.unmake(*mv, undo);
        if score > best {
            best = score;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break;
        }
    }
    best
}

/// Negamax with alpha-beta cutoffs, TT, killer moves, and MVV-LVA move
/// ordering. Evaluation is from the side-to-move's perspective; recursive
/// calls negate the score (negamax framing).
///
/// `ply` is the search distance from the root, used to make mates that
/// are closer to the root score higher than distant ones (so the engine
/// doesn't stall when a mate is available).
fn negamax(
    board: &mut Board,
    depth: u32,
    mut alpha: i32,
    mut beta: i32,
    ply: u32,
    ctx: &mut SearchCtx,
) -> i32 {
    ctx.nodes += 1;
    if let Some(o) = board.outcome() {
        return match o {
            Outcome::Win(winner) => {
                let side = board.side_to_move;
                if winner == side {
                    MATE - ply as i32
                } else {
                    -(MATE - ply as i32)
                }
            }
            Outcome::Draw => 0,
        };
    }
    if depth == 0 {
        // Drop into quiescence: resolve any pending captures so the eval
        // is called on a tactically quiet position.
        return qsearch(board, alpha, beta, ply, ctx);
    }

    // TT probe.
    let key = board.zobrist;
    let alpha_orig = alpha;
    let mut tt_best: Option<Move> = None;
    {
        let entry = &ctx.tt[(key as usize) & SEARCH_TT_MASK];
        ctx.tt_probes += 1;
        if entry.key == key && entry.bound != TTBound::Empty {
            ctx.tt_hits += 1;
            tt_best = entry.best;
            if entry.depth as u32 >= depth {
                match entry.bound {
                    TTBound::Exact => return entry.score,
                    TTBound::Lower => alpha = alpha.max(entry.score),
                    TTBound::Upper => beta = beta.min(entry.score),
                    TTBound::Empty => {}
                }
                if alpha >= beta {
                    return entry.score;
                }
            }
        }
    }

    let mut moves: Vec<Move> = Vec::with_capacity(64);
    board.legal_moves_into(&mut moves);
    if moves.is_empty() {
        return 0;
    }

    let ply_idx = ply as usize;
    let killers = ctx
        .killers
        .get(ply_idx)
        .copied()
        .unwrap_or([None, None]);
    order_moves(board, &mut moves, &killers);
    // If the TT gave us a previous best move, bubble it to the front —
    // but only if the natural top of the list is *not* already a capture.
    // Promoting a quiet TT-best ahead of a real capture loses MVV-LVA's
    // tightening of α and ends up costing more nodes than it saves.
    if let Some(tt_mv) = tt_best
        && let Some(idx) = moves.iter().position(|&m| m == tt_mv)
        && idx != 0
        && score_move(board, moves[0], &killers) < score_move(board, tt_mv, &killers)
    {
        moves.swap(0, idx);
    }

    let mut best = -INF;
    let mut best_move: Option<Move> = None;
    for mv in &moves {
        let undo = board.apply_legal(*mv);
        let score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, ctx);
        board.unmake(*mv, undo);
        if score > best {
            best = score;
            best_move = Some(*mv);
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            // β-cutoff. Record quiet (non-capture) moves as killers for
            // this ply so siblings try them first.
            let is_capture = matches!(*mv, Move::Deplacement { to, .. } if board.at(to).is_some());
            if !is_capture && ply_idx < ctx.killers.len() {
                let slot = &mut ctx.killers[ply_idx];
                if slot[0] != Some(*mv) {
                    slot[1] = slot[0];
                    slot[0] = Some(*mv);
                }
            }
            break;
        }
    }

    // TT store. Classify the result vs the original α/β window.
    let bound = if best <= alpha_orig {
        TTBound::Upper
    } else if best >= beta {
        TTBound::Lower
    } else {
        TTBound::Exact
    };
    ctx.tt[(key as usize) & SEARCH_TT_MASK] = SearchTTEntry {
        key,
        depth: depth.min(u16::MAX as u32) as u16,
        bound,
        score: best,
        best: best_move,
    };

    best
}

/// Fixed-depth alpha-beta agent.
pub struct AlphaBetaPlayer {
    pub depth: u32,
    /// Used to break ties between equally-scored moves so two AB players
    /// with the same depth don't produce identical replies forever in
    /// self-play.
    rng: RandomPlayer,
    /// Leaf evaluator. `full_eval` (default) uses all positional terms;
    /// `material_eval` is the legacy material-only function, kept around
    /// for A/B testing.
    eval_fn: fn(&Board) -> i32,
    name_tag: String,
    /// Persistent transposition table — kept across `choose_move` calls so
    /// turn N+1 can reuse entries from turn N. Always-replace; collisions
    /// are caught by the full-key check in `probe`.
    tt: Box<[SearchTTEntry]>,
    /// Stats from the last `choose_move` call. Useful for benchmarking
    /// the search heuristics we add (move ordering, killers, TT, …).
    pub last_nodes: u64,
    pub last_qnodes: u64,
    pub last_elapsed_ms: u64,
    pub last_tt_probes: u64,
    pub last_tt_hits: u64,
}

impl AlphaBetaPlayer {
    /// AB with the full positional eval (recommended default).
    pub fn new(depth: u32, seed: u64) -> Self {
        Self::with_eval(depth, seed, full_eval, "ab".to_string())
    }

    /// AB with material-only eval, kept so we can measure how much the
    /// positional terms are worth in self-play.
    pub fn new_material_only(depth: u32, seed: u64) -> Self {
        Self::with_eval(depth, seed, material_eval, "ab-mat".to_string())
    }

    fn with_eval(depth: u32, seed: u64, eval_fn: fn(&Board) -> i32, name_tag: String) -> Self {
        Self {
            depth,
            rng: RandomPlayer::new(seed),
            eval_fn,
            name_tag,
            tt: vec![SearchTTEntry::default(); SEARCH_TT_SIZE].into_boxed_slice(),
            last_nodes: 0,
            last_qnodes: 0,
            last_elapsed_ms: 0,
            last_tt_probes: 0,
            last_tt_hits: 0,
        }
    }
}

impl Player for AlphaBetaPlayer {
    fn choose_move(&mut self, board: &Board) -> Option<Move> {
        let start = std::time::Instant::now();
        let mut b = board.clone();
        let mut moves = Vec::with_capacity(64);
        b.legal_moves_into(&mut moves);
        if moves.is_empty() {
            return None;
        }

        let mut ctx = SearchCtx {
            eval_fn: self.eval_fn,
            nodes: 0,
            qnodes: 0,
            killers: vec![[None; 2]; self.depth as usize + 2],
            tt: &mut self.tt,
            tt_probes: 0,
            tt_hits: 0,
        };

        // Order root moves too — captures first improves α at the very
        // first child, tightening the search for all siblings.
        order_moves(&b, &mut moves, &[None, None]);
        // If the TT remembers a best move from a previous search at this
        // exact root position, try it first.
        let key = b.zobrist;
        let tt_root_best = {
            let entry = &ctx.tt[(key as usize) & SEARCH_TT_MASK];
            if entry.key == key && entry.bound != TTBound::Empty {
                entry.best
            } else {
                None
            }
        };
        if let Some(tt_mv) = tt_root_best
            && let Some(idx) = moves.iter().position(|&m| m == tt_mv)
            && idx != 0
            && score_move(&b, moves[0], &[None, None]) < score_move(&b, tt_mv, &[None, None])
        {
            moves.swap(0, idx);
        }

        let mut alpha = -INF;
        let beta = INF;
        let mut best_score = -INF;
        let mut best_indices: Vec<usize> = Vec::new();

        for (i, mv) in moves.iter().enumerate() {
            let undo = b.apply_legal(*mv);
            // Pass ply=1 because we've made one move from the root.
            let score = -negamax(
                &mut b,
                self.depth.saturating_sub(1),
                -beta,
                -alpha,
                1,
                &mut ctx,
            );
            b.unmake(*mv, undo);

            if score > best_score {
                best_score = score;
                best_indices.clear();
                best_indices.push(i);
                if best_score > alpha {
                    alpha = best_score;
                }
            } else if score == best_score {
                best_indices.push(i);
            }
        }

        // Random tiebreak among equally-good moves.
        let pick = if best_indices.len() == 1 {
            best_indices[0]
        } else {
            best_indices[(self.rng.next_u64() % best_indices.len() as u64) as usize]
        };
        // Also store the picked root move so future searches start with it.
        ctx.tt[(key as usize) & SEARCH_TT_MASK] = SearchTTEntry {
            key,
            depth: self.depth.min(u16::MAX as u32) as u16,
            bound: TTBound::Exact,
            score: best_score,
            best: Some(moves[pick]),
        };

        self.last_nodes = ctx.nodes;
        self.last_qnodes = ctx.qnodes;
        self.last_elapsed_ms = start.elapsed().as_millis() as u64;
        self.last_tt_probes = ctx.tt_probes;
        self.last_tt_hits = ctx.tt_hits;
        Some(moves[pick])
    }
    fn name(&self) -> String {
        format!("{}(d={})", self.name_tag, self.depth)
    }
}

// ---------------------------------------------------------------------------
// MctsPlayer
// ---------------------------------------------------------------------------

/// How long an MCTS search runs for.
#[derive(Clone, Copy, Debug)]
pub enum MctsBudget {
    /// Run exactly `n` UCT iterations regardless of wall-clock time.
    Iterations(u32),
    /// Keep iterating until at least `ms` milliseconds have elapsed. The
    /// clock is checked every 64 iterations to keep the overhead off the
    /// hot path; a single iteration that takes longer than `ms` cannot
    /// be cut short.
    TimeMs(u64),
}

/// Classical UCT (Upper Confidence bounds for Trees). Four-phase loop:
/// **selection** via UCB1, **expansion** of one new child, random
/// **rollout** to a terminal, **backprop** of the result. Default
/// exploration constant c = √2 (textbook UCT).
///
/// The arena holds one node per *expanded* board position. Each node
/// stores wins on a `[0, 1]` scale from the perspective of the player
/// who *chose* to play the move into this node (i.e. the parent's side
/// to move). Backprop flips signs accordingly.
pub struct MctsPlayer {
    pub budget: MctsBudget,
    pub c: f64,
    rng: RandomPlayer,
    /// Iterations actually executed during the last `choose_move` call.
    /// Useful for reporting "I ran N iters in T ms" with a time budget.
    pub last_iterations: u32,
    pub last_elapsed_ms: u64,
}

impl MctsPlayer {
    /// Iteration-budgeted MCTS.
    pub fn new(iterations: u32, seed: u64) -> Self {
        Self::with_budget(MctsBudget::Iterations(iterations), seed)
    }

    /// Time-budgeted MCTS — keep iterating for at least `ms` milliseconds
    /// per move.
    pub fn with_time_ms(ms: u64, seed: u64) -> Self {
        Self::with_budget(MctsBudget::TimeMs(ms), seed)
    }

    pub fn with_budget(budget: MctsBudget, seed: u64) -> Self {
        Self {
            budget,
            c: std::f64::consts::SQRT_2,
            rng: RandomPlayer::new(seed),
            last_iterations: 0,
            last_elapsed_ms: 0,
        }
    }
}

struct MctsNode {
    visits: u32,
    /// Sum of rewards from the perspective of the player who chose the
    /// move into this node (i.e. the parent's side to move). Rewards
    /// are on `[0, 1]`: 1.0 = win for that player, 0.0 = loss, 0.5 = draw.
    wins: f64,
    /// `Some(mv)` for non-root nodes — the move that produced this state.
    mv: Option<Move>,
    parent: Option<usize>,
    children: Vec<usize>,
    /// Moves at this node not yet promoted to children.
    unexpanded: Vec<Move>,
    /// Side to move at this node *after* `mv` was played. Used to interpret
    /// the rollout result from this node's perspective during backprop.
    stm: Color,
    is_terminal: bool,
    /// Cached outcome of a terminal node — saves an `outcome()` call.
    terminal_outcome: Option<Outcome>,
}

impl MctsPlayer {
    /// UCB1 score of `child` under `parent_visits` parent visits.
    /// `wins` is on `[0, visits]`, so `mean ∈ [0, 1]`; with `c = √2`
    /// the exploration term has the same units.
    #[inline]
    fn ucb1(child: &MctsNode, parent_visits: u32, c: f64) -> f64 {
        if child.visits == 0 {
            return f64::INFINITY;
        }
        let mean = child.wins / child.visits as f64;
        let explore = ((parent_visits as f64).ln() / child.visits as f64).sqrt();
        mean + c * explore
    }

    /// One full UCT iteration. Mutates the arena and returns nothing.
    fn iterate(&mut self, root_board: &Board, arena: &mut Vec<MctsNode>) {
        // Local board scratchpad: we descend then mutate in place; the local
        // board is discarded at the end of the iteration.
        let mut board = root_board.clone();
        let mut node_id = 0usize;

        // 1. Selection — descend through fully expanded, non-terminal nodes.
        loop {
            let node = &arena[node_id];
            if node.is_terminal || !node.unexpanded.is_empty() || node.children.is_empty() {
                break;
            }
            let parent_visits = node.visits;
            let best_child = node
                .children
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    let ua = Self::ucb1(&arena[a], parent_visits, self.c);
                    let ub = Self::ucb1(&arena[b], parent_visits, self.c);
                    ua.partial_cmp(&ub).unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("non-empty children");
            let mv = arena[best_child].mv.expect("non-root has a move");
            board.apply_legal(mv);
            node_id = best_child;
        }

        // 2. Expansion — if non-terminal and has unexpanded moves, pop one.
        if !arena[node_id].is_terminal && !arena[node_id].unexpanded.is_empty() {
            let pick =
                (self.rng.next_u64() % arena[node_id].unexpanded.len() as u64) as usize;
            let mv = arena[node_id].unexpanded.swap_remove(pick);
            board.apply_legal(mv);
            let outcome = board.outcome();
            let new_node = MctsNode {
                visits: 0,
                wins: 0.0,
                mv: Some(mv),
                parent: Some(node_id),
                children: Vec::new(),
                unexpanded: if outcome.is_some() {
                    Vec::new()
                } else {
                    board.legal_moves()
                },
                stm: board.side_to_move,
                is_terminal: outcome.is_some(),
                terminal_outcome: outcome,
            };
            let new_id = arena.len();
            arena.push(new_node);
            arena[node_id].children.push(new_id);
            node_id = new_id;
        }

        // 3. Rollout — random play to a terminal. If we landed on a
        //    terminal during expansion, skip the rollout.
        let final_outcome = if arena[node_id].is_terminal {
            arena[node_id]
                .terminal_outcome
                .expect("terminal node carries outcome")
        } else {
            rollout(&mut board, &mut self.rng)
        };

        // 4. Backprop — at each node, accumulate the reward from the
        //    perspective of the player who *chose* the move into it.
        let mut current = Some(node_id);
        while let Some(id) = current {
            arena[id].visits += 1;
            if let Some(pid) = arena[id].parent {
                let mover = arena[pid].stm;
                let reward = match final_outcome {
                    Outcome::Win(winner) => {
                        if winner == mover {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Outcome::Draw => 0.5,
                };
                arena[id].wins += reward;
            }
            current = arena[id].parent;
        }
    }
}

impl Player for MctsPlayer {
    fn choose_move(&mut self, board: &Board) -> Option<Move> {
        let root_outcome = board.outcome();
        if root_outcome.is_some() {
            return None;
        }
        let root_moves = board.legal_moves();
        if root_moves.is_empty() {
            return None;
        }

        let initial_capacity = match self.budget {
            MctsBudget::Iterations(n) => n as usize + 1,
            MctsBudget::TimeMs(_) => 1024,
        };
        let mut arena: Vec<MctsNode> = Vec::with_capacity(initial_capacity);
        arena.push(MctsNode {
            visits: 0,
            wins: 0.0,
            mv: None,
            parent: None,
            children: Vec::new(),
            unexpanded: root_moves.clone(),
            stm: board.side_to_move,
            is_terminal: false,
            terminal_outcome: None,
        });

        let start = std::time::Instant::now();
        let mut iters: u32 = 0;
        match self.budget {
            MctsBudget::Iterations(n) => {
                for _ in 0..n {
                    self.iterate(board, &mut arena);
                    iters += 1;
                }
            }
            MctsBudget::TimeMs(ms) => {
                let deadline = std::time::Duration::from_millis(ms);
                loop {
                    self.iterate(board, &mut arena);
                    iters += 1;
                    // Check the clock every 64 iters to keep the overhead
                    // negligible; a single iter is microseconds-cheap.
                    if iters & 63 == 0 && start.elapsed() >= deadline {
                        break;
                    }
                }
            }
        }
        self.last_iterations = iters;
        self.last_elapsed_ms = start.elapsed().as_millis() as u64;

        // Pick the most-visited root child (more robust than best-mean
        // when iteration count is small — high-visit children are
        // exhaustively evaluated).
        let root = &arena[0];
        let best = root
            .children
            .iter()
            .max_by_key(|&&cid| arena[cid].visits)?;
        arena[*best].mv
    }

    fn name(&self) -> String {
        match self.budget {
            MctsBudget::Iterations(n) => format!("mcts(n={n})"),
            MctsBudget::TimeMs(ms) => format!("mcts(t={ms}ms)"),
        }
    }
}

/// Random rollout from `board` to a terminal state. Returns the final
/// outcome. Mutates the board in place — caller passes a scratch copy.
fn rollout(board: &mut Board, rng: &mut RandomPlayer) -> Outcome {
    let mut moves: Vec<Move> = Vec::with_capacity(64);
    loop {
        if let Some(o) = board.outcome() {
            return o;
        }
        moves.clear();
        board.legal_moves_into(&mut moves);
        if moves.is_empty() {
            return Outcome::Draw; // defensive
        }
        let i = (rng.next_u64() % moves.len() as u64) as usize;
        board.apply_legal(moves[i]);
    }
}

// ---------------------------------------------------------------------------
// Game harness
// ---------------------------------------------------------------------------

/// How the game ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WinType {
    /// One side's capitaine was captured.
    CapitaineCaptured,
    /// One side ran out of equipiers.
    EquipiersEliminated,
    /// 30-ply draw counter ran out.
    DrawCounter,
    /// Used when a player returns no move (shouldn't happen in ADIX).
    NoMove,
}

/// One game's worth of bookkeeping for callers that want a trace.
#[derive(Debug, Clone)]
pub struct GameRecord {
    pub outcome: Outcome,
    pub plies: u32,
    pub moves: Vec<Move>,
    /// Final alive counts: ((clair_cap, clair_eq), (fonce_cap, fonce_eq)).
    pub final_alive: ((u32, u32), (u32, u32)),
    pub win_type: WinType,
    /// Plies at which a capture (deplacement onto a piece) occurred.
    /// Useful for "is the game won by attrition or sudden tactic?"
    pub capture_plies: Vec<u32>,
}

fn classify_outcome(outcome: Outcome, board: &Board) -> WinType {
    let cl = board.alive_counts(Color::Clair);
    let fo = board.alive_counts(Color::Fonce);
    match outcome {
        Outcome::Draw => WinType::DrawCounter,
        Outcome::Win(_) => {
            // The losing side has a zeroed count somewhere.
            if cl.0 == 0 || fo.0 == 0 {
                WinType::CapitaineCaptured
            } else if cl.1 == 0 || fo.1 == 0 {
                WinType::EquipiersEliminated
            } else {
                // Shouldn't happen, but be defensive.
                WinType::NoMove
            }
        }
    }
}

/// Drive a game from the initial position. `clair_player` moves first.
/// Returns the outcome and a record of moves played.
///
/// We trust each player to return a legal move (they should call
/// `board.legal_moves()` internally). If a player returns `None`, we treat
/// it as a draw — though ADIX positions always have legal moves until
/// terminal.
pub fn play_game(
    clair_player: &mut dyn Player,
    fonce_player: &mut dyn Player,
) -> GameRecord {
    let mut board = Board::initial();
    let mut moves_played: Vec<Move> = Vec::new();
    let mut capture_plies: Vec<u32> = Vec::new();

    loop {
        if let Some(o) = board.outcome() {
            let win_type = classify_outcome(o, &board);
            return GameRecord {
                outcome: o,
                plies: board.ply,
                moves: moves_played,
                final_alive: (board.alive_counts(Color::Clair), board.alive_counts(Color::Fonce)),
                win_type,
                capture_plies,
            };
        }
        let player: &mut dyn Player = match board.side_to_move {
            Color::Clair => clair_player,
            Color::Fonce => fonce_player,
        };
        let Some(mv) = player.choose_move(&board) else {
            return GameRecord {
                outcome: Outcome::Draw,
                plies: board.ply,
                moves: moves_played,
                final_alive: (board.alive_counts(Color::Clair), board.alive_counts(Color::Fonce)),
                win_type: WinType::NoMove,
                capture_plies,
            };
        };
        // Detect if this move is a capture before applying (the captured
        // piece will go into the board's captured list).
        let is_capture = matches!(mv, Move::Deplacement { to, .. } if board.at(to).is_some());
        // apply() validates — if a player produced an illegal move, we panic
        // here loudly. This is a harness, not the hot search path.
        board
            .apply(mv)
            .unwrap_or_else(|e| panic!("{} produced illegal move {mv:?}: {e:?}", player.name()));
        if is_capture {
            capture_plies.push(board.ply);
        }
        moves_played.push(mv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_plays_a_legal_game_to_completion() {
        let mut white = RandomPlayer::new(42);
        let mut black = RandomPlayer::new(43);
        let rec = play_game(&mut white, &mut black);
        // ADIX with the 30-ply draw rule + capture-driven progress always
        // terminates. Random play tends to draw most of the time.
        assert!(rec.plies > 0);
        // outcome is one of the three valid ones (compiler-enforced).
        let _ = rec.outcome;
    }

    #[test]
    fn alpha_beta_d1_beats_random_in_a_quick_match() {
        // Not a guarantee in any one game, but over a handful AB(d=1)
        // should at least win some. The point of this test is that the
        // engine completes a game without panicking on an illegal move.
        let mut ab = AlphaBetaPlayer::new(1, 7);
        let mut rng = RandomPlayer::new(7);
        let rec = play_game(&mut ab, &mut rng);
        let _ = rec.outcome;
    }

    #[test]
    fn mcts_plays_a_legal_game_to_completion() {
        // A tiny iteration count so the test stays fast; just verifies
        // every move chosen by MCTS is legal.
        let mut mcts = MctsPlayer::new(100, 11);
        let mut rng = RandomPlayer::new(11);
        let rec = play_game(&mut mcts, &mut rng);
        let _ = rec.outcome;
    }
}
