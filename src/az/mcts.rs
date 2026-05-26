//! PUCT Monte Carlo Tree Search guided by a policy/value network.
//!
//! Same arena layout as [`crate::agent::MctsPlayer`], but the selection
//! score is **PUCT** instead of UCB1:
//!
//! ```text
//! PUCT(child) = Q(child) + c_puct · P(child) · √N_parent / (1 + N(child))
//! ```
//!
//! where `Q` is the mean value from the perspective of the player who
//! moved *into* the child, `P` is the prior probability assigned by the
//! parent's policy head, and `N` are visit counts. There are **no**
//! random rollouts: when we hit an unexpanded leaf we call the network
//! once, get `(policy, value)`, expand all legal children with their
//! priors, and back up `value` directly.
//!
//! Lives behind the `tch` feature because it consumes an [`AzNet`].
//! The MCTS algorithm itself is network-agnostic via the [`Evaluator`]
//! trait — useful for unit-testing with a uniform/random stub.

use crate::agent::{Player, RandomPlayer};
use crate::board::{Board, Outcome};
use crate::moves::Move;
use crate::piece::Color;

use super::encoding::move_to_index;
use super::net::AzNet;

// --- evaluator abstraction ------------------------------------------------

/// Anything that scores a position into `(policy[ACTIONS], value[-1,1])`.
/// `AzNet` is the production implementation; tests pass uniform stubs.
pub trait Evaluator {
    fn evaluate(&self, board: &Board) -> (Vec<f32>, f32);
}

impl Evaluator for AzNet {
    fn evaluate(&self, board: &Board) -> (Vec<f32>, f32) {
        self.forward_board(board)
    }
}

/// Batched evaluator: scores N positions in one shot. The point is to
/// amortize per-call overhead — GPU kernel launches, host↔device
/// transfer — across many positions. Implementors are expected to
/// hit the underlying network exactly once.
pub trait BatchedEvaluator {
    fn evaluate_batch(&self, boards: &[&Board]) -> Vec<(Vec<f32>, f32)>;
}

impl BatchedEvaluator for AzNet {
    fn evaluate_batch(&self, boards: &[&Board]) -> Vec<(Vec<f32>, f32)> {
        self.forward_boards(boards)
    }
}

// --- config ---------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct PuctConfig {
    /// Number of PUCT iterations to run per move.
    pub iterations: u32,
    /// Exploration constant. AZ paper used ~1.0–4.0; start at 1.5.
    pub c_puct: f32,
}

impl Default for PuctConfig {
    fn default() -> Self {
        Self { iterations: 400, c_puct: 1.5 }
    }
}

// --- arena ----------------------------------------------------------------

pub(crate) struct PuctNode {
    pub(crate) visits: u32,
    /// Sum of values from the perspective of the player who *moved
    /// into* this node (i.e. the parent's side to move). Stored as
    /// f64 so a long search doesn't lose precision in the sum.
    pub(crate) value_sum: f64,
    /// Prior assigned by the parent's policy head. Root has 1.0 by
    /// convention (never read).
    pub(crate) prior: f32,
    pub(crate) mv: Option<Move>,
    pub(crate) parent: Option<usize>,
    pub(crate) children: Vec<usize>,
    /// Side to move at this node (i.e. who plays the *next* move from
    /// here). Used to interpret leaf-value during backup.
    pub(crate) stm: Color,
    pub(crate) is_terminal: bool,
    pub(crate) terminal_outcome: Option<Outcome>,
    pub(crate) expanded: bool,
}

/// Build the root node for a search rooted at `board`. Shared by
/// [`PuctPlayer`] and self-play.
pub(crate) fn puct_root_node(board: &Board) -> PuctNode {
    PuctNode {
        visits: 0,
        value_sum: 0.0,
        prior: 1.0,
        mv: None,
        parent: None,
        children: Vec::new(),
        stm: board.side_to_move,
        is_terminal: false,
        terminal_outcome: None,
        expanded: false,
    }
}

#[inline]
fn puct_score(child: &PuctNode, parent_visits: u32, c_puct: f32) -> f32 {
    let q = if child.visits == 0 {
        0.0
    } else {
        (child.value_sum / child.visits as f64) as f32
    };
    let u = c_puct * child.prior * (parent_visits as f32).sqrt() / (1.0 + child.visits as f32);
    q + u
}

// PUCT primitives, split into selection / expansion / backup so the
// batched self-play loop can interleave eval calls across many trees.
// [`puct_iterate`] is the sequential wrapper used by [`PuctPlayer`].

/// Select from `arena[0]` (root) down to a leaf (a node that is either
/// unexpanded or terminal), advancing a scratch board along the path.
/// Returns the leaf node id and the board state at that leaf.
pub(crate) fn puct_select(
    arena: &Vec<PuctNode>,
    root_board: &Board,
    c_puct: f32,
) -> (Board, usize) {
    let mut board = root_board.clone();
    let mut node_id = 0usize;
    loop {
        let node = &arena[node_id];
        if !node.expanded || node.is_terminal {
            break;
        }
        let parent_visits = node.visits;
        let best = node
            .children
            .iter()
            .copied()
            .max_by(|&a, &b| {
                let sa = puct_score(&arena[a], parent_visits, c_puct);
                let sb = puct_score(&arena[b], parent_visits, c_puct);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("expanded internal node has children");
        let mv = arena[best].mv.expect("non-root child has a move");
        board.apply_legal(mv);
        node_id = best;
    }
    (board, node_id)
}

/// Materialize children of `leaf_id` from its legal moves, attributing
/// priors from `policy`. The leaf must be non-terminal and unexpanded.
pub(crate) fn puct_expand(
    arena: &mut Vec<PuctNode>,
    leaf_id: usize,
    leaf_board: &Board,
    policy: &[f32],
) {
    let mut moves: Vec<Move> = Vec::with_capacity(64);
    leaf_board.legal_moves_into(&mut moves);
    for mv in moves {
        let idx = move_to_index(mv);
        let prior = policy.get(idx).copied().unwrap_or(0.0);
        let mut child_board = leaf_board.clone();
        child_board.apply_legal(mv);
        let outcome = child_board.outcome();
        let child = PuctNode {
            visits: 0,
            value_sum: 0.0,
            prior,
            mv: Some(mv),
            parent: Some(leaf_id),
            children: Vec::new(),
            stm: child_board.side_to_move,
            is_terminal: outcome.is_some(),
            terminal_outcome: outcome,
            expanded: false,
        };
        let new_id = arena.len();
        arena.push(child);
        arena[leaf_id].children.push(new_id);
    }
    arena[leaf_id].expanded = true;
}

/// Walk from `leaf_id` to the root, incrementing visits and adding the
/// signed leaf value into each ancestor's `value_sum`. `leaf_value` is
/// from the leaf's STM perspective (+1 = "leaf STM wins").
pub(crate) fn puct_backup(arena: &mut Vec<PuctNode>, leaf_id: usize, leaf_value: f64) {
    let leaf_stm = arena[leaf_id].stm;
    let mut current = Some(leaf_id);
    while let Some(id) = current {
        arena[id].visits += 1;
        if let Some(pid) = arena[id].parent {
            let mover_into = arena[pid].stm;
            let v = if mover_into == leaf_stm { leaf_value } else { -leaf_value };
            arena[id].value_sum += v;
        }
        current = arena[id].parent;
    }
}

/// Score of a terminal node from its own STM perspective. Used to feed
/// backup when selection landed on an end-of-game state.
#[inline]
pub(crate) fn puct_terminal_value(node: &PuctNode) -> f64 {
    debug_assert!(node.is_terminal);
    match node.terminal_outcome.expect("terminal carries outcome") {
        Outcome::Win(winner) => if winner == node.stm { 1.0 } else { -1.0 },
        Outcome::Draw => 0.0,
    }
}

/// One PUCT iteration: select to a leaf, expand+evaluate, back up.
/// Sequential / single-game variant; the batched self-play loop uses
/// the three primitives above directly.
pub(crate) fn puct_iterate<E: Evaluator + ?Sized>(
    arena: &mut Vec<PuctNode>,
    root_board: &Board,
    eval: &E,
    c_puct: f32,
) {
    let (leaf_board, leaf_id) = puct_select(arena, root_board, c_puct);
    let leaf_value: f64 = if arena[leaf_id].is_terminal {
        puct_terminal_value(&arena[leaf_id])
    } else {
        let (policy, value) = eval.evaluate(&leaf_board);
        puct_expand(arena, leaf_id, &leaf_board, &policy);
        value as f64
    };
    puct_backup(arena, leaf_id, leaf_value);
}

// --- player ---------------------------------------------------------------

pub struct PuctPlayer<E: Evaluator> {
    pub config: PuctConfig,
    eval: E,
    rng: RandomPlayer,
    /// Iterations actually executed in the last `choose_move`.
    pub last_iterations: u32,
    pub last_elapsed_ms: u64,
}

impl<E: Evaluator> PuctPlayer<E> {
    pub fn new(eval: E, config: PuctConfig, seed: u64) -> Self {
        Self {
            config,
            eval,
            rng: RandomPlayer::new(seed),
            last_iterations: 0,
            last_elapsed_ms: 0,
        }
    }
}

impl<E: Evaluator> Player for PuctPlayer<E> {
    fn choose_move(&mut self, board: &Board) -> Option<Move> {
        if board.outcome().is_some() {
            return None;
        }
        let root_moves = board.legal_moves();
        if root_moves.is_empty() {
            return None;
        }

        let start = std::time::Instant::now();
        let mut arena: Vec<PuctNode> = Vec::with_capacity(self.config.iterations as usize + 8);
        arena.push(puct_root_node(board));

        for _ in 0..self.config.iterations {
            puct_iterate(&mut arena, board, &self.eval, self.config.c_puct);
        }
        self.last_iterations = self.config.iterations;
        self.last_elapsed_ms = start.elapsed().as_millis() as u64;

        // Pick the most-visited root child; break ties randomly.
        let root = &arena[0];
        let max_visits = root.children.iter().map(|&c| arena[c].visits).max()?;
        let candidates: Vec<usize> = root
            .children
            .iter()
            .copied()
            .filter(|&c| arena[c].visits == max_visits)
            .collect();
        let pick = if candidates.len() == 1 {
            candidates[0]
        } else {
            let r = (self.rng.next_u64() % candidates.len() as u64) as usize;
            candidates[r]
        };
        arena[pick].mv
    }

    fn name(&self) -> String {
        format!("puct(n={})", self.config.iterations)
    }
}

// --- tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az::encoding::{ACTIONS, fill_legal_mask};

    /// Uniform evaluator: returns mask/sum as policy and 0 value.
    /// Lets us unit-test the MCTS machinery without a real network.
    struct UniformEval;
    impl Evaluator for UniformEval {
        fn evaluate(&self, board: &Board) -> (Vec<f32>, f32) {
            let mut mask = vec![0.0_f32; ACTIONS];
            fill_legal_mask(board, &mut mask);
            let sum: f32 = mask.iter().sum();
            if sum > 0.0 {
                for v in mask.iter_mut() {
                    *v /= sum;
                }
            }
            (mask, 0.0)
        }
    }

    #[test]
    fn puct_plays_a_legal_game_to_completion() {
        // 50 iterations / move — keeps the test fast while still
        // exercising selection + expansion + backup at each move.
        let mut puct = PuctPlayer::new(
            UniformEval,
            PuctConfig { iterations: 50, c_puct: 1.5 },
            17,
        );
        let mut rng = RandomPlayer::new(17);
        let rec = crate::agent::play_game(&mut puct, &mut rng);
        let _ = rec.outcome;
    }
}
