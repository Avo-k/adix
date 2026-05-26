//! Self-play game generation for AlphaZero training.
//!
//! Plays one game with the current network against itself, recording
//! a training sample at every position visited. Each sample carries:
//!
//! - the encoded state at that ply,
//! - the **visit-count policy** at MCTS root (normalised; the AZ
//!   training target — strictly stronger than the network's raw policy
//!   because it's been refined by the tree search), and
//! - the **value target** — the eventual game outcome, signed from
//!   the perspective of the player who was about to move at that ply.
//!
//! Two self-play–specific knobs on top of the inference-time PUCT:
//!
//! - For the first `temperature_plies` plies, the next move is sampled
//!   in proportion to root visit counts (opens up exploration during
//!   the opening). After that we switch to argmax-visits.
//! - A hard `max_plies` cap so a pathological early-training game can't
//!   loop indefinitely. ADIX terminates by §10-2-1 anyway; the cap is
//!   defensive.
//!
//! Dirichlet noise injection at the root is a known follow-up; with
//! `tch`'s stdlib-only dependency story it'd need a custom Gamma
//! sampler. v1 ships without it — exploration relies on temperature
//! sampling alone.

use crate::agent::RandomPlayer;
use crate::board::{Board, Outcome};
use crate::moves::Move;
use crate::piece::Color;

use super::dirichlet::symmetric_dirichlet;
use super::encoding::{ACTIONS, INPUT_SIZE, encode_state, move_to_index};
use super::mcts::{
    BatchedEvaluator, Evaluator, PuctConfig, PuctNode, promote_child_to_root, puct_expand,
    puct_iterate, puct_backup, puct_root_node, puct_select, puct_terminal_value,
};

/// One AlphaZero training sample. All vectors have the canonical sizes
/// — `state.len() == INPUT_SIZE`, `policy_target.len() == ACTIONS`.
#[derive(Clone, Debug)]
pub struct Sample {
    pub state: Vec<f32>,
    pub policy_target: Vec<f32>,
    pub value_target: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SelfPlayConfig {
    pub puct: PuctConfig,
    /// Plies during which to sample moves proportionally to visit counts
    /// (encourages opening diversity). After this, argmax-visits.
    pub temperature_plies: u32,
    /// Hard cap on plies. ADIX's 30-ply draw counter terminates games
    /// anyway; this is a defensive safety net for early-training noise.
    pub max_plies: u32,
    /// Symmetric Dirichlet α applied to the root's prior at the start
    /// of every search. AZ paper rule of thumb: ~10 / branching_factor;
    /// ADIX has ~50, so 0.2 is a sensible default.
    pub dirichlet_alpha: f64,
    /// Mixing weight for the Dirichlet noise: `p' = (1-ε)·p + ε·noise`.
    /// AZ paper used 0.25. Set to 0 to disable noise (e.g. for
    /// inference / evaluation runs).
    pub dirichlet_eps: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            puct: PuctConfig::default(),
            temperature_plies: 20,
            max_plies: 400,
            dirichlet_alpha: 0.2,
            dirichlet_eps: 0.25,
        }
    }
}

/// Outcome bundle for a self-played game.
pub struct SelfPlayResult {
    pub samples: Vec<Sample>,
    pub outcome: Outcome,
    pub plies: u32,
}

/// Play a single self-play game with `eval` on both sides. Returns the
/// per-ply training samples plus the game's outcome.
pub fn play_one_game<E: Evaluator + ?Sized>(
    eval: &E,
    config: &SelfPlayConfig,
    rng: &mut RandomPlayer,
) -> SelfPlayResult {
    let mut board = Board::initial();
    // (state, policy_target, stm at that state) — value_target filled in below.
    let mut records: Vec<(Vec<f32>, Vec<f32>, Color)> = Vec::new();

    while board.outcome().is_none() && board.ply < config.max_plies {
        let mv = match search_one_move(eval, config, &board) {
            Some(out) => {
                let mut state = vec![0.0_f32; INPUT_SIZE];
                encode_state(&board, &mut state);
                records.push((state, out.policy_target, board.side_to_move));
                pick_move_with_temperature(&out.root_visits, config, &board, rng)
            }
            None => break,
        };
        board.apply_legal(mv);
    }

    let outcome = board.outcome().unwrap_or(Outcome::Draw);
    let winner = match outcome {
        Outcome::Win(w) => Some(w),
        Outcome::Draw => None,
    };
    let samples: Vec<Sample> = records
        .into_iter()
        .map(|(state, policy_target, stm)| {
            let value_target = match winner {
                Some(w) => if w == stm { 1.0 } else { -1.0 },
                None => 0.0,
            };
            Sample { state, policy_target, value_target }
        })
        .collect();

    SelfPlayResult { samples, outcome, plies: board.ply }
}

// --- internals ------------------------------------------------------------

struct OneSearch {
    policy_target: Vec<f32>,
    root_visits: Vec<(Move, u32)>,
}

/// Run PUCT once at `board`, return the visit-count policy target plus
/// the per-child visit list (for temperature sampling).
fn search_one_move<E: Evaluator + ?Sized>(
    eval: &E,
    config: &SelfPlayConfig,
    board: &Board,
) -> Option<OneSearch> {
    let mut arena: Vec<PuctNode> = Vec::with_capacity(config.puct.iterations as usize + 8);
    arena.push(puct_root_node(board));
    for _ in 0..config.puct.iterations {
        puct_iterate(
            &mut arena,
            board,
            eval,
            config.puct.c_puct,
            config.puct.fpu_reduction,
        );
    }

    let root = &arena[0];
    let mut root_visits: Vec<(Move, u32)> = Vec::with_capacity(root.children.len());
    let mut total: u64 = 0;
    for &cid in &root.children {
        let child = &arena[cid];
        if let Some(mv) = child.mv {
            root_visits.push((mv, child.visits));
            total += child.visits as u64;
        }
    }
    if total == 0 || root_visits.is_empty() {
        return None;
    }

    let mut policy_target = vec![0.0_f32; ACTIONS];
    for &(mv, v) in &root_visits {
        policy_target[move_to_index(mv)] = v as f32 / total as f32;
    }
    Some(OneSearch { policy_target, root_visits })
}

/// Mix symmetric Dirichlet(α) noise into the root's children's priors:
/// `p' = (1 - eps) · p + eps · noise`. Called once per move per worker,
/// just after the root is first expanded. Standard AZ exploration knob.
fn inject_root_dirichlet(
    arena: &mut [PuctNode],
    alpha: f64,
    eps: f32,
    rng: &mut RandomPlayer,
) {
    let n = arena[0].children.len();
    if n == 0 {
        return;
    }
    let noise = symmetric_dirichlet(alpha, n, rng);
    let children: Vec<usize> = arena[0].children.clone();
    for (i, &cid) in children.iter().enumerate() {
        let p = arena[cid].prior;
        arena[cid].prior = (1.0 - eps) * p + eps * noise[i];
    }
}

/// Sample proportionally to visits during the opening; argmax thereafter.
fn pick_move_with_temperature(
    root_visits: &[(Move, u32)],
    config: &SelfPlayConfig,
    board: &Board,
    rng: &mut RandomPlayer,
) -> Move {
    if board.ply < config.temperature_plies {
        let total: u64 = root_visits.iter().map(|&(_, v)| v as u64).sum();
        if total == 0 {
            return root_visits[0].0;
        }
        let target = rng.next_u64() % total;
        let mut cum: u64 = 0;
        for &(mv, v) in root_visits {
            cum += v as u64;
            if cum > target {
                return mv;
            }
        }
        root_visits.last().expect("non-empty").0
    } else {
        // Argmax visits (break ties to first — deterministic given seed).
        root_visits
            .iter()
            .max_by_key(|&&(_, v)| v)
            .expect("non-empty")
            .0
    }
}

// --- batched (multi-game parallel) self-play ------------------------------

/// One self-play game in flight. Holds the live board, the active MCTS
/// arena for the current move, the per-ply records buffered for this
/// game, and the iteration count for the in-flight search.
struct Worker {
    board: Board,
    arena: Vec<PuctNode>,
    iter_count: u32,
    /// `(state, policy_target, stm)` triplets for the current game.
    records: Vec<(Vec<f32>, Vec<f32>, Color)>,
    /// How many plies this game has played so far (for stats; ply count
    /// also lives on the board).
    ply: u32,
    /// Root Dirichlet noise hasn't been mixed into the priors yet for
    /// this move. We can't inject at construction time because the
    /// root might still be unexpanded — defer until just after the
    /// root's first expansion (or immediately if tree-reuse handed us
    /// an already-expanded root).
    noise_pending: bool,
}

impl Worker {
    fn new() -> Self {
        let board = Board::initial();
        let arena = vec![puct_root_node(&board)];
        Self {
            board,
            arena,
            iter_count: 0,
            records: Vec::new(),
            ply: 0,
            noise_pending: true,
        }
    }

    fn reset(&mut self) {
        self.board = Board::initial();
        self.arena = vec![puct_root_node(&self.board)];
        self.iter_count = 0;
        self.records.clear();
        self.ply = 0;
        self.noise_pending = true;
    }

    fn reset_arena(&mut self) {
        self.arena = vec![puct_root_node(&self.board)];
        self.iter_count = 0;
        self.noise_pending = true;
    }
}

/// Aggregate output of a batched self-play run.
pub struct BatchedSelfPlayResult {
    pub samples: Vec<Sample>,
    pub outcomes: Vec<Outcome>,
    pub plies_per_game: Vec<u32>,
    /// Number of network forward passes performed.
    pub batches: u64,
    /// Total positions evaluated by the network (sum of batch sizes).
    pub evaluated_positions: u64,
}

/// Run `n_workers` self-play games in parallel against `eval`, stopping
/// after `target_games` complete games have been collected. Per tick,
/// each active worker contributes one PUCT iteration (one leaf, or a
/// terminal short-circuit) — so each network call is a batch of up to
/// `n_workers` positions.
///
/// Trade-off vs [`play_one_game`]: same total search work, but the
/// network is hit ~`n_workers`× fewer times per ply, which is the
/// shape GPUs reward. CPU stays neutral.
pub fn play_batched<E: BatchedEvaluator + ?Sized>(
    eval: &E,
    config: &SelfPlayConfig,
    n_workers: usize,
    target_games: usize,
    rng: &mut RandomPlayer,
) -> BatchedSelfPlayResult {
    assert!(n_workers > 0);
    assert!(target_games > 0);
    let mut workers: Vec<Worker> = (0..n_workers).map(|_| Worker::new()).collect();

    let mut samples_out: Vec<Sample> = Vec::new();
    let mut outcomes: Vec<Outcome> = Vec::new();
    let mut plies_per_game: Vec<u32> = Vec::new();
    let mut games_done: usize = 0;
    let mut batches: u64 = 0;
    let mut evaluated_positions: u64 = 0;

    while games_done < target_games {
        // 1. Selection on every active worker. Yields a leaf board per
        //    worker that's still mid-search; terminal leaves are
        //    handled inline (no eval needed).
        let mut to_eval: Vec<(usize, Board, usize)> = Vec::new();
        let mut to_backup_terminal: Vec<(usize, usize, f64)> = Vec::new();
        for (i, w) in workers.iter().enumerate() {
            if w.iter_count >= config.puct.iterations {
                continue; // search already finished this move
            }
            if w.board.outcome().is_some() {
                continue; // game finished, awaiting reset
            }
            let (leaf_board, leaf_id) = puct_select(
                &w.arena,
                &w.board,
                config.puct.c_puct,
                config.puct.fpu_reduction,
            );
            if w.arena[leaf_id].is_terminal {
                let v = puct_terminal_value(&w.arena[leaf_id]);
                to_backup_terminal.push((i, leaf_id, v));
            } else {
                to_eval.push((i, leaf_board, leaf_id));
            }
        }

        // 2. Batched network call.
        if !to_eval.is_empty() {
            let board_refs: Vec<&Board> = to_eval.iter().map(|(_, b, _)| b).collect();
            let results = eval.evaluate_batch(&board_refs);
            batches += 1;
            evaluated_positions += results.len() as u64;
            for ((wid, board, leaf_id), (policy, value)) in
                to_eval.into_iter().zip(results.into_iter())
            {
                puct_expand(&mut workers[wid].arena, leaf_id, &board, &policy);
                puct_backup(&mut workers[wid].arena, leaf_id, value as f64);
                workers[wid].iter_count += 1;
            }
        }

        // 3. Terminal-leaf backups (no network needed).
        for (wid, leaf_id, v) in to_backup_terminal {
            puct_backup(&mut workers[wid].arena, leaf_id, v);
            workers[wid].iter_count += 1;
        }

        // 4. For each worker whose search just finished, commit the
        //    move, record the sample, and maybe finalize the game.
        for i in 0..workers.len() {
            if workers[i].iter_count < config.puct.iterations {
                continue;
            }
            if workers[i].board.outcome().is_some() {
                // Game already over; will be reset in step 5.
                continue;
            }

            // Read root visit distribution.
            let mut visits: Vec<(Move, u32)> =
                Vec::with_capacity(workers[i].arena[0].children.len());
            let mut total: u64 = 0;
            for &cid in &workers[i].arena[0].children.clone() {
                let child = &workers[i].arena[cid];
                if let Some(mv) = child.mv {
                    visits.push((mv, child.visits));
                    total += child.visits as u64;
                }
            }
            if total == 0 || visits.is_empty() {
                // Degenerate — no eval landed before iter_count hit budget
                // (only happens if every selected leaf was terminal). Reset
                // arena and re-search; if that loops, we'll exceed max_plies.
                workers[i].reset_arena();
                continue;
            }

            // Build the policy target and the encoded state for this ply.
            let mut policy_target = vec![0.0_f32; ACTIONS];
            for &(mv, v) in &visits {
                policy_target[move_to_index(mv)] = v as f32 / total as f32;
            }
            let mut state = vec![0.0_f32; INPUT_SIZE];
            encode_state(&workers[i].board, &mut state);
            let stm = workers[i].board.side_to_move;
            workers[i].records.push((state, policy_target, stm));

            // Pick the actual move with the configured temperature schedule.
            let mv = pick_move_with_temperature(&visits, config, &workers[i].board, rng);
            // Tree reuse: find the arena id of the child that corresponds
            // to the picked move, then promote its subtree to be the new
            // root. Salvages prior visits / Q-values / cached
            // expansions instead of rebuilding from scratch.
            let chosen_child_id = workers[i].arena[0]
                .children
                .iter()
                .copied()
                .find(|&cid| workers[i].arena[cid].mv == Some(mv))
                .expect("picked move must match a root child");
            workers[i].board.apply_legal(mv);
            workers[i].ply = workers[i].board.ply;
            let old_arena = std::mem::take(&mut workers[i].arena);
            workers[i].arena = promote_child_to_root(old_arena, chosen_child_id);
            // Lc0-style budgeting: count the carried-over visits toward
            // the new move's iteration budget. The transplanted subtree
            // already represents that much search work; we only owe the
            // delta. Cap defensively if reuse somehow exceeds the
            // budget (shouldn't happen unless config changed mid-game).
            let carried = workers[i].arena[0].visits;
            workers[i].iter_count = carried.min(config.puct.iterations);
            // New search root, so we owe a fresh round of root noise.
            workers[i].noise_pending = true;
        }

        // After the eval step has expanded any just-reached leaves, the
        // root may have become expanded (fresh-search case). Inject the
        // Dirichlet noise once per move into expanded roots that still
        // have noise pending. With tree reuse this fires immediately
        // because the transplanted root is already expanded.
        if config.dirichlet_eps > 0.0 {
            for w in workers.iter_mut() {
                if w.noise_pending && w.arena[0].expanded && !w.arena[0].children.is_empty() {
                    inject_root_dirichlet(
                        &mut w.arena,
                        config.dirichlet_alpha,
                        config.dirichlet_eps,
                        rng,
                    );
                    w.noise_pending = false;
                }
            }
        }

        // 5. Game-over workers: emit samples, restart the board.
        for i in 0..workers.len() {
            let game_over =
                workers[i].board.outcome().is_some() || workers[i].board.ply >= config.max_plies;
            if !game_over {
                continue;
            }
            let outcome = workers[i].board.outcome().unwrap_or(Outcome::Draw);
            let winner = match outcome {
                Outcome::Win(w) => Some(w),
                Outcome::Draw => None,
            };
            for (state, policy_target, sample_stm) in workers[i].records.drain(..) {
                let value_target = match winner {
                    Some(w) => if w == sample_stm { 1.0 } else { -1.0 },
                    None => 0.0,
                };
                samples_out.push(Sample { state, policy_target, value_target });
            }
            outcomes.push(outcome);
            plies_per_game.push(workers[i].board.ply);
            games_done += 1;
            workers[i].reset();
            if games_done >= target_games {
                break;
            }
        }
    }

    BatchedSelfPlayResult {
        samples: samples_out,
        outcomes,
        plies_per_game,
        batches,
        evaluated_positions,
    }
}

// --- tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az::encoding::fill_legal_mask;

    /// Same uniform evaluator as in mcts tests — lets us exercise
    /// the self-play loop without a real network.
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

    impl BatchedEvaluator for UniformEval {
        fn evaluate_batch(&self, boards: &[&Board]) -> Vec<(Vec<f32>, f32)> {
            boards.iter().map(|b| self.evaluate(b)).collect()
        }
    }

    #[test]
    fn batched_selfplay_collects_target_games_with_well_shaped_samples() {
        let cfg = SelfPlayConfig {
            puct: PuctConfig { iterations: 20, c_puct: 1.5, fpu_reduction: 0.2 },
            temperature_plies: 4,
            max_plies: 80,
            dirichlet_alpha: 0.2,
            dirichlet_eps: 0.25,
        };
        let mut rng = RandomPlayer::new(2024);
        let res = play_batched(&UniformEval, &cfg, 4, 3, &mut rng);

        assert_eq!(res.outcomes.len(), 3);
        assert_eq!(res.plies_per_game.len(), 3);
        assert!(res.batches > 0);
        assert!(res.evaluated_positions >= res.batches);

        for s in &res.samples {
            assert_eq!(s.state.len(), INPUT_SIZE);
            assert_eq!(s.policy_target.len(), ACTIONS);
            let sum: f32 = s.policy_target.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "policy target should sum to 1, got {sum}"
            );
            assert!(s.value_target == -1.0 || s.value_target == 0.0 || s.value_target == 1.0);
        }
    }

    #[test]
    fn selfplay_produces_well_shaped_samples() {
        let cfg = SelfPlayConfig {
            puct: PuctConfig { iterations: 30, c_puct: 1.5, fpu_reduction: 0.2 },
            temperature_plies: 4,
            max_plies: 60, // keep the test fast
            // play_one_game (sequential) doesn't inject root noise; the
            // fields are still set so the struct is fully initialized.
            dirichlet_alpha: 0.2,
            dirichlet_eps: 0.0,
        };
        let mut rng = RandomPlayer::new(1337);
        let res = play_one_game(&UniformEval, &cfg, &mut rng);

        assert!(res.plies > 0);
        assert_eq!(res.samples.len() as u32, res.plies);
        for s in &res.samples {
            assert_eq!(s.state.len(), INPUT_SIZE);
            assert_eq!(s.policy_target.len(), ACTIONS);
            // Policy target sums to 1 over the legal moves.
            let sum: f32 = s.policy_target.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "policy target should sum to 1, got {sum}"
            );
            // Value target is in {-1, 0, 1}.
            assert!(
                s.value_target == -1.0 || s.value_target == 0.0 || s.value_target == 1.0,
                "unexpected value target {}",
                s.value_target
            );
        }
    }
}
