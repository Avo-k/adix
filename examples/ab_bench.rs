//! Benchmark AlphaBetaPlayer at varying depths: nodes/move and time/move.
use adix::agent::{AlphaBetaPlayer, Player, RandomPlayer};
use adix::board::Board;

fn bench(depth: u32, n_moves: u32) {
    let mut ab = AlphaBetaPlayer::new(depth, 42);
    let mut rng = RandomPlayer::new(43);
    let mut board = Board::initial();
    let mut total_nodes: u64 = 0;
    let mut total_qnodes: u64 = 0;
    let mut total_ms: u64 = 0;
    let mut total_probes: u64 = 0;
    let mut total_hits: u64 = 0;
    let mut moves_done = 0;
    while moves_done < n_moves && board.outcome().is_none() {
        let mv = match ab.choose_move(&board) {
            Some(m) => m,
            None => break,
        };
        total_nodes += ab.last_nodes;
        total_qnodes += ab.last_qnodes;
        total_ms += ab.last_elapsed_ms;
        total_probes += ab.last_tt_probes;
        total_hits += ab.last_tt_hits;
        moves_done += 1;
        board.apply(mv).unwrap();
        if board.outcome().is_some() { break; }
        // Random opponent move
        if let Some(mv) = rng.choose_move(&board) {
            board.apply(mv).unwrap();
        }
    }
    let avg_nodes = total_nodes / moves_done as u64;
    let avg_qnodes = total_qnodes / moves_done as u64;
    let avg_ms = total_ms / moves_done as u64;
    let total_all = total_nodes + total_qnodes;
    let knps = if total_ms > 0 { total_all / total_ms } else { 0 };
    let q_pct = if total_all > 0 { 100.0 * total_qnodes as f64 / total_all as f64 } else { 0.0 };
    let hit_rate = if total_probes > 0 {
        100.0 * total_hits as f64 / total_probes as f64
    } else { 0.0 };
    println!(
        "d={depth}: {moves_done} mv, main {avg_nodes:>7} q {avg_qnodes:>7} ({q_pct:>4.1}% q), {avg_ms:>5} ms, {knps:>4} kn/s, TT {hit_rate:>4.1}%"
    );
}

fn main() {
    println!("--- AlphaBetaPlayer benchmark ---");
    for d in [2u32, 3, 4, 5] {
        bench(d, 4);
    }
}
