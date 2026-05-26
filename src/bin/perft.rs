use std::time::Instant;

use adix::board::Board;
use adix::notation::fmt_move;
use adix::perft::{perft, perft_divide};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_depth: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let divide_at: Option<u32> = args.get(2).and_then(|s| s.parse().ok());

    let board = Board::initial();

    println!("ADIX perft — initial position, depth 1..={}", max_depth);
    println!("{:>5}  {:>15}  {:>10}  {:>12}", "depth", "nodes", "time(ms)", "Mnodes/s");
    for d in 1..=max_depth {
        let t0 = Instant::now();
        let n = perft(&board, d);
        let elapsed = t0.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        let mnps = if elapsed.as_secs_f64() > 0.0 {
            (n as f64) / elapsed.as_secs_f64() / 1.0e6
        } else {
            0.0
        };
        println!("{:>5}  {:>15}  {:>10.1}  {:>12.2}", d, n, ms, mnps);
    }

    if let Some(d) = divide_at {
        println!();
        println!("divide at depth {}:", d);
        let rows = perft_divide(&board, d);
        let total: u64 = rows.iter().map(|(_, n)| *n).sum();
        for (mv, n) in &rows {
            println!("  {:<8} {}", fmt_move(*mv), n);
        }
        println!("  {:<8} {}  (total)", "==", total);
    }
}
