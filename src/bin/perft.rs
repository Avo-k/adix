use std::time::Instant;

use adix::board::Board;
use adix::notation::fmt_move;
use adix::perft::{PerftTT, perft, perft_divide, perft_search, perft_tt, unique_exact, unique_hll};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_depth: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    // Positional 2nd arg is the divide-depth only if it's a bare integer.
    let divide_at: Option<u32> = args.get(2)
        .filter(|s| !s.starts_with("--"))
        .and_then(|s| s.parse().ok());
    let search_mode = std::env::args().any(|a| a == "--search");
    let tt_mb: Option<usize> = std::env::args().find_map(|a| {
        a.strip_prefix("--tt=").and_then(|n| n.parse().ok())
    });
    let tt_mode = tt_mb.is_some() || std::env::args().any(|a| a == "--tt");
    // Opt-in: also count positions deduplicated by Zobrist hash. Exact
    // (HashSet) at the depths it fits in RAM, HLL beyond. Cutoff is
    // configurable via `--unique-exact-max=N` (default 5).
    let unique_mode = std::env::args().any(|a| a == "--unique");
    let unique_exact_max: u32 = std::env::args()
        .find_map(|a| a.strip_prefix("--unique-exact-max=").and_then(|n| n.parse().ok()))
        .unwrap_or(5);

    let board = Board::initial();

    let label = if tt_mode {
        "perft (TT)"
    } else if search_mode {
        "perft (search mode)"
    } else {
        "perft"
    };
    let mut tt = if tt_mode {
        Some(PerftTT::with_mb(tt_mb.unwrap_or(64)))
    } else {
        None
    };
    if let Some(tt) = tt.as_ref() {
        println!(
            "ADIX {label} — initial position, depth 1..={max_depth}, TT {} slots (~{} MB)",
            tt.len(),
            tt_mb.unwrap_or(64),
        );
        print!(
            "{:>5}  {:>15}  {:>10}  {:>12}  {:>10}  {:>8}",
            "depth", "nodes", "time(ms)", "Mnodes/s", "probes", "hit%"
        );
    } else {
        println!("ADIX {label} — initial position, depth 1..={max_depth}");
        print!("{:>5}  {:>15}  {:>10}  {:>12}", "depth", "nodes", "time(ms)", "Mnodes/s");
    }
    if unique_mode {
        print!("  {:>15}  {:>10}  {:>8}", "unique", "uniq(ms)", "mode");
    }
    println!();
    for d in 1..=max_depth {
        if let Some(tt) = tt.as_mut() {
            tt.reset_stats();
        }
        let t0 = Instant::now();
        let n = if let Some(tt) = tt.as_mut() {
            perft_tt(&board, d, tt)
        } else if search_mode {
            perft_search(&board, d)
        } else {
            perft(&board, d)
        };
        let elapsed = t0.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        let mnps = if elapsed.as_secs_f64() > 0.0 {
            (n as f64) / elapsed.as_secs_f64() / 1.0e6
        } else {
            0.0
        };
        if let Some(tt) = tt.as_ref() {
            print!(
                "{:>5}  {:>15}  {:>10.1}  {:>12.2}  {:>10}  {:>7.1}%",
                d, n, ms, mnps, tt.probes, tt.hit_rate() * 100.0
            );
        } else {
            print!("{:>5}  {:>15}  {:>10.1}  {:>12.2}", d, n, ms, mnps);
        }
        if unique_mode {
            let u0 = Instant::now();
            let (uniq, mode) = if d <= unique_exact_max {
                (unique_exact(&board, d), "exact")
            } else {
                (unique_hll(&board, d), "hll")
            };
            let ums = u0.elapsed().as_secs_f64() * 1000.0;
            print!("  {:>15}  {:>10.1}  {:>8}", uniq, ums, mode);
        }
        println!();
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
