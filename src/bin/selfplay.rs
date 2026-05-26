//! Self-play harness: play N games between two agents, report W/L/D.
//!
//! Usage:
//!   selfplay <white-spec> <black-spec> [N] [--swap]
//!
//! Agent specs:
//!   random        - uniform-random player
//!   ab:<depth>    - alpha-beta with material eval, depth `depth`
//!
//! Examples:
//!   selfplay random random 100
//!   selfplay ab:2 random 20
//!   selfplay ab:3 ab:1 10 --swap     # alternate colors between games
//!
//! Each game uses a unique seed so different games diverge even when
//! both agents are stochastic.

use std::time::Instant;

use adix::agent::{AlphaBetaPlayer, MctsPlayer, Player, RandomPlayer, WinType, play_game};
use adix::board::Outcome;
use adix::piece::Color;

fn parse_player(spec: &str, seed: u64) -> Box<dyn Player> {
    if spec == "random" {
        Box::new(RandomPlayer::new(seed))
    } else if let Some(d) = spec.strip_prefix("ab:") {
        let depth: u32 = d.parse().unwrap_or_else(|_| {
            eprintln!("bad depth in '{spec}'");
            std::process::exit(2);
        });
        Box::new(AlphaBetaPlayer::new(depth, seed))
    } else if let Some(d) = spec.strip_prefix("ab-mat:") {
        let depth: u32 = d.parse().unwrap_or_else(|_| {
            eprintln!("bad depth in '{spec}'");
            std::process::exit(2);
        });
        Box::new(AlphaBetaPlayer::new_material_only(depth, seed))
    } else if let Some(n) = spec.strip_prefix("mcts:") {
        let iters: u32 = n.parse().unwrap_or_else(|_| {
            eprintln!("bad iteration count in '{spec}'");
            std::process::exit(2);
        });
        Box::new(MctsPlayer::new(iters, seed))
    } else if let Some(t) = spec.strip_prefix("mcts-t:") {
        let ms: u64 = t.parse().unwrap_or_else(|_| {
            eprintln!("bad ms count in '{spec}'");
            std::process::exit(2);
        });
        Box::new(MctsPlayer::with_time_ms(ms, seed))
    } else {
        eprintln!(
            "unknown agent spec '{spec}'. Try 'random', 'ab:<depth>', 'mcts:<iters>', or 'mcts-t:<ms>'."
        );
        std::process::exit(2);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: selfplay <white> <black> [N] [--swap]");
        eprintln!("  white, black ∈ {{ random, ab:<depth>, ab-mat:<depth>, mcts:<iters>, mcts-t:<ms> }}");
        eprintln!("    ab       — alpha-beta with full positional eval");
        eprintln!("    ab-mat   — alpha-beta with material-only eval (baseline)");
        eprintln!("    mcts:N   — MCTS, fixed iteration budget per move");
        eprintln!("    mcts-t:T — MCTS, fixed time budget T ms per move");
        std::process::exit(2);
    }
    let white_spec = args[1].clone();
    let black_spec = args[2].clone();
    let n: u32 = args.get(3)
        .filter(|s| !s.starts_with("--"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let swap = args.iter().any(|a| a == "--swap");

    println!("selfplay: {white_spec} (W) vs {black_spec} (B), {n} games{}",
        if swap { ", colors swap each game" } else { "" });

    let mut clair_wins = 0u32;
    let mut fonce_wins = 0u32;
    let mut draws = 0u32;
    let mut total_plies = 0u32;
    let mut white_agent_wins = 0u32;
    let mut black_agent_wins = 0u32;

    // Game-dynamics aggregates.
    let mut win_by_cap = 0u32;
    let mut win_by_eq = 0u32;
    let mut win_by_draw = 0u32;
    let mut total_winner_eq_left: u32 = 0;
    let mut total_loser_eq_left: u32 = 0;
    let mut total_captures: u32 = 0;
    let mut first_capture_plies: Vec<u32> = Vec::new();

    let t0 = Instant::now();
    for g in 0..n {
        let seed_w = 0xC0FFEE ^ (g as u64) << 1;
        let seed_b = 0xBADC0DE ^ (g as u64) << 1;

        // If swapping, even games have white_spec as Clair; odd games have
        // black_spec as Clair. Track which agent won regardless of color.
        let (clair_spec, fonce_spec, white_is_clair) = if swap && g % 2 == 1 {
            (&black_spec, &white_spec, false)
        } else {
            (&white_spec, &black_spec, true)
        };

        let mut clair = parse_player(clair_spec, seed_w);
        let mut fonce = parse_player(fonce_spec, seed_b);

        let rec = play_game(clair.as_mut(), fonce.as_mut());
        total_plies += rec.plies;
        match rec.outcome {
            Outcome::Win(Color::Clair) => {
                clair_wins += 1;
                if white_is_clair { white_agent_wins += 1; } else { black_agent_wins += 1; }
            }
            Outcome::Win(Color::Fonce) => {
                fonce_wins += 1;
                if white_is_clair { black_agent_wins += 1; } else { white_agent_wins += 1; }
            }
            Outcome::Draw => draws += 1,
        }
        match rec.win_type {
            WinType::CapitaineCaptured => win_by_cap += 1,
            WinType::EquipiersEliminated => win_by_eq += 1,
            WinType::DrawCounter | WinType::NoMove => win_by_draw += 1,
        }
        let (clair_alive, fonce_alive) = rec.final_alive;
        let (winner_eq, loser_eq) = match rec.outcome {
            Outcome::Win(Color::Clair) => (clair_alive.1, fonce_alive.1),
            Outcome::Win(Color::Fonce) => (fonce_alive.1, clair_alive.1),
            Outcome::Draw => (clair_alive.1, fonce_alive.1),
        };
        total_winner_eq_left += winner_eq;
        total_loser_eq_left += loser_eq;
        total_captures += rec.capture_plies.len() as u32;
        if let Some(&p) = rec.capture_plies.first() {
            first_capture_plies.push(p);
        }

        let label = match rec.outcome {
            Outcome::Win(Color::Clair) => format!("W({})", clair.name()),
            Outcome::Win(Color::Fonce) => format!("B({})", fonce.name()),
            Outcome::Draw => "Draw".to_string(),
        };
        let win_tag = match rec.win_type {
            WinType::CapitaineCaptured => "cap",
            WinType::EquipiersEliminated => "eq",
            WinType::DrawCounter => "draw",
            WinType::NoMove => "stall",
        };
        println!(
            "  game {:>3}: {:>5} plies, {:<3}, {:>2} captures, end {} {} v {} {} ({})",
            g + 1,
            rec.plies,
            win_tag,
            rec.capture_plies.len(),
            clair_alive.0,
            clair_alive.1,
            fonce_alive.0,
            fonce_alive.1,
            label
        );
    }
    let elapsed = t0.elapsed().as_secs_f64();

    println!();
    println!("By color:");
    println!("  Clair wins: {clair_wins:>3} / {n}");
    println!("  Fonce wins: {fonce_wins:>3} / {n}");
    println!("  draws     : {draws:>3} / {n}");
    if swap {
        println!();
        println!("By agent (color-corrected):");
        println!("  {white_spec:>10}: {white_agent_wins:>3} / {n}");
        println!("  {black_spec:>10}: {black_agent_wins:>3} / {n}");
        println!("  draws     : {draws:>3} / {n}");
    }
    println!();
    println!("Win type:");
    println!("  capitaine captured  : {win_by_cap:>3} / {n}");
    println!("  equipiers eliminated: {win_by_eq:>3} / {n}");
    println!("  draw                : {win_by_draw:>3} / {n}");
    println!();
    println!("Game dynamics:");
    println!("  avg plies/game            : {:.1}", total_plies as f64 / n as f64);
    println!("  avg captures/game         : {:.2}", total_captures as f64 / n as f64);
    if !first_capture_plies.is_empty() {
        let avg_first = first_capture_plies.iter().sum::<u32>() as f64 / first_capture_plies.len() as f64;
        println!("  avg ply of first capture  : {avg_first:.1}");
    }
    println!(
        "  avg winner equipiers left : {:.2} / 9",
        total_winner_eq_left as f64 / n as f64
    );
    println!(
        "  avg loser  equipiers left : {:.2} / 9",
        total_loser_eq_left as f64 / n as f64
    );
    println!();
    println!("Wall clock: {elapsed:.2}s ({:.2}s/game)", elapsed / n as f64);
}
