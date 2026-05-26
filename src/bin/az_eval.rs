//! Evaluation tournament: PUCT-with-checkpoint vs alpha-beta baseline.
//!
//! Loads a trained AzNet checkpoint and plays a head-to-head series
//! against [`AlphaBetaPlayer`] (the existing classical agent in
//! [`adix::agent`]). Colors swap every other game so the result isn't
//! biased by who moves first.
//!
//! Usage:
//! ```sh
//! cargo run --release --features tch --bin az_eval -- \
//!     <ckpt.ot> [games=20] [puct_iters=400] [ab_depth=3]
//! ```
//!
//! Output: per-game line + final W/D/L tally for the AZ side.
//!
//! This is *the* progress metric for the training loop — the loss
//! numbers from `az_train` only tell us we're optimizing something;
//! beating ab:3 tells us we're optimizing the right thing.

use std::env;
use std::path::Path;
use std::time::Instant;

use adix::agent::{AlphaBetaPlayer, play_game};
use adix::az::{
    mcts::{PuctConfig, PuctPlayer},
    net::{AzNet, DEFAULT_CHANNELS, DEFAULT_RES_BLOCKS},
};
use adix::board::Outcome;
use adix::piece::Color;

use tch::{Device, nn};

fn pick_device() -> Device {
    if env::var("ADIX_AZ_FORCE_CPU").ok().as_deref() == Some("1") {
        return Device::Cpu;
    }
    force_load_libtorch_cuda();
    Device::cuda_if_available()
}

fn force_load_libtorch_cuda() {
    #[cfg(unix)]
    {
        const RTLD_NOW: std::os::raw::c_int = 2;
        const RTLD_GLOBAL: std::os::raw::c_int = 0x100;
        unsafe extern "C" {
            fn dlopen(
                filename: *const std::os::raw::c_char,
                flag: std::os::raw::c_int,
            ) -> *mut std::os::raw::c_void;
        }
        unsafe {
            dlopen(c"libtorch_cuda.so".as_ptr(), RTLD_NOW | RTLD_GLOBAL);
        }
    }
}

fn outcome_for_az(outcome: Outcome, az_color: Color) -> &'static str {
    match outcome {
        Outcome::Win(c) if c == az_color => "W",
        Outcome::Win(_) => "L",
        Outcome::Draw => "D",
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: az_eval <ckpt.ot> [games=20] [puct_iters=400] [ab_depth=3]");
        std::process::exit(1);
    }
    let ckpt_path = Path::new(&args[0]).to_path_buf();
    let games: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let puct_iters: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(400);
    let ab_depth: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    let device = pick_device();
    println!(
        "device={:?} ckpt={} games={games} puct={puct_iters} ab_depth={ab_depth}",
        device,
        ckpt_path.display(),
    );

    let mut vs = nn::VarStore::new(device);
    let net = AzNet::new(&vs.root(), DEFAULT_RES_BLOCKS, DEFAULT_CHANNELS);
    vs.load(&ckpt_path)
        .unwrap_or_else(|e| panic!("failed to load {}: {e}", ckpt_path.display()));

    // Evaluation runs with eps=0 (no Dirichlet) — we want the net's
    // best deterministic play, not exploration noise. The PUCT itself
    // remains stochastic only via the tie-break random; argmax visits
    // dominates at this iteration count.
    let cfg = PuctConfig { iterations: puct_iters, c_puct: 1.5, fpu_reduction: 0.2 };

    let mut az_w = 0u32;
    let mut az_l = 0u32;
    let mut draws = 0u32;
    let total_start = Instant::now();

    for g in 0..games {
        let swap = g % 2 == 1;
        let mut puct = PuctPlayer::new(&net, cfg, 0xBEEF_u64 ^ g as u64);
        let mut ab = AlphaBetaPlayer::new(ab_depth, 0xDEAD_u64 ^ g as u64);
        let game_start = Instant::now();
        let rec = if swap {
            play_game(&mut ab, &mut puct)
        } else {
            play_game(&mut puct, &mut ab)
        };
        let az_color = if swap { Color::Fonce } else { Color::Clair };
        let tag = outcome_for_az(rec.outcome, az_color);
        match tag {
            "W" => az_w += 1,
            "L" => az_l += 1,
            "D" => draws += 1,
            _ => unreachable!(),
        }
        println!(
            "game {g} (AZ={}): {} plies, outcome={}, {} ms",
            if swap { "B" } else { "W" },
            rec.plies,
            tag,
            game_start.elapsed().as_millis(),
        );
    }

    let total_elapsed = total_start.elapsed();
    let win_rate = 100.0 * az_w as f32 / games as f32;
    let non_loss = 100.0 * (az_w + draws) as f32 / games as f32;
    println!(
        "\nPUCT(n={puct_iters}) vs ab:{ab_depth} on {games} games in {:.1}s — W/D/L = {az_w}/{draws}/{az_l} (win {win_rate:.1}%, non-loss {non_loss:.1}%)",
        total_elapsed.as_secs_f32(),
    );
}
