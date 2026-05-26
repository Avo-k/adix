//! AlphaZero training driver — batched self-play + replay buffer + checkpoints.
//!
//! Usage:
//! ```sh
//! # default smoke run: 3 iterations × 8 games × 16 parallel workers
//! cargo run --release --features tch --bin az_train
//!
//! # explicit positional args (any prefix length works, the rest take defaults):
//! cargo run --release --features tch --bin az_train -- \
//!     <iterations=3> <games/iter=8> <workers=16> \
//!     <puct_iters=100> <batch=64> <steps/iter=20>
//! ```
//!
//! Environment variables:
//! - `ADIX_AZ_FORCE_CPU=1` — pin to CPU even if CUDA is available.
//! - `ADIX_AZ_BUFFER=N` — replay buffer capacity (default 50 000).
//! - `ADIX_AZ_CKPT_DIR=path` — save the VarStore as `iter_NNNN.ot`
//!   under this directory after each iteration. No checkpointing if unset.
//! - `ADIX_AZ_LOAD=path.ot` — load this VarStore before the first
//!   iteration (resume from previous run).
//!
//! Each iteration:
//! 1. Run batched self-play to produce a fresh batch of games. Each
//!    network call is sized up to `n_workers` (this is the only place
//!    GPU vs CPU really matters).
//! 2. Push the new samples into the replay buffer (FIFO eviction).
//! 3. Run `steps_per_iter` gradient steps, each on a `batch_size`
//!    minibatch drawn uniformly at random from the buffer.
//! 4. Optionally checkpoint the VarStore.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use adix::agent::RandomPlayer;
use adix::az::{
    mcts::PuctConfig,
    net::{AzNet, DEFAULT_CHANNELS, DEFAULT_RES_BLOCKS, make_optimizer},
    replay::ReplayBuffer,
    selfplay::{SelfPlayConfig, play_batched},
    train::Trainer,
};
use adix::board::Outcome;
use adix::piece::Color;

use tch::{Device, nn};

struct Args {
    iterations: u32,
    games_per_iter: u32,
    n_workers: usize,
    puct_iters: u32,
    batch_size: usize,
    steps_per_iter: u32,
}

impl Args {
    fn parse() -> Self {
        let mut a = Self {
            iterations: 3,
            games_per_iter: 8,
            n_workers: 16,
            puct_iters: 100,
            batch_size: 64,
            steps_per_iter: 20,
        };
        let argv: Vec<String> = env::args().skip(1).collect();
        if let Some(v) = argv.first() {
            a.iterations = v.parse().expect("iterations: u32");
        }
        if let Some(v) = argv.get(1) {
            a.games_per_iter = v.parse().expect("games_per_iter: u32");
        }
        if let Some(v) = argv.get(2) {
            a.n_workers = v.parse().expect("n_workers: usize");
        }
        if let Some(v) = argv.get(3) {
            a.puct_iters = v.parse().expect("puct_iters: u32");
        }
        if let Some(v) = argv.get(4) {
            a.batch_size = v.parse().expect("batch_size: usize");
        }
        if let Some(v) = argv.get(5) {
            a.steps_per_iter = v.parse().expect("steps_per_iter: u32");
        }
        a
    }
}

fn pick_device() -> Device {
    if env::var("ADIX_AZ_FORCE_CPU").ok().as_deref() == Some("1") {
        return Device::Cpu;
    }
    force_load_libtorch_cuda();
    Device::cuda_if_available()
}

/// Force libtorch_cuda.so into the process so libtorch_cpu's runtime
/// CUDA detection can see it.
///
/// torch-sys's build script emits `-l torch_cuda`, but the modern Linux
/// linker drops it with `--as-needed` (default) because no Rust symbol
/// references it directly — libtorch_cpu loads CUDA dynamically at
/// runtime, *if it's already in the process*. Doing a manual dlopen
/// here keeps `cargo run` working without `LD_PRELOAD` gymnastics.
///
/// Failure is fine: on a CPU-only libtorch build the .so won't be found
/// and we just fall through to CPU.
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
        // SAFETY: dlopen with a static C string is safe; ignore the
        // returned handle. If the .so isn't there (CPU-only libtorch),
        // dlopen just returns null and we silently fall back to CPU.
        unsafe {
            dlopen(c"libtorch_cuda.so".as_ptr(), RTLD_NOW | RTLD_GLOBAL);
        }
    }
}

fn tally_outcomes(outcomes: &[Outcome]) -> (u32, u32, u32) {
    let mut w = 0;
    let mut d = 0;
    let mut b = 0;
    for &o in outcomes {
        match o {
            Outcome::Win(Color::Clair) => w += 1,
            Outcome::Draw => d += 1,
            Outcome::Win(Color::Fonce) => b += 1,
        }
    }
    (w, d, b)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let args = Args::parse();
    let device = pick_device();
    let buffer_cap = env_usize("ADIX_AZ_BUFFER", 50_000);
    let ckpt_dir: Option<PathBuf> = env::var("ADIX_AZ_CKPT_DIR").ok().map(PathBuf::from);
    let load_path: Option<PathBuf> = env::var("ADIX_AZ_LOAD").ok().map(PathBuf::from);

    println!(
        "device={:?} iter={} games/iter={} workers={} puct={} batch={} steps/iter={} buffer={}",
        device,
        args.iterations,
        args.games_per_iter,
        args.n_workers,
        args.puct_iters,
        args.batch_size,
        args.steps_per_iter,
        buffer_cap,
    );
    if let Some(p) = &ckpt_dir {
        println!("checkpoint dir: {}", p.display());
    }
    if let Some(p) = &load_path {
        println!("resume from: {}", p.display());
    }

    let mut vs = nn::VarStore::new(device);
    let net = AzNet::new(&vs.root(), DEFAULT_RES_BLOCKS, DEFAULT_CHANNELS);
    if let Some(p) = &load_path {
        vs.load(p).unwrap_or_else(|e| panic!("failed to load {}: {e}", p.display()));
        println!("loaded VarStore from {}", p.display());
    }
    let optimizer = make_optimizer(&vs, 1.0e-3);
    let mut trainer = Trainer::new(net, optimizer);

    if let Some(p) = &ckpt_dir {
        std::fs::create_dir_all(p)
            .unwrap_or_else(|e| panic!("failed to create checkpoint dir {}: {e}", p.display()));
    }

    let mut buffer = ReplayBuffer::new(buffer_cap);
    let mut rng = RandomPlayer::new(0xCAFEBABE);

    for it in 1..=args.iterations {
        // --- self-play phase --------------------------------------------
        let sp_start = Instant::now();
        let cfg = SelfPlayConfig {
            puct: PuctConfig {
                iterations: args.puct_iters,
                c_puct: 1.5,
                fpu_reduction: 0.2,
            },
            temperature_plies: 20,
            max_plies: 400,
            dirichlet_alpha: 0.2,
            dirichlet_eps: 0.25,
            augment_symmetry: true,
        };
        let res = play_batched(
            &trainer.net,
            &cfg,
            args.n_workers,
            args.games_per_iter as usize,
            &mut rng,
        );
        let sp_elapsed = sp_start.elapsed();
        let n_samples = res.samples.len();
        let avg_plies = if res.plies_per_game.is_empty() {
            0.0
        } else {
            res.plies_per_game.iter().sum::<u32>() as f32 / res.plies_per_game.len() as f32
        };
        let avg_batch = if res.batches > 0 {
            res.evaluated_positions as f64 / res.batches as f64
        } else {
            0.0
        };
        let (w, d, b) = tally_outcomes(&res.outcomes);
        println!(
            "iter {it} self-play: {games} games, avg plies={apl:.1}, {samples} samples in {ms} ms ({sps:.0} samples/s); batches={batches} avg_size={avg_batch:.1}; W/D/B = {w}/{d}/{b}",
            games = res.outcomes.len(),
            apl = avg_plies,
            samples = n_samples,
            ms = sp_elapsed.as_millis(),
            sps = n_samples as f64 / sp_elapsed.as_secs_f64().max(1e-6),
            batches = res.batches,
            avg_batch = avg_batch,
        );
        buffer.extend(res.samples);

        // --- training phase ---------------------------------------------
        if buffer.len() < args.batch_size {
            println!(
                "iter {it} train: buffer too small ({}/{}) — skipping",
                buffer.len(),
                args.batch_size,
            );
            continue;
        }
        let tr_start = Instant::now();
        let mut p_acc = 0.0_f32;
        let mut v_acc = 0.0_f32;
        for step in 1..=args.steps_per_iter {
            let batch = buffer.sample_batch(args.batch_size, &mut rng);
            let stats = trainer.train_step(&batch);
            p_acc += stats.policy_loss;
            v_acc += stats.value_loss;
            if step % 50 == 0 {
                println!(
                    "  iter {it} step {step}: policy={:.4} value={:.4}",
                    stats.policy_loss, stats.value_loss,
                );
            }
        }
        let tr_elapsed = tr_start.elapsed();
        let n = args.steps_per_iter.max(1) as f32;
        println!(
            "iter {it} train: {steps} steps in {ms} ms (buffer={buf}/{cap}), avg policy={:.4} avg value={:.4}",
            p_acc / n,
            v_acc / n,
            steps = args.steps_per_iter,
            ms = tr_elapsed.as_millis(),
            buf = buffer.len(),
            cap = buffer.capacity(),
        );

        // --- checkpoint --------------------------------------------------
        if let Some(dir) = &ckpt_dir {
            let p = dir.join(format!("iter_{it:04}.ot"));
            vs.save(&p)
                .unwrap_or_else(|e| eprintln!("checkpoint save failed at {}: {e}", p.display()));
            println!("iter {it} saved checkpoint: {}", p.display());
        }
    }
}
