//! Policy/value network for ADIX AlphaZero.
//!
//! Standard AZ architecture, downsized for a 9×9 board with no human
//! prior to bootstrap from:
//!
//! - Stem: 3×3 conv → BN → ReLU (input planes → `channels`).
//! - Tower: `num_res_blocks` residual blocks (conv-BN-ReLU-conv-BN +
//!   skip → ReLU).
//! - Policy head: 1×1 conv to 2 channels → BN → ReLU → linear to
//!   `ACTIONS` logits.
//! - Value head: 1×1 conv to 1 channel → BN → ReLU → linear to 64 →
//!   ReLU → linear to 1 → tanh.
//!
//! Forward returns `(policy_logits[B, ACTIONS], value[B, 1])`. The
//! [`forward_board`] convenience wraps a single position end-to-end:
//! `&Board` → encoded tensor → forward → softmax over legal moves +
//! scalar value.

use tch::{
    Device, Kind, Tensor,
    nn::{self, OptimizerConfig},
};

use crate::board::Board;

use super::encoding::{
    ACTIONS, BOARD_H, BOARD_W, INPUT_PLANES, INPUT_SIZE, encode_state,
};

// --- defaults -------------------------------------------------------------

/// Default trunk width. Modest; bump once we have something training.
pub const DEFAULT_CHANNELS: i64 = 64;

/// Default residual-block count.
pub const DEFAULT_RES_BLOCKS: i64 = 6;

// --- building blocks ------------------------------------------------------

fn conv3(p: &nn::Path, in_c: i64, out_c: i64) -> nn::Conv2D {
    nn::conv2d(
        p,
        in_c,
        out_c,
        3,
        nn::ConvConfig { padding: 1, bias: false, ..Default::default() },
    )
}

fn conv1(p: &nn::Path, in_c: i64, out_c: i64) -> nn::Conv2D {
    nn::conv2d(
        p,
        in_c,
        out_c,
        1,
        nn::ConvConfig { padding: 0, bias: false, ..Default::default() },
    )
}

struct ResBlock {
    c1: nn::Conv2D,
    bn1: nn::BatchNorm,
    c2: nn::Conv2D,
    bn2: nn::BatchNorm,
}

impl ResBlock {
    fn new(p: &nn::Path, channels: i64) -> Self {
        Self {
            c1: conv3(&(p / "c1"), channels, channels),
            bn1: nn::batch_norm2d(p / "bn1", channels, Default::default()),
            c2: conv3(&(p / "c2"), channels, channels),
            bn2: nn::batch_norm2d(p / "bn2", channels, Default::default()),
        }
    }

    fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        let h = x.apply(&self.c1).apply_t(&self.bn1, train).relu();
        let h = h.apply(&self.c2).apply_t(&self.bn2, train);
        (h + x).relu()
    }
}

// --- top-level net --------------------------------------------------------

pub struct AzNet {
    // stem
    stem_conv: nn::Conv2D,
    stem_bn: nn::BatchNorm,
    // tower
    blocks: Vec<ResBlock>,
    // policy head
    p_conv: nn::Conv2D,
    p_bn: nn::BatchNorm,
    p_fc: nn::Linear,
    // value head
    v_conv: nn::Conv2D,
    v_bn: nn::BatchNorm,
    v_fc1: nn::Linear,
    v_fc2: nn::Linear,
    // bookkeeping
    pub device: Device,
}

impl AzNet {
    pub fn new(vs: &nn::Path, num_res_blocks: i64, channels: i64) -> Self {
        let device = vs.device();
        let board_area = (BOARD_W * BOARD_H) as i64;
        let stem_conv = conv3(&(vs / "stem_conv"), INPUT_PLANES as i64, channels);
        let stem_bn = nn::batch_norm2d(vs / "stem_bn", channels, Default::default());
        let blocks: Vec<ResBlock> = (0..num_res_blocks)
            .map(|i| ResBlock::new(&(vs / format!("res{i}")), channels))
            .collect();
        // policy head: 2 channels then a big linear into ACTIONS.
        let p_conv = conv1(&(vs / "p_conv"), channels, 2);
        let p_bn = nn::batch_norm2d(vs / "p_bn", 2, Default::default());
        let p_fc = nn::linear(
            vs / "p_fc",
            2 * board_area,
            ACTIONS as i64,
            Default::default(),
        );
        // value head: 1 channel → 64 hidden → 1 scalar.
        let v_conv = conv1(&(vs / "v_conv"), channels, 1);
        let v_bn = nn::batch_norm2d(vs / "v_bn", 1, Default::default());
        let v_fc1 = nn::linear(vs / "v_fc1", board_area, 64, Default::default());
        let v_fc2 = nn::linear(vs / "v_fc2", 64, 1, Default::default());
        Self {
            stem_conv,
            stem_bn,
            blocks,
            p_conv,
            p_bn,
            p_fc,
            v_conv,
            v_bn,
            v_fc1,
            v_fc2,
            device,
        }
    }

    /// Default-sized constructor: 6 residual blocks, 64 channels.
    pub fn new_default(vs: &nn::Path) -> Self {
        Self::new(vs, DEFAULT_RES_BLOCKS, DEFAULT_CHANNELS)
    }

    /// Forward pass on a batched input tensor of shape
    /// `[B, INPUT_PLANES, 9, 9]`. Returns `(policy_logits[B, ACTIONS],
    /// value[B, 1])`. `train` toggles BatchNorm behavior; pass `false`
    /// for inference (MCTS), `true` during training steps.
    pub fn forward_t(&self, x: &Tensor, train: bool) -> (Tensor, Tensor) {
        // Trunk.
        let mut h = x.apply(&self.stem_conv).apply_t(&self.stem_bn, train).relu();
        for blk in &self.blocks {
            h = blk.forward_t(&h, train);
        }
        // Policy.
        let p = h.apply(&self.p_conv).apply_t(&self.p_bn, train).relu();
        let p = p.flatten(1, -1).apply(&self.p_fc);
        // Value.
        let v = h.apply(&self.v_conv).apply_t(&self.v_bn, train).relu();
        let v = v.flatten(1, -1).apply(&self.v_fc1).relu().apply(&self.v_fc2).tanh();
        (p, v)
    }

    /// Single-position convenience: encode `board`, forward, softmax
    /// over **legal** moves only (illegal logits masked to -inf), return
    /// the (`policy[ACTIONS]`, scalar `value`) for MCTS consumption.
    ///
    /// `value` is in `[-1, 1]` from the side-to-move's perspective: +1
    /// is "I expect to win from here", -1 is "I expect to lose".
    /// Batched single-pass forward over many boards. Same semantics as
    /// [`forward_board`] applied position-by-position, but the network
    /// is hit exactly once with a `[N, INPUT_PLANES, 9, 9]` input — the
    /// shape modern GPUs actually like. Used by the batched self-play
    /// loop, where each game contributes one leaf per tick.
    ///
    /// Returns a `Vec` of `(policy[ACTIONS], value)` tuples in the same
    /// order as the input slice.
    pub fn forward_boards(&self, boards: &[&Board]) -> Vec<(Vec<f32>, f32)> {
        if boards.is_empty() {
            return Vec::new();
        }
        let n = boards.len();

        // Pack inputs + legal-mask bias on the CPU.
        let mut state_buf = vec![0.0_f32; n * INPUT_SIZE];
        let mut bias_buf = vec![0.0_f32; n * ACTIONS];
        for (i, board) in boards.iter().enumerate() {
            let s_slice = &mut state_buf[i * INPUT_SIZE..(i + 1) * INPUT_SIZE];
            encode_state(board, s_slice);
            // legal mask → bias: 0 where legal, -1e9 where illegal.
            let mut mask = vec![0.0_f32; ACTIONS];
            super::encoding::fill_legal_mask(board, &mut mask);
            let b_slice = &mut bias_buf[i * ACTIONS..(i + 1) * ACTIONS];
            for (j, &m) in mask.iter().enumerate() {
                if m == 0.0 {
                    b_slice[j] = -1.0e9;
                }
            }
        }

        let _no_grad = tch::no_grad_guard();
        let x = Tensor::from_slice(&state_buf)
            .view([n as i64, INPUT_PLANES as i64, BOARD_H as i64, BOARD_W as i64])
            .to_device(self.device);
        let (logits, value) = self.forward_t(&x, false);
        let bias = Tensor::from_slice(&bias_buf)
            .view([n as i64, ACTIONS as i64])
            .to_device(self.device);
        let masked = logits + bias;
        let probs = masked.softmax(-1, Kind::Float).to_device(Device::Cpu);
        let value_cpu = value.view([n as i64]).to_device(Device::Cpu);

        let mut all_probs = vec![0.0_f32; n * ACTIONS];
        probs.copy_data(&mut all_probs, n * ACTIONS);
        let mut all_values = vec![0.0_f32; n];
        value_cpu.copy_data(&mut all_values, n);

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let policy = all_probs[i * ACTIONS..(i + 1) * ACTIONS].to_vec();
            out.push((policy, all_values[i]));
        }
        out
    }

    pub fn forward_board(&self, board: &Board) -> (Vec<f32>, f32) {
        // Build input + a precomputed log-space bias on the CPU: 0 for
        // legal entries, -1e9 for illegal ones. Adding this bias to the
        // logits before softmax zeros out illegal moves exactly. Doing
        // it CPU-side keeps the GPU path purely pointwise-add, robust
        // across tch versions.
        let mut buf = vec![0.0_f32; INPUT_SIZE];
        encode_state(board, &mut buf);
        let mut mask_buf = vec![0.0_f32; ACTIONS];
        super::encoding::fill_legal_mask(board, &mut mask_buf);
        let mut bias_buf = vec![0.0_f32; ACTIONS];
        for (i, &m) in mask_buf.iter().enumerate() {
            if m == 0.0 {
                bias_buf[i] = -1.0e9;
            }
        }

        let _no_grad = tch::no_grad_guard();
        let x = Tensor::from_slice(&buf)
            .view([1, INPUT_PLANES as i64, BOARD_H as i64, BOARD_W as i64])
            .to_device(self.device);
        let (logits, value) = self.forward_t(&x, false);

        let bias = Tensor::from_slice(&bias_buf)
            .view([1, ACTIONS as i64])
            .to_device(self.device);
        let masked = logits + bias;
        let probs = masked.softmax(-1, Kind::Float);

        let mut policy = vec![0.0_f32; ACTIONS];
        probs.to_device(Device::Cpu).copy_data(&mut policy, ACTIONS);
        let v = value.double_value(&[0, 0]) as f32;
        (policy, v)
    }
}

// --- training stub --------------------------------------------------------

/// Minimal Adam optimizer factory — the training loop itself is a follow-up
/// step (replay buffer, batches, loss = policy KL + value MSE).
pub fn make_optimizer(vs: &nn::VarStore, lr: f64) -> tch::nn::Optimizer {
    nn::Adam::default().build(vs, lr).expect("optimizer build")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az::encoding::{ACTIONS, fill_legal_mask};

    /// Forward-pass smoke test: builds an un-trained net, runs the
    /// initial position through it, and checks output shapes / sanity
    /// invariants (policy sums to ~1 over legal moves, illegal entries
    /// are ~0, value lies in [-1, 1]).
    #[test]
    fn forward_initial_position_shapes() {
        let vs = nn::VarStore::new(Device::Cpu);
        let net = AzNet::new(&vs.root(), 2, 16); // tiny net for speed
        let board = Board::initial();
        let (policy, value) = net.forward_board(&board);
        assert_eq!(policy.len(), ACTIONS);

        let sum: f32 = policy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "policy should sum to 1, got {sum}"
        );

        let mut mask = vec![0.0_f32; ACTIONS];
        fill_legal_mask(&board, &mut mask);
        for i in 0..ACTIONS {
            if mask[i] == 0.0 {
                assert!(
                    policy[i] < 1e-6,
                    "illegal index {i} has non-zero prob {}",
                    policy[i]
                );
            }
        }
        assert!(
            (-1.0..=1.0).contains(&value),
            "value out of tanh range: {value}"
        );
    }
}
