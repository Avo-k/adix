//! Trainer: one gradient step on a batch of [`Sample`] records.
//!
//! Loss = **policy cross-entropy** (target = MCTS visit distribution,
//! prediction = network policy softmax) + **value MSE** (target =
//! eventual game outcome z ∈ {-1, 0, +1}, prediction = network value).
//!
//! The standard AZ loss also adds an L2 weight-decay term. We get
//! that "for free" by enabling weight decay on the optimizer instead
//! of folding it into the forward pass — see [`make_optimizer`].
//!
//! No explicit legal-move mask on the policy head during training: the
//! visit-count targets only have mass on legal moves (zero elsewhere),
//! so illegal logits contribute via the softmax denominator only. The
//! network learns to suppress illegal moves implicitly. Masking
//! could speed convergence — TODO once we have a baseline.

use tch::{Kind, Tensor};

use super::encoding::{ACTIONS, BOARD_H, BOARD_W, INPUT_PLANES, INPUT_SIZE};
use super::net::AzNet;
use super::selfplay::Sample;

pub struct Trainer {
    pub net: AzNet,
    pub optimizer: tch::nn::Optimizer,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LossStats {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub total_loss: f32,
}

impl Trainer {
    pub fn new(net: AzNet, optimizer: tch::nn::Optimizer) -> Self {
        Self { net, optimizer }
    }

    /// Run one Adam step on `batch`. Returns the per-component loss for
    /// logging. Both the forward pass and the gradient step happen on
    /// `self.net.device` (CPU or CUDA, picked at VarStore construction).
    pub fn train_step(&mut self, batch: &[Sample]) -> LossStats {
        assert!(!batch.is_empty(), "empty batch");
        let b = batch.len();
        let device = self.net.device;

        // Pack the batch into three flat buffers, then upload once.
        let mut states = Vec::with_capacity(b * INPUT_SIZE);
        let mut policies = Vec::with_capacity(b * ACTIONS);
        let mut values = Vec::with_capacity(b);
        for s in batch {
            states.extend_from_slice(&s.state);
            policies.extend_from_slice(&s.policy_target);
            values.push(s.value_target);
        }

        let states_t = Tensor::from_slice(&states)
            .view([b as i64, INPUT_PLANES as i64, BOARD_H as i64, BOARD_W as i64])
            .to_device(device);
        let policies_t = Tensor::from_slice(&policies)
            .view([b as i64, ACTIONS as i64])
            .to_device(device);
        let values_t = Tensor::from_slice(&values)
            .view([b as i64, 1])
            .to_device(device);

        let (logits, v_pred) = self.net.forward_t(&states_t, true);

        // Policy: mean over batch of -Σ_a π_target(a) · log softmax(logits)(a).
        let log_probs = logits.log_softmax(-1, Kind::Float);
        let policy_loss = -(policies_t * log_probs).sum(Kind::Float) / (b as f64);
        // Value: mean squared error on the scalar.
        let diff = v_pred - values_t;
        let value_loss = (&diff * &diff).mean(Kind::Float);

        let total = &policy_loss + &value_loss;
        self.optimizer.zero_grad();
        total.backward();
        self.optimizer.step();

        LossStats {
            policy_loss: policy_loss.double_value(&[]) as f32,
            value_loss: value_loss.double_value(&[]) as f32,
            total_loss: total.double_value(&[]) as f32,
        }
    }
}
