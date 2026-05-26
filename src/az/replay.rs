//! FIFO replay buffer for AlphaZero training.
//!
//! Self-play emits samples bursts (one whole game's worth at a time);
//! training consumes uniform random minibatches. The buffer bridges
//! the two: a sliding window of the most recent N samples, with
//! uniform-random sampling for batch construction.
//!
//! Capacity is set per the usual AZ heuristic: a few times the average
//! number of samples produced per training iteration, so training sees
//! several "epochs"-worth of fresh data per gradient step.

use std::collections::VecDeque;

use crate::agent::RandomPlayer;

use super::selfplay::Sample;

pub struct ReplayBuffer {
    samples: VecDeque<Sample>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self { samples: VecDeque::with_capacity(capacity), capacity }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Push samples; older ones drop out FIFO once we exceed capacity.
    pub fn extend(&mut self, samples: Vec<Sample>) {
        for s in samples {
            if self.samples.len() == self.capacity {
                self.samples.pop_front();
            }
            self.samples.push_back(s);
        }
    }

    /// Sample `batch_size` minibatch items uniformly at random (with
    /// replacement — typical for AZ, simpler than reservoir sampling
    /// and the duplicates are statistically harmless at our scale).
    pub fn sample_batch(&self, batch_size: usize, rng: &mut RandomPlayer) -> Vec<Sample> {
        if self.samples.is_empty() {
            return Vec::new();
        }
        let n = self.samples.len();
        (0..batch_size)
            .map(|_| {
                let idx = (rng.next_u64() % n as u64) as usize;
                self.samples[idx].clone()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az::encoding::{ACTIONS, INPUT_SIZE};

    fn mk(value: f32) -> Sample {
        Sample {
            state: vec![0.0; INPUT_SIZE],
            policy_target: vec![0.0; ACTIONS],
            value_target: value,
        }
    }

    #[test]
    fn fifo_eviction_keeps_most_recent() {
        let mut buf = ReplayBuffer::new(3);
        buf.extend(vec![mk(1.0), mk(2.0), mk(3.0), mk(4.0)]);
        assert_eq!(buf.len(), 3);
        // First sample (1.0) should have been evicted.
        let values: Vec<f32> = buf.samples.iter().map(|s| s.value_target).collect();
        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn sample_batch_respects_size_and_picks_from_buffer() {
        let mut buf = ReplayBuffer::new(10);
        buf.extend((0..5).map(|i| mk(i as f32)).collect());
        let mut rng = RandomPlayer::new(42);
        let batch = buf.sample_batch(7, &mut rng);
        assert_eq!(batch.len(), 7);
        for s in batch {
            assert!((0.0..=4.0).contains(&s.value_target));
        }
    }
}
