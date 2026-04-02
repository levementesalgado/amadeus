use crate::tensor::ops::matmul;
use crate::tensor::Tensor;

pub struct TraceSediment {
    pub projection: Tensor,
    pub base_lr: f32,
}

impl TraceSediment {
    pub fn new(n_embd: usize, d_affect: usize, d_trace: usize, rng: &mut fastrand::Rng) -> Self {
        Self {
            projection: super::rng::init_linear(n_embd + d_affect, d_trace, rng),
            base_lr: 0.01,
        }
    }

    pub fn update(&self, trace: &mut [f32], mood: &[f32], hidden: &[f32]) {
        let cat_len = hidden.len() + mood.len();
        let mut cat = Vec::with_capacity(cat_len);
        cat.extend_from_slice(hidden);
        cat.extend_from_slice(mood);
        let cat_t = Tensor::from_vec(cat, vec![1, cat_len]);
        let delta = matmul(&self.projection, &cat_t);

        let intensity: f32 = mood.iter().map(|v| v * v).sum::<f32>().sqrt() / (mood.len() as f32).sqrt();
        let lr = self.base_lr * (1.0 + intensity);

        for i in 0..trace.len() {
            trace[i] += lr * delta.as_slice()[i];
            trace[i] = trace[i].clamp(-5.0, 5.0);
        }
    }
}
