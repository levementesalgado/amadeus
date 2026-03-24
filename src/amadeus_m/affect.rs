use crate::tensor::Tensor;
use crate::tensor::ops::matmul;

pub struct AffectModule {
    pub mood: Vec<f32>,
    pub w_in: Tensor,
    pub b_in: Tensor,
    pub w_out: Tensor,
    pub b_out: Tensor,
    pub alpha: f32,
}

impl AffectModule {
    pub fn new(d_affect: usize, n_embd: usize, rng: &mut fastrand::Rng) -> Self {
        let w_in = super::rng::init_linear(d_affect + n_embd, d_affect * 2, rng);
        let b_in = super::rng::init_bias(d_affect * 2, rng);
        let w_out = super::rng::init_linear(d_affect * 2, d_affect, rng);
        let b_out = super::rng::init_bias(d_affect, rng);
        Self {
            mood: vec![0.0; d_affect],
            w_in,
            b_in,
            w_out,
            b_out,
            alpha: 0.9,
        }
    }

    pub fn evolve(&mut self, hidden: &[f32]) {
        let d = self.mood.len();
        let mut input = Vec::with_capacity(d + hidden.len());
        input.extend_from_slice(&self.mood);
        input.extend_from_slice(hidden);

        let input_t = Tensor::from_vec(input, vec![1, d + hidden.len()]);
        let mut h = matmul(&self.w_in, &input_t);
        for (i, v) in h.as_mut_slice().iter_mut().enumerate() {
            *v += self.b_in.as_slice()[i];
            *v = v.tanh();
        }
        let mut delta = matmul(&self.w_out, &h);
        for (i, v) in delta.as_mut_slice().iter_mut().enumerate() {
            *v += self.b_out.as_slice()[i];
            *v = v.tanh() * 0.1;
        }

        for i in 0..d {
            self.mood[i] = self.alpha * self.mood[i] + (1.0 - self.alpha) * delta.as_slice()[i];
            self.mood[i] = self.mood[i].clamp(-3.0, 3.0);
        }
    }

    pub fn intensity(&self) -> f32 {
        let sum_sq: f32 = self.mood.iter().map(|v| v * v).sum();
        (sum_sq / self.mood.len() as f32).sqrt()
    }

    pub fn modulate(&self, x: &mut [f32]) {
        let d = self.mood.len();
        if x.len() == d {
            for i in 0..d {
                x[i] *= 1.0 + 0.1 * self.mood[i];
            }
        }
    }
}
