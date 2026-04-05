use crate::tensor::ops::{matmul, rms_norm_inplace};
use crate::tensor::Tensor;

pub struct SynthesisLayer {
    pub w_gate: Tensor,
    pub w_mood: Tensor,
    pub w_intent: Tensor,
    pub w_ffn_gate: Tensor,
    pub w_ffn_up: Tensor,
    pub w_ffn_down: Tensor,
    pub norm_weight: Tensor,
}

impl SynthesisLayer {
    pub fn new(n_embd: usize, d_affect: usize, d_intent: usize, n_intermediate: usize, rng: &mut fastrand::Rng) -> Self {
        Self {
            w_gate: super::rng::init_linear(n_embd * 2, 3, rng),
            w_mood: super::rng::init_linear(d_affect, n_embd, rng),
            w_intent: super::rng::init_linear(d_intent, n_embd, rng),
            w_ffn_gate: super::rng::init_linear(n_embd, n_intermediate, rng),
            w_ffn_up: super::rng::init_linear(n_embd, n_intermediate, rng),
            w_ffn_down: super::rng::init_linear(n_intermediate, n_embd, rng),
            norm_weight: super::rng::init_rms_weight(n_embd),
        }
    }

    pub fn forward(
        &self,
        hidden: &mut [f32],
        residual: &[f32],
        habit: &[f32],
        recollection: &[f32],
        mood: &[f32],
        intent_emb: &[f32],
    ) {
        let n_embd = hidden.len();

        let mut cat = Vec::with_capacity(n_embd * 2);
        cat.extend_from_slice(hidden);
        cat.extend_from_slice(habit);
        let cat_t = Tensor::from_vec(cat, vec![1, n_embd * 2]);
        let mut gate_logits = matmul(&self.w_gate, &cat_t);
        for v in gate_logits.as_mut_slice().iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
        let gates = gate_logits.as_slice().to_vec();

        let mut combined = vec![0.0; n_embd];
        for i in 0..n_embd {
            combined[i] = gates[0] * hidden[i] + gates[1] * habit[i] + gates[2] * recollection[i];
        }

        let mood_bias_t = matmul(&self.w_mood, &Tensor::from_vec(mood.to_vec(), vec![1, mood.len()]));
        let intent_bias_t = matmul(&self.w_intent, &Tensor::from_vec(intent_emb.to_vec(), vec![1, intent_emb.len()]));
        let mood_bias = mood_bias_t.as_slice().to_vec();
        let intent_bias = intent_bias_t.as_slice().to_vec();
        for i in 0..n_embd {
            combined[i] += mood_bias[i] + intent_bias[i];
        }

        let mut combined_t = Tensor::from_vec(combined, vec![1, n_embd]);
        rms_norm_inplace(&mut combined_t, self.norm_weight.as_slice(), 1e-6);
        let normed = combined_t.as_slice().to_vec();

        let mut gate_ffn = matmul(&self.w_ffn_gate, &Tensor::from_vec(normed.clone(), vec![1, n_embd]));
        for v in gate_ffn.as_mut_slice().iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
        let up = matmul(&self.w_ffn_up, &Tensor::from_vec(normed, vec![1, n_embd]));
        let gated_len = gate_ffn.as_slice().len();
        let mut gated = gate_ffn.as_slice().to_vec();
        for i in 0..gated_len {
            gated[i] *= up.as_slice()[i];
        }
        let gated_t = Tensor::from_vec(gated, vec![1, gated_len]);
        let down = matmul(&self.w_ffn_down, &gated_t);

        for i in 0..n_embd {
            hidden[i] = residual[i] + down.as_slice()[i];
        }
    }
}


