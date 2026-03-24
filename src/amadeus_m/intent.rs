use crate::tensor::ops::{matmul, softmax_inplace};
use crate::tensor::Tensor;

pub struct Intentionality {
    pub classifier: Tensor,
    pub intent_embed: Tensor,
}

impl Intentionality {
    pub fn new(n_embd: usize, d_intent: usize, n_intents: usize, rng: &mut fastrand::Rng) -> Self {
        let classifier = super::rng::init_linear(n_embd, n_intents, rng);
        let intent_embed = super::rng::init_embedding(n_intents, d_intent, rng);
        Self { classifier, intent_embed }
    }

    pub fn classify(&self, hidden: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let x = Tensor::from_vec(hidden.to_vec(), shape_2d(hidden));
        let mut logits = matmul(&self.classifier, &x);
        softmax_inplace(&mut logits);
        let dist = logits.as_slice().to_vec();

        let n_intents = self.intent_embed.shape()[0];
        let d_intent = self.intent_embed.shape()[1];

        let mut intent_emb = vec![0.0; d_intent];
        for i in 0..n_intents {
            let w = dist[i];
            let start = i * d_intent;
            let slice = &self.intent_embed.as_slice()[start..start + d_intent];
            for j in 0..d_intent {
                intent_emb[j] += w * slice[j];
            }
        }
        (dist, intent_emb)
    }
}

fn shape_2d(hidden: &[f32]) -> Vec<usize> {
    vec![1, hidden.len()]
}
