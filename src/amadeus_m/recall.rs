use crate::tensor::ops::matmul;
use crate::tensor::Tensor;

pub struct PureRecollection {
    pub trace_proj: Tensor,
    pub query_proj: Tensor,
    pub decoder: Tensor,
}

impl PureRecollection {
    pub fn new(n_embd: usize, d_affect: usize, d_intent: usize, d_trace: usize, d_hidden: usize, rng: &mut fastrand::Rng) -> Self {
        Self {
            trace_proj: super::rng::init_linear(d_trace, d_hidden, rng),
            query_proj: super::rng::init_linear(d_affect + d_intent + n_embd, d_hidden, rng),
            decoder: super::rng::init_linear(d_hidden, n_embd, rng),
        }
    }

    pub fn reconstruct(&self, hidden: &[f32], mood: &[f32], intent_emb: &[f32], trace: &[f32]) -> Vec<f32> {
        let d_hidden = self.query_proj.shape()[0];

        let qlen = mood.len() + intent_emb.len() + hidden.len();
        let mut query_vec = Vec::with_capacity(qlen);
        query_vec.extend_from_slice(mood);
        query_vec.extend_from_slice(intent_emb);
        query_vec.extend_from_slice(hidden);
        let query_t = Tensor::from_vec(query_vec, vec![1, qlen]);

        let query = matmul(&self.query_proj, &query_t);
        let mut q_slice = query.as_slice().to_vec();

        let trace_t = Tensor::from_vec(trace.to_vec(), vec![1, trace.len()]);
        let trace_ctx = matmul(&self.trace_proj, &trace_t);
        let t_slice = trace_ctx.as_slice();

        for i in 0..d_hidden {
            q_slice[i] = (q_slice[i] * t_slice[i]).tanh();
        }

        let combined = Tensor::from_vec(q_slice, vec![1, d_hidden]);
        let recollection = matmul(&self.decoder, &combined);
        recollection.as_slice().to_vec()
    }
}
