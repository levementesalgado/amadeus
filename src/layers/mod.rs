pub mod attention;
pub mod ffn;
pub mod rope;

use crate::layers::attention::Attention;
use crate::layers::ffn::Ffn;
use crate::model::config::ModelConfig;
use crate::model::KvCache;
use crate::tensor::ops::{add_inplace, rms_norm_inplace};
use crate::tensor::Tensor;

/// One transformer decoder layer:
///   x = x + attention(rms_norm(x))
///   x = x + ffn(rms_norm(x))
pub struct TransformerLayer {
    pub attention: Attention,
    pub ffn: Ffn,
    pub attention_norm: Tensor,
    pub ffn_norm: Tensor,
    pub norm_eps: f32,
}

impl TransformerLayer {
    pub fn new(config: &ModelConfig) -> Self {
        let n_embd = config.n_embd;
        Self {
            attention: Attention::new(config),
            ffn: Ffn::new(config),
            attention_norm: Tensor::zeros(vec![n_embd]),
            ffn_norm: Tensor::zeros(vec![n_embd]),
            norm_eps: config.norm_eps,
        }
    }

    pub fn forward(&self, x: &mut Tensor, kv_cache: &mut KvCache, pos: usize) {
        // pre-attention norm
        let residual = x.clone();
        rms_norm_inplace(x, self.attention_norm.as_slice(), self.norm_eps);
        self.attention.forward(x, kv_cache, pos);
        add_inplace(x, &residual);

        // pre-ffn norm
        let residual = x.clone();
        rms_norm_inplace(x, self.ffn_norm.as_slice(), self.norm_eps);
        self.ffn.forward(x);
        add_inplace(x, &residual);
    }
}
