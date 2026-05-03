pub mod config;
pub mod loader;

use crate::layers::TransformerLayer;
use crate::tensor::ops::{matmul, rms_norm_inplace};
use crate::tensor::Tensor;

pub struct Model {
    pub config: config::ModelConfig,
    pub layers: Vec<TransformerLayer>,
    pub tok_embeddings: Tensor,
    pub output_weight: Tensor,
    pub norm: Tensor,
    pub norm_eps: f32,
    pub kv_caches: Vec<KvCache>,
}

pub struct KvCache {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl KvCache {
    pub fn new(max_seq_len: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let cap = max_seq_len * n_kv_heads * head_dim;
        Self {
            keys: vec![0.0; cap],
            values: vec![0.0; cap],
            seq_len: 0,
            max_seq_len,
            n_kv_heads,
            head_dim,
        }
    }

    pub fn key_data(&mut self) -> &mut [f32] {
        &mut self.keys
    }

    pub fn value_data(&mut self) -> &mut [f32] {
        &mut self.values
    }

    pub fn kv_len(&self) -> usize {
        self.seq_len * self.n_kv_heads * self.head_dim
    }
}

impl Model {
    pub fn forward(&mut self, x: &mut Tensor, pos: usize) {
        for (layer, cache) in self.layers.iter().zip(self.kv_caches.iter_mut()) {
            layer.forward(x, cache, pos);
        }
    }

    pub fn forward_token(&mut self, token: u32, pos: usize) -> Tensor {
        let n_embd = self.config.n_embd;
        let idx = token as usize;
        let start = idx * n_embd;
        let end = start + n_embd;
        let emb_slice = &self.tok_embeddings.as_slice()[start..end];
        let mut x = Tensor::from_vec(emb_slice.to_vec(), vec![1, n_embd]);

        self.forward(&mut x, pos);

        rms_norm_inplace(&mut x, self.norm.as_slice(), self.norm_eps);

        matmul(&self.output_weight, &x)
    }
}
