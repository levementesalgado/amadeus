pub struct AmadeusMConfig {
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_intermediate: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub window_size: usize,
    pub d_affect: usize,
    pub d_intent: usize,
    pub d_trace: usize,
    pub n_intents: usize,
}

impl AmadeusMConfig {
    pub fn small() -> Self {
        let n_embd = 128;
        let head_dim = 32;
        Self {
            n_layers: 4,
            n_embd,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim,
            n_intermediate: 512,
            vocab_size: 256,
            max_seq_len: 2048,
            window_size: 256,
            d_affect: 16,
            d_intent: 16,
            d_trace: 64,
            n_intents: 8,
        }
    }

    pub fn medium() -> Self {
        let n_embd = 512;
        let head_dim = 64;
        Self {
            n_layers: 8,
            n_embd,
            n_heads: 8,
            n_kv_heads: 4,
            head_dim,
            n_intermediate: 2048,
            vocab_size: 32000,
            max_seq_len: 8192,
            window_size: 1024,
            d_affect: 32,
            d_intent: 32,
            d_trace: 128,
            n_intents: 8,
        }
    }

    pub fn n_gqa(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}
