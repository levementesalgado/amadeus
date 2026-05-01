/// Sampling strategies for token generation.
#[derive(Debug, Clone, Copy)]
pub enum SamplerKind {
    Greedy,
    Temperature { temp: f32 },
    TopK { k: usize },
    TopP { p: f32 },
    Mirostat { tau: f32, rate: f32 },
}

/// Token sampler with state (for mirostat).
#[allow(unused)]
pub struct Sampler {
    pub kind: SamplerKind,
    rng: fastrand::Rng,
    mirostat_mu: f32,
    recent: Vec<u32>,
    max_recent: usize,
}

impl Sampler {
    pub fn new(kind: SamplerKind) -> Self {
        Self {
            kind,
            rng: fastrand::Rng::default(),
            mirostat_mu: 3.0,
            recent: Vec::new(),
            max_recent: 64,
        }
    }

    /// Sample a token index from logits (in-place, will be modified).
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        match self.kind {
            SamplerKind::Greedy => self.sample_greedy(logits),
            SamplerKind::Temperature { temp } => self.sample_temp(logits, temp),
            SamplerKind::TopK { k } => self.sample_topk(logits, k),
            SamplerKind::TopP { p } => self.sample_topp(logits, p),
            SamplerKind::Mirostat { tau, rate } => self.sample_mirostat(logits, tau, rate),
        }
    }

    fn sample_greedy(&self, logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let a = if a.is_nan() { &f32::NEG_INFINITY } else { a };
                let b = if b.is_nan() { &f32::NEG_INFINITY } else { b };
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    fn sample_temp(&mut self, logits: &mut [f32], temp: f32) -> u32 {
        if temp <= 0.0 {
            return self.sample_greedy(logits);
        }
        // softmax with temperature
        let inv_temp = 1.0 / temp;
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for x in logits.iter_mut() {
            *x = ((*x - max_val) * inv_temp).exp();
            sum += *x as f64;
        }
        let r: f64 = self.rng.f64() * sum;
        let mut acc = 0.0f64;
        for (i, &p) in logits.iter().enumerate() {
            acc += p as f64;
            if acc > r {
                return i as u32;
            }
        }
        (logits.len() - 1) as u32
    }

    fn sample_topk(&mut self, _logits: &mut [f32], _k: usize) -> u32 {
        todo!("Top-K sampling")
    }

    fn sample_topp(&mut self, _logits: &mut [f32], _p: f32) -> u32 {
        todo!("Top-P (nucleus) sampling")
    }

    fn sample_mirostat(&mut self, _logits: &mut [f32], _tau: f32, _rate: f32) -> u32 {
        todo!("Mirostat sampling")
    }
}
