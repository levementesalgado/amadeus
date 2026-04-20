use crate::tensor::ops::{matmul, rms_norm_inplace};
use crate::tensor::Tensor;
use super::config::AmadeusMConfig;
use super::affect::AffectModule;
use super::intent::Intentionality;
use super::habit::HabitMemory;
use super::recall::PureRecollection;
use super::synthesis::SynthesisLayer;
use super::sediment::TraceSediment;
use super::multicode::MultiCoding;
use super::rng;

pub struct AmadeusMModel {
    pub config: AmadeusMConfig,
    pub multicode: MultiCoding,
    pub output_weight: Tensor,
    pub output_norm: Tensor,
    pub affect: AffectModule,
    pub intentionality: Intentionality,
    pub recollection: PureRecollection,
    pub sediment: TraceSediment,
    pub layers: Vec<HabitLayer>,
    pub trace: Vec<f32>,
}

pub struct HabitLayer {
    pub habit: HabitMemory,
    pub synthesis: SynthesisLayer,
}

impl AmadeusMModel {
    pub fn new(cfg: AmadeusMConfig) -> Self {
        let mut rng = fastrand::Rng::new();
        let n_embd = cfg.n_embd;
        let d_trace = cfg.d_trace;
        let d_hidden = 64;

        let multicode = MultiCoding::new(cfg.vocab_size, n_embd, cfg.d_affect, &mut rng);
        let output_weight = rng::init_linear(n_embd, cfg.vocab_size, &mut rng);
        let output_norm = rng::init_rms_weight(n_embd);

        let affect = AffectModule::new(cfg.d_affect, n_embd, &mut rng);
        let intentionality = Intentionality::new(n_embd, cfg.d_intent, cfg.n_intents, &mut rng);
        let recollection = PureRecollection::new(n_embd, cfg.d_affect, cfg.d_intent, d_trace, d_hidden, &mut rng);
        let sediment = TraceSediment::new(n_embd, cfg.d_affect, d_trace, &mut rng);

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            layers.push(HabitLayer {
                habit: HabitMemory::new(&cfg, &mut rng),
                synthesis: SynthesisLayer::new(n_embd, cfg.d_affect, cfg.d_intent, cfg.n_intermediate, &mut rng),
            });
        }

        let trace = vec![0.0; d_trace];

        Self {
            config: cfg,
            multicode,
            output_weight,
            output_norm,
            affect,
            intentionality,
            recollection,
            sediment,
            layers,
            trace,
        }
    }

    pub fn forward_token(&mut self, token: u32, pos: usize) -> Vec<f32> {
        self.forward_with_gate(token, pos, 0.3, 0.5, 0.5)
    }

    pub fn forward_with_gate(&mut self, token: u32, pos: usize, excursion_phase: f32, curiosity: f32, focus_attention: f32) -> Vec<f32> {
        let cfg = &self.config;
        let n_embd = cfg.n_embd;

        let mut hidden = self.multicode.forward(token as usize, excursion_phase, curiosity, focus_attention);

        for layer in self.layers.iter_mut() {
            let residual = hidden.clone();

            let (_intent_dist, intent_emb) = self.intentionality.classify(&hidden);

            let habit_out = layer.habit.attend(&hidden, pos, n_embd);

            let recollection_out = self.recollection.reconstruct(&hidden, &self.affect.mood, &intent_emb, &self.trace);

            layer.synthesis.forward(
                &mut hidden,
                &residual,
                &habit_out,
                &recollection_out,
                &self.affect.mood,
                &intent_emb,
            );
        }

        let mut hidden_t = Tensor::from_vec(hidden, vec![1, n_embd]);
        rms_norm_inplace(&mut hidden_t, self.output_norm.as_slice(), 1e-6);
        let normed = hidden_t.as_slice().to_vec();

        self.affect.evolve(&normed);
        self.sediment.update(&mut self.trace, &self.affect.mood, &normed);

        let logits = matmul(&self.output_weight, &Tensor::from_vec(normed, vec![1, n_embd]));
        logits.as_slice().to_vec()
    }

    pub fn reset(&mut self) {
        self.affect.mood.fill(0.0);
        self.trace.fill(0.0);
        for layer in &mut self.layers {
            layer.habit.reset_cache();
        }
    }
}
