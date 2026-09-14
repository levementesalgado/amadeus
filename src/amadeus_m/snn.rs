use std::collections::HashMap;

// ─── Parâmetros do SNN ───

const DEFAULT_TAU: f32 = 5.0;
const DEFAULT_THRESHOLD: f32 = 1.0;
const DEFAULT_REFRAC: u32 = 2;
const DEFAULT_DT: f32 = 1.0;
const DEFAULT_TSTEPS: usize = 16;

// ─── Spike Train (taxa codificada) ───

#[derive(Debug, Clone)]
pub struct SpikeTrain {
    pub timesteps: usize,
    pub spikes: Vec<bool>,
}

impl SpikeTrain {
    pub fn from_rate(rate: f32, timesteps: usize, rng: &mut fastrand::Rng) -> Self {
        let prob = rate.clamp(0.0, 1.0);
        let spikes = (0..timesteps).map(|_| rng.f32() < prob).collect();
        Self { timesteps, spikes }
    }

    pub fn spike_count(&self) -> u32 {
        self.spikes.iter().filter(|&&s| s).count() as u32
    }

    pub fn first_spike(&self) -> Option<usize> {
        self.spikes.iter().position(|&s| s)
    }
}

// ─── Neurônio LIF (Leaky Integrate-and-Fire) ───

#[derive(Debug, Clone)]
pub struct Neuron {
    pub v: f32,
    pub threshold: f32,
    pub tau: f32,
    pub refrac_steps: u32,
    pub refrac_left: u32,
    pub fired: bool,
    pub spike_times: Vec<usize>,
    pub last_spike_time: usize,
}

impl Neuron {
    pub fn new(threshold: f32, tau: f32, refrac_steps: u32) -> Self {
        Self {
            v: 0.0,
            threshold,
            tau,
            refrac_steps,
            refrac_left: 0,
            fired: false,
            spike_times: Vec::new(),
            last_spike_time: 0,
        }
    }

    pub fn step(&mut self, input: f32, dt: f32, t: usize) {
        self.fired = false;
        if self.refrac_left > 0 {
            self.refrac_left -= 1;
            return;
        }
        self.v += (-self.v / self.tau + input) * dt;
        if self.v >= self.threshold {
            self.fired = true;
            self.spike_times.push(t);
            self.last_spike_time = t;
            self.v = 0.0;
            self.refrac_left = self.refrac_steps;
        }
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
        self.refrac_left = 0;
        self.fired = false;
        self.spike_times.clear();
    }
}

// ─── Rede Neural Spiking ───

#[derive(Debug, Clone)]
pub struct SpikingNetwork {
    pub input_neurons: Vec<Neuron>,
    pub hidden_neurons: Vec<Neuron>,
    pub output_neurons: Vec<Neuron>,
    pub synapses_ih: Vec<Vec<f32>>,
    pub synapses_ho: Vec<Vec<f32>>,
    pub dt: f32,
    pub tsteps: usize,
    pub output_labels: Vec<u32>,
}

impl SpikingNetwork {
    pub fn new(
        n_input: usize,
        n_hidden: usize,
        output_labels: Vec<u32>,
        tau: f32,
        threshold: f32,
        refrac: u32,
        dt: f32,
        tsteps: usize,
    ) -> Self {
        let n_out = output_labels.len();
        Self {
            input_neurons: vec![Neuron::new(threshold, tau, refrac); n_input],
            hidden_neurons: vec![Neuron::new(threshold, tau, refrac); n_hidden],
            output_neurons: vec![Neuron::new(threshold, tau, refrac); n_out],
            synapses_ih: vec![vec![0.0; n_hidden]; n_input],
            synapses_ho: vec![vec![0.0; n_out]; n_hidden],
            dt,
            tsteps,
            output_labels,
        }
    }

    pub fn reset(&mut self) {
        for n in self.input_neurons.iter_mut() { n.reset(); }
        for n in self.hidden_neurons.iter_mut() { n.reset(); }
        for n in self.output_neurons.iter_mut() { n.reset(); }
    }

    // ─── GGUF Serialization ───

    pub fn flat_ih(&self) -> Vec<f32> {
        self.synapses_ih.iter().flat_map(|row| row.iter().copied()).collect()
    }

    pub fn flat_ho(&self) -> Vec<f32> {
        self.synapses_ho.iter().flat_map(|row| row.iter().copied()).collect()
    }

    pub fn flat_labels(&self) -> Vec<f32> {
        self.output_labels.iter().map(|&id| id as f32).collect()
    }

    pub fn from_flat_ih(flat: &[f32], n_input: usize, n_hidden: usize) -> Vec<Vec<f32>> {
        flat.chunks(n_hidden).take(n_input).map(|row| row.to_vec()).collect()
    }

    pub fn from_flat_ho(flat: &[f32], n_hidden: usize, n_out: usize) -> Vec<Vec<f32>> {
        flat.chunks(n_out).take(n_hidden).map(|row| row.to_vec()).collect()
    }

    pub fn from_flat_labels(flat: &[f32]) -> Vec<u32> {
        flat.iter().map(|&v| v as u32).collect()
    }

    pub fn run(&mut self, input_spikes: &[SpikeTrain]) -> Vec<Vec<usize>> {
        self.reset();
        let mut output_spike_times = vec![Vec::new(); self.output_neurons.len()];

        // Pré-computar spikes de input como f32 (evita branch no loop)
        let mut input_fired = vec![0.0f32; self.input_neurons.len()];

        for t in 0..self.tsteps {
            // Input layer
            for (i, train) in input_spikes.iter().enumerate() {
                let spike_in = if t < train.timesteps && train.spikes[t] { 1.0 } else { 0.0 };
                self.input_neurons[i].step(spike_in, self.dt, t);
                input_fired[i] = if self.input_neurons[i].fired { 1.0 } else { 0.0 };
            }

            // Input → Hidden (SIMD-friendly: flatten + chunk)
            let n_hidden = self.hidden_neurons.len();
            let mut hidden_input = vec![0.0f32; n_hidden];

            // Usar chunks para processar 4 pesos por vez (AVX2)
            for (i, ni_fired) in input_fired.iter().enumerate() {
                if *ni_fired > 0.0 {
                    let row = &self.synapses_ih[i];
                    let chunks = row.len() / 4;
                    for c in 0..chunks {
                        let base = c * 4;
                        let w0 = row[base];
                        let w1 = row[base + 1];
                        let w2 = row[base + 2];
                        let w3 = row[base + 3];
                        hidden_input[base] += w0;
                        hidden_input[base + 1] += w1;
                        hidden_input[base + 2] += w2;
                        hidden_input[base + 3] += w3;
                    }
                    // Resto
                    for h in (chunks * 4)..n_hidden {
                        hidden_input[h] += row[h];
                    }
                }
            }

            for (h, nh) in self.hidden_neurons.iter_mut().enumerate() {
                nh.step(hidden_input[h], self.dt, t);
            }

            // Hidden → Output (SIMD-friendly)
            let n_out = self.output_neurons.len();
            let mut output_input = vec![0.0f32; n_out];

            for (h, nh) in self.hidden_neurons.iter().enumerate() {
                if nh.fired {
                    let row = &self.synapses_ho[h];
                    let chunks = row.len() / 4;
                    for c in 0..chunks {
                        let base = c * 4;
                        let w0 = row[base];
                        let w1 = row[base + 1];
                        let w2 = row[base + 2];
                        let w3 = row[base + 3];
                        output_input[base] += w0;
                        output_input[base + 1] += w1;
                        output_input[base + 2] += w2;
                        output_input[base + 3] += w3;
                    }
                    for o in (chunks * 4)..n_out {
                        output_input[o] += row[o];
                    }
                }
            }

            for (o, no) in self.output_neurons.iter_mut().enumerate() {
                no.step(output_input[o], self.dt, t);
                if no.fired {
                    output_spike_times[o].push(t);
                }
            }
        }

        output_spike_times
    }

    /// Run SNN without resetting membrane potential (recurrent memory)
    pub fn run_recurrent(&mut self, input_spikes: &[SpikeTrain]) -> Vec<Vec<usize>> {
        let mut output_spike_times = vec![Vec::new(); self.output_neurons.len()];

        let mut input_fired = vec![0.0f32; self.input_neurons.len()];

        for t in 0..self.tsteps {
            for (i, train) in input_spikes.iter().enumerate() {
                let spike_in = if t < train.timesteps && train.spikes[t] { 1.0 } else { 0.0 };
                self.input_neurons[i].step(spike_in, self.dt, t);
                input_fired[i] = if self.input_neurons[i].fired { 1.0 } else { 0.0 };
            }

            let n_hidden = self.hidden_neurons.len();
            let mut hidden_input = vec![0.0f32; n_hidden];
            for (i, ni_fired) in input_fired.iter().enumerate() {
                if *ni_fired > 0.0 {
                    let row = &self.synapses_ih[i];
                    let chunks = row.len() / 4;
                    for c in 0..chunks {
                        let base = c * 4;
                        hidden_input[base] += row[base];
                        hidden_input[base + 1] += row[base + 1];
                        hidden_input[base + 2] += row[base + 2];
                        hidden_input[base + 3] += row[base + 3];
                    }
                    for h in (chunks * 4)..n_hidden {
                        hidden_input[h] += row[h];
                    }
                }
            }

            for (h, nh) in self.hidden_neurons.iter_mut().enumerate() {
                nh.step(hidden_input[h], self.dt, t);
            }

            let n_out = self.output_neurons.len();
            let mut output_input = vec![0.0f32; n_out];
            for (h, nh) in self.hidden_neurons.iter().enumerate() {
                if nh.fired {
                    let row = &self.synapses_ho[h];
                    let chunks = row.len() / 4;
                    for c in 0..chunks {
                        let base = c * 4;
                        output_input[base] += row[base];
                        output_input[base + 1] += row[base + 1];
                        output_input[base + 2] += row[base + 2];
                        output_input[base + 3] += row[base + 3];
                    }
                    for o in (chunks * 4)..n_out {
                        output_input[o] += row[o];
                    }
                }
            }

            for (o, no) in self.output_neurons.iter_mut().enumerate() {
                no.step(output_input[o], self.dt, t);
                if no.fired {
                    output_spike_times[o].push(t);
                }
            }
        }

        output_spike_times
    }

    /// Reset only input neurons (for recurrent mode between tokens)
    pub fn reset_inputs_only(&mut self) {
        for n in self.input_neurons.iter_mut() { n.reset(); }
    }

    // ─── STDP: Aprendizado por Timing de Spikes ───
    pub fn stdp_train(&mut self, input_spikes: &[SpikeTrain], target_output: usize, learning_rate: f32) {
        let output_spike_times = self.run(input_spikes);

        // Parâmetros STDP
        let tau_plus = 20.0; // Constante de tempo para potenciação
        let tau_minus = 20.0; // Constante de tempo para depressão
        let a_plus = learning_rate; // Amplitude de potenciação
        let a_minus = learning_rate * 0.85; // Amplitude de depressão

        // Para cada par de spikes (pre, post)
        for (h, nh) in self.hidden_neurons.iter().enumerate() {
            if !nh.fired { continue; }

            // Encontrar tempo de spike mais próximo no output alvo
            if let Some(&spike_time) = output_spike_times.get(target_output)
                .and_then(|times| times.first())
            {
                let dt_spike = spike_time as f32 - nh.last_spike_time as f32;

                // Atualizar pesos synapses_ho
                if dt_spike > 0.0 {
                    // Post dispara após pre → potenciação (LTP)
                    let delta_w = a_plus * (-dt_spike / tau_plus).exp();
                    for o in 0..self.synapses_ho[h].len() {
                        self.synapses_ho[h][o] += delta_w;
                        self.synapses_ho[h][o] = self.synapses_ho[h][o].clamp(-1.0, 1.0);
                    }
                } else if dt_spike < 0.0 {
                    // Pre dispara após post → depressão (LTD)
                    let delta_w = -a_minus * (dt_spike / tau_minus).exp();
                    for o in 0..self.synapses_ho[h].len() {
                        self.synapses_ho[h][o] += delta_w;
                        self.synapses_ho[h][o] = self.synapses_ho[h][o].clamp(-1.0, 1.0);
                    }
                }
            }
        }

        // Atualizar pesos synapses_ih
        for (i, ni) in self.input_neurons.iter().enumerate() {
            if !ni.fired { continue; }

            for (h, nh) in self.hidden_neurons.iter().enumerate() {
                if !nh.fired { continue; }

                let dt_spike = nh.last_spike_time as f32 - ni.last_spike_time as f32;

                if dt_spike > 0.0 {
                    let delta_w = a_plus * (-dt_spike / tau_plus).exp();
                    self.synapses_ih[i][h] += delta_w;
                    self.synapses_ih[i][h] = self.synapses_ih[i][h].clamp(-1.0, 1.0);
                } else if dt_spike < 0.0 {
                    let delta_w = -a_minus * (dt_spike / tau_minus).exp();
                    self.synapses_ih[i][h] += delta_w;
                    self.synapses_ih[i][h] = self.synapses_ih[i][h].clamp(-1.0, 1.0);
                }
            }
        }
    }

    // ─── Treinamento iterativo STDP ───

    /// STDP de 3 fatores: pré × pós × modulador
    /// modulator: sinal de fricção/reforço (0.0=fricção alta, 1.0=reforço)
    pub fn stdp_train_3factor(
        &mut self,
        input_spikes: &[SpikeTrain],
        target_output: usize,
        learning_rate: f32,
        modulator: f32,
    ) {
        let output_spike_times = self.run(input_spikes);

        let tau_plus = 20.0;
        let tau_minus = 20.0;
        let a_plus = learning_rate * modulator; // Modulador escala LTP
        let a_minus = learning_rate * 0.85 * (1.0 - modulator * 0.5); // LTD reduzido com reforço

        // Hidden → Output (3-factor)
        for (h, nh) in self.hidden_neurons.iter().enumerate() {
            if !nh.fired { continue; }

            if let Some(&spike_time) = output_spike_times.get(target_output)
                .and_then(|times| times.first())
            {
                let dt_spike = spike_time as f32 - nh.last_spike_time as f32;

                if dt_spike > 0.0 {
                    let delta_w = a_plus * (-dt_spike / tau_plus).exp();
                    for o in 0..self.synapses_ho[h].len() {
                        self.synapses_ho[h][o] += delta_w;
                        self.synapses_ho[h][o] = self.synapses_ho[h][o].clamp(-1.0, 1.0);
                    }
                } else if dt_spike < 0.0 {
                    let delta_w = -a_minus * (dt_spike / tau_minus).exp();
                    for o in 0..self.synapses_ho[h].len() {
                        self.synapses_ho[h][o] += delta_w;
                        self.synapses_ho[h][o] = self.synapses_ho[h][o].clamp(-1.0, 1.0);
                    }
                }
            }
        }

        // Input → Hidden (3-factor)
        for (i, ni) in self.input_neurons.iter().enumerate() {
            if !ni.fired { continue; }

            for (h, nh) in self.hidden_neurons.iter().enumerate() {
                if !nh.fired { continue; }

                let dt_spike = nh.last_spike_time as f32 - ni.last_spike_time as f32;

                if dt_spike > 0.0 {
                    let delta_w = a_plus * (-dt_spike / tau_plus).exp();
                    self.synapses_ih[i][h] += delta_w;
                    self.synapses_ih[i][h] = self.synapses_ih[i][h].clamp(-1.0, 1.0);
                } else if dt_spike < 0.0 {
                    let delta_w = -a_minus * (dt_spike / tau_minus).exp();
                    self.synapses_ih[i][h] += delta_w;
                    self.synapses_ih[i][h] = self.synapses_ih[i][h].clamp(-1.0, 1.0);
                }
            }
        }
    }

    /// Treinamento iterativo STDP
    /// Treina o SNN em múltiplas épocas para classificação POS (7 classes)
    /// Retorna (época, acurácia_top1, acurácia_top3) por época
    pub fn iterative_train(
        &mut self,
        enc: &MorphEncoding,
        grammar_lexicon: &HashMap<(u16, u8, u16), HashMap<u32, f32>>,
        epochs: usize,
        learning_rate: f32,
        samples_per_epoch: usize,
    ) -> Vec<(usize, f32, f32)> {
        let mut results = Vec::new();
        let mut rng = fastrand::Rng::with_seed(42);

        // Extrair exemplos: (morph, syn_func, style, class_id)
        let mut training_examples: Vec<(u16, u8, u16, u32)> = Vec::new();
        for (&(morph, syn_func, style), _candidates) in grammar_lexicon {
            let class_id = (morph & 0x7) as u32;
            training_examples.push((morph, syn_func, style, class_id));
        }

        if training_examples.is_empty() {
            return results;
        }

        println!("    Exemplos de treinamento: {} (classificação POS)", training_examples.len());

        for epoch in 0..epochs {
            // Fisher-Yates shuffle
            let n = training_examples.len();
            for i in (1..n).rev() {
                let j = rng.usize(0..=i);
                training_examples.swap(i, j);
            }

            let n_samples = samples_per_epoch.min(training_examples.len());
            let mut correct_top1 = 0usize;
            let mut correct_top3 = 0usize;

            for idx in 0..n_samples {
                let (morph, syn_func, style, target_class) = training_examples[idx];

                let trains = enc.encode(morph, syn_func, style, self.tsteps, &mut rng);
                let output_spikes = self.run(&trains);

                // Scores das 7 classes
                let mut scores: Vec<(u32, f32)> = self.output_neurons.iter().enumerate()
                    .map(|(i, n)| {
                        let class_id = self.output_labels[i];
                        let spike_count = output_spikes.get(i).map(|s| s.len()).unwrap_or(0) as f32;
                        let potential: f32 = n.spike_times.iter().map(|&t| (-(t as f32) / 5.0).exp()).sum();
                        (class_id, spike_count + potential)
                    })
                    .collect();
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                // Verificar acurácia
                for (rank, &(class_id, _)) in scores.iter().take(3).enumerate() {
                    if class_id == target_class {
                        if rank == 0 { correct_top1 += 1; }
                        correct_top3 += 1;
                        break;
                    }
                }

                // STDP de 3 fatores: modulador = confiança da predição
                if let Some(target_idx) = self.output_labels.iter().position(|&id| id == target_class) {
                    // Modulador: 1.0 se acertou (reforço), 0.3 se errou (fricção)
                    let predicted_class = scores.first().map(|(c, _)| *c).unwrap_or(0);
                    let modulator = if predicted_class == target_class { 1.0 } else { 0.3 };
                    self.stdp_train_3factor(&trains, target_idx, learning_rate, modulator);
                }
            }

            let acc_top1 = correct_top1 as f32 / n_samples as f32;
            let acc_top3 = correct_top3 as f32 / n_samples as f32;
            results.push((epoch + 1, acc_top1, acc_top3));

            if (epoch + 1) % 5 == 0 || epoch == 0 {
                println!("    Época {}: Top-1={:.1}% Top-3={:.1}%",
                    epoch + 1, acc_top1 * 100.0, acc_top3 * 100.0);
            }
        }

        results
    }
}

// ─── Features morfológicos (índices no vetor de input) ───

#[derive(Debug, Clone)]
pub struct MorphEncoding {
    pub class_start: usize,
    pub class_count: usize,
    pub gender_start: usize,
    pub gender_count: usize,
    pub number_start: usize,
    pub number_count: usize,
    pub tense_start: usize,
    pub tense_count: usize,
    pub person_start: usize,
    pub person_count: usize,
    pub style_start: usize,
    pub style_count: usize,
    pub syn_start: usize,
    pub syn_count: usize,
    pub total: usize,
}

impl MorphEncoding {
    pub fn new() -> Self {
        let mut off = 0;
        let class_start = off; let class_count = 7; off += class_count;
        let gender_start = off; let gender_count = 2; off += gender_count;
        let number_start = off; let number_count = 2; off += number_count;
        let tense_start = off; let tense_count = 5; off += tense_count;
        let person_start = off; let person_count = 3; off += person_count;
        let style_start = off; let style_count = 3; off += style_count;
        let syn_start = off; let syn_count = 8; off += syn_count;
        Self { class_start, class_count, gender_start, gender_count, number_start, number_count, tense_start, tense_count, person_start, person_count, style_start, style_count, syn_start, syn_count, total: off }
    }

    pub fn encode(&self, morph: u16, syn_func: u8, style: u16, tsteps: usize, rng: &mut fastrand::Rng) -> Vec<SpikeTrain> {
        let mut trains = Vec::with_capacity(self.total);
        let morph_bits = morph as u32;

        // Class (bits 0-2) → one-hot rate
        let class_id = (morph_bits & 0x7) as usize;
        for i in 0..self.class_count {
            let rate = if i == class_id { 0.8 } else { 0.05 };
            trains.push(SpikeTrain::from_rate(rate, tsteps, rng));
        }

        // Gender (bit 3) → one-hot
        let gender = ((morph_bits >> 3) & 0x1) as usize;
        for i in 0..self.gender_count {
            let rate = if i == gender { 0.7 } else { 0.05 };
            trains.push(SpikeTrain::from_rate(rate, tsteps, rng));
        }

        // Number (bit 4) → one-hot
        let number = ((morph_bits >> 4) & 0x1) as usize;
        for i in 0..self.number_count {
            let rate = if i == number { 0.7 } else { 0.05 };
            trains.push(SpikeTrain::from_rate(rate, tsteps, rng));
        }

        // Tense (bits 5-7) → one-hot
        let tense = ((morph_bits >> 5) & 0x7) as usize;
        for i in 0..self.tense_count {
            let rate = if i == tense.min(self.tense_count - 1) { 0.7 } else { 0.05 };
            trains.push(SpikeTrain::from_rate(rate, tsteps, rng));
        }

        // Person (bits 8-9) → one-hot
        let person = ((morph_bits >> 8) & 0x3) as usize;
        for i in 0..self.person_count {
            let rate = if i == person.min(self.person_count - 1) { 0.6 } else { 0.05 };
            trains.push(SpikeTrain::from_rate(rate, tsteps, rng));
        }

        // Style → one-hot
        let style_id = (style as usize).min(self.style_count - 1);
        for i in 0..self.style_count {
            let rate = if i == style_id { 0.7 } else { 0.05 };
            trains.push(SpikeTrain::from_rate(rate, tsteps, rng));
        }

        // Syntactic function → one-hot
        let syn_id = (syn_func as usize).min(self.syn_count - 1);
        for i in 0..self.syn_count {
            let rate = if i == syn_id { 0.6 } else { 0.05 };
            trains.push(SpikeTrain::from_rate(rate, tsteps, rng));
        }

        trains
    }
}

// ─── Treinamento: pesos sinápticos a partir de estatísticas ───

pub fn build_snn_from_grammar(
    lexicon_exact: &HashMap<(u16, u8, u16), HashMap<u32, f32>>,
    lexicon_css: &HashMap<(u8, u8, u16), HashMap<u32, f32>>,
    lexicon_cs: &HashMap<(u8, u8), HashMap<u32, f32>>,
    lexicon_cst: &HashMap<(u8, u16), HashMap<u32, f32>>,
    lexicon_class: &HashMap<u8, HashMap<u32, f32>>,
    tau: f32,
    threshold: f32,
    refrac: u32,
    dt: f32,
    tsteps: usize,
) -> (SpikingNetwork, MorphEncoding) {
    let enc = MorphEncoding::new();
    let n_input = enc.total;

    // 7 classes POS como outputs (SUBST=0, VERBO=1, ADJ=2, ART=3, ADV=4, PREP/CONJ=5, PONT=6)
    let n_out = 7;
    let class_labels: Vec<u32> = (0..n_out as u32).collect();

    // Hidden layer: T3 keys + classe neurons
    let n_class_neurons = 7;
    let n_t3_neurons = lexicon_exact.len().min(256);
    let n_hidden = n_class_neurons + n_t3_neurons;

    let mut net = SpikingNetwork::new(n_input, n_hidden, class_labels, tau, threshold, refrac, dt, tsteps);

    // ── Pesos I→H: classe neurons ──
    for class_id in 0..n_class_neurons {
        let h_idx = class_id;
        net.synapses_ih[enc.class_start + class_id][h_idx] = 0.5;
        net.synapses_ih[enc.gender_start][h_idx] = 0.1;
        net.synapses_ih[enc.gender_start + 1][h_idx] = 0.1;
        net.synapses_ih[enc.number_start][h_idx] = 0.1;
        net.synapses_ih[enc.number_start + 1][h_idx] = 0.1;
    }

    // ── Pesos I→H: T3 neurons ──
    let t3_keys: Vec<&(u16, u8, u16)> = lexicon_exact.keys().collect();
    for (t3_idx, &key) in t3_keys.iter().enumerate().take(n_t3_neurons) {
        let h_idx = n_class_neurons + t3_idx;
        let (morph, syn_func, style) = *key;
        let class_id = (morph & 0x7) as usize;
        if class_id < enc.class_count {
            net.synapses_ih[enc.class_start + class_id][h_idx] = 0.4;
        }
        let gender = ((morph >> 3) & 0x1) as usize;
        if gender < enc.gender_count {
            net.synapses_ih[enc.gender_start + gender][h_idx] = 0.2;
        }
        let number = ((morph >> 4) & 0x1) as usize;
        if number < enc.number_count {
            net.synapses_ih[enc.number_start + number][h_idx] = 0.2;
        }
        let syn_id = (syn_func as usize).min(enc.syn_count - 1);
        net.synapses_ih[enc.syn_start + syn_id][h_idx] = 0.3;
        let style_id = (style as usize).min(enc.style_count - 1);
        net.synapses_ih[enc.style_start + style_id][h_idx] = 0.2;
    }

    // ── Pesos H→O: cada hidden neuron → classe correspondente ──
    // Neurônios de classe → sua própria classe
    for class_id in 0..n_class_neurons {
        net.synapses_ho[class_id][class_id] = 0.8;
    }

    // T3 neurons → classe do seu morph
    for (t3_idx, &key) in t3_keys.iter().enumerate().take(n_t3_neurons) {
        let h_idx = n_class_neurons + t3_idx;
        let (morph, _, _) = *key;
        let class_id = (morph & 0x7) as usize;
        if class_id < n_out {
            net.synapses_ho[h_idx][class_id] = 0.5;
        }
    }

    (net, enc)
}

// ─── Inferência SNN → ranking de candidatos ───

pub fn infer_snn(
    net: &mut SpikingNetwork,
    enc: &MorphEncoding,
    morph: u16,
    syn_func: u8,
    style: u16,
    rng: &mut fastrand::Rng,
) -> Vec<(u32, f32)> {
    let trains = enc.encode(morph, syn_func, style, net.tsteps, rng);

    let mut output_potentials = vec![0.0f32; net.output_neurons.len()];
    let mut output_spikes = vec![0u32; net.output_neurons.len()];

    net.reset();

    for t in 0..net.tsteps {
        for (i, train) in trains.iter().enumerate() {
            let spike_in = if t < train.timesteps && train.spikes[t] { 1.0 } else { 0.0 };
            net.input_neurons[i].step(spike_in, net.dt, t);
        }

        let mut hidden_input = vec![0.0f32; net.hidden_neurons.len()];
        for (i, ni) in net.input_neurons.iter().enumerate() {
            if ni.fired {
                for (h, &w) in net.synapses_ih[i].iter().enumerate() {
                    hidden_input[h] += w;
                }
            }
        }
        for (h, nh) in net.hidden_neurons.iter_mut().enumerate() {
            nh.step(hidden_input[h], net.dt, t);
        }

        let mut output_input = vec![0.0f32; net.output_neurons.len()];
        for (h, nh) in net.hidden_neurons.iter().enumerate() {
            if nh.fired {
                for (o, &w) in net.synapses_ho[h].iter().enumerate() {
                    output_input[o] += w;
                }
            }
        }
        for (o, no) in net.output_neurons.iter_mut().enumerate() {
            no.step(output_input[o], net.dt, t);
            output_potentials[o] += output_input[o];
            if no.fired {
                output_spikes[o] += 1;
            }
        }
    }

    // Score: potencial + spikes → 7 classes POS
    let scores: Vec<(u32, f32)> = output_potentials.iter().enumerate().map(|(i, &pot)| {
        let spike_bonus = output_spikes[i] as f32 * 2.0;
        let score = pot + spike_bonus;
        (net.output_labels[i], score)
    }).filter(|(_, s)| *s > 0.01).collect();

    let mut sorted = scores;
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

/// Inferência SNN recorrente: mantém potencial de membrana entre tokens
pub fn infer_snn_recurrent(
    net: &mut SpikingNetwork,
    enc: &MorphEncoding,
    morph: u16,
    syn_func: u8,
    style: u16,
    rng: &mut fastrand::Rng,
) -> Vec<(u32, f32)> {
    let trains = enc.encode(morph, syn_func, style, net.tsteps, rng);

    let mut output_potentials = vec![0.0f32; net.output_neurons.len()];
    let mut output_spikes = vec![0u32; net.output_neurons.len()];

    // NÃO reseta — membrana persiste entre tokens
    for t in 0..net.tsteps {
        for (i, train) in trains.iter().enumerate() {
            let spike_in = if t < train.timesteps && train.spikes[t] { 1.0 } else { 0.0 };
            net.input_neurons[i].step(spike_in, net.dt, t);
        }

        let mut hidden_input = vec![0.0f32; net.hidden_neurons.len()];
        for (i, ni) in net.input_neurons.iter().enumerate() {
            if ni.fired {
                for (h, &w) in net.synapses_ih[i].iter().enumerate() {
                    hidden_input[h] += w;
                }
            }
        }
        for (h, nh) in net.hidden_neurons.iter_mut().enumerate() {
            nh.step(hidden_input[h], net.dt, t);
        }

        let mut output_input = vec![0.0f32; net.output_neurons.len()];
        for (h, nh) in net.hidden_neurons.iter().enumerate() {
            if nh.fired {
                for (o, &w) in net.synapses_ho[h].iter().enumerate() {
                    output_input[o] += w;
                }
            }
        }
        for (o, no) in net.output_neurons.iter_mut().enumerate() {
            no.step(output_input[o], net.dt, t);
            output_potentials[o] += output_input[o];
            if no.fired {
                output_spikes[o] += 1;
            }
        }
    }

    let scores: Vec<(u32, f32)> = output_potentials.iter().enumerate().map(|(i, &pot)| {
        let spike_bonus = output_spikes[i] as f32 * 2.0;
        let score = pot + spike_bonus;
        (net.output_labels[i], score)
    }).filter(|(_, s)| *s > 0.01).collect();

    let mut sorted = scores;
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

// ─── Interface unificada: substitui T3 cascade ───

pub fn snn_sample_lex(
    net: &mut SpikingNetwork,
    enc: &MorphEncoding,
    morph: u16,
    syn_func: u8,
    style: u16,
    candidate: u32,
    temperature: f32,
    exploration_rate: f32,
    rng: &mut fastrand::Rng,
) -> u32 {
    let class_scores = infer_snn(net, enc, morph, syn_func, style, rng);

    if rng.f32() < exploration_rate {
        if !class_scores.is_empty() {
            let idx = rng.usize(0..class_scores.len());
            return class_scores[idx].0;
        }
        return candidate;
    }

    if let Some(&(best_class, _score)) = class_scores.first() {
        return best_class;
    }

    candidate
}

/// Amostragem SNN recorrente: membrana persiste entre chamadas
pub fn snn_sample_lex_recurrent(
    net: &mut SpikingNetwork,
    enc: &MorphEncoding,
    morph: u16,
    syn_func: u8,
    style: u16,
    candidate: u32,
    temperature: f32,
    exploration_rate: f32,
    rng: &mut fastrand::Rng,
) -> u32 {
    let class_scores = infer_snn_recurrent(net, enc, morph, syn_func, style, rng);

    if rng.f32() < exploration_rate {
        if !class_scores.is_empty() {
            let idx = rng.usize(0..class_scores.len());
            return class_scores[idx].0;
        }
        return candidate;
    }

    if let Some(&(best_class, _score)) = class_scores.first() {
        return best_class;
    }

    candidate
}
