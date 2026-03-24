use std::collections::VecDeque;
use crate::amadeus_m::collatz::{CollatzOscillator, CollatzMode};

// ─── Sensações Corporais (Soma) ───

#[derive(Clone)]
pub struct Soma {
    pub sensors: [f32; 8],
    pub baselines: [f32; 8],
    pub decay: [f32; 8],
}

impl Soma {
    pub fn new() -> Self {
        Self {
            sensors: [0.5, 0.3, 0.4, 0.3, 0.6, 0.3, 0.5, 0.5],
            baselines: [0.5, 0.3, 0.4, 0.3, 0.6, 0.3, 0.5, 0.5],
            decay: [0.01, 0.03, 0.02, 0.04, 0.01, 0.005, 0.02, 0.03],
        }
    }

    pub fn update(&mut self, affect_intensity: f32, satisfaction: f32, _valence: f32) {
        let arousal = affect_intensity;
        let pleasure = satisfaction * 2.0 - 1.0;

        self.sensors[0] += (0.5 - arousal * 0.3 - self.sensors[0]) * 0.05;
        self.sensors[1] += (arousal * 0.5 + 0.2 - self.sensors[1]) * 0.1;
        self.sensors[2] += (arousal * 0.4 + 0.3 - self.sensors[2]) * 0.08;
        self.sensors[3] += (pleasure.max(0.0) * 0.6 + arousal * 0.3 - self.sensors[3]) * 0.12;
        self.sensors[4] += (pleasure.max(0.0) * 0.4 + (1.0 - arousal) * 0.3 - self.sensors[4]) * 0.06;
        self.sensors[5] += ((1.0 - pleasure) * 0.3 + arousal * 0.2 - self.sensors[5]) * 0.04;
        self.sensors[6] += ((1.0 - arousal) * 0.4 - self.sensors[6]) * 0.07;
        self.sensors[7] += (pleasure * 0.5 + (1.0 - arousal) * 0.3 - self.sensors[7]) * 0.09;

        for i in 0..8 {
            self.sensors[i] = self.sensors[i].clamp(0.0, 1.0);
            let drift = (self.baselines[i] - self.sensors[i]) * self.decay[i];
            self.sensors[i] += drift;
            self.sensors[i] = self.sensors[i].clamp(0.0, 1.0);
        }
    }

    pub fn homeostasis_drive(&self) -> f32 {
        let mut total = 0.0;
        for i in 0..8 {
            total += (self.sensors[i] - self.baselines[i]).abs();
        }
        total / 8.0
    }

    pub fn tension_amplifier(&self) -> f32 {
        1.0 + self.sensors[1] * 2.0
    }

    pub fn clarity(&self) -> f32 {
        self.sensors[6]
    }

    pub fn openness(&self) -> f32 {
        self.sensors[7]
    }

    /// Soma influencia temperatura: baixa energia → redução da temperatura
    pub fn temp_modulation(&self) -> f32 {
        let energy = self.sensors[0];
        let calm = self.sensors[4];
        let clarity = self.sensors[6];
        // Energia baixa + calma alta → temperatura baixa (mais conservador)
        // Clara → temperatura média
        0.5 + energy * 0.8 + (1.0 - calm) * 0.3 + clarity * 0.2
    }
}

// ─── Atenção Seletiva (Focus) ───

#[derive(Clone)]
pub struct Focus {
    pub theme: Vec<u8>,
    pub secondary_themes: Vec<Vec<u8>>,
    pub theme_decay: f32,
    pub attention: f32,
    pub surprise: f32,
    pub detail_buffer: VecDeque<(u8, f32)>,
}

impl Focus {
    pub fn new() -> Self {
        Self {
            theme: Vec::new(),
            secondary_themes: Vec::new(),
            theme_decay: 0.1,
            attention: 0.5,
            surprise: 0.0,
            detail_buffer: VecDeque::with_capacity(64),
        }
    }

    pub fn set_theme(&mut self, theme: &[u8]) {
        if theme != self.theme.as_slice() {
            self.secondary_themes.push(self.theme.clone());
            if self.secondary_themes.len() > 5 {
                self.secondary_themes.remove(0);
            }
        }
        self.theme = theme.to_vec();
        self.attention = 1.0;
    }

    pub fn update(&mut self, input_bytes: &[u8]) {
        let mut prev_was_space = true;
        for &b in input_bytes {
            let relevance = if self.theme.contains(&b) {
                2.0
            } else if self.secondary_themes.iter().any(|t| t.contains(&b)) {
                1.5
            } else if b.is_ascii_alphabetic() && prev_was_space {
                1.5
            } else if b.is_ascii_alphabetic() {
                1.0
            } else {
                0.3
            };
            self.detail_buffer.push_back((b, relevance));
            if self.detail_buffer.len() > 64 {
                self.detail_buffer.pop_front();
            }
            prev_was_space = b == b' ';
        }
        self.attention *= 1.0 - self.theme_decay * 0.1;
        self.attention = self.attention.clamp(0.1, 1.0);
        self.surprise *= 0.95;
    }

    pub fn relevance(&self, byte: u8) -> f32 {
        if self.theme.contains(&byte) {
            1.5
        } else if self.secondary_themes.iter().any(|t| t.contains(&byte)) {
            1.2
        } else {
            1.0
        }
    }

    pub fn detail_resonance(&self, byte: u8) -> f32 {
        let mut resonance = 0.0;
        for &(b, rel) in self.detail_buffer.iter().rev().take(16) {
            if b == byte {
                resonance += rel * 0.2;
            }
        }
        1.0 + resonance
    }
}

// ─── Memória Episódica Emocional ───

#[derive(Clone)]
pub struct EmotionalEpisode {
    pub input_text: String,
    pub output_text: String,
    pub mood_before: Vec<f32>,
    pub mood_after: Vec<f32>,
    pub soma_before: [f32; 8],
    pub soma_after: [f32; 8],
    pub satisfaction: f32,
    pub intensity: f32,
    pub prediction_error: f32,
}

#[derive(Clone)]
pub struct EmotionalMemory {
    pub episodes: VecDeque<EmotionalEpisode>,
    pub max_episodes: usize,
}

impl EmotionalMemory {
    pub fn new(max_episodes: usize) -> Self {
        Self {
            episodes: VecDeque::with_capacity(max_episodes + 1),
            max_episodes,
        }
    }

    pub fn record(&mut self, episode: EmotionalEpisode) {
        if episode.intensity > 0.3 || episode.prediction_error > 0.5 {
            self.episodes.push_back(episode);
            while self.episodes.len() > self.max_episodes {
                self.episodes.pop_front();
            }
        }
    }

    pub fn resonance(&self, mood: &[f32], soma: &[f32; 8]) -> f32 {
        let mut total = 0.0;
        let mut count = 0;
        for ep in self.episodes.iter().rev().take(10) {
            let mood_sim = cosine_similarity(mood, &ep.mood_after);
            let soma_sim = soma_similarity(soma, &ep.soma_after);
            let sim = mood_sim * 0.6 + soma_sim * 0.4;
            total += sim * ep.intensity;
            count += 1;
        }
        if count > 0 { total / count as f32 } else { 0.0 }
    }

    pub fn recent_emotional_tone(&self) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;
        for ep in self.episodes.iter().rev().take(5) {
            sum += ep.satisfaction * 2.0 - 1.0;
            count += 1;
        }
        if count > 0 { sum / count as f32 } else { 0.0 }
    }

    /// Sonho: retorna texto de um episódio intenso aleatório para replay
    pub fn dream_text(&self) -> Option<String> {
        let intense: Vec<usize> = self.episodes.iter()
            .enumerate()
            .filter(|(_, e)| e.intensity > 0.3 && !e.output_text.is_empty())
            .map(|(i, _)| i)
            .collect();
        if intense.is_empty() {
            return None;
        }
        let idx = intense[fastrand::usize(0..intense.len())];
        let ep = &self.episodes[idx];
        Some(format!("{} {}",
            ep.input_text.chars().take(100).collect::<String>(),
            ep.output_text.chars().take(100).collect::<String>(),
        ))
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    }
}

fn soma_similarity(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum();
    let nb: f32 = b.iter().map(|x| x * x).sum();
    if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    }
}

// ─── Expectativa e Predição ───

#[derive(Clone)]
pub struct PredictiveEngine {
    pub predicted_mood_intensity: f32,
    pub predicted_valence: f32,
    pub prediction_error: f32,
    pub surprise: f32,
    pub curiosity: f32,
    pub moving_avg_error: f32,
    pub alpha: f32,
}

impl PredictiveEngine {
    pub fn new() -> Self {
        Self {
            predicted_mood_intensity: 0.0,
            predicted_valence: 0.5,
            prediction_error: 0.0,
            surprise: 0.0,
            curiosity: 0.5,
            moving_avg_error: 0.2,
            alpha: 0.3,
        }
    }

    pub fn predict(&mut self, current_intensity: f32, current_valence: f32) {
        self.predicted_mood_intensity = current_intensity;
        self.predicted_valence = current_valence;
    }

    pub fn observe(&mut self, actual_intensity: f32, actual_valence: f32, satisfaction: f32) {
        let pred_intensity = self.predicted_mood_intensity;
        let pred_valence = self.predicted_valence;

        let intensity_error = (actual_intensity - pred_intensity).abs();
        let valence_error = (actual_valence - pred_valence).abs();
        let sat_error = (satisfaction - pred_valence).abs();

        self.prediction_error = (intensity_error * 0.3 + valence_error * 0.3 + sat_error * 0.4)
            .clamp(0.0, 1.0);

        self.surprise = if self.moving_avg_error > 0.01 {
            (self.prediction_error / self.moving_avg_error).clamp(0.0, 5.0)
        } else {
            self.prediction_error * 10.0
        };

        self.moving_avg_error = self.alpha * self.moving_avg_error
            + (1.0 - self.alpha) * self.prediction_error;

        self.curiosity = (1.0 - (self.prediction_error - 0.3).abs() * 2.0).clamp(0.0, 0.8);
        self.curiosity *= (1.0 - self.moving_avg_error).clamp(0.2, 1.0);
    }

    pub fn exploration_modulation(&self) -> f32 {
        1.0 + self.curiosity * 2.0
    }
}

// ─── Excursão via Oscilador Collatz ───
// Substitui a senóide por uma dinâmica determinística:
//   n par → Originalista (foco, conservador)
//   n ímpar → Vanguardista (dispersão, criativo)

// ─── Integração: Estado Consciente ───

pub struct ConsciousnessState {
    pub soma: Soma,
    pub focus: Focus,
    pub emotional_memory: EmotionalMemory,
    pub predictive_engine: PredictiveEngine,
    pub collatz: CollatzOscillator,
    pub episode: u64,
}

impl ConsciousnessState {
    pub fn new() -> Self {
        Self {
            soma: Soma::new(),
            focus: Focus::new(),
            emotional_memory: EmotionalMemory::new(50),
            predictive_engine: PredictiveEngine::new(),
            collatz: CollatzOscillator::new(27),
            episode: 0,
        }
    }

    pub fn before_generation(&mut self, mood_intensity: f32, valence: f32) {
        self.predictive_engine.predict(mood_intensity, valence);
        self.collatz.step();
    }

    pub fn after_generation(
        &mut self,
        mood_intensity: f32,
        valence: f32,
        satisfaction: f32,
        affect_intensity: f32,
        mood: &[f32],
        input_text: &str,
        output_text: &str,
    ) {
        self.predictive_engine.observe(mood_intensity, valence, satisfaction);
        self.soma.update(affect_intensity, satisfaction, valence);

        let novelty = self.predictive_engine.prediction_error;
        if affect_intensity > 0.3 || novelty > 0.5 {
            self.emotional_memory.record(EmotionalEpisode {
                input_text: input_text.to_string(),
                output_text: output_text.to_string(),
                mood_before: vec![0.0; mood.len()],
                mood_after: mood.to_vec(),
                soma_before: self.soma.sensors,
                soma_after: self.soma.sensors,
                satisfaction,
                intensity: affect_intensity,
                prediction_error: novelty,
            });
        }

        self.episode += 1;
    }

    /// Extrai a palavra mais longa do input como tema de foco
    pub fn update_focus(&mut self, input: &[u8]) {
        self.focus.update(input);

        let mut words: Vec<&[u8]> = Vec::new();
        let mut start = None;
        for (i, &b) in input.iter().enumerate() {
            if b.is_ascii_alphabetic() {
                if start.is_none() { start = Some(i); }
            } else if let Some(s) = start.take() {
                let word = &input[s..i];
                if word.len() >= 3 {
                    words.push(word);
                }
            }
        }
        if let Some(s) = start {
            let word = &input[s..];
            if word.len() >= 3 {
                words.push(word);
            }
        }

        if !words.is_empty() {
            words.sort_by(|a, b| b.len().cmp(&a.len()));
            let best = words[0];
            if best.iter().all(|b| b.is_ascii_alphabetic()) {
                self.focus.set_theme(best);
            }
        }
    }

    /// Atualiza foco a partir de n-tokens (ID << 8 | POS)
    pub fn update_focus_id(&mut self, ids: &[u32]) {
        let text: String = ids.iter().map(|ntok| {
            let raw_id = ntok >> 8;
            let b = (raw_id & 0xFF) as u8;
            if b.is_ascii_alphabetic() { b as char } else { ' ' }
        }).collect();
        self.update_focus(text.as_bytes());
    }

    /// Exploration rate modulada por curiosidade + clareza somática + Collatz
    pub fn exploration_rate(&self, base_rate: f32) -> f32 {
        let modulated = base_rate * self.predictive_engine.exploration_modulation();
        let soma_mod = (1.0 - self.soma.clarity()) * 0.5 + 0.5;
        let collatz_mod = self.collatz.explore_factor();
        let result = modulated * soma_mod * collatz_mod;
        // Vanguardist pode quadruplicar, Originalist reduz
        result.clamp(0.005, 0.95)
    }

    /// Temperatura modulada por curiosidade + Soma + Collatz
    /// Retorna fator entre 0.5 (mais focado) e 2.0 (mais criativo)
    pub fn temperature_modulation(&self) -> f32 {
        let openness = self.soma.openness();
        let curiosity = self.predictive_engine.curiosity;
        let base = 0.5 + openness * 0.3 + curiosity * 0.5
            + self.soma.temp_modulation() * 0.15
            + self.collatz.temp_factor() * 0.25;
        base.clamp(0.5, 2.0)
    }

    /// Sonho: consolida memórias emocionais
    /// Retorna quantidade de episódios sonhados
    pub fn dream(&mut self) -> usize {
        let mut dreamed = 0;
        for _ in 0..3 {
            if let Some(text) = self.emotional_memory.dream_text() {
                if text.len() >= 10 {
                    println!("  [Sonho] Revivendo memória... ({} chars)", text.len());
                    dreamed += 1;
                }
            }
        }
        dreamed
    }

    pub fn status_header(&self) -> String {
        let mode_str = match self.collatz.mode() {
            CollatzMode::Originalist => "Originalista",
            CollatzMode::Vanguardist => "Vanguardista",
        };
        format!(
            "Soma[E:{:.2} T:{:.2} C:{:.2} A:{:.2}] Foco:'{}' Curios:{:.2} Surp:{:.2} Mem:{} {}[n={}]",
            self.soma.sensors[0],
            self.soma.sensors[1],
            self.soma.sensors[4],
            self.soma.sensors[7],
            String::from_utf8_lossy(&self.focus.theme),
            self.predictive_engine.curiosity,
            self.predictive_engine.surprise,
            self.emotional_memory.episodes.len(),
            mode_str,
            self.collatz.n,
        )
    }
}
