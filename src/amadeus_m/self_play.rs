use std::collections::HashMap;
use crate::amadeus_m::token7::Token7;

/// Resultado da avaliação de texto gerado
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// Score total (0.0 - 1.0)
    pub score: f32,
    /// Aprovação (score > threshold)
    pub passed: bool,
    /// Métricas individuais
    pub structure_score: f32,
    pub diversity_score: f32,
    pub coherence_score: f32,
    pub length_score: f32,
    /// Razão da rejeição (se houver)
    pub reason: String,
}

/// Filtro de self-play: avalia texto gerado sem API externa
pub struct SelfPlayFilter {
    /// Threshold mínimo para aprovação
    pub threshold: f32,
    ///GRAPH embeddings para coerência
    graph: HashMap<u32, u32>,
    /// Estatísticas do corpus de treino
    corpus_freq: HashMap<u8, f32>, // classe POS → frequência relativa
    /// Número mínimo de tokens
    pub min_tokens: usize,
    /// Número máximo de tokens
    pub max_tokens: usize,
}

impl SelfPlayFilter {
    pub fn new(graph: HashMap<u32, u32>) -> Self {
        // Frequências esperadas de classes POS em português
        let mut corpus_freq = HashMap::new();
        corpus_freq.insert(0, 0.45); // SUBST
        corpus_freq.insert(1, 0.15); // VERBO
        corpus_freq.insert(2, 0.08); // ADJ
        corpus_freq.insert(3, 0.10); // ART
        corpus_freq.insert(4, 0.05); // ADV
        corpus_freq.insert(5, 0.12); // PREP/CONJ
        corpus_freq.insert(6, 0.05); // PONT

        Self {
            threshold: 0.75, // Mais exigente: só aprova textos bons
            graph,
            corpus_freq,
            min_tokens: 5,
            max_tokens: 20,
        }
    }

    /// Avaliar texto gerado
    pub fn evaluate(&self, tokens: &[Token7]) -> EvalResult {
        if tokens.is_empty() {
            return EvalResult {
                score: 0.0,
                passed: false,
                structure_score: 0.0,
                diversity_score: 0.0,
                coherence_score: 0.0,
                length_score: 0.0,
                reason: "vazio".to_string(),
            };
        }

        // 1. Comprimento
        let length_score = self.evaluate_length(tokens.len());

        // 2. Estrutura POS
        let structure_score = self.evaluate_structure(tokens);

        // 3. Diversidade léxica
        let diversity_score = self.evaluate_diversity(tokens);

        // 4. Coerência semântica (GRAPH)
        let coherence_score = self.evaluate_coherence(tokens);

        // Score final: média ponderada
        let score = structure_score * 0.35
            + diversity_score * 0.25
            + coherence_score * 0.25
            + length_score * 0.15;

        let passed = score >= self.threshold;

        let reason = if !passed {
            if structure_score < 0.3 {
                "estrutura POS ruim".to_string()
            } else if diversity_score < 0.3 {
                "repetitivo".to_string()
            } else if coherence_score < 0.3 {
                "sem coerência semântica".to_string()
            } else if length_score < 0.3 {
                "comprimento inadequado".to_string()
            } else {
                "score baixo".to_string()
            }
        } else {
            "aprovado".to_string()
        };

        EvalResult {
            score,
            passed,
            structure_score,
            diversity_score,
            coherence_score,
            length_score,
            reason,
        }
    }

    /// Avaliar comprimento
    fn evaluate_length(&self, len: usize) -> f32 {
        if len < self.min_tokens {
            return 0.2;
        }
        if len > self.max_tokens {
            return 0.5;
        }
        // Score ideal: 5-15 tokens
        if len >= 5 && len <= 15 {
            1.0
        } else if len < 5 {
            0.6
        } else {
            0.8
        }
    }

    /// Avaliar estrutura POS
    fn evaluate_structure(&self, tokens: &[Token7]) -> f32 {
        let mut score = 0.0f32;
        let mut checks = 0;

        // Verificar se começa com ART ou SUBST (comum em português)
        if let Some(first) = tokens.first() {
            let cls = first.morph & 0x7;
            if cls == 3 || cls == 0 { // ART ou SUBST
                score += 0.3;
            }
            checks += 1;
        }

        // Verificar se tem verbo (nem toda frase precisa, mas ajuda)
        let has_verb = tokens.iter().any(|t| (t.morph & 0x7) == 1);
        if has_verb {
            score += 0.3;
        }
        checks += 1;

        // Verificar proporção de classes (não pode ser 100% de uma classe)
        let mut class_counts = [0u32; 7];
        for t in tokens {
            let cls = (t.morph & 0x7) as usize;
            if cls < 7 {
                class_counts[cls] += 1;
            }
        }
        let total = tokens.len() as f32;
        let max_class = class_counts.iter().copied().max().unwrap_or(0) as f32;
        if max_class / total < 0.7 { // Nenhuma classe > 70%
            score += 0.4;
        }
        checks += 1;

        score / checks as f32
    }

    /// Avaliar diversidade léxica
    fn evaluate_diversity(&self, tokens: &[Token7]) -> f32 {
        let total = tokens.len();
        if total == 0 { return 0.0; }

        let unique: usize = tokens.iter()
            .map(|t| t.lex)
            .collect::<std::collections::HashSet<_>>()
            .len();

        let ratio = unique as f32 / total as f32;

        // Tipo-token ratio: ideal > 0.7
        if ratio > 0.8 {
            1.0
        } else if ratio > 0.6 {
            0.8
        } else if ratio > 0.4 {
            0.5
        } else {
            0.2 // Muito repetitivo
        }
    }

    /// Avaliar coerência semântica via GRAPH
    fn evaluate_coherence(&self, tokens: &[Token7]) -> f32 {
        if tokens.len() < 2 || self.graph.is_empty() {
            return 0.5; // Neutro se sem dados
        }

        let mut total_hamming = 0u32;
        let mut pairs = 0u32;

        for i in 0..tokens.len() - 1 {
            let g1 = self.graph.get(&tokens[i].lex).copied().unwrap_or(0);
            let g2 = self.graph.get(&tokens[i + 1].lex).copied().unwrap_or(0);

            // Pular pares onde ambos são 0
            if g1 == 0 && g2 == 0 { continue; }

            let hamming = (g1 ^ g2).count_ones();
            total_hamming += hamming;
            pairs += 1;
        }

        if pairs == 0 {
            return 0.5;
        }

        let avg_hamming = total_hamming as f32 / pairs as f32;

        // Hamming ideal: 10-20 (nem muito similar nem muito diferente)
        if avg_hamming >= 8.0 && avg_hamming <= 20.0 {
            1.0
        } else if avg_hamming < 8.0 {
            0.6 // Muito similar (repetitivo semanticamente)
        } else {
            0.4 // Muito diferente (sem coerência)
        }
    }
}

impl Default for SelfPlayFilter {
    fn default() -> Self {
        Self::new(HashMap::new())
    }
}

/// Self-play loop: gerar → avaliar → retreinar no que passou
pub fn self_play_round(
    generate_fn: &mut dyn FnMut() -> Vec<Token7>,
    filter: &SelfPlayFilter,
    n_attempts: usize,
) -> (Vec<Vec<Token7>>, Vec<EvalResult>) {
    let mut accepted = Vec::new();
    let mut all_results = Vec::new();

    for _ in 0..n_attempts {
        let text = generate_fn();
        let result = filter.evaluate(&text);

        all_results.push(result.clone());
        if result.passed {
            accepted.push(text);
        }
    }

    (accepted, all_results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_empty() {
        let filter = SelfPlayFilter::default();
        let result = filter.evaluate(&[]);
        assert!(!result.passed);
    }

    #[test]
    fn test_filter_short() {
        let filter = SelfPlayFilter::default();
        let tokens = vec![Token7::new(1, 0)];
        let result = filter.evaluate(&tokens);
        assert!(!result.passed);
    }
}
