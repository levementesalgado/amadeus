use crate::amadeus_m::token7::Token7;

/// Papel retórico de uma sentença dentro do parágrafo
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RhetoricRole {
    /// Abertura: apresenta o tópico
    Introduction,
    /// Desenvolvimento: expande a ideia
    Development,
    /// Contraste: apresenta visão oposta
    Contrast,
    /// Exemplo: ilustra com caso concreto
    Example,
    /// Transição: conecta com próxima ideia
    Transition,
    /// Conclusão: fecha o raciocínio
    Conclusion,
}

/// Template de estrutura: sequência de classes POS esperadas
#[derive(Debug, Clone)]
pub struct RhetoricTemplate {
    pub role: RhetoricRole,
    /// Classes POS esperadas em ordem (valores de 0-6)
    /// None = qualquer classe
    pub pos_sequence: Vec<Option<u8>>,
    /// Comprimento mínimo da sentença
    pub min_len: usize,
    /// Comprimento máximo da sentença
    pub max_len: usize,
    /// Probabilidade de usar esta estrutura
    pub weight: f32,
}

impl RhetoricTemplate {
    pub const fn new(role: RhetoricRole, pos_sequence: Vec<Option<u8>>, min_len: usize, max_len: usize, weight: f32) -> Self {
        Self { role, pos_sequence, min_len, max_len, weight }
    }
}

/// Templates retóricos para português
pub fn portuguese_templates() -> Vec<RhetoricTemplate> {
    vec![
        // INTRODUÇÃO: ART + SUBST + VERB + ...
        // "O Brasil é um país grande"
        RhetoricTemplate::new(
            RhetoricRole::Introduction,
            vec![Some(3), Some(0), Some(1), None, None], // ART SUBS VERB
            3, 8, 0.25,
        ),
        // "A natureza mostra que..."
        RhetoricTemplate::new(
            RhetoricRole::Introduction,
            vec![Some(3), Some(0), Some(1), None, None, None],
            4, 10, 0.15,
        ),

        // DESENVOLVIMENTO: SUBS + VERB + SUBS/OBJ
        // "as pessoas gostam de natureza"
        RhetoricTemplate::new(
            RhetoricRole::Development,
            vec![Some(0), Some(1), None, None, None],
            3, 8, 0.20,
        ),
        // "isso mostra a importância da vida"
        RhetoricTemplate::new(
            RhetoricRole::Development,
            vec![None, Some(1), Some(3), Some(0), None, None],
            4, 10, 0.15,
        ),

        // CONTRASTE: CONJ + ART + SUBS + VERB
        // "mas a realidade é diferente"
        RhetoricTemplate::new(
            RhetoricRole::Contrast,
            vec![Some(5), Some(3), Some(0), Some(1), None],
            3, 7, 0.10,
        ),

        // EXEMPLO: ART + SUBS + VERB + PREP
        // "o caso do Brasil mostra"
        RhetoricTemplate::new(
            RhetoricRole::Example,
            vec![Some(3), Some(0), Some(1), None, None],
            3, 8, 0.10,
        ),

        // TRANSIÇÃO: ADV + VERB + ...
        // "assim podemos ver que"
        RhetoricTemplate::new(
            RhetoricRole::Transition,
            vec![Some(4), Some(1), None, None, None],
            3, 6, 0.05,
        ),

        // CONCLUSÃO: ART + SUBS + VERB + ADJ
        // "essa é a verdade real"
        RhetoricTemplate::new(
            RhetoricRole::Conclusion,
            vec![Some(3), Some(0), Some(1), None, None],
            3, 7, 0.10,
        ),
    ]
}

/// Planejador retórico: decide a estrutura antes de gerar
pub struct RhetoricPlanner {
    templates: Vec<RhetoricTemplate>,
    /// Papel atual no parágrafo
    current_role: RhetoricRole,
    /// Sentenças geradas no parágrafo
    sentences_in_paragraph: usize,
    /// Total de sentenças antes de mudar de parágrafo
    paragraph_length: usize,
    /// RNG
    rng: fastrand::Rng,
}

impl RhetoricPlanner {
    pub fn new() -> Self {
        Self {
            templates: portuguese_templates(),
            current_role: RhetoricRole::Introduction,
            sentences_in_paragraph: 0,
            paragraph_length: 4, // 4-6 sentenças por parágrafo
            rng: fastrand::Rng::with_seed(42),
        }
    }

    /// Decidir o próximo papel retórico
    pub fn next_role(&mut self) -> RhetoricRole {
        self.sentences_in_paragraph += 1;

        self.current_role = if self.sentences_in_paragraph == 1 {
            RhetoricRole::Introduction
        } else if self.sentences_in_paragraph >= self.paragraph_length - 1 {
            RhetoricRole::Conclusion
        } else if self.rng.f32() < 0.15 {
            RhetoricRole::Contrast
        } else if self.rng.f32() < 0.25 {
            RhetoricRole::Example
        } else if self.rng.f32() < 0.10 {
            RhetoricRole::Transition
        } else {
            RhetoricRole::Development
        };

        // Reset paragraph se necessário
        if self.sentences_in_paragraph >= self.paragraph_length {
            self.sentences_in_paragraph = 0;
            self.paragraph_length = 4 + self.rng.usize(0..3); // 4-6 sentenças
        }

        self.current_role
    }

    /// Selecionar template para o papel atual
    pub fn select_template(&mut self) -> &RhetoricTemplate {
        let candidates: Vec<&RhetoricTemplate> = self.templates.iter()
            .filter(|t| t.role == self.current_role)
            .collect();

        if candidates.is_empty() {
            return &self.templates[0]; // fallback
        }

        // Selecionar por peso
        let total_weight: f32 = candidates.iter().map(|t| t.weight).sum();
        let mut r = self.rng.f32() * total_weight;
        let mut last = candidates[0];
        for &t in &candidates {
            r -= t.weight;
            if r <= 0.0 {
                return t;
            }
            last = t;
        }
        last
    }

    /// Gerar esqueleto de sentença baseado no template
    pub fn plan_sentence(&mut self) -> Vec<Option<u8>> {
        let template = self.select_template();
        let mut skeleton = template.pos_sequence.clone();
        let min_len = template.min_len;
        let max_len = template.max_len;

        // Ajustar comprimento
        let target_len = self.rng.usize(min_len..=max_len);
        while skeleton.len() < target_len {
            skeleton.push(None); // qualquer classe
        }
        skeleton.truncate(target_len);

        skeleton
    }

    /// Verificar se um token segue o esqueleto
    pub fn token_matches_skeleton(&self, token: &Token7, skeleton_pos: Option<u8>) -> bool {
        match skeleton_pos {
            None => true, // qualquer classe
            Some(expected) => {
                let actual = (token.morph & 0x7) as u8;
                actual == expected
            }
        }
    }

    /// Reset para novo parágrafo
    pub fn reset_paragraph(&mut self) {
        self.sentences_in_paragraph = 0;
        self.current_role = RhetoricRole::Introduction;
    }

    /// Obter posição atual no parágrafo (0.0 - 1.0)
    pub fn paragraph_progress(&self) -> f32 {
        self.sentences_in_paragraph as f32 / self.paragraph_length as f32
    }
}

impl Default for RhetoricPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planner_creates_templates() {
        let planner = RhetoricPlanner::new();
        assert!(!planner.templates.is_empty());
    }

    #[test]
    fn test_planner_roles() {
        let mut planner = RhetoricPlanner::new();
        let role = planner.next_role();
        assert_eq!(role, RhetoricRole::Introduction); // primeira sentença
    }

    #[test]
    fn test_planner_skeleton() {
        let mut planner = RhetoricPlanner::new();
        planner.next_role();
        let skeleton = planner.plan_sentence();
        assert!(!skeleton.is_empty());
    }
}
