use crate::amadeus_m::collatz::CollatzOscillator;
use crate::amadeus_m::token7::{
    CLASS_SUBSTANTIVO, CLASS_VERBO, CLASS_ARTIGO,
    CLASS_PREPOSICAO, CLASS_OUTRO,
};

pub struct ContextPattern {
    pub prev_pos: Option<u8>,
    pub curr_pos: Option<u8>,
    pub prev_bits_mask: Option<u128>,
}

pub enum ConstraintAction {
    ForcePOS(u8),
    BlockPOS(u8),
    WeightMul(f32),
}

pub struct ConstitRule {
    pub name: String,
    pub pattern: ContextPattern,
    pub action: ConstraintAction,
}

pub struct Amendment {
    pub name: String,
    pub pattern: ContextPattern,
    pub action: ConstraintAction,
    pub weight: f32,
    pub votes_for: u32,
    pub votes_against: u32,
    pub active: bool,
}

pub struct Constitution {
    pub rules: Vec<ConstitRule>,
    pub amendments: Vec<Amendment>,
}

impl Default for Constitution {
    fn default() -> Self {
        Self::new()
    }
}

impl Constitution {
    pub fn new() -> Self {
        let rules = vec![
            ConstitRule {
                name: "artigo_rege_substantivo".into(),
                pattern: ContextPattern {
                    prev_pos: Some(CLASS_ARTIGO),
                    curr_pos: None,
                    prev_bits_mask: None,
                },
                action: ConstraintAction::ForcePOS(CLASS_SUBSTANTIVO),
            },
            ConstitRule {
                name: "preposicao_rege_termo".into(),
                pattern: ContextPattern {
                    prev_pos: Some(CLASS_PREPOSICAO),
                    curr_pos: None,
                    prev_bits_mask: None,
                },
                action: ConstraintAction::BlockPOS(CLASS_VERBO),
            },
            ConstitRule {
                name: "artigo_concorda_com_substantivo".into(),
                pattern: ContextPattern {
                    prev_pos: Some(CLASS_ARTIGO),
                    curr_pos: Some(CLASS_SUBSTANTIVO),
                    prev_bits_mask: None,
                },
                action: ConstraintAction::WeightMul(1.5),
            },
        ];

        Self {
            rules,
            amendments: Vec::new(),
        }
    }

    pub fn add_amendment(&mut self, name: &str, pattern: ContextPattern, action: ConstraintAction) {
        self.amendments.push(Amendment {
            name: name.into(),
            pattern,
            action,
            weight: 0.5,
            votes_for: 0,
            votes_against: 0,
            active: true,
        });
    }

    pub fn vote(&mut self, index: usize, approve: bool) {
        if index >= self.amendments.len() { return; }
        if approve {
            self.amendments[index].votes_for += 1;
        } else {
            self.amendments[index].votes_against += 1;
        }
    }

    pub fn apply_amendments(&mut self, collatz: &CollatzOscillator) {
        let vote_weight = match collatz.mode() {
            crate::amadeus_m::collatz::CollatzMode::Originalist => 0.8,
            crate::amadeus_m::collatz::CollatzMode::Vanguardist => 1.2,
        };

        for a in &mut self.amendments {
            if a.votes_for + a.votes_against == 0 { continue; }
            let ratio = a.votes_for as f32 / (a.votes_for + a.votes_against) as f32;
            let delta = (ratio - 0.5) * 2.0 * vote_weight * 0.1;
            a.weight = (a.weight + delta).clamp(0.0, 1.0);
            a.active = a.weight > 0.3;
        }
    }

    pub fn check_rules(&self, prev_pos: u8, curr_pos: u8, prev_bits: u128) -> Vec<&ConstraintAction> {
        let mut actions = Vec::new();

        for rule in &self.rules {
            if Self::matches(&rule.pattern, prev_pos, curr_pos, prev_bits) {
                actions.push(&rule.action);
            }
        }

        for a in &self.amendments {
            if !a.active { continue; }
            if fastrand::f32() < a.weight {
                if Self::matches(&a.pattern, prev_pos, curr_pos, prev_bits) {
                    actions.push(&a.action);
                }
            }
        }

        actions
    }

    fn matches(pattern: &ContextPattern, prev_pos: u8, curr_pos: u8, prev_bits: u128) -> bool {
        if let Some(pp) = pattern.prev_pos {
            if pp != prev_pos { return false; }
        }
        if let Some(cp) = pattern.curr_pos {
            if cp != curr_pos { return false; }
        }
        if let Some(pb) = pattern.prev_bits_mask {
            if (prev_bits & pb) == 0 { return false; }
        }
        true
    }

    pub fn status(&self) -> String {
        let mut out = format!("  Constituição: {} regras\n", self.rules.len());
        let active = self.amendments.iter().filter(|a| a.active).count();
        let total = self.amendments.len();
        out.push_str(&format!("  Emendas: {} ativas / {} total\n", active, total));
        for a in &self.amendments {
            let status = if a.active { "ativa" } else { "inativa" };
            out.push_str(&format!("    {}: peso={:.1} votos={}/{} ({})\n",
                a.name, a.weight, a.votes_for, a.votes_against, status));
        }
        out
    }
}

pub fn default_constitution() -> Constitution {
    let mut c = Constitution::new();

    c.add_amendment(
        "preferir_verbo_apos_pronome",
        ContextPattern {
            prev_pos: Some(CLASS_OUTRO),
            curr_pos: None,
            prev_bits_mask: None,
        },
        ConstraintAction::ForcePOS(CLASS_VERBO),
    );

    c.add_amendment(
        "preferir_adverbio_apos_verbo",
        ContextPattern {
            prev_pos: Some(CLASS_VERBO),
            curr_pos: None,
            prev_bits_mask: None,
        },
        ConstraintAction::ForcePOS(CLASS_OUTRO),
    );

    c
}
