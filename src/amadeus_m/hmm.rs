/// HMM (Hidden Markov Model) para POS Tagging
///
/// Aprende classes morfológicas automaticamente do texto bruto,
/// sem necessidade de rótulos manuais.
///
/// States: SUBST(0), VERBO(1), ADJ(2), ART(3), ADV(4), PREP(5), CONJ(6), PONT(7)

use std::collections::HashMap;

pub const N_TAGS: usize = 8;
pub const TAG_SUBST: usize = 0;
pub const TAG_VERBO: usize = 1;
pub const TAG_ADJ: usize = 2;
pub const TAG_ART: usize = 3;
pub const TAG_ADV: usize = 4;
pub const TAG_PREP: usize = 5;
pub const TAG_CONJ: usize = 6;
pub const TAG_PONT: usize = 7;

/// HMM treinado para POS tagging
pub struct HmmPosTagger {
    /// P(tag_i | tag_{i-1}) — transições
    pub transition: Vec<Vec<f64>>,
    /// P(word | tag) — emissões (Top-K por tag)
    pub emission: Vec<HashMap<String, f64>>,
    /// Prior P(tag)
    pub prior: Vec<f64>,
    /// Contadores para treinamento
    trans_count: Vec<Vec<f64>>,
    emit_count: Vec<HashMap<String, f64>>,
    prior_count: Vec<f64>,
    /// Tabela de sufixos para suavização
    suffix_table: HashMap<String, HashMap<String, f64>>,
}

impl HmmPosTagger {
    pub fn new() -> Self {
        let mut hmm = Self {
            transition: vec![vec![0.0; N_TAGS]; N_TAGS],
            emission: vec![HashMap::new(); N_TAGS],
            prior: vec![0.0; N_TAGS],
            trans_count: vec![vec![0.0; N_TAGS]; N_TAGS],
            emit_count: vec![HashMap::new(); N_TAGS],
            prior_count: vec![0.0; N_TAGS],
            suffix_table: HashMap::new(),
        };
        hmm.init_seeds();
        hmm
    }

    /// Inicializar com seeds de palavras funcionais (alta confiança)
    fn init_seeds(&mut self) {
        // ─── PRIOR INICIAL (uniforme — balanced sampling) ───
        // Todas as classes começam com peso igual
        for tag in 0..N_TAGS {
            self.prior_count[tag] = 10.0;
        }

        // Artigos — inambíguos
        let arts = ["o", "os", "a", "as", "um", "uns", "uma", "umas"];
        for w in &arts {
            self.emit_count[TAG_ART].insert(w.to_string(), 100.0);
        }

        // Preposições — inambíguas
        let preps = [
            "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
            "por", "para", "com", "sem", "sob", "entre", "até", "desde",
            "contra", "perante", "após", "sobre", "sob",
        ];
        for w in &preps {
            self.emit_count[TAG_PREP].insert(w.to_string(), 100.0);
        }

        // Conjunções — inambíguas
        let conjs = [
            "e", "ou", "mas", "porém", "então", "contudo", "todavia",
            "que", "se", "como", "quando", "onde", "porque", "embora",
            "enquanto", "caso", "mesmo",
        ];
        for w in &conjs {
            self.emit_count[TAG_CONJ].insert(w.to_string(), 100.0);
        }

        // Pronomes — alta confiança
        let prons = [
            "eu", "tu", "ele", "ela", "nós", "vós", "eles", "elas",
            "me", "te", "se", "lhe", "nos", "vos", "lhes",
            "meu", "minha", "teu", "tua", "seu", "sua",
        ];
        for w in &prons {
            self.emit_count[TAG_ART].entry(w.to_string()).and_modify(|e| *e += 50.0).or_insert(50.0);
        }

        // Verbos comuns — alta confiança (infinitivo e formas conjugadas)
        let verbs = [
            "ser", "estar", "ter", "fazer", "dizer", "ir", "vir", "poder",
            "saber", "querer", "ver", "dar", "ficar", "parecer", "haver",
            "trazer", "encontrar", "deixar", "chegar", "passar", "viver",
            "produzir", "existir", "começar", "considerar", "apresentar",
            "desenvolver", "partir", "manter", "criar", "perder", "acabar",
            "formar", "trabalhar", "realizar", "morrer", "nascer", "andar",
            "cair", "ler", "abranger", "basear", "depender", "gerar",
            "variar", "permitir", "constituir", "processar", "estabelecer",
            "contribuir", "significar", "adquirir", "substituir", "compor",
            // Formas conjugadas MUITO comuns em Wikipedia
            "é", "são", "está", "estão", "foi", "foram", "será", "tinha",
            "tem", "têm", "há", "pode", "podem", "deve", "devem",
            "fez", "faz", "fazem", "disse", "diz", "dizem",
            "ficou", "fica", "ficam", "chegou", "chega", "chegam",
            "passou", "passa", "passam", "entrou", "entra", "entraram",
            "começou", "começa", "começaram", "apareceu", "aparece",
            "realizou", "realiza", "desenvolveu", "desenvolve",
            "encontrou", "encontra", "produziu", "produz",
            "existiu", "existem", "existia", "existiam",
            "existiu", "existem", "estava", "estavam",
            "tiveram", "tenham", "seja", "sejam",
            "era", "eram", "seria", "seriam",
            "estaria", "estariam", "teria", "teriam",
            "poderia", "poderiam", "deveria", "deveriam",
            "fez", "fizeram", "tenha", "tenham",
            "sendo", "tendo", "fazendo", "dizendo",
            "feito", "dito", "posto", "visto",
            // Participios comuns
            "utilizado", "utilizada", "utilizados", "utilizadas",
            "considerado", "considerada", "considerados", "consideradas",
            "realizado", "realizada", "realizados", "realizadas",
            "apresentado", "apresentada", "apresentados", "apresentadas",
            "encontrado", "encontrada", "encontrados", "encontradas",
            "desenvolvido", "desenvolvida", "desenvolvidos", "desenvolvidas",
            "criado", "criada", "criados", "criadas",
            "perdido", "perdida", "perdidos", "perdidas",
            "acabado", "acabada", "acabados", "acabadas",
            "formado", "formada", "formados", "formadas",
            "trabalhado", "trabalhada", "trabalhados", "trabalhadas",
            // Gerundios comuns
            "sendo", "tendo", "fazendo", "dizendo", "trabalhando",
            "passando", "chegando", "entrando", "começando",
        ];
        for w in &verbs {
            self.emit_count[TAG_VERBO].insert(w.to_string(), 100.0);
        }

        // Adjetivos comuns — alta confiança
        let adjs = [
            "grande", "pequeno", "novo", "nova", "primeiro", "primeira",
            "último", "última", "maior", "menor", "melhor", "pior",
            "bom", "boa", "mau", "má", "bonito", "bonita",
            "importante", "diferente", "próximo", "próxima",
            "possível", "necessário", "necessária", "principal",
            "geral", "próprio", "própria", "humano", "humana",
            "natural", "político", "política", "social", "econômico",
            "grande", "pequeno", "alto", "baixo", "largo", "estrito",
            "forte", "fraco", "rápido", "lento", "duro", "macio",
            "quente", "frio", "novo", "velho", "jovem", "antigo",
            "claro", "escuro", "limpo", "sujo", "cheio", "vazio",
            // Adjetivos muito comuns em Wikipedia
            "brasileiro", "brasileira", "brasileiros", "brasileiras",
            "internacional", "internacionais",
            "nacional", "nacionais",
            "federal", "federais",
            "estadual", "estaduais",
            "municipal", "municipais",
            "público", "pública", "públicos", "públicas",
            "privado", "privada", "privados", "privadas",
            "grande", "grandes",
            "pequeno", "pequena", "pequenos", "pequenas",
            "novo", "nova", "novos", "novas",
            "antigo", "antiga", "antigos", "antigas",
            "diferentes", "diferentes",
            "importante", "importantes",
            "maiores", "menores", "melhores", "piores",
            "primeiro", "primeira", "primeiros", "primeiras",
            "último", "última", "últimos", "últimas",
            "próximo", "próxima", "próximos", "próximas",
            "anterior", "anteriores",
            "posterior", "posteriores",
            "principais", "principais",
            "diversos", "diversas",
            "vários", "várias",
            "outro", "outra", "outros", "outras",
            "mesmo", "mesma", "mesmos", "mesmas",
            "todo", "toda", "todos", "todas",
            "cada", "qualquer",
            "certo", "certa", "certos", "certas",
            "próprio", "própria", "próprios", "próprias",
        ];
        for w in &adjs {
            self.emit_count[TAG_ADJ].insert(w.to_string(), 80.0);
        }

        // Advérbios comuns — alta confiança
        let advs = [
            "não", "sim", "também", "muito", "mais", "menos", "bem", "mal",
            "já", "ainda", "sempre", "nunca", "aqui", "lá", "onde",
            "agora", "então", "depois", "antes", "hoje", "ontem",
            "quase", "apenas", "talvez", "realmente",
        ];
        for w in &advs {
            self.emit_count[TAG_ADV].insert(w.to_string(), 100.0);
        }

        // Pontuação
        let pons = [",", ".", "!", "?", ";", ":", "—", "-", "\"", "(", ")"];
        for w in &pons {
            self.emit_count[TAG_PONT].insert(w.to_string(), 100.0);
        }

        // Transições iniciais suaves (uniforme com viés gramatical)
        // ART → SUBST (artigo antes de substantivo)
        self.trans_count[TAG_ART][TAG_SUBST] = 50.0;
        // PREP → SUBST (preposição antes de substantivo)
        self.trans_count[TAG_PREP][TAG_SUBST] = 30.0;
        // PREP → ART (preposição antes de artigo: "de o" → "do")
        self.trans_count[TAG_PREP][TAG_ART] = 20.0;
        // SUBST → VERBO (sujeito antes de verbo)
        self.trans_count[TAG_SUBST][TAG_VERBO] = 30.0;
        // VERBO → SUBST (verbo antes de objeto)
        self.trans_count[TAG_VERBO][TAG_SUBST] = 25.0;
        // SUBST → PREP (substantivo antes de preposição)
        self.trans_count[TAG_SUBST][TAG_PREP] = 20.0;
        // ADJ → SUBST (adjetivo depois de substantivo: "casa grande")
        self.trans_count[TAG_SUBST][TAG_ADJ] = 15.0;
        // SUBST → CONJ
        self.trans_count[TAG_SUBST][TAG_CONJ] = 10.0;
        // VERBO → CONJ
        self.trans_count[TAG_VERBO][TAG_CONJ] = 10.0;
        // PONT → ART (frase começa com artigo)
        self.trans_count[TAG_PONT][TAG_ART] = 30.0;
        // PONT → SUBST
        self.trans_count[TAG_PONT][TAG_SUBST] = 20.0;
        // PONT → PREP
        self.trans_count[TAG_PONT][TAG_PREP] = 10.0;
    }

    /// Treinar o HMM com texto bruto (tokenizado)
    /// Usa contagem de contexto (não Viterbi) para evitar auto-reforço
    pub fn train(&mut self, sentences: &[Vec<String>]) {
        // Fase 1: Usar seeds para taggear APENAS palavras conhecidas
        // Palavras desconhecidas ficam como "observadas" sem tag fixa
        for sent in sentences {
            let mut prev_tag: Option<usize> = None;

            for word in sent {
                // Verificar se a palavra é um seed conhecido
                let known_tag = self.known_tag(word);

                let tag = if let Some(t) = known_tag {
                    // Palavra conhecida — usar tag do seed
                    t
                } else {
                    // Palavra desconhecida — usar previsão baseada em contexto
                    self.predict_tag(word, prev_tag)
                };

                self.prior_count[tag] += 1.0;
                self.emit_count[tag].entry(word.clone()).and_modify(|e| *e += 1.0).or_insert(1.0);

                if let Some(pt) = prev_tag {
                    self.trans_count[pt][tag] += 1.0;
                }
                prev_tag = Some(tag);
            }
        }

        // Fase 2: Aprender sufixos (suavização para palavras desconhecidas)
        self.learn_suffixes();

        // Fase 3: Normalizar probabilidades
        self.normalize();
    }

    /// Verificar se uma palavra é um seed conhecido e retornar sua tag
    fn known_tag(&self, word: &str) -> Option<usize> {
        // Verificar em cada tag se a palavra tem contagem alta (seed)
        for tag in 0..N_TAGS {
            if let Some(&count) = self.emit_count[tag].get(word) {
                if count >= 40.0 { // Threshold de seed
                    return Some(tag);
                }
            }
        }
        None
    }

    /// Aprender probabilidades de sufixos a partir das emissões
    fn learn_suffixes(&mut self) {
        // Para cada tag, coletar sufixos de palavras conhecidas
        let suffixes = ["ção", "ções", "mento", "mentos", "ismo", "idade",
                        "oso", "osa", "vel", "ável", "ível", "al",
                        "ar", "er", "ir", "or",
                        "ando", "endo", "indo",
                        "ado", "ido",
                        "mente", "amente", "eira", "eiro"];

        for tag in 0..N_TAGS {
            let mut suffix_freq: HashMap<String, f64> = HashMap::new();
            let total: f64 = self.emit_count[tag].values().sum();

            for (word, &count) in &self.emit_count[tag] {
                for suffix in &suffixes {
                    if word.ends_with(suffix) && word.len() > suffix.len() {
                        *suffix_freq.entry(suffix.to_string()).or_insert(0.0) += count;
                    }
                }
            }

            // Normalizar suffix frequencies
            if total > 0.0 {
                for (_, count) in &mut suffix_freq {
                    *count /= total;
                }
            }

            self.suffix_table.insert(
                format!("tag_{}", tag),
                suffix_freq
            );
        }
    }

    /// Normalizar contadores em probabilidades
    fn normalize(&mut self) {
        // ─── PRIORS BALANCEADOS ───
        // Forçar priors uniformes para balanced sampling
        let uniform = 1.0 / N_TAGS as f64;
        for i in 0..N_TAGS {
            self.prior[i] = uniform;
        }

        // Normalizar transições
        for i in 0..N_TAGS {
            let total: f64 = self.trans_count[i].iter().sum();
            for j in 0..N_TAGS {
                if total > 0.0 {
                    self.transition[i][j] = self.trans_count[i][j] / total;
                } else {
                    self.transition[i][j] = 1.0 / N_TAGS as f64;
                }
                if self.transition[i][j] < 1e-10 {
                    self.transition[i][j] = 1e-10;
                }
            }
        }

        // Normalizar emissões (manter Top-K)
        for tag in 0..N_TAGS {
            let total: f64 = self.emit_count[tag].values().sum();
            if total > 0.0 {
                let mut entries: Vec<(String, f64)> = self.emit_count[tag]
                    .iter()
                    .map(|(w, &c)| (w.clone(), c / total))
                    .collect();
                entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                entries.truncate(10000); // Top 10K por tag
                self.emission[tag] = entries.into_iter().collect();
            }
        }
    }

    /// Obter probabilidade de emissão P(word | tag)
    fn emit_prob(&self, word: &str, tag: usize) -> f64 {
        // 1. Verificar se a palavra está nas emissões aprendidas
        if let Some(&prob) = self.emission[tag].get(word) {
            return prob;
        }

        // 2. Usar sufixos como suavização
        if let Some(suffix_freq) = self.suffix_table.get(&format!("tag_{}", tag)) {
            for (suffix, &suffix_prob) in suffix_freq {
                if word.ends_with(suffix) && word.len() > suffix.len() {
                    return suffix_prob * 0.01; // Fator de suavização
                }
            }
        }

        // 3. Probabilidade mínima (suavização de Laplace)
        1e-8
    }

    /// Algoritmo de Viterbi — melhor sequência de tags
    pub fn viterbi(&self, words: &[String]) -> Vec<usize> {
        let n = words.len();
        if n == 0 { return Vec::new(); }

        // dp[t][tag] = log probabilidade máxima até tempo t com tag
        let mut dp: Vec<Vec<f64>> = vec![vec![0.0; N_TAGS]; n];
        let mut backptr: Vec<Vec<usize>> = vec![vec![0; N_TAGS]; n];

        // Inicialização (t=0)
        for tag in 0..N_TAGS {
            dp[0][tag] = self.prior[tag].ln() + self.emit_prob(&words[0], tag).ln();
        }

        // Recursão
        for t in 1..n {
            for tag in 0..N_TAGS {
                let mut best_prob = f64::NEG_INFINITY;
                let mut best_prev = 0;

                for prev_tag in 0..N_TAGS {
                    let prob = dp[t - 1][prev_tag]
                        + self.transition[prev_tag][tag].ln()
                        + self.emit_prob(&words[t], tag).ln();

                    if prob > best_prob {
                        best_prob = prob;
                        best_prev = prev_tag;
                    }
                }

                dp[t][tag] = best_prob;
                backptr[t][tag] = best_prev;
            }
        }

        // Backtrack
        let mut tags = vec![0usize; n];
        let mut best_last_tag = 0;
        let mut best_last_prob = f64::NEG_INFINITY;
        for tag in 0..N_TAGS {
            if dp[n - 1][tag] > best_last_prob {
                best_last_prob = dp[n - 1][tag];
                best_last_tag = tag;
            }
        }
        tags[n - 1] = best_last_tag;

        for t in (1..n).rev() {
            tags[t - 1] = backptr[t][tags[t]];
        }

        tags
    }

    /// Classificar uma palavra individual (para uso no compiler)
    pub fn predict_tag(&self, word: &str, prev_tag: Option<usize>) -> usize {
        let mut best_tag = TAG_SUBST;
        let mut best_prob = f64::NEG_INFINITY;

        for tag in 0..N_TAGS {
            let trans_prob = if let Some(pt) = prev_tag {
                self.transition[pt][tag]
            } else {
                self.prior[tag]
            };

            let prob = trans_prob.ln() + self.emit_prob(word, tag).ln();

            if prob > best_prob {
                best_prob = prob;
                best_tag = tag;
            }
        }

        best_tag
    }

    /// Exportar estatísticas para debug
    pub fn stats(&self) -> HmmStats {
        let mut word_counts = [0usize; N_TAGS];
        for tag in 0..N_TAGS {
            word_counts[tag] = self.emission[tag].len();
        }

        HmmStats {
            prior: self.prior.clone(),
            word_counts,
            top_words: self.top_words_per_tag(5),
        }
    }

    /// Top N palavras mais prováveis por tag
    pub fn top_words_per_tag(&self, n: usize) -> Vec<Vec<(String, f64)>> {
        let mut result = Vec::new();
        for tag in 0..N_TAGS {
            let mut words: Vec<(String, f64)> = self.emission[tag]
                .iter()
                .map(|(w, &p)| (w.clone(), p))
                .collect();
            words.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            words.truncate(n);
            result.push(words);
        }
        result
    }
}

pub struct HmmStats {
    pub prior: Vec<f64>,
    pub word_counts: [usize; N_TAGS],
    pub top_words: Vec<Vec<(String, f64)>>,
}
