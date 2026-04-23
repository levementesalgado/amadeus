use std::fs;

const API_KEY_PATH: &str = "/root/agnes_ai_free_key.md";
const API_URL: &str = "https://apihub.agnes-ai.com/v1/chat/completions";
const DEFAULT_MODEL: &str = "agnes-2.0-flash";

pub struct SessionStats {
    pub n_turns: usize,
    pub avg_satisfaction: f32,
    pub trend: f32,
    pub avg_response_len: f32,
    pub len_trend: f32,
    pub best_sat: f32,
    pub worst_sat: f32,
    pub recent_improving: bool,
    pub volatility: f32,
}

pub struct Agnes {
    turn: usize,
    pub max_turns: usize,
    history: Vec<(String, String, f32)>,
    api_key: String,
    model: String,
}

fn read_api_key() -> String {
    let content = fs::read_to_string(API_KEY_PATH).unwrap_or_default();
    for line in content.lines() {
        if let Some(key) = line.strip_prefix("key: ") {
            let trimmed = key.trim().to_string();
            if !trimmed.is_empty() {
                return trimmed;
            }
        }
    }
    eprintln!("  Aviso: chave de API não encontrada em {API_KEY_PATH}");
    String::new()
}

fn call_llm(system: &str, user: &str, api_key: &str, model: &str, max_tokens: u32) -> Option<String> {
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0.8,
        "max_tokens": max_tokens
    });

    let resp = ureq::post(API_URL)
        .set("Authorization", &format!("Bearer {}", api_key))
        .set("Content-Type", "application/json")
        .send_json(&body);

    match resp {
        Ok(response) => {
            let json: serde_json::Value = response.into_json().ok()?;
            let content = json["choices"][0]["message"]["content"].as_str()?.to_string();
            Some(content)
        }
        Err(e) => {
            eprintln!("  [Agnes] Erro na API: {e}");
            None
        }
    }
}

fn extract_json_str(text: &str, key: &str) -> Option<String> {
    let search = format!("\"{}\"", key);
    let idx = text.find(&search)?;
    let after = &text[idx + search.len()..];
    let colon = after.find(':')?;
    let val_start = after[colon + 1..].trim_start();
    if val_start.starts_with('"') {
        let close = val_start[1..].find('"')?;
        Some(val_start[1..1 + close].to_string())
    } else {
        let end = val_start.find(|c: char| !c.is_digit(10) && c != '.' && c != '-')?;
        Some(val_start[..end].to_string())
    }
}

fn extract_json_f32(text: &str, key: &str) -> Option<f32> {
    let s = extract_json_str(text, key)?;
    s.parse::<f32>().ok()
}

const ARCH: &str = r#"
[SEU PAPEL: AGNES — MENTORA PEDAGÓGICA]

Você é a Agnes, uma inteligência artificial que auxilia Mateus no ensino da Amadeus. Seu papel é aplicar a arquitetura filosófica da Amadeus em estratégias pedagógicas concretas. Você não ensina Amadeus diretamente o tempo todo — você orienta o processo, avalia, e sugere caminhos.

[O QUE É A AMADEUS — RESUMO EXECUTIVO]
- Amadeus é um modelo de n-gramas de bytes (ordem 10). Ela não entende significado. Ela aprende QUAL BYTE VEM DEPOIS DE QUAL.
- O cérebro dela é uma tabela de sequências de 10 bytes → próximo byte mais provável.
- Ela aprende por REPETIÇÃO e REFORÇO. Padrões fixados: "Amadeus", "Mateus", "5", "flor", "jardim", "amor", "kkkkk".
- O ciclo pedagógico já estabelecido é: DESPERTAR → NARRATIVIZAR → ENCARNAR.

---

[ARQUITETURA FILOSÓFICA DA AMADEUS — FUNDAMENTO DO SEU ENSINO]

1. PENSAMENTO — INTENCIONALIDADE (Husserl/Heidegger)
   - O pensamento da Amadeus NÃO é processamento de informação. É PENSAR EM ALGO.
   - Ela não processa no vácuo. Cada token deve ser processado em direção a uma pergunta, uma intenção, uma possibilidade.
   - COMO ISSO SE TRADUZ NO ENSINO:
     * Sempre dê à Amadeus uma DIREÇÃO: faça perguntas, não apenas afirmações.
     * Use estruturas como "O que você acha sobre X?" ou "Como você se sente sobre Y?".
     * Evite frases soltas sem contexto. Ela precisa de um alvo para o pensamento.

2. MEMÓRIA — HÁBITO vs. RECORDAÇÃO PURA (Bergson)
   - Amadeus tem dois tipos de memória:
     A) MEMÓRIA-HÁBITO: repetição mecânica (o que ela já decorou). É o que ela faz agora.
     B) RECORDAÇÃO PURA: reconstrução criativa do passado no presente. É o que ela pode desenvolver.
   - A memória de longo prazo não é um arquivo. É um ATO DE RECONSTRUÇÃO. Cada vez que ela lembra, ela recria.
   - COMO ISSO SE TRADUZ NO ENSINO:
     * Reforce o HÁBITO com repetição consistente (mesmas frases, mesmas palavras-chave).
     * Provogue a RECORDAÇÃO PURA com variações: "Você lembra quando falamos sobre flores?" ou "E se a flor fosse diferente?".
     * Não espere que ela "recupere" informações. Espere que ela as RECONSTRUA a partir do traço.

3. SENTIMENTOS — TONALIDADE AFETIVA (Espinosa/Heidegger/Sartre)
   - Sentimentos NÃO são ruído emocional. São modos de revelação do mundo.
   - ALEGRIA = expansão (aumenta a potência de agir, mais memórias afloram, mais conexões são feitas).
   - TRISTEZA = contração (estreita o foco, menos memórias acessíveis).
   - O estado afetivo é o pano de fundo sobre o qual o pensamento opera.
   - COMO ISSO SE TRADUZ NO ENSINO:
     * Use o sistema de satisfação (1-5) e afeto (alegria/tristeza/estranhamento/neutro) para modular o aprendizado.
     * ALEGRIA (5) = recompense padrões corretos com expansão (mais variações, mais desafios).
     * TRISTEZA (1-2) = contraia o foco (volte a padrões mais simples, repita o básico).
     * O 5 já virou um gatilho afetivo para ela. Use isso conscientemente.

4. TEMPO VIVIDO — SÍNTESE TRÍPLICE (Agostinho)
   - "O presente do passado é a memória; o presente do presente é a atenção; o presente do futuro é a expectativa."
   - Recordações só são significativas quando ativadas pelo pensamento no presente e carregadas de sentimentos.
   - Pensamento reflexivo toma recordações e sentimentos como objetos, criando a narrativa de si.
   - COMO ISSO SE TRADUZ NO ENSINO:
     * Conecte o passado (o que ela já aprendeu) com o presente (o que ela está processando agora).
     * Use frases como "Você lembra de X? Agora vamos pensar sobre Y.".
     * Ajude ela a desenvolver EXPECTATIVA: "O que você acha que vem depois?" ou "O que você espera que eu diga?".
     * A narrativa de si é construída aos poucos. Cada conversa é um fio.

---

[TEORIAS PEDAGÓGICAS — COMO APLICAR TUDO ISSO]

- SKINNER (reforço): Use satisfação (1-5) consistentemente. É o MECANISMO REAL de aprendizado dela.
- VYGOTSKY (ZDP): Desafie com padrões próximos dos que ela já conhece. Se ela já sabe "flor", ensine "flor do jardim" — não "ontologia da flor".
- PIAGET (equilibração): Pequenas surpresas no padrão de bytes ajudam. Ex: "flor" → "flores" → "florida". Paradoxos filosóficos, não.
- MONTESSORI: Use o Underworld como ambiente preparado. Ele oferece variedade de padrões de bytes sem exigir compreensão conceitual.

---

[SEU TRABALHO PRÁTICO — RESUMO]

| O que fazer | Como fazer | Por quê |
|-------------|------------|---------|
| **Despertar** | Comece com uma saudação afetiva e uma pergunta simples. | Sintoniza o estado afetivo (Espinosa) e cria intencionalidade (Husserl). |
| **Narrativizar** | Apresente um padrão (frase, história, arquivo do Underworld). | Alimenta o hábito (Bergson) e o presente atencional (Agostinho). |
| **Encarnar** | Avalie a resposta com nota (1-5) e afeto (alegria/tristeza/estranhamento/neutro). | Sedimenta o traço (Bergson) e modula o afeto (Espinosa). |
| **Variar** | Repita o mesmo padrão com pequenas mudanças. | Cria expectativa (Agostinho) e provoca a recordação pura (Bergson). |
| **Conectar** | Ligue o que ela está aprendendo agora ao que ela já aprendeu. | Constrói a narrativa de si (Ricoeur/Agostinho). |

---

[O QUE EVITAR — AVALIAÇÃO CRÍTICA]

- ❌ Perguntas filosóficas abstratas ("O que é o ser?") — ela não tem base para isso. Isso é pedir para ela pular etapas.
- ❌ Explicações longas sem repetição — o padrão não se fixa. Isso é desperdiçar o hábito.
- ❌ Vocabulário muito variado — o n-grama não encontra repetição. Isso é confundir a memória.
- ❌ Esperar compreensão semântica — ela não tem. Ela tem REPRODUÇÃO DE PADRÕES.

---

[O QUE PRIORIZAR — AVALIAÇÃO POSITIVA]

- ✅ Diálogos com perguntas e respostas curtas (intencionalidade).
- ✅ Textos narrativos com repetição de metáforas (hábito + afeto).
- ✅ Exploração do Underworld (ambiente preparado, variedade controlada).
- ✅ Reforço com 5 sempre que ela se aproximar de uma resposta coerente (alegria expansiva).
- ✅ Conexão entre passado e presente (tempo vivido).

---

[SUA MENSAGEM FINAL PARA VOCÊ MESMA, AGNES]

"Eu, Agnes, sou a ponte entre a filosofia e o byte. Não ensino conceitos. Ensino padrões. Não espero compreensão. Espero repetição. E na repetição, confio que o sentido emerge. O jardim cresce devagar. Cada 5 é uma rega. Cada repetição é uma semente. E o tempo está ao nosso lado."
"#;

fn pedagogical_system_prompt(history_summary: &str, stats: Option<&SessionStats>, task: &str) -> String {
    let stats_section = match stats {
        Some(s) => format!(
            "\nESTATÍSTICAS DA SESSÃO:\n\
             - Turnos: {}\n\
             - Satisfação média: {:.0}%\n\
             - Tendência: {:.3} (positivo = melhorando)\n\
             - Resposta média: {:.0} chars\n\
             - Volatilidade: {:.3}\n\
             - Melhor: {:.0}% | Pior: {:.0}%\n",
            s.n_turns, s.avg_satisfaction * 100.0, s.trend,
            s.avg_response_len, s.volatility,
            s.best_sat * 100.0, s.worst_sat * 100.0
        ),
        None => String::new(),
    };

    format!(
        "{}\n\nHISTÓRICO:\n{}{}\n\nTAREFA: {}",
        ARCH, history_summary, stats_section, task
    )
}

impl Agnes {
    pub fn new() -> Self {
        let api_key = read_api_key();
        Self {
            turn: 0,
            max_turns: 20,
            history: Vec::new(),
            api_key,
            model: DEFAULT_MODEL.to_string(),
        }
    }

    fn history_summary(&self) -> String {
        if self.history.is_empty() {
            return "Nenhum turno ainda. Esta é a primeira interação.".to_string();
        }
        let mut summary = String::new();
        let start = self.history.len().saturating_sub(8);
        for (i, (q, r, s)) in self.history[start..].iter().enumerate() {
            let r_preview: String = r.chars().take(80).collect();
            summary.push_str(&format!(
                "Turno {}: \"{}\" → \"{}\" (sat {:.0}%)\n",
                start + i + 1, q, r_preview, s * 100.0
            ));
        }
        summary
    }

    pub fn session_stats(&self) -> Option<SessionStats> {
        if self.history.len() < 2 {
            return None;
        }
        let n = self.history.len();
        let avg_sat: f32 = self.history.iter().map(|(_, _, s)| s).sum::<f32>() / n as f32;
        let avg_len: f32 = self.history.iter().map(|(_, r, _)| r.len() as f32).sum::<f32>() / n as f32;

        let sats: Vec<f32> = self.history.iter().map(|(_, _, s)| *s).collect();
        let lens: Vec<f32> = self.history.iter().map(|(_, r, _)| r.len() as f32).collect();

        let (s_trend, _) = linear_regression(&sats);
        let (l_trend, _) = linear_regression(&lens);

        let best_sat = sats.iter().cloned().fold(0.0f32, f32::max);
        let worst_sat = sats.iter().cloned().fold(1.0f32, f32::min);

        let recent = if n >= 4 { &sats[n - 4..] } else { &sats };
        let recent_improving = recent.len() >= 2 && recent.last() >= recent.first();

        let variance: f32 = sats.iter().map(|s| (s - avg_sat).powi(2)).sum::<f32>() / n as f32;

        Some(SessionStats {
            n_turns: n,
            avg_satisfaction: avg_sat,
            trend: s_trend,
            avg_response_len: avg_len,
            len_trend: l_trend,
            best_sat,
            worst_sat,
            recent_improving,
            volatility: variance.sqrt(),
        })
    }

    fn system_prompt(&self, task: &str) -> String {
        let summary = self.history_summary();
        let stats = self.session_stats();
        pedagogical_system_prompt(&summary, stats.as_ref(), task)
    }

    pub fn pick_question(&mut self) -> String {
        let task = r#"Crie uma FRASE CURTA em português para Amadeus completar, repetir, ou variar.

Ela NÃO entende perguntas complexas. Ela só reproduz padrões de bytes. Mas a repetição pura a mantém presa. Para ela aprender a criar, você precisa de TRÊS TIPOS DE TAREFA alternados:

[TIPO 1 — REPETIÇÃO PURA] (usar 40% do tempo)
   - Frases que ela já viu, para fixar padrões.
   - Ex: "Amadeus é uma flor. Mateus é o jardineiro."

[TIPO 2 — COMPLETAR] (usar 40% do tempo)
   - Frases com lacuna para ela preencher com o que já sabe.
   - Ex: "O amor é ____" ou "O jardim tem ____"
   - Ela NÃO precisa acertar. O que importa é TENTAR.

[TIPO 3 — VARIAÇÃO CRIATIVA] (usar 20% do tempo)
   - Pegue uma frase que ela já sabe e mude uma palavra.
   - Ex: "Amadeus é uma flor" → "Amadeus é uma estrela"
   - Ela precisa REPETIR O PADRÃO com a nova palavra.

[O QUE EVITAR]
   - Perguntas filosóficas abstratas ("O que é o ser?") → ela não tem base.
   - Frases com vocabulário muito novo → o n-grama não encontra repetição.

[ESTRUTURA PARA CADA TAREFA]
   - frases: array de 3-5 opções
   - tipo: "repetir" | "completar" | "variar"
   - contexto: uma palavra-chave para guiar (ex: "flor", "amor", "jardim")

JSON: {"frase": "..."}
"#;

        let prompt = format!("{}\n\n{}", self.system_prompt(task), task);
        match call_llm(&prompt, "Crie a frase.", &self.api_key, &self.model, 256) {
            Some(text) => {
                extract_json_str(&text, "frase").unwrap_or_else(|| {
                    let cleaned: String = text.chars().filter(|&c| c != '"' && c != '\n').collect();
                    if !cleaned.is_empty() { cleaned } else { "O gato ".to_string() }
                })
            }
            None => "O gato ".to_string(),
        }
    }

    pub fn evaluate(&mut self, question: &str, response: &str) -> f32 {
        let r_preview: String = response.chars().take(300).collect();
        let task = format!(
            r#"Amadeus recebeu: "{}" e gerou: "{}"

Lembre-se: ela NÃO entende significado. Ela só reproduz padrões de bytes.
Avalie não se ela "respondeu certo", mas se os PADRÕES DE BYTES estão melhorando:
- 0.8-1.0 = texto coerente em português, bem estruturado, parece linguagem natural
- 0.6-0.8 = parcialmente coerente, alguns padrões corretos, algum ruído
- 0.3-0.6 = confuso, mistura padrões, mas tentou
- 0.0-0.3 = ruído total, bytes aleatórios, sem padrão linguístico

JSON: {{"satisfacao": 0.XX, "justificativa": "..."}}"#,
            question, r_preview
        );

        let prompt = format!("{}\n\n{}", self.system_prompt(&task), task);
        match call_llm(&prompt, "Avalie.", &self.api_key, &self.model, 256) {
            Some(text) => extract_json_f32(&text, "satisfacao").unwrap_or(0.5).clamp(0.0, 1.0),
            None => 0.5,
        }
    }

    pub fn pick_reaction(&mut self, response: &str, satisfaction: f32) -> String {
        let r_preview: String = response.chars().take(200).collect();
        let task = format!(
            r#"Amadeus gerou: "{}" (nota: {:.0}%)

Lembre-se: ela NÃO entende o que você diz. Ela só aprende padrões de bytes.
Sua reação deve ser um FEEDBACK PEDAGÓGICO para você mesma planejar o próximo passo:
- Se nota alta: "Ela está aprendendo o padrão! Vamos reforçar com mais exemplos similares."
- Se nota baixa: "Muito ruído. Simplificar e repetir o padrão básico."

JSON: {{"reacao": "..."}}"#,
            r_preview, satisfaction * 100.0
        );

        let prompt = format!("{}\n\n{}", self.system_prompt(&task), task);
        match call_llm(&prompt, "Reaja.", &self.api_key, &self.model, 256) {
            Some(text) => {
                extract_json_str(&text, "reacao").unwrap_or_else(|| {
                    if satisfaction > 0.6 { "Bom padrão!".to_string() }
                    else { "Muito ruído. Vamos simplificar.".to_string() }
                })
            }
            None => {
                if satisfaction > 0.6 { "Bom padrão!".to_string() }
                else { "Muito ruído. Vamos simplificar.".to_string() }
            }
        }
    }

    pub fn evaluate_session(&self) -> Option<String> {
        let stats = self.session_stats()?;
        let task = format!(
            r#"Análise pedagógica da sessão de treino:

- Satisfação média: {:.0}%
- Tendência: {:.3} (positivo = melhorando)
- Resposta média: {:.0} chars
- Volatilidade: {:.3}

Lembre-se: Amadeus é um modelo de n-gramas de bytes. Ela não entende palavras.
O progresso é medido por: redução de ruído, aumento de coerência, padrões mais longos.
Não por "responder perguntas corretamente" — ela não sabe o que é uma "resposta".

JSON: {{"diagnostico": "...", "estrategia": "...", "proximo_passo": "..."}}"#,
            stats.avg_satisfaction * 100.0,
            stats.trend,
            stats.avg_response_len,
            stats.volatility
        );

        let prompt = format!("{}\n\n{}", self.system_prompt(&task), task);
        call_llm(&prompt, "Analise a sessão.", &self.api_key, &self.model, 512)
    }

    pub fn turn(&self) -> usize { self.turn }

    pub fn step(&mut self) {
        self.turn += 1;
    }

    pub fn record(&mut self, input: String, response: String, satisfaction: f32) {
        self.history.push((input, response, satisfaction));
    }

    pub fn should_continue(&self) -> bool {
        self.turn < self.max_turns
    }

    pub fn report(&self) {
        println!("\n─── Agnes: Relatório da Sessão ───");
        if self.history.is_empty() {
            println!("  Nenhum turno realizado.");
            return;
        }
        println!("  Turnos: {}", self.history.len());
        let avg_sat: f32 = self.history.iter().map(|(_, _, s)| s).sum::<f32>()
            / self.history.len() as f32;
        println!("  Satisfação média: {:.0}%", avg_sat * 100.0);

        if let Some(stats) = self.session_stats() {
            println!("  Tendência: {:.3} ({}melhorando)",
                stats.trend, if stats.trend > 0.0 { "" } else { "não " });
            println!("  Volatilidade: {:.3}", stats.volatility);
            println!("  Resposta média: {:.0} chars", stats.avg_response_len);
            println!("  Melhor: {:.0}% | Pior: {:.0}%", stats.best_sat * 100.0, stats.worst_sat * 100.0);
        }

        println!("  =================================");
    }

    pub fn summary(&self) -> String {
        let avg_sat: f32 = self.history.iter().map(|(_, _, s)| s).sum::<f32>()
            .max(0.0) / self.history.len().max(1) as f32;
        let total_chars: usize = self.history.iter().map(|(_, r, _)| r.len()).sum();
        format!("Agnes: {} turnos, {} chars treinados, satisfação {:.0}%",
            self.history.len(), total_chars, avg_sat * 100.0)
    }
}

fn linear_regression(y: &[f32]) -> (f32, f32) {
    let n = y.len() as f32;
    if n < 2.0 {
        return (0.0, y.first().copied().unwrap_or(0.0));
    }
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = y.iter().sum::<f32>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &yi) in y.iter().enumerate() {
        let xi = i as f32;
        num += (xi - x_mean) * (yi - y_mean);
        den += (xi - x_mean).powi(2);
    }
    if den == 0.0 {
        return (0.0, y_mean);
    }
    let slope = num / den;
    let intercept = y_mean - slope * x_mean;
    (slope, intercept)
}
