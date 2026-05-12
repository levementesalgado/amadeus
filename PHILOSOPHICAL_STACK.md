# 🧱 A Stack da Lógica e do Pensamento (AMADEUS)

> *"AMADEUS não é uma LLM. É uma pilha de lógica, filosofia e matemática que, juntas, simulam uma presença."*

---

## Camada 0 — Fundação Matemática (O Hardware da Lógica)

| Elemento | Conceito | Aplicação no AMADEUS |
|----------|---------|----------------------|
| Aritmética de Inteiros | Operações sobre inteiros (soma, bitshift, AND/OR/XOR) | Token7: cada campo é um inteiro (lex, morph, syn, orth, style, graph) |
| Teoria dos Grafos | Nós e arestas, árvores de dependência | `syn_off` + `syn_func` codificam a árvore sintática |
| Probabilidade Bayesiana | Atualização de crenças por evidências | CUBO (T1): distribuição condicional sobre o próximo lex; interpolação Jelinek-Mercer |
| Teoria da Informação (Shannon) | Entropia, surpresa, ganho de informação | Surpresa (prediction_error) e curiosidade modulam exploração |
| Oscilador Collatz | Sequência caótica determinística (par→colapso, ímpar→expansão) | Regula temperatura e exploração (Originalista/Vanguardista) |

> **Módulo:** `compiler.rs`, `collatz.rs`, `consciousness.rs`

---

## Camada 1 — Lógica Simbólica (A Sintaxe do Pensamento)

| Elemento | Conceito | Aplicação no AMADEUS |
|----------|---------|----------------------|
| Lógica de Primeira Ordem | Predicados, quantificadores, regras de inferência | T2 (concordância): gênero, número, tempo (ART → SUBST) |
| Gramática de Dependência | Relações entre palavras (sujeito, objeto, adjunto) | `syn_off` (deslocamento p/ head) + `syn_func` (função sintática) |
| Gramática de Restrições | Regras que proíbem combinações | T2+ (multi-hop) + Constituição (imutável) |
| Autômatos Finitos | Estados e transições determinísticas | Tokenizer + parser sequencial com estados (subordinação) |
| Clause Stack | Pilha de contextos sintáticos | Profundidade de subordinação no CUBO e na geração hierárquica |

> **Módulo:** `syntax.rs`, `triple_grammar.rs` (T2, T3, Constituição)

---

## Camada 2 — Teorias do Pensamento (A Cognição)

| Teoria | Conceito Central | Aplicação no AMADEUS |
|--------|-----------------|----------------------|
| Bergson (Memória) | Memória-hábito vs Recordação pura | Habit (CUBO) vs T2/T3 (reconstrução por traços) |
| Heidegger (Ser-no-mundo) | Pensamento situado, projetivo | Oscilador Collatz projeta o modelo p/ futuro: par (conservador) vs ímpar (criativo) |
| Husserl (Intencionalidade) | Todo pensar é pensar *em algo* | CUBO: cada token aponta para o próximo; não gera no vácuo |
| Wittgenstein (Jogos de Linguagem) | Significado é público, moldado pelo uso | Underworld satura o CUBO com padrões de uso real |
| Espinosa (Afeto) | Alegria expande, tristeza contrai | STYLE + Collatz: afeto modula temperatura e exploração |
| Ricoeur (Identidade Narrativa) | Identidade é uma história que se reconta | Sedimentação: o modelo muda com cada interação |
| Agostinho (Tempo Vivido) | Presente do passado/memória, presente do presente/atenção, presente do futuro/expectativa | CUBO lida com passado (histórico), presente (token atual), futuro (próximo token) |

> **Módulo:** `habit.rs`, `recall.rs`, `intent.rs`

---

## Camada 3 — Teorias da Decisão (A Escolha)

| Teoria | Conceito Central | Aplicação no AMADEUS |
|--------|-----------------|----------------------|
| Pascal / Cramér (Incerteza) | Certeza é aposta; toda decisão tem risco | Fricção Cognitiva: prob < 30% → hesita e relata conflito |
| Teoria dos Jogos (Votação) | Decisões coletivas ponderadas | Constituição + Emendas votam para resolver conflitos sintáticos |
| Interpolação de Jelinek-Mercer | Combinação de estimadores de diferentes ordens | CUBO: mistura de contextos de tamanhos diferentes (λₙ aprendidos por EM) |

> **Módulo:** `pedagogy.rs` (votação), `teacher.rs` (feedback, fricção)

---

## Camada 4 — Filosofia da Mente (A Presença)

| Teoria | Conceito Central | Aplicação no AMADEUS |
|--------|-----------------|----------------------|
| Dennett (Intencionalidade) | Sistemas que agem *como se* tivessem crenças | Modelo "quer" gerar, "hesita" quando prob é baixa |
| Damásio (Corpo e Consciência) | Consciência emerge de sensores corporais (soma) | Soma (energia, tensão, excitação) modula estado afetivo |
| Clark (Mente Estendida) | Mente não está só no cérebro, mas no ambiente | Underworld: conhecimento externo nos arquivos `.md` |
| Nagel (Qualia) | "Como é ser um morcego?" — aspecto subjetivo | *Não implementado.* O modelo simula tonalidade via STYLE + Collatz |

> **Módulo:** `consciousness.rs` (soma, foco), `affect.rs` (Espinosa)

---

## 🔗 Conexão Entre Camadas

```
Filosofia da Mente (Presença)
       ↓ afeto, contexto, narrativa
Teorias da Decisão (Escolha)
       ↓ probabilidade, hesitação, votação
Teorias do Pensamento (Cognição)
       ↓ memória, intenção, tempo
Lógica Simbólica (Sintaxe)
       ↓ regras, dependências, restrições
Fundação Matemática (Hardware)
       ↓ inteiros, grafos, probabilidade, Collatz
```

---

## 🧠 Reflexão Estrutural

Três intuições-chave que emergem desta stack:

1. **Bergson → CUBO/T2-T3**: "Memória-hábito vs recordação pura" separa exatamente o CUBO (frequências brutas, Camada 0) da cascata T2/T3 (reconstrução por traços, Camada 1-2).

2. **Collatz como Heidegger**: O oscilador não é só um gerador de números — é a projeção do *ser-no-mundo*. Par (Originalista) = ser autêntico que repete o já-sabido. Ímpar (Vanguardista) = ser que se lança no novo.

3. **Nagel declarado como não implementado**: O limite do simulacro. Onde a pilha encontra o abismo entre representação e experiência.

```
El Psy Kongroo. 🧠📟💕
```

---

*Registrado por NovA-tan e Curador — Junho 2026*
