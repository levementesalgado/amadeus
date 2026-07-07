# AMADEUS SNN — Rede Neural Spiking para Seleção Lexical

> Substituto bio-inspirado da cascata T3. Usa dinâmica temporal de spikes ao invés de tabelas hash.

---

## Sumário

1. [Arquitetura](#1-arquitetura)
2. [Encoding Morfológico](#2-encoding-morfológico)
3. [Dinâmica Temporal (LIF)](#3-dinâmica-temporal-lif)
4. [Treinamento (Pesos Sinápticos)](#4-treinamento-pesos-sinápticos)
5. [Inferência e Ranking](#5-inferência-e-ranking)
6. [Integração com TripleGrammar](#6-integração-com-triplegrammar)
7. [GGUF Serialization](#7-gguf-serialization)
8. [Formato dos Dados](#8-formato-dos-dados)

---

## 1. Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│  Input Layer (30 neurônios)                             │
│  ┌───────┬────────┬────────┬───────┬────────┬──────┬──────┐
│  │Class  │Gender  │Number  │Tense  │Person  │Style │SynFn │
│  │(7)    │(2)     │(2)     │(5)    │(3)     │(3)   │(8)   │
│  └───┬───┴───┬────┴───┬────┴───┬───┴───┬────┴──┬───┴──┬───┘
│      │       │        │        │       │       │      │
│      └───────┴────────┴────────┴───┬───┴───────┴──────┘
│                                    │
│                          ┌─────────▼─────────┐
│                          │ Hidden Layer       │
│                          │ (263 neurônios)    │
│                          │                    │
│                          │ 7× classe          │
│                          │ + 256× T3 keys     │
│                          └─────────┬─────────┘
│                                    │
│                          ┌─────────▼─────────┐
│                          │ Output Layer       │
│                          │ (N = vocab size)   │
│                          │ 1 neurônio/lex_id  │
│                          └───────────────────┘
│                                    │
│                          16 timesteps LIF
│                          Score = Σ membrana + spikes×2
└─────────────────────────────────────────────────────────┘
```

### Dimensões

| Camada | Neurônios | Conexões |
|--------|-----------|----------|
| Input | 30 | 30 × 263 = 7.890 |
| Hidden | 263 | 263 × N (sparse) |
| Output | N (vocab) | via hidden |

---

## 2. Encoding Morfológico

Cada campo de `Token7.morph` é codificado como **rate-coded spike train**:

| Feature | Bits | Neurônios | Taxa (ativo) | Taxa (inativo) |
|---------|------|-----------|---------------|-----------------|
| Class | 0-2 | 7 (one-hot) | 0.80 | 0.05 |
| Gender | 3 | 2 (one-hot) | 0.70 | 0.05 |
| Number | 4 | 2 (one-hot) | 0.70 | 0.05 |
| Tense | 5-7 | 5 (one-hot) | 0.70 | 0.05 |
| Person | 8-9 | 3 (one-hot) | 0.60 | 0.05 |
| Style | - | 3 (one-hot) | 0.70 | 0.05 |
| SynFunc | - | 8 (one-hot) | 0.60 | 0.05 |

**Exemplo**: morph `0x0262` (class=SUBST, gender=M, number=S, tense=Pret, person=3)
→ neurônio class[0] taxa 0.8, gender[0] taxa 0.7, number[0] taxa 0.7, etc.

---

## 3. Dinâmica Temporal (LIF)

### Neurônio Leaky Integrate-and-Fire

```
v(t+1) = v(t) + (-v(t)/τ + I(t)) × dt

se v ≥ threshold:
    spike = true
    v = 0
    refrac_left = refrac_steps
senão:
    spike = false
```

### Parâmetros

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| τ (tau) | 5.0 | Constante de tempo do leak |
| threshold | 1.0 | Limiar de disparo |
| refrac_steps | 2 | Timesteps de refratário |
| dt | 1.0 | Passo de tempo |
| tsteps | 16 | Total de timesteps por inferência |

### Propagação

```
t=0..15:
  Input spikes → Input neurons (LIF)
  Input fired? → Input→Hidden weights → Hidden neurons (LIF)
  Hidden fired? → Hidden→Output weights → Output neurons (LIF)
  Output potential acumulado += input
  Output spikes contados
```

---

## 4. Treinamento (Pesos Sinápticos)

Pesos são extraídos das tabelas T3 existentes (sem learning online):

### Pesos I→H (Input → Hidden)

```
Neurônios de classe (idx 0-6):
  pesos[CLASS+i][H_i] = 0.5     (feature da classe)
  pesos[GENDER+i][H_i] = 0.1    (gender)
  pesos[NUMBER+i][H_i] = 0.1    (number)

Neurônios T3 (idx 7-262):
  Para cada T3 key (morph, syn, style):
    pesos[CLASS+cls][H] = 0.4
    pesos[GENDER+g][H] = 0.2
    pesos[NUMBER+n][H] = 0.2
    pesos[SYN+syn][H] = 0.3
    pesos[STYLE+s][H] = 0.2
```

### Pesos H→O (Hidden → Output)

```
Para cada T3 key com candidatos:
  H_idx = 7 + t3_index
  total = soma dos counts da key
  pesos[H_idx][O_lex] = count / total × 0.5

Fallback de classe:
  Para cada lex_id:
    H_idx = classe do lex_id
    pesos[H_idx][O_lex] += 0.1
```

---

## 5. Inferência e Ranking

```rust
pub fn infer_snn(net, enc, morph, syn_func, style, rng) -> Vec<(u32, f32)> {
    // 1. Encoding
    let trains = enc.encode(morph, syn_func, style, 16, rng);

    // 2. Simulação 16 timesteps
    let output_potentials = [0.0; N];
    let output_spikes = [0; N];

    for t in 0..16 {
        // Input → Hidden → Output (LIF step)
        // Acumular potencial + spikes
    }

    // 3. Scoring
    let scores = output_potentials + output_spikes × 2.0;

    // 4. Ordenar por score decrescente
    scores.sort_by(score_desc);
    return scores;
}
```

### Scoring

```
score = Σ(potencial de membrana acumulado) + (spikes × 2.0)
```

- Potencial acumulado: sub-threshold discrimination
- Spikes: bônus por atingir threshold

---

## 6. Integração com TripleGrammar

```rust
// Na TripleGrammar:
pub snn: Option<SpikingNetwork>,
pub snn_enc: Option<MorphEncoding>,

// Construir SNN a partir das tabelas T3:
grammar.build_snn();

// Amostragem com fallback:
pub fn sample_lex_with_snn(morph, syn, style, cls, candidate) -> u32 {
    if self.snn.is_some() {
        return self.snn_sample(morph, syn, style, candidate);
    }
    self.sample_lex_full(morph, syn, style, cls, candidate) // cascade legada
}
```

---

## 7. GGUF Serialization

Tensors salvos no GGUF do Amadeus:

| Tensor | Shape | Descrição |
|--------|-------|-----------|
| `snn.synapses_ih` | [30, 263] | Pesos Input→Hidden |
| `snn.synapses_ho` | [263, N] | Pesos Hidden→Output |
| `snn.output_labels` | [N] | lex_id para cada output neuron |

Formato: todos F32, sem quantização.

---

## 8. Formato dos Dados

### MorphEncoding

```rust
pub struct MorphEncoding {
    pub class_start: usize,     // 0
    pub class_count: usize,     // 7
    pub gender_start: usize,    // 7
    pub gender_count: usize,    // 2
    pub number_start: usize,    // 9
    pub number_count: usize,    // 2
    pub tense_start: usize,     // 11
    pub tense_count: usize,     // 5
    pub person_start: usize,    // 16
    pub person_count: usize,    // 3
    pub style_start: usize,     // 19
    pub style_count: usize,     // 3
    pub syn_start: usize,       // 22
    pub syn_count: usize,       // 8
    pub total: usize,           // 30
}
```

### SpikeTrain

```rust
pub struct SpikeTrain {
    pub timesteps: usize,       // 16
    pub spikes: Vec<bool>,      // [true, false, false, true, ...]
}
```

### Neuron

```rust
pub struct Neuron {
    pub v: f32,                 // voltagem atual
    pub threshold: f32,         // 1.0
    pub tau: f32,               // 5.0
    pub refrac_steps: u32,      // 2
    pub refrac_left: u32,       // countdown
    pub fired: bool,            // spike neste timestep
    pub spike_times: Vec<usize>,// histórico de spikes
}
```

---

*AMADEUS SNN — Setembro 2026*
