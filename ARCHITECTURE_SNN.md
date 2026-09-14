# AMADEUS SNN — Rede Neural Spiking para Classificação POS

> Classificador bio-inspirado de 7 classes POS. Usa STDP para aprender, modula o grammar na geração.

---

## Sumário

1. [Arquitetura (v6)](#1-arquitetura-v6)
2. [Encoding Morfológico](#2-encoding-morfológico)
3. [Dinâmica Temporal (LIF)](#3-dinâmica-temporal-lif)
4. [Treinamento STDP](#4-treinamento-stdp)
5. [Pipeline de Geração](#5-pipeline-de-geração)
6. [GGUF Serialization](#6-gguf-serialization)
7. [Métricas](#7-métricas)

---

## 1. Arquitetura (v6)

### Re-escrita: 7 outputs (classes POS)

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
│                          │ (7 neurônios)      │
│                          │                    │
│                          │ SUBST(0) VERBO(1)  │
│                          │ ADJ(2)   ART(3)    │
│                          │ ADV(4)   PREP(5)   │
│                          │ PONT(6)            │
│                          └───────────────────┘
│                                    │
│                          16 timesteps LIF
│                          STDP lr=0.001
│                          Top-1: 82.8%
│                          Top-3: 86.0%
└─────────────────────────────────────────────────────────┘
```

### Antes vs Agora

| Métrica | v5 (50K outputs) | v6 (7 outputs) |
|---------|------------------|----------------|
| Output neurons | 50.042 (1/lexema) | **7** (1/classe) |
| STDP Top-1 | 0.0% | **82.8%** |
| STDP Top-3 | 0.0% | **86.0%** |
| Exemplos treino | 57.445 | **1.055** |
| Velocidade | 15min+ (timeout) | **~30s** |
| Aprendizagem | Não converge | **Converge** |

---

## 2. Encoding Morfológico

Mesmo encoding rate-coded (30 neurônios):

| Feature | Bits | Neurônios | Taxa (ativo) |
|---------|------|-----------|--------------|
| Class | 0-2 | 7 (one-hot) | 0.80 |
| Gender | 3 | 2 (one-hot) | 0.70 |
| Number | 4 | 2 (one-hot) | 0.70 |
| Tense | 5-7 | 5 (one-hot) | 0.70 |
| Person | 8-9 | 3 (one-hot) | 0.60 |
| Style | - | 3 (one-hot) | 0.70 |
| SynFunc | - | 8 (one-hot) | 0.60 |

---

## 3. Dinâmica Temporal (LIF)

```
v(t+1) = v(t) + (-v(t)/τ + I(t)) × dt
se v ≥ threshold: spike, v=0, refrac=2
```

| Parâmetro | Valor |
|-----------|-------|
| τ | 5.0 |
| threshold | 1.0 |
| refrac_steps | 2 |
| tsteps | 16 |

---

## 4. Treinamento STDP

### Pesos iniciais (extraídos do grammar)

```
Pesos I→H:
  Neurônios de classe (0-6):
    pesos[CLASS+i][H_i] = 0.5
    pesos[GENDER+i][H_i] = 0.1
    pesos[NUMBER+i][H_i] = 0.1

  Neurônios T3 (7-262):
    pesos[CLASS+cls][H] = 0.4
    pesos[SYN+syn][H] = 0.3
    pesos[STYLE+s][H] = 0.2

Pesos H→O:
  Neurônios de classe → sua classe (0.8)
  T3 neurons → classe do morph (0.5)
```

### STDP iterativo

```rust
for epoch in 0..10 {
    // Embaralhar 1.055 exemplos
    for (morph, syn_func, style, class_id) in examples {
        let trains = enc.encode(morph, syn_func, style, 16, rng);
        let output_spikes = self.run(&trains);
        
        // Calcular top-1 e top-3
        // STDP: atualizar pesos sinápticos
        self.stdp_train(&trains, target_idx, 0.001);
    }
}
```

### Resultados

```
Época 1:  Top-1=77.8%  Top-3=83.8%
Época 5:  Top-1=84.4%  Top-3=87.4%  ← pico
Época 10: Top-1=82.8%  Top-3=86.0%  ← estável
```

---

## 5. Pipeline de Geração

```
1. CUBO hierárquico → prediz classe POS (estrutura sintática)
2. T3 lexicon → top-16 candidatos P(palavra | morph, syn_func, style)
3. SNN 7-outputs → confirma classe (82.8%)
4. Se classe do SNN == classe do T3 → usa T3
5. Se não → fallback para frequency-based
6. GRAPH → modula CUBO via Hamming distance
```

### Backoff

```
CUBO → classe → T3 refina → SNN confirma → GRAPH desempata
```

---

## 6. GGUF Serialization

| Tensor | Shape | Descrição |
|--------|-------|-----------|
| `snn.synapses_ih` | [30, 263] | Pesos Input→Hidden |
| `snn.synapses_ho` | [263, 7] | Pesos Hidden→Output |
| `snn.output_labels` | [7] | Classe POS (0-6) |

---

## 7. Métricas

### Corpus de treinamento

| Fonte | Tokens |
|-------|--------|
| Wikipédia PT | 359K |
| Romances Gutenberg | 151K |
| **Total** | **510K** |

### HMM POS Tagger

| Classe | Prior | Palavras |
|--------|-------|----------|
| SUBST | 56.1% | 10.000 |
| VERBO | 4.4% | 181 |
| ADJ | 2.6% | 129 |
| ART | 12.5% | 29 |
| ADV | 2.9% | 25 |
| PREP | 14.1% | 22 |
| CONJ | 7.4% | 17 |
| PONT | 0.1% | 11 |

### Gramática treinada

| Tabela | Entradas |
|--------|----------|
| T3 (exact) | 1.055 |
| T2 (concordância) | 3.906 |
| GRAPH (PPMI) | 50.030 |
| Bigramas | ~50K pares |

### STDP

| Métrica | Valor |
|---------|-------|
| Top-1 accuracy | 82.8% |
| Top-3 accuracy | 86.0% |
| Learning rate | 0.001 |
| Epochs | 10 |
| Samples/epoch | 1.055 |

---

*AMADEUS SNN v6 — Setembro 2026*
