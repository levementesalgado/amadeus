# AMADEUS SNN — Rede Neural Spiking Recorrente para Classificação POS

> Classificador bio-inspirado de 7 classes POS. Usa STDP de 3 fatores com friction, memória recorrente entre tokens.

---

## Sumário

1. [Arquitetura (v6.2)](#1-arquitetura-v62)
2. [Encoding Morfológico](#2-encoding-morfológico)
3. [Dinâmica Temporal (LIF)](#3-dinâmica-temporal-lif)
4. [Recorrência (Membrana Persistente)](#4-recorrência-membrana-persistente)
5. [Treinamento 3-Factor STDP](#5-treinamento-3-factor-stdp)
6. [Pipeline de Geração](#6-pipeline-de-geração)
7. [GGUF Serialization](#7-gguf-serialization)
8. [Métricas](#8-métricas)

---

## 1. Arquitetura (v6.2)

### Recorrência + 3-Factor STDP

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
│                          │                    │
│                          │ ★ MEMBRANA         │
│                          │   PERSISTE         │
│                          │   ENTRE TOKENS     │
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
│                          3-Factor STDP
│                          Top-1: 89.2%
│                          Top-3: 91.2%
└─────────────────────────────────────────────────────────┘
```

### Antes vs Agora

| Métrica | v5 (50K outputs) | v6.0 (7 outputs) | v6.2 (recorrente) |
|---------|------------------|------------------|-------------------|
| Output neurons | 50.042 | **7** | **7** |
| STDP Top-1 | 0.0% | 82.0% | **89.2%** |
| STDP Top-3 | 0.0% | 85.6% | **91.2%** |
| Recorrência | Não | Não | **Sim** |
| STDP rule | 2-fator | 2-fator | **3-fator** |
| Memória entre tokens | Não | Não | **Sim** |

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

## 4. Recorrência (Membrana Persistente)

### Problema

SNN antigo rodava do zero a cada token:
```rust
// Antigo
net.reset();  // ← perde toda memória
for t in 0..16 { ... }
```

### Solução

Manter potencial de membrana entre tokens:
```rust
// Novo
// NÃO chama reset() — membrana carrega resíduo
for t in 0..16 { ... }
```

### Impacto

- **Memória de longo alcance**: potencial acumulado entre tokens
- **Sem overhead**: não aumenta neurons nem dimensões
- **Biológico**: neurônios reais não reseta entre estímulos

---

## 5. Treinamento 3-Factor STDP

### STDP clássico (2 fatores)

```
Δw = f(pre, post)
```

Apenas timing de spikes. Não sabe se a predição estava certa.

### 3-Factor STDP (com modulador)

```
Δw = f(pre, post) × modulator
```

**Modulador** = sinal de fricção/reforço:
- `modulator = 1.0` → predição correta (reforço)
- `modulator = 0.3` → predição errada (fricção)

### Efeito

| Condição | LTP (potenciação) | LTD (depressão) |
|----------|-------------------|-----------------|
| Acertou (mod=1.0) | Forte (1.0×) | Fraca (0.15×) |
| Errou (mod=0.3) | Fraca (0.3×) | Moderada (0.85×) |

**Resultado**: rede aprende mais rápido com exemplos que acerta, esquece menos quando confiante.

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

### STDP iterativo com 3 fatores

```rust
for epoch in 0..10 {
    for (morph, syn_func, style, class_id) in examples {
        let trains = enc.encode(morph, syn_func, style, 16, rng);
        let output_spikes = self.run(&trains);
        
        // Predição
        let predicted = scores.first().map(|(c, _)| *c).unwrap_or(0);
        
        // Modulador: 1.0 se acertou, 0.3 se errou
        let modulator = if predicted == class_id { 1.0 } else { 0.3 };
        
        // STDP de 3 fatores
        self.stdp_train_3factor(&trains, target_idx, 0.001, modulator);
    }
}
```

### Resultados

```
Época 1:  Top-1=84.0%  Top-3=89.8%
Época 5:  Top-1=86.8%  Top-3=91.4%
Época 10: Top-1=89.2%  Top-3=91.2%
```

---

## 6. Pipeline de Geração

```
1. CUBO hierárquico → prediz classe POS (estrutura sintática)
2. T3 lexicon → top-16 candidatos P(palavra | morph, syn_func, style)
3. SNN recorrente → confirma classe (89.2%) ★ memória entre tokens
4. Se classe do SNN == classe do T3 → usa T3
5. Se não → fallback para frequency-based
6. GRAPH (SVD) → modula CUBO via Hamming distance
```

### Backoff

```
CUBO → classe → T3 refina → SNN recorrente confirma → GRAPH (SVD) desempata
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
| T3 (exact) | 1.350 |
| T2 (concordância) | 4.800 |
| GRAPH (SVD) | 2.000 |
| Bigramas | ~50K pares |

### STDP

| Métrica | v6.0 | v6.2 |
|---------|------|------|
| Top-1 accuracy | 82.0% | **89.2%** |
| Top-3 accuracy | 85.6% | **91.2%** |
| Learning rate | 0.001 | 0.001 |
| STDP rule | 2-fator | **3-fator** |
| Recorrência | Não | **Sim** |
| Epochs | 10 | 10 |
| Samples/epoch | 1.055 | 1.350 |

---

*AMADEUS SNN v6.2 — Setembro 2026*
