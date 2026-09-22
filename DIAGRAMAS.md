# Amadeus - Diagramas

## Arquitetura Geral

```mermaid
graph TB
    subgraph "Entrada"
        A[Texto] --> B[Token7]
    end

    subgraph "Compilação"
        B --> C[SimpleCompiler]
        C --> D[TripleGrammar]
        C --> E[HMM]
        C --> F[MorphVocab]
    end

    subgraph "Modelos"
        D --> G[GRAPH / SVD]
        D --> H[SNN]
        D --> I[Bigrams]
        F --> G
        E --> D
    end

    subgraph "Geração"
        G --> J[Rhetoric]
        H --> J
        I --> J
        J --> K[SelfPlay]
        K --> L[Texto]
    end

    style G fill:#f9f,stroke:#333
    style H fill:#bbf,stroke:#333
    style J fill:#bfb,stroke:#333
```

## Pipeline de Treinamento

```mermaid
flowchart TD
    A[Corpus Bruto] --> B[Strip Gutenberg]
    B --> C{Português?}
    C -- Não --> D[Descartar]
    C -- Sim --> E[Token7]
    E --> F[SimpleCompiler]
    F --> G[HMM Treinar]
    F --> H[MorphVocab]
    F --> I[TripleGrammar]
    G --> I
    H --> I
    I --> J[GRAM SVD]
    I --> K[SNN Treinar]
    I --> L[Bigrams]
    J --> M[Modelo Salvo]
    K --> M
    L --> M

    style M fill:#f9f,stroke:#333
```

## Geração de Texto

```mermaid
flowchart TD
    A[Input] --> B[Rhetoric Role]
    B --> C[TEMPLATE Pos Sequence]
    C --> D[CUBO Predict Class]
    D --> E[T3 Refine]
    E --> F[GRAPH Hamming]
    F --> G[SelfPlay Filter]
    G -- score >= 0.75 --> H[Accept]
    G -- score < 0.75 --> I[Reject]
    I --> E
    H --> J[Output]

    style H fill:#bfb,stroke:#333
    style I fill:#fbb,stroke:#333
```

## Classes Principais

```mermaid
classDiagram
    class TripleGrammar {
        +train(corpus)
        +generate_with_snn()
        +generate_phrases()
        +graph: Graph
        +snn: SpikingNetwork
        +hmm: HmmPosTagger
        +bigrams: HashMap
    }

    class SpikingNetwork {
        +weights: Matrix
        +membrane: Vec
        +stdp_train_3factor()
        +iterative_train() Best epoch
        +predict(input) Vec
    }

    class Graph {
        +embeddings: HashMap
        +svd: Matrix
        +ppmi: Matrix
        +distance(a, b) f32
    }

    class HmmPosTagger {
        +transitions: Matrix
        +emissions: Matrix
        +priors: Vec
        +viterbi(obs) Vec
    }

    class AffixVocabulary {
        +roots: Vec
        +prefixes: Vec
        +suffixes: Vec
        +tokenize(word) Vec
    }

    class Rhetoric {
        +templates: Vec
        +plan() RhetoricRole
        +pos_sequence() Vec
    }

    class SelfPlayFilter {
        +evaluate(text) f32
        +score: f32
        +threshold: f32
    }

    TripleGrammar --> SpikingNetwork
    TripleGrammar --> Graph
    TripleGrammar --> HmmPosTagger
    TripleGrammar --> AffixVocabulary
    TripleGrammar --> Rhetoric
    TripleGrammar --> SelfPlayFilter
```

## SNN - Treinamento Iterativo

```mermaid
flowchart TD
    A[Init Weights] --> B[Epoch 1]
    B --> C{Top-1 > 90%?}
    C -- Não --> D[STDP 3-Factor]
    D --> E[Save Weights if Best]
    E --> F[Next Epoch]
    F --> B
    C -- Sim --> G[Early Stop]
    G --> H[Restore Best Weights]
    H --> I[Done]

    style G fill:#bfb,stroke:#333
```

## GRAPH - Embeddings

```mermaid
flowchart TD
    A[Corpus] --> B[Co-occurrence Matrix]
    B --> C[PPMI]
    C --> D[Row Normalize]
    D --> E[Power Iteration SVD]
    E --> F[32-dim Embeddings]
    A --> G[Morphological XOR]
    G --> H[All 50K Words]
    F --> I[Top 2000 Overlay]
    H --> I
    I --> J[Final Embeddings]

    style J fill:#f9f,stroke:#333
```

## HMM - POS Tagging

```mermaid
flowchart TD
    A[Word] --> B[Character Features]
    B --> C[Feature Vector]
    C --> D[Transition Matrix]
    C --> E[Emission Matrix]
    D --> F[Viterbi]
    E --> F
    F --> G[POS Tag]

    style G fill:#bbf,stroke:#333
```

## MorphVocab - Tokenização

```mermaid
flowchart TD
    A[Word "correndo"] --> B[Match Suffix]
    B --> C["correndo" → "corrend" + "o"]
    C --> D[Match Prefix]
    D --> E["corrend" → "co" + "rrend"]
    E --> F[Components]
    F --> G[ID: 12345]

    style G fill:#bfb,stroke:#333
```

## Data Flow

```mermaid
graph LR
    subgraph "Disk"
        A[corpus.txt]
        B[model.bin]
        C[vocab.bin]
    end

    subgraph "Memory"
        D[TripleGrammar]
        E[Graph]
        F[SNN]
        G[HMM]
    end

    subgraph "Output"
        H[Generated Text]
    end

    A --> D
    D --> B
    D --> C
    B --> D
    C --> D
    D --> E
    D --> F
    D --> G
    E --> H
    F --> H
    G --> H
```

## Módulos

```mermaid
graph TB
    subgraph "Núcleo"
        A[triple_grammar.rs]
        B[token7.rs]
        C[compiler.rs]
        D[syntax.rs]
    end

    subgraph "Modelos"
        E[snn.rs]
        F[graph via SVD]
        G[hmm.rs]
        H[morph_vocab.rs]
        I[pcfg.rs]
    end

    subgraph "Geração"
        J[rhetoric.rs]
        K[self_play.rs]
        L[synthesis.rs]
    end

    subgraph "Infra"
        M[dual_loader.rs]
        N[embedding_compressor.rs]
        O[rng.rs]
        P[config.rs]
    end

    A --> E
    A --> F
    A --> G
    A --> H
    A --> I
    A --> J
    J --> K
    K --> L
    M --> A
    N --> F
    O --> A
    P --> A
```
