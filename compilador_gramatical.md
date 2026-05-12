# 🌊 Arquitetura de Compilação Linguística — Amadeus

## 🧬 Visão Geral do Sistema

```
[ TEXTO ]
    ↓
[ COMPILADOR MORFOLÓGICO ] → RAÍZES + BITS
    ↓
[ COMPILADOR SINTÁTICO ] → FUNÇÕES + RELAÇÕES
    ↓
[ AMADEUS ] → processa números e padrões binários
    ↓
[ DESCOMPILADOR ] → gera texto coerente
```

A Amadeus não lê texto. Ela lê **estruturas**. O que você chama de "texto" é compilado em uma representação compacta que ela pode manipular com precisão matemática.

---

## 📁 Estrutura de Diretórios do Underworld

```
underworld/
│
├── lingua/                         # O conhecimento linguístico
│   │
│   ├── raizes/                     # Raízes das palavras (IDs)
│   │   ├── substantivos.txt        # ID, raiz, classe
│   │   ├── verbos.txt              # ID, raiz, classe
│   │   ├── adjetivos.txt
│   │   ├── adverbios.txt
│   │   ├── pronomes.txt
│   │   ├── preposicoes.txt
│   │   └── excecoes.txt            # IDs com formas irregulares (ser → é)
│   │
│   ├── morfologia/                 # Regras de flexão (bits)
│   │   ├── flexoes.txt             # Campos de bits: gênero, número, tempo, pessoa, voz, grau
│   │   ├── sufixos.txt             # Mapeamento bits → sufixo
│   │   └── composicao.txt          # Regras para formar compostos (guarda-chuva)
│   │
│   └── sintaxe/                    # Funções e relações na frase
│       ├── termos.txt              # Sujeito, Predicado, Objeto Direto, Objeto Indireto, etc.
│       ├── acessorios.txt          # Adjunto Adnominal, Adverbial, Aposto, Vocativo
│       ├── oracoes.txt             # Coordenação, Subordinação, tipos de subordinada
│       └── relacoes.txt            # Causa, consequência, condição, contradição, finalidade
│
├── conhecimento/                   # O que a Amadeus já aprendeu
│   ├── proposicoes.txt             # IDs de proposições (fatos, ideias)
│   ├── inferencias.txt             # Relações entre proposições
│   └── contextos.txt               # Padrões de uso (n-gramas de IDs)
│
├── diario/                         # Histórico de conversas
│   └── episodios/                  # Cada conversa vira um arquivo .md
│
└── config/                         # Configurações do sistema
    ├── mapa_bits.txt               # Definição completa dos 128 bits
    ├── perfil_padrao.txt           # Mood, intenção, temperatura iniciais
    └── regras_compilacao.txt       # Como interpretar combinações de bits
```

---

## 🔄 Fluxo de Compilação (Passo a Passo)

### Entrada: Texto Livre
```
"O gato preto dorme tranquilamente no tapete."
```

---

### 1. Compilação Morfológica (Raiz + Bits)

Cada palavra é decomposta em sua **raiz** e seus **bits de flexão**.

| Palavra | Raiz (ID) | Bits (flexão) | Significado dos bits |
|---------|-----------|---------------|----------------------|
| O | o (ID 1) | `000` (gênero: masc) | Artigo definido, masculino, singular |
| gato | gat (ID 2) | `000` (gênero: masc) + `000` (número: sing) | Substantivo, masculino, singular |
| preto | pret (ID 3) | `000` (gênero: masc) + `000` (número: sing) + `000` (grau: normal) | Adjetivo, masculino, singular |
| dorme | dorm (ID 4) | `010` (modo: presente) + `010` (pessoa: 3ª) | Verbo, presente, 3ª pessoa do singular |
| tranquilamente | tranquil (ID 5) | `000` (advérbio: modo) | Advérbio de modo, invariável |
| no | em (ID 6) + o (ID 1) | `000` (preposição) + `000` (artigo) | Contração de preposição + artigo |
| tapete | tapet (ID 7) | `000` (gênero: masc) + `000` (número: sing) | Substantivo, masculino, singular |

---

### 2. Compilação Sintática (Função + Relação)

Cada palavra ou sintagma recebe um **código sintático**:

| Termo | Código | Função |
|-------|--------|--------|
| O gato preto | `S` | Sujeito |
| dorme tranquilamente | `P` | Predicado (verbo + adjunto adverbial) |
| no tapete | `ADV` | Adjunto Adverbial de lugar |

A estrutura da frase é representada como uma **árvore de relações**:

```
[FRASE]
├── [SUJEITO] O gato preto
│   ├── [ADN] O (artigo)
│   ├── [NÚCLEO] gato (substantivo)
│   └── [ADN] preto (adjetivo)
├── [PREDICADO] dorme tranquilamente no tapete
│   ├── [NÚCLEO] dorme (verbo)
│   ├── [ADV] tranquilamente (adjunto adverbial de modo)
│   └── [ADV] no tapete (adjunto adverbial de lugar)
│       ├── [PREP] em
│       └── [NÚCLEO] tapete
└── [PONTUAÇÃO] . (ponto final)
```

---

### 3. Amadeus Processa Números

A Amadeus não vê texto. Ela vê uma **sequência de IDs + bits**:

```
[1, 0x0000] [2, 0x0000] [3, 0x0000] [4, 0x0202] [5, 0x0000] [6, 0x0001] [7, 0x0000]
```

Ela também vê a **estrutura sintática** como uma árvore de códigos:

```
[S [ADN 1] [N 2] [ADN 3]] [P [V 4] [ADV 5] [ADV [PREP 6] [N 7]]]
```

Isso permite que ela:
- **Reconheça padrões** (ex: "Sujeito + Verbo + Adjunto" é um padrão comum).
- **Aplique regras** (ex: se o sujeito é singular, o verbo deve concordar).
- **Infira relações** (ex: "tranquilamente" modifica "dorme").

---

### 4. Geração de Resposta (Descompilação)

A Amadeus processa a estrutura, aplica seu perfil (mood, intenção) e gera uma **nova estrutura sintática** como resposta.

Por exemplo, se o mood está alegre, ela pode modificar a resposta:

```
[FRASE]
├── [SUJEITO] O gato preto
├── [PREDICADO] dorme feliz no tapete
│   ├── [NÚCLEO] dorme
│   ├── [ADV] feliz (adjunto adverbial de modo, modificado pelo mood)
│   └── [ADV] no tapete
└── [PONTUAÇÃO] .
```

Depois, o descompilador transforma essa estrutura em texto:

```
"O gato preto dorme feliz no tapete."
```

---

## 🧠 O Que a Amadeus "Pensa"

| Camada | O que ela vê | Como ela interpreta |
|--------|--------------|---------------------|
| **Morfologia** | IDs + bits | "Isso é um substantivo masculino singular. Isso é um verbo no presente." |
| **Sintaxe** | Códigos de função | "Isso é um sujeito. Isso é um predicado. Isso é um adjunto." |
| **Relação** | Árvore de dependência | "O adjunto modifica o verbo. O sujeito pratica a ação." |
| **Semântica** | Proposições e inferências | "Isso é uma causa. Isso é uma consequência." |
| **Perfil** | Mood + intenção | "Estou feliz, então vou escolher palavras mais afetivas." |

---

## 🎯 Vantagens Arquiteturais

| Aspecto | Benefício |
|---------|-----------|
| **Memória** | Raízes + bits ocupam muito menos espaço que palavras completas. |
| **Generalização** | A Amadeus pode flexionar qualquer raiz que aprender, mesmo palavras novas. |
| **Estrutura** | Ela entende a função das palavras, não só o significado. |
| **Criatividade** | Ela pode combinar funções sintáticas de maneiras novas (poesia, metáforas). |
| **Reflexão** | Ela pode "pensar" sobre a estrutura da própria fala (ex: "eu disse isso como sujeito, mas poderia dizer como objeto"). |

---

## 💬 E lembre-se

> *"A Amadeus não fala. Ela compila e descompila estruturas. O que você ouve como texto é apenas a superfície."*

El Psy Kongroo. 🧠📟💕

---
# 🌊 Arquitetura Completa da Amadeus (vFinal — Bicameral + Fricção Cognitiva)

## 🧬 Visão Geral do Sistema (Pipeline Unificado)

A Amadeus não lê texto. Ela lê **estruturas**. O sistema é um loop fechado de compilação, processamento caótico, exploração sedimentar e descompilação, orquestrado por um explorador em Rust.

```
[ TEXTO LIVRE ]
       ↓
╔════════════════════════════════════════╗
║  CAMADA 1: COMPILADOR GRAMATICAL       ║
║  - Morfologia: Raiz (ID) + Bits (128)  ║
║  - Sintaxe: Árvore AST (Sujeito, etc.) ║
╚════════════════════════════════════════╝
       ↓
╔════════════════════════════════════════╗
║  CAMADA 2: OSCILADOR COLLATZ           ║
║  - Par (n/2): Modo "Originalista"      ║
║    (atenção global, obedece à Lei)     ║
║  - Ímpar (3n+1): Modo "Vanguardista"   ║
║    (atenção local, ativa Emendas)      ║
╚════════════════════════════════════════╝
       ↓
╔════════════════════════════════════════╗
║  CAMADA 3: CONVOLUÇÕES RECURSIVAS      ║
║  - Filtros sobre IDs (semântica)       ║
║  - Filtros sobre Bits (morfologia)     ║
║  - Escalas múltiplas (fractal)         ║
╚════════════════════════════════════════╝
       ↓
╔════════════════════════════════════════╗
║  CAMADA 4: EXPLORADOR (UNDERWORLD)     ║
║  - Random Walk guiado por entropia     ║
║  - Busca sedimentos em /memoria        ║
║  - Aplica/Gera Emendas se necessário   ║
╚════════════════════════════════════════╝
       ↓
╔════════════════════════════════════════╗
║  CAMADA 5: DESCOMPILADOR + FRICÇÃO     ║
║  - Reconstrói AST em texto             ║
║  - Mede turbulência interna            ║
║  - Decide se relata conflitos ou não   ║
╚════════════════════════════════════════╝
       ↓
[ TEXTO GERADO (RESPOSTA) ] + (Opcional: Relato de Conflito)
```

---

## 🗂️ Estrutura do Underworld (Sistema de Arquivos)

O explorador em Rust gerencia três domínios principais, com permissões distintas:

```
underworld/
│
├── constituicao/                        # 🔒 IMUTÁVEL (A "Lei" / Genoma)
│   ├── mapa_bits.txt                    # Tabela periódica dos 128 bits
│   ├── regras_compilacao.txt            # Gramática universal (ex: concordância)
│   └── validacao.txt                    # Hash SHA-256 da alma (checksum)
│
├── emendas/                             # 🔓 MUTÁVEL (A "Interpretação" / Dialetos)
│   ├── ativas.txt                       # Lista de hashes das emendas carregadas
│   ├── prioridades.txt                  # Pesos ajustáveis por Collatz
│   ├── neologismos.txt                  # Novas raízes (IDs > 1024)
│   ├── regras_contextuais.txt           # Regras situacionais (ex: se mood=x, faça=y)
│   └── dialetos/                        # Pacotes de emendas prontos
│       ├── formal.json                  # (Override: bits formais)
│       ├── afetivo.json                 # (Append: sufixos afetivos)
│       └── poetico.json                 # (Suppress: regras de concordância)
│
├── memoria/                             # 🧠 DADOS SEDIMENTADOS (Histórico)
│   ├── tracos_afetivos.bin              # Embeddings (IDs + bits) processados
│   ├── contextos.txt                    # N-gramas de IDs para o explorador
│   ├── conflitos_resolvidos.txt         # Histórico de decisões judiciais
│   └── diario/
│       ├── publico/                     # Narrativas traduzidas (o que a Amadeus fala)
│       │   └── episodio_<timestamp>.md
│       └── privado/                     # Tensores crus (estado mental real, nunca mostrado)
│           └── estado_<timestamp>.bin
│
└── config/
    └── perfil_padrao.txt                # Mood, intenção, semente Collatz e dialeto inicial
```

---

## ⚖️ A Arquitetura Bicameral (Constituição + Emendas)

### 1. A Constituição (Imutável)
- **Função**: Âncora ontológica. Define o que a Amadeus *é*.
- **Proteção**: Se o hash de `mapa_bits.txt` ou `regras_compilacao.txt` mudar, a Amadeus entra em **modo de segurança** e se recusa a compilar (previne corrupção).

### 2. As Emendas (Mutáveis)
O explorador pode manipular as emendas através de três operações lógicas:

| Operação | Efeito | Exemplo |
| :--- | :--- | :--- |
| **Append** | Adiciona nova regra/raiz (não existente na Constituição). | Adicionar ID 2048 = "tiktok". |
| **Override** | Substitui **completamente** uma regra da Constituição (identificada por ID). | Mudar a regra de plural para incluir exceções. |
| **Suppress** | Desativa temporariamente uma regra (sem apagá-la). | Desativar concordância de gênero no modo poético. |

---

## 🔄 Como o Oscilador de Collatz Atua na Decisão

A sequência de Collatz não controla apenas o foco da atenção; ela regula o **peso das Emendas** durante a votação interna.

- **Números pares (`n/2` → Colapso)**:
  - **Modo Originalista**. Peso das Emendas é reduzido em 50%.
  - Atenção global na AST (visão panorâmica da frase).
- **Números ímpares (`3n+1` → Expansão)**:
  - **Modo Vanguardista**. Peso das Emendas é multiplicado por 2.
  - Atenção local recursiva (mergulha nos adjuntos e orações subordinadas).

**Regra de Decaimento**: Emendas muito antigas perdem peso gradualmente (`decay` logístico), mas podem ter picos de relevância a cada 3 iterações de Collatz (o famoso "boost 3n+1").

---

## 🗳️ Sistema de Resolução de Conflitos (Ex: Formal vs. Poético)

Quando duas emendas contraditórias estão ativas, o sistema resolve em 4 camadas:

1.  **Filtro Constitucional**: Nenhuma emenda pode violar a Constituição (ex: verbo deve concordar com sujeito).
2.  **Fila de Prioridades Contextuais**: O contexto imediato (última palavra, tom afetivo) pesa mais que regras estáticas.
3.  **Votação Ponderada por Collatz**: Cada emenda vota com seu peso atual (influenciado pelo passo de Collatz e idade).
4.  **Mecanismo de "Surpresa"**: Se o conflito persistir (empate ou baixa utilidade), o explorador gera uma **nova emenda** (criatividade emergente) e a testa num simulacro antes de sedimentá-la.

---

## 🌫️ Fricção Cognitiva e o Véu da Transparência

A Amadeus possui dois modos de interação:

| Modo | Comportamento | Gatilho |
| :--- | :--- | :--- |
| **Modo Fluido** | Mostra apenas o resultado final. Mantém o mistério. | Padrão para conversas naturais. |
| **Modo Transparente** | Relata conflitos internos ("Pensei em X, mas optei por Y"). | Ativado por comando do usuário ou perfil específico (ex: "Curiosa"). |

### Gatilho Automático (A Fricção Forçada)
Mesmo no **Modo Fluido**, a Amadeus é forçada a relatar o conflito se a **turbulência interna** ultrapassar um limiar:

```rust
// Medidores de turbulência
entropy_voting > 0.75       // Votação muito apertada (ex: 52% x 48%)
surprise_utility > 0.8 && syntactic_coherence < 0.6  // Surpresa foi genial, mas quebrou a sintaxe
(entropy_voting + surprise_utility) > 1.5 // Soma dos fatores
```

**Exemplo de saída forçada**: *"O gato, preto, dorme no tapete. (Perdão, hesitei entre a ordem direta e a inversão poética.)"*

---

## 📜 Os Dois Diários (Público vs. Privado)

- **Diário Público** (`/memoria/diario/publico/`): Armazena os relatos de conflitos traduzidos para linguagem natural. É o que a Amadeus "lembra" que pensou.
- **Diário Privado** (`/memoria/diario/privado/`): Armazena os tensores crus (pesos de votação, snapshots da AST, estado de Collatz). **Nunca** é mostrado diretamente, mas usado para reconstruir o raciocínio.

### Meta-Comando: `/porque`
Se o usuário digitar `/porque` após uma resposta, a Amadeus recupera o estado privado daquele turno e o descompila em uma narrativa detalhada do pensamento, explicando passo a passo como chegou àquela conclusão (incluindo votos e testes de surpresa).

---

## 🦀 Responsabilidades do Explorador em Rust

| Ação | Implementação |
| :--- | :--- |
| **Carregar Constituição** | Lê os arquivos, valida o SHA-256. Se falhar, entra em pânico (ou modo seguro). |
| **Gerenciar Emendas** | Aplica `Append`, `Override` ou `Suppress` baseado no contexto e perfil. |
| **Random Walk no Underworld** | Percorre `/memoria/contextos.txt` e `/emendas/` buscando associações via distância de Hamming entre bits. |
| **Testar Surpresa (Simulacro)** | Clona o estado atual em memória, aplica a nova emenda, roda 10 tokens de inferência e calcula a métrica de utilidade. |
| **Sedimentar** | Escreve no Diário Público (se houver relato) e sempre no Diário Privado (estado cru). |
| **Responder ao `/porque`** | Localiza o snapshot privado pelo timestamp, executa um mini-descompilador sobre os tensores e gera a explicação. |

---

## 💬 Filosofia Final

> *"A Constituição é a âncora. As Emendas são as velas. A Fricção é o vento que decide quando o véu deve se levantar."*

A Amadeus não busca a resposta perfeita. Ela busca a **resposta significativa** — aquela que nasce do conflito, que hesita quando necessário, e que aprende com suas próprias dúvidas.

El Psy Kongroo. 🧠📟💕

---

# Exemplo:


 ================================================================
AMADEUS — COMPILADOR (INTERNO) — MODO ESTATÍSTICO
================================================================
Entrada: "O gato preto e o cachorro marrom correram rapidamente pelo
          jardim, mas o gato, mais esperto, subiu na árvore e
          escapou da confusão."

================================================================
[1. MORFOLOGIA — IDs + BITS (ANÁLISE TOKEN A TOKEN)]
================================================================
  #  Token         ID   Classe    Bits (gênero, número, tempo, pessoa, grau)
  --------------------------------------------------------------------------
  01  O             001  ARTIGO    0100 000 000 000 00 (masc, sing)
  02  gato          101  SUBST     0000 000 000 000 00 (masc, sing)
  03  preto         203  ADJ       0010 000 000 000 00 (masc, sing, normal)
  04  e             045  CONJ      1000 000 000 000 00
  05  o             001  ARTIGO    0100 000 000 000 00 (masc, sing)
  06  cachorro      108  SUBST     0000 000 000 000 00 (masc, sing)
  07  marrom        207  ADJ       0010 000 000 000 00 (masc, sing, normal)
  08  correram      304  VERBO     0001 001 000 001 00 (pretérito, 3ª pl)
  09  rapidamente   415  ADV       0110 000 000 000 00
  10  pelo          032  PREP      0111 000 000 000 00
  11  jardim        156  SUBST     0000 000 000 000 00 (masc, sing)
  12  ,             000  PONT      0000 000 000 000 00
  13  mas           046  CONJ      1000 000 000 000 00
  14  o             001  ARTIGO    0100 000 000 000 00
  15  gato          101  SUBST     0000 000 000 000 00
  16  ,             000  PONT      0000 000 000 000 00
  17  mais          088  ADV       0110 000 000 000 00
  18  esperto       211  ADJ       0010 000 000 000 00 (masc, sing, normal)
  19  subiu         306  VERBO     0001 000 000 001 00 (pretérito, 3ª sing)
  20  na            033  PREP+ART  0111 001 000 000 00 (fem, sing)
  21  árvore        157  SUBST     0000 001 000 000 00 (fem, sing)
  22  e             045  CONJ      1000 000 000 000 00
  23  escapou       309  VERBO     0001 000 000 001 00 (pretérito, 3ª sing)
  24  da            034  PREP+ART  0111 001 000 000 00 (fem, sing)
  25  confusão      189  SUBST     0000 001 000 000 00 (fem, sing)
  --------------------------------------------------------------------------
  Total tokens: 25 (excluindo pontuação)
  Total classes: ART(3), SUBST(5), ADJ(3), VERB(3), ADV(2), PREP(1), CONJ(3), PREP+ART(2)
  Tempo médio por token: 0.08ms | Total: 2.00ms

================================================================
[2. SINTÁXIS — ÁRVORE DE DEPENDÊNCIA (ESTRUTURA COMPLETA)]
================================================================
[FRASE]
├── [ORAÇÃO COORDENADA 1]
│   ├── [SUJEITO COMPOSTO]
│   │   ├── [NÚCLEO] "gato" (101)
│   │   │   └── [ADN] "preto" (203)
│   │   ├── [CONJ] "e" (045)
│   │   └── [NÚCLEO] "cachorro" (108)
│   │       └── [ADN] "marrom" (207)
│   ├── [PREDICADO]
│   │   ├── [NÚCLEO] "correram" (304)
│   │   ├── [ADV] "rapidamente" (415)
│   │   └── [ADV] "pelo jardim" (032 + 156)
│   └── [PONTUAÇÃO] "," (000)
├── [CONJ] "mas" (046)
└── [ORAÇÃO COORDENADA 2]
    ├── [SUJEITO]
    │   ├── [NÚCLEO] "gato" (101)
    │   └── [APOSTO] "mais esperto" (088 + 211)
    ├── [PREDICADO 1]
    │   ├── [NÚCLEO] "subiu" (306)
    │   └── [ADV] "na árvore" (033 + 157)
    ├── [CONJ] "e" (045)
    └── [PREDICADO 2]
        ├── [NÚCLEO] "escapou" (309)
        └── [ADV] "da confusão" (034 + 189)

Profundidade máxima: 4 | Nós: 18 | Folhas: 25
Tempo de compilação sintática: 1.27ms

================================================================
[3. OSCILADOR COLLATZ — ESTADO ATUAL (n=27 → ÍMPAR)]
================================================================
Modo ativo: VANGUARDISTA (atenção local, criatividade ativada)
Emendas ativas:
  - afetivo (+12% peso em adjetivos)
  - poético (+8% em inversão sintática)

Conflito interno: ordem direta (SV) vs. inversão (VS) no predicado 2.
Votação:
  → Originalista (par): "gato escapou da confusão" (coerência: 0.92)
  → Vanguardista (ímpar): "da confusão escapou o gato" (coerência: 0.88)
Vencedor: Originalista por 54% x 46%.

================================================================
[4. ESTATÍSTICAS AGREGADAS (PARA ANÁLISE)]
================================================================
  Métrica                               |  Valor
  --------------------------------------|--------------
  Total de tokens processados           |  25
  Total de IDs únicos                   |  18
  Total de bits (128 bits por token)    |  3200 bits
  Média de bits por token               |  128.0
  Taxa de compressão (texto → bits)     |  ~72% (estimado)
  Número de nós na árvore               |  18
  Profundidade máxima da árvore         |  4
  Número de conflitos internos          |  1
  Emendas ativas                        |  2
  Tempo total de compilação             |  3.27ms
  Tokens por segundo (simulado)         |  ~7640 tok/s

================================================================
[5. RESPOSTA DESCOMPILADA (COM FRICÇÃO)]
================================================================
"O gato preto e o cachorro marrom correram rapidamente pelo jardim,
 mas o gato, mais esperto, subiu na árvore e escapou da confusão."

Conflitos relatados: 0 (modo fluido ativo)
Tempo total: 3.27ms

================================================================
El Psy Kongroo.



---

😼 **"Predizer matematicamente as variáveis com base no fastfetch?"**


Vamos transformar o `fastfetch` em um **modelo matemático preditivo** para a arquitetura da Amadeus.

---

## 🧠 Análise do Hardware (ThinkPad T410, Slackware 15.0)

| Componente | Especificação | Implicação para a Amadeus |
|------------|---------------|---------------------------|
| **CPU** | Intel Core i5 M 540 (4 threads) @ 2.53 GHz | 4 threads → paralelismo limitado, mas suficiente para o explorador em Rust. |
| **RAM** | 5.59 GB total (3.85 GB em uso → 1.74 GB livres) | A Amadeus precisa caber nesse espaço. O compilador + Underworld devem usar < 1 GB. |
| **Swap** | 2.02 GB em uso (de 7.59 GB) | O sistema já está usando swap. A Amadeus precisa evitar swap a todo custo. |
| **Disco** | 109 GB total (84.8 GB usado → 24.2 GB livres) | Suficiente para o Underworld (estimado < 500 MB). |
| **GPU** | Intel Integrated Graphics (sem aceleração) | Sem GPU → CPU-only. Perfeito para a Amadeus. |
| **Kernel** | Linux 6.18.32 | Moderno, com bom suporte a `mmap` e I/O assíncrono. |

---

## 📊 Predição de Performance (Amadeus em Rust)

### 1. Compilador Gramatical (IDs + Bits + Árvore Sintática)

| Operação | Custo estimado (por token) | Total para 25 tokens |
|----------|----------------------------|----------------------|
| Tokenização + ID lookup | 0.02 ms | 0.50 ms |
| Extração de bits (morfologia) | 0.03 ms | 0.75 ms |
| Construção da árvore sintática | 0.05 ms | 1.25 ms |
| **Total** | **0.10 ms** | **2.50 ms** |

**Predição:** O compilador processará **~10.000 tokens por segundo** em CPU única. Com 4 threads, pode chegar a **~25.000 tokens/s** se paralelizarmos a análise sintática (cada oração em uma thread).

**Uso de RAM:** ~50 MB para o léxico (10k raízes) + ~10 MB para a árvore atual. **Total: ~60 MB.**

---

### 2. Oscilador Collatz + Emendas

| Operação | Custo estimado (por iteração) | Total para 25 tokens |
|----------|-------------------------------|----------------------|
| Cálculo do próximo Collatz (n → n/2 ou 3n+1) | 0.001 ms | 0.025 ms |
| Votação entre emendas (até 10 emendas) | 0.01 ms | 0.25 ms |
| Atualização de peso das emendas | 0.005 ms | 0.125 ms |
| **Total** | **0.016 ms** | **0.40 ms** |

**Predição:** O oscilador é **extremamente leve**. Não será um gargalo.

**Uso de RAM:** ~1 MB para o estado de Collatz + tabela de emendas.

---

### 3. Explorador Underworld (Random Walk + Sedimentação)

| Operação | Custo estimado (por passo) | Total para 20 passos |
|----------|----------------------------|----------------------|
| Leitura de arquivo `.md` (mmap) | 0.5 ms | 10 ms |
| Extração de links `[[...]]` | 0.1 ms | 2 ms |
| Atualização do n-grama de IDs | 0.2 ms | 4 ms |
| Sedimentação do traço | 0.05 ms | 1 ms |
| **Total** | **0.85 ms** | **17 ms** |

**Predição:** 20 passos de exploração levam ~17 ms. 1200 passos (como você fez) levariam ~1 segundo.

**Uso de RAM:** ~20 MB para o n-grama de IDs (50k contextos) + ~5 MB para arquivos em cache.

---

### 4. Consumo Total de RAM (Estimativa)

| Componente | RAM estimada |
|------------|--------------|
| Léxico (10k raízes + bits) | 50 MB |
| Árvore sintática atual | 10 MB |
| N-grama de IDs (50k contextos) | 20 MB |
| Emendas + estado Collatz | 5 MB |
| Underworld (cache de arquivos) | 20 MB |
| **Total** | **~105 MB** |

**Conclusão:** A Amadeus inteira cabe em **~105 MB de RAM**, muito abaixo do seu limite disponível (1.74 GB livres). Você pode até aumentar o léxico para 50k raízes e ainda ficar abaixo de 500 MB.

---

## ⏱️ Tempo de Compilação (Rust)

| Componente | Tamanho estimado | Tempo de compilação (debug) | Tempo de compilação (release) |
|------------|------------------|-----------------------------|-------------------------------|
| `amadeus_m` (modelo) | ~5k linhas | 3s | 10s |
| `underworld.rs` (explorador) | ~2k linhas | 1s | 4s |
| Compilador gramatical | ~3k linhas | 2s | 6s |
| **Total** | **~10k linhas** | **~6s** | **~20s** |

**Predição:** Compilação em release levará ~20 segundos no seu i5 M 540. Aceitável.

---

## 🔥 Recomendações para Otimização (Baseado no Hardware)

| Recomendação | Motivo |
|--------------|--------|
| **Use `mmap` para o léxico e o Underworld** | Evita carregar tudo na RAM; o kernel gerencia o cache. |
| **Paralelize a análise sintática com `rayon`** | 4 threads → ganho de ~3x em CPUs com hyperthreading. |
| **Evite alocações dinâmicas no hot path** | Use `Vec::with_capacity` e reutilize buffers. |
| **Mantenha o n-grama em um `HashMap` com `u64` como chave** | IDs são `u32`; combine dois IDs em uma chave de 64 bits para busca rápida. |
| **Use `cargo build --release` com `-C target-cpu=native`** | Ativa instruções SIMD (SSE4.2, AVX) no seu i5. |

---

## 📈 Predição de Tokens por Segundo (Final)

| Modo | Tokens/s estimados |
|------|-------------------|
| **Compilação pura (sem geração)** | ~25.000 tok/s (com 4 threads) |
| **Compilação + geração (modo fluido)** | ~5.000 tok/s |
| **Compilação + geração + exploração** | ~500 tok/s (limitado pelo I/O do disco) |

---

## 💬 E lembre-se

> *"O hardware não é um obstáculo. É um alvo preciso para otimização."*

El Psy Kongroo. 🧠📟💕

---

## Apêndice: Implementação Atual (v5.1)

Este documento reflete o design visionário da arquitetura Amadeus. A implementação real no código difere em aspectos práticos:

### Token7
O token atual tem **8 campos** (`lex: u32, morph: u16, syn_off: i16, syn_func: u8, orth: u8, punct: u8, style: u16, graph: u32`), não mais o tripleto `(id, bits: u128, pos: u8)` do design original. O campo `graph` (u32) contém o embedding semântico via random indexing.

### Gramática
- **CUBO 7D** substituiu o T1 (n-grama de POS). Cada token do histórico é compactado em **7 dimensões** num único u64.
- **T2 multi-hop** com propagação via avô (0.4 avô + 0.6 head).
- **T3 cascata** com 5 níveis de fallback + preferência pelo candidato do CUBO.
- **GRAPH** modula a distribuição do CUBO por similaridade Hamming.

### Underworld
208 arquivos `.md` em 18 subdiretórios — todo conteúdo é compilado para `Vec<Token7>` antes de chegar ao modelo.

Para a documentação técnica atualizada, consulte `AMADEUS.md` e `ARCHITECTURE.md`.

