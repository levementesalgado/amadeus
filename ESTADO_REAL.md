# Estado Real do Amadeus — medido em 2026-09-28

Documento de diagnóstico. A `ARCHITECTURE.md` descreve o projeto como ele foi
desenhado; este descreve o que foi medido no código, com números.

Comandos usados para produzir cada número estão indicados.

---

## 1. O caminho de execução

O projeto tem **um** caminho de inferência que funciona: o pipeline de gramática
tabular (CUBO + T2/T3 + SNN). Ele opera sobre `Token7` — 8 campos inteiros — e
não é uma rede neural.

```bash
cargo run --release --bin train_wikipedia      # gera o modelo
cargo run --release --bin benchmark_wikipedia  # avalia
```

```
▸ FASE 1: Carregando gramática Wikipédia...
  Gramática carregada: 1379 lex, 4968 T2, 99301 GRAPH
  SNN construída

  "o Brasil é um país"
    Cascade: o brasil é um país brasil
    SNN:     o brasil é um país
```

### O que foi removido em 2026-09-28

Havia um segundo caminho, um transformer decoder, com dois binários próprios
(`amadeus_m` e `amadeus_m_train`). Foram apagados junto com mais cinco
demonstrações. Motivo: não funcionavam e não eram usados.

O `amadeus_m` rodava com pesos **aleatórios** — `AmadeusMModel::new(cfg)` cria
pesos novos, e o modelo treinado de 795 MB nunca era carregado. A saída era
`//////////r[231]`:

```
Generating 20 tokens...
//////////r[231]/r[231]/r[231]/r
```

O `amadeus_m_train` tinha interface de REPL (`Você>`, comandos `/porque`,
`/ast`, `temp`) mas **não conversava**: cada turno chamava `grammar.train()`
antes de responder, o léxico tinha 17 entradas, e o corpus de raízes estava
vazio. A forma era de conversa; o conteúdo, um playground de gramática treinado
ao vivo.

Sobrou o código de transformer em `src/model/`, `src/layers/`,
`src/sampler/`, `src/tokenizer/`, `src/quant/` e `src/main.rs`, sem binário que
o exercite. Ver a seção 6.

O `.bin` que o `benchmark_wikipedia` carrega **não inclui o CUBO**, e é por isso
que esse caminho roda sem consumo excessivo de memória.

---

## 2. O modelo de 795 MB e seu custo de memória

`training/wikipedia_grammar.gguf` — 775 MB, 19 tensores, GGUF v3.

### Onde estão os bytes

```
cubo.clause.blob     721,5 MB   97,5%   <- 189.135.413 bytes
lexicon.forms          3,0 MB
t2.head.data           2,3 MB
t3.exact.data          2,2 MB
t3.css.data            2,2 MB
demais (14 tensores)   ~8 MB
```

### Onde está a RAM

```
arquivo lido:      739 MB
rss apos read:     741 MB      <- I/O e quantização não são o gargalo
load_gguf total:  3632 MB      <- +2,9 GB na deserialização
cubo clause table: 7670280    <- 7,67 milhões de contextos
```

Deserializar 7,67M entradas em `HashMap<Vec<u128>, HashMap<u32, f32>>` custa
~380 bytes por entrada. Não é quantização: `snn.synapses_ih` tem 7.890
elementos e ocupa 0,03 MB.

### Verificação: o round-trip do blob está correto

Vale registrar porque a hipótese natural ("o reader corrompe o blob") é falsa:

```
save:  numel = data.len()/4   -> shape=[189135413]
read:  byte_size = numel*4    -> 756541652 bytes = tamanho real do blob
n_contexts declarado:  7.670.280
contextos na tabela:   7.670.280
```

Simétrico, correto, sem leitura fora dos limites.

### Distribuição de frequência

```
contextos totais:  7.670.280
mediana:  total=1        <- metade dos contextos aparece UMA vez
p75:      total=1
p90:      total=1
p99:      total=3

poda total<2:  mantém   220.782 ( 2,9%) contextos, 15% da massa
poda total<3:  mantém    92.307 ( 1,2%) contextos, 12% da massa
```

90% dos contextos são únicas. Podar `total<2` remove 97% das entradas e 85%
da massa — não é remoção de ruído, é descarte do aprendido. Por isso a poda
foi descartada e o sharding foi escolhido.

---

## 3. Sharding (implementado, commit `91277ee`)

`src/amadeus_m/shard.rs`. Um arquivo `.cuboshard` por flush, índice
contexto→offset em RAM, registros lidos via mmap.

```
memória real  3,6 GB -> 1,0 GB
geração       4,0s  -> 3,9s   (1,02x)
saída         idêntica
```

A memória real foi confirmada com `malloc_trim` (2,9 GB → 1,0 GB) e com uma
alocação de 2 GB subsequente, que não aumentou o RSS.

**Ressalva:** o spill roda *depois* do pico de RAM. Não resolve OOM quando dois
processos carregam o modelo juntos (testes em paralelo). Resolver exigiria ler
o blob direto do mmap sem desserializar — não implementado.

---

## 4. Correção de não-determinismo na amostragem

Bug pré-existente, encontrado pelo teste de paridade do sharding.

`HyperCube::sample_lex_modulated` percorria `dist` (um `HashMap`) na ordem de
iteração para a varredura cumulativa da amostra. A ordem de iteração de um
`HashMap` depende do layout interno, que muda entre construção por
`entry().or_insert()` e por `with_capacity()`. Resultado: dois CUBos com o
mesmo conteúdo, montados de formas diferentes, geravam sequências distintas
com a mesma semente.

Corrigido ordenando por lexema em quatro pontos: `probs` na amostragem, as
chaves do ramo de exploração, o top-k da modulação GRAPH e o desempate do
`sort_by` (que é instável).

A geração agora é reprodutível entre builds de tabela distintos. Este é o
ganho de mais valor do trabalho, e é independente do sharding.

---

## 5. Como rodar cada coisa

Os seis binários que restaram:

| Binário | Função |
|---------|--------|
| `train_wikipedia` | Treina o modelo a partir do corpus Wikipédia (813 linhas, ~30s) |
| `benchmark_wikipedia` | Avalia: gera texto, mede qualidade e tempo |
| `train_iterative` | Treino iterativo de morfologia |
| `test_compiler` | Inspeção: decompõe uma frase em `Token7` (ID, morph hex, classe) |
| `ollama_ls` | Lista modelos `.gguf` do Ollama com tamanho e quantização |
| `agnes_train` | Pedagoga com LLM externo (requer API key) |

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release

cargo run --release --bin train_wikipedia
cargo run --release --bin benchmark_wikipedia
cargo run --release --bin train_iterative
cargo run --release --bin test_compiler
cargo run --release --bin ollama_ls
```

Não há modo de conversa. O REPL que existia foi removido porque treinava com
cada frase recebida antes de responder, e o corpus de raízes estava vazio:

```
─── Episódio 1 ───
  Você: /porque gato
  Amadeus: gato gato gato gato porque gato porque gato gato gato gato ...
```

### Testes

```bash
# 5 testes de caracterização do formato do artefato real
cargo test --test gguf_characterization --release

# 4 testes de round-trip do GGUF (save → load)
cargo test --test gguf_roundtrip --release

# 7 testes de paridade do shard
cargo test --test shard_parity --release

# 3 testes de paridade de geração RAM vs sharded
cargo test --test shard_generation --release

# carga do modelo real (lento, ~100s, precisa de ~4 GB)
cargo test --test grammar_load --release -- --test-threads=1
```

---

## 6. Lacunas conhecidas

| Lacuna | Impacto | Onde |
|--------|---------|------|
| Código de transformer órfão | 1.332 linhas sem binário que as exercite | `src/model/`, `src/layers/`, `src/sampler/`, `src/tokenizer/`, `src/quant/`, `src/main.rs` |
| `main.rs` não tokeniza prompt | Prompt virado byte cru | `src/main.rs:50` |
| `grammar7` é checkpoint de teste | 448 B, léxico de 17 entradas | `amadeus.grammar7` |
| Corpus Underworld vazio | Compilador sem raízes para `--train-morphology` | `training/empty_underworld/lingua/raizes/` |
| 9 docs de arquitetura, 2 sobrepostos | Risco de ler o errado | `ARCHITECTURE_HYBRID.md` (jul), `PHILOSOPHICAL_STACK.md` (mai) |

Vigente: `ARCHITECTURE.md` (v6.3, 16/set). `ARCHITECTURE_SNN.md` (14/set) e
`AMADEUS.md` (14/set) são complementares. `ARCHITECTURE_HYBRID.md` (jul) e
`PHILOSOPHICAL_STACK.md` (mai) estão superados — os dois já carregam aviso no
topo apontando para este documento.

### Binários removidos

Sete binários foram apagados em 2026-09-28. Cinco eram demonstrações
(`test_pipeline`, `test_generate`, `test_snn`, `test_hybrid`, `test_gguf`) e dois
eram o caminho transformer (`amadeus_m`, `amadeus_m_train`). O `test_gguf` foi
convertido em `tests/gguf_roundtrip.rs` antes de sair, porque validava o
round-trip do GGUF que nenhum outro teste cobria.

---

## 7. Cobertura de testes

Antes desta sessão: 7 `#[test]` em 63 arquivos, concentrados em `self_play`,
`morph_vocab` e `rhetoric`. Nenhum verificava geração.

Adicionados 19:

| Arquivo | Testes | Verifica |
|---------|--------|----------|
| `gguf_characterization.rs` | 5 | Formato, tensores e dimensões do artefato real |
| `gguf_roundtrip.rs` | 4 | Save → load campo a campo; GGUF e .bin com o mesmo conteúdo |
| `shard_parity.rs` | 7 | Conteúdo lido do shard == HashMap, cascade, corrompido |
| `shard_generation.rs` | 3 | Geração idêntica RAM vs sharded, auto-flush |

O `gguf_roundtrip` veio do binário `test_gguf`, que treinava com PCFG sintético,
salvava, recarregava e comparava 20 campos imprimindo um contador de erros. Como
teste, a mesma verificação quebra o build. Roda em 0,18s contra ~30s do binário.

`grammar_load.rs` foi reescrito: o teste anterior carregava dois modelos de
795 MB simultâneos e era morto por OOM. Agora carrega um e gera a partir dele.

---

## 8. Hipercubo vs transformer: medição (2026-09-28)

Baseline permanente em `src/bin/baseline_transformer.rs`, para que a comparação
seja repetível em vez de afirmada. `src/bin/mede_cubo.rs` faz o mesmo pelo
lado do hipercubo.

###throughput e memória

Mesmo hardware (4 cores, i5 540, 5,7 GB RAM), geração
autoregressiva de 500 tokens. Pesos aleatórios nos dois: mede custo de
inferência, não qualidade.

**Hipercubo** (modelo real, 4,64M contextos, chave `u64`):

| Configuração | RSS | tok/s | ms/token |
|---|---|---|---|
| com sharding | 298 MB | 44,3 | 22,6 |
| sem sharding | 561 MB | 40,9 | 24,4 |

**Transformer decoder** (llama-like, `baseline_transformer`):

| Vocabulário | Layers | Embd | RSS | tok/s | ms/token |
|---|---|---|---|---|---|
| 1.379 | 4 | 256 | **7 MB** | **129,3** | 7,7 |
| 75.741 | 4 | 256 | 152 MB | 31,7 | 31,5 |
| 75.741 | 8 | 512 | 307 MB | 12,1 | 82,3 |
| 75.741 | 12 | 768 | 464 MB | 5,2 | 190,7 |

### Uso de CPU: a primeira comparação era injusta

O transformer usa **rayon** nas operações de tensor (`src/tensor/ops.rs`), que
satura os 4 cores. O hipercubo é sequencial. Medir só tok/s favorece quem
paraleliza — e "CPU-only sem GPU" significa justamente poder usar todos os
núcleos, então o consumo importa tanto quanto a velocidade.

Ambos os binários agora reportam tempo de CPU de processo (user + system):

| | tok/s | CPU | uso | tok/s por core | RAM |
|---|---|---|---|---|---|
| Hipercubo | 65,2 | 7,66 s | **100%** | **16,3** | 298 MB |
| Transformer (4 cores) | 31,6 | 33,08 s | **209%** | 7,9 | 152 MB |
| Transformer (1 core) | 22,6 | 21,97 s | 99% | 5,7 | 152 MB |

Com um core, o transformer é 22,6 tok/s. Com quatro, 31,6. Ou seja, o rayon
compra **1,4x** de velocidade e cobra 2,09x de CPU.

Contra o hipercubo, no mesmo vocabulário: o hipercubo é **2x mais rápido por
core** (16,3 contra 7,9) e usa metade da CPU. O transformer ganha só em RAM,
e porque tem metade das camadas, não por eficiência.

O que a benchmark original omitiu, então: o hipercubo é sequencial por
construção, e isso é uma escolha de arquitetura com custo e benefício. O
custo é não usar os outros 3 cores. O benefício é metade do consumo para
tok/s por core maior.

### O que os números dizem

**No vocabulário pequeno (1.379), o transformer ganha de RAM:** 7 MB contra
298 MB, e 129 tok/s contra 65. O tokenizer de 1.379 palavras torna o
embedding trivial, e aí não há nada que iguale.

**No vocabulário do modelo real (75.741), a troca é direta:** o transformer
usa 152 MB contra 298 MB, mas o hipercubo é 2x mais rápido por core e usa
metade da CPU. Se o objetivo é não prender os 4 núcleos, o hipercubo ganha.
Se o objetivo é a menor footprint absoluto possível, o transformer ganha.

**O gargalo do transformer é o custo quadrático do embedding:** com
vocabulário grande, `tok_embeddings` e `output_weight` sozinhos são
`2 * vocab * embd` f32 — 147 MB só com 75.741 × 256. É por isso que
vocabulário pequeno é dobramente vantajoso para ele.

**O gargalo do hipercubo é a explosão combinatória:** 4,64 milhões de
contextos para 75.741 palavras. É superlinear no corpus, e é a razão de o
modelo completo original ter 7,67 milhões de contextos e 2 GB.

### A resposta honesta

A theory original — "hipercubo é mais vantajoso para CPU sem GPU" — **se
sustenta em parte, e é mais estreita do que se supunha.** A leitura correta
dos números:

- **RAM:** o transformer vence (152 MB contra 298 MB). A theory previa o
  contrário.
- **Velocidade por core:** o hipercubo vence 2x (16,3 contra 7,9 tok/s/core).
- **Consumo total:** o hipercubo usa metade da CPU para ser 2x mais rápido
  por core.
- **Paralelismo:** o hipercubo não usa. Essa é uma escolha, com custo.

O que o hipercubo tem que o transformer não tem, e que este baseline **não
mede** porque usa pesos aleatórios: nada. A qualidade — gramática, semântica,
coerência de texto longo — depende de treino, e o transformer não foi treinado
neste repositório.

A comparação honesta de qualidade exigiria treinar os dois no mesmo corpus, o
que não foi feito. O que se pode afirmar hoje é sobre **custo**, e o custo é
dividido: o transformer ganha em RAM, o hipercubo ganha em velocidade por core
e em consumo de CPU.

### Quando o hipercubo ainda faz sentido

- Corpus muito pequeno (centenas de milhares de tokens), onde a explosão
  combinatória ainda é pequena e o `train` é de segundos.
- Quando o número de tokens distintos é pequeno e a repetição alta: a tabela
  de frequência é uma Representação muito eficiente nesse regime, e o
  transformer desperdiça capacidade em parâmetros que nunca ativam.

Isso é hipótese, não medição. Para decidir de verdade: treinar os dois no
mesmo corpus pequeno e comparar a qualidade do texto gerado.

### Colisão da chave canônica: medida, não estimada

`pack8d` usa 50 bits por token, então ordem 3 são 150 bits e não cabem num
`u64`. A chave canônica colide de propósito com FxHash.

Medido no corpus completo (2.067.373 contextos **distintos**):

```
chaves canônicas distintas:      2.067.373
chaves com >1 contexto distinto: 0
CONTEXTOS PERDIDOS:              0
```

**Zero colisões.** O que aparece como "39% de colisão" em medição ingênua é
duplicata no dado: o compilador dá id de lexema por posição na frase, então
"o gato" nas posições 0-1 de frases diferentes produz o mesmo contexto, e
48,3% dos contextos gerados são idênticos byte a byte. Somar as contagens é o
comportamento correto de uma tabela de frequência. O teste
`tests/chave_canonica.rs` fixa essa verificação.
