# Pesquisa: um modelo pequeno como copiloto de contexto para LLMs em ficção

> Status: **estudo concluído, implementação não iniciada.** O plano de execução está em [PLANO.md](./PLANO.md).
> Data do levantamento: setembro de 2026.

## Resumo

A proposta é treinar um modelo da família MiniMind (~70–125M de parâmetros) que **não escreve a história**. Quem escreve é um LLM de fronteira (Claude, GPT, Gemini etc.), escolhido depois. O modelo pequeno atua **antes** de cada chamada ao LLM grande, mantendo a memória da história e montando um contexto enxuto. Com isso ele **reduz tokens** e **melhora a consistência** da narrativa.

A literatura sustenta a ideia, desde que três escolhas sejam respeitadas:

1. **Selecionar trechos originais** (extração) em vez de podar palavras ou reescrever resumos.
2. **Manter um estado estruturado** da história (personagens, locais, objetos, fios de enredo).
3. **Preservar o prompt caching** do provedor. Sem isso, a "economia" pode sair mais cara que mandar o histórico inteiro.

O modelo precisa ser **bilíngue (PT + EN)** e, para isso, exige **novo tokenizer e novo pré-treino**: o MiniMind atual foi treinado quase só em chinês.

## 1. Formulação do problema

Numa sessão de escrita, o histórico `H` cresce a cada turno. Roleplay e coautoria chegam fácil a 50k–200k tokens. A cada turno o LLM grande precisa de um contexto `C = f(H, pedido)`. Queremos um `f` barato, executado pelo modelo pequeno, que minimize `custo(C)` sem perder (e idealmente ganhando) **consistência narrativa**: nomes, fatos, relações, objetos, tom e fios em aberto.

Premissas do cenário:
- os usuários escrevem em português, e o produto quer alcançar também o público internacional em inglês;
- o LLM grande ainda não está definido, então a interface precisa ser **só texto**, sem acesso a logits ou pesos;
- o modelo roda como microserviço num app próprio;
- a prosa e o vocabulário ficam com o LLM grande, e o modelo pequeno só precisa **entender e extrair**.

## 2. Papéis possíveis para um modelo pequeno

| Papel | Evidência | Funciona com API fechada? | Veredito |
|---|---|---|---|
| **Memória estruturada / estado da história** | Re3 e DOC mostram que um plano/estado explícito melhora muito a coerência de histórias longas (DOC: +22,5 pontos em coerência de enredo sobre Re3). MemGPT e sumarização recursiva sustentam diálogos longos com memória externa. | Sim | **Núcleo** |
| **Seleção extrativa de trechos antigos** (reranker) | Jha et al. 2024: compressão extrativa com reranker chega a **até 10x** com perda mínima, às vezes com *ganho* de acurácia, e supera poda de tokens e resumos abstrativos. BOOKMARKS (2026): resumir de forma incremental **descarta detalhes**, e buscar sob demanda no histórico completo sobe o raciocínio de 39,4% para 45,4%. | Sim | **Núcleo** |
| **Poda de tokens** (LLMLingua, LLMLingua-2, Selective Context) | LLMLingua-2 atinge 2–5x de compressão com encoders de 110–355M. LongLLMLingua dá +21,4% com ~4x menos tokens em QA. Em compensação, a poda quebra a gramática (Jha et al.), e em ficção o LLM grande tende a imitar o texto telegráfico. | Sim | Opcional, só na zona de notas |
| **Roteamento / intenção do pedido** (FrugalGPT, RouteLLM) | Classificação é fácil para modelos pequenos, e um pedido como "reescreva este parágrafo" quase não precisa de histórico. | Sim | Barato e com alto retorno |
| Decodificação especulativa | Exige logits do modelo alvo e o mesmo vocabulário | **Não** | Descartado |
| Gist tokens / ICAE / soft prompts | Exigem acesso aos embeddings internos do alvo | **Não** | Descartado |

Visão geral da colaboração entre modelos pequenos e LLMs: Chen & Varoquaux (2024).

## 3. Achados que moldam o desenho

1. **Modelos minúsculos dão conta de narrativa quando o domínio é restrito.** O TinyStories mostrou modelos com menos de 10M parâmetros gerando histórias coerentes. O TF3-RO-50M (2026) repetiu o resultado em romeno, com ~50M. Aqui a exigência é ainda menor, já que o modelo só precisa entender e extrair.
2. **Destilação específica de tarefa funciona.** Em Distilling Step-by-Step, um T5 de 770M supera o PaLM de 540B na tarefa destilada. Os rótulos das nossas tarefas virão de um LLM professor.
3. **Extrair é melhor que abstrair e melhor que podar tokens** (Jha et al. 2024).
4. **Contexto longo atrapalha o próprio LLM grande.** O efeito *Lost in the Middle* mostra que um contexto enxuto e bem ordenado pode render resultado **melhor** que o histórico inteiro, e não só mais barato.
5. **Um decoder pode virar encoder.** O LLM2Vec mostra que basta ligar a atenção bidirecional e fazer um treino curto. No MiniMind isso é uma flag: `self.is_causal = True` em [`model/model_minimind.py:99`](../../model/model_minimind.py#L99).
6. **Reranker gerativo.** Pontuar relevância pela probabilidade do token "sim" (estilo monoT5) dispensa mudanças de arquitetura.

## 4. A economia real: prompt caching muda a conta

Os provedores cobram muito menos por prefixos repetidos. Na Anthropic, a leitura em cache custa **0,1x** o preço normal, a escrita custa 1,25x (TTL de 5 min) ou 2x (TTL de 1 h), e o cache só vale para **prefixo exato**. Consequências:

- Como o histórico completo só cresce no final, ele já é "cacheável". **O baseline honesto é "histórico completo + cache"**.
- Um compressor que **reescreve o contexto a cada turno destrói o cache** e pode sair mais caro.
- Escritores fazem pausas longas com frequência. Pausas de mais de 5 min expiram o cache, e o histórico inteiro volta a ser cobrado (a 1,25x). **O tempo real entre turnos dos usuários decide a economia**, então precisa ser medido nos logs do app.

Por isso o contexto montado pelo modelo pequeno segue um layout que preserva o cache (ver [PLANO.md](./PLANO.md#22-layout-do-prompt-enviado-ao-llm-grande-preserva-o-cache)).

## 5. Idioma e tokenizer

Medição feita com [`medir_tokenizer.py`](./medir_tokenizer.py), sobre o mesmo parágrafo de ficção nos dois idiomas:

| Idioma | Caracteres/token (tokenizer atual, 6.400 tokens) |
|---|---|
| Português | **~1,98** |
| Inglês | ~2,85 |

O script reimplementa o BPE a partir de `model/tokenizer.json` com uma regex aproximada de pré-tokenização. Os valores absolutos são estimativas; a diferença entre os idiomas é o que conta.

Conclusões:
- em português, o tokenizer atual desperdiça cerca de metade da janela de contexto;
- os pesos atuais foram pré-treinados quase só em chinês e não servem de ponto de partida, então é preciso **tokenizer novo e pré-treino do zero**;
- há precedentes em português: TeenyTinyLlama (160M/460M, 6,2B tokens em PT, Apache 2.0) e Tucano (até 2,4B, sobre o GigaVerbo de 200B tokens);
- proposta: BPE byte-level **bilíngue com 16.384 tokens**, com ablação contra 8k e 32k.

## 6. Dados (atenção a licenças, já que o produto é comercial)

| Fonte | Idioma | Tamanho | Licença / observação |
|---|---|---|---|
| Portuguese-PD (PleIAs) | PT | 672M palavras, anterior a 1884 | Domínio público; OCR sujo, precisa de limpeza |
| PPORTAL / Domínio Público / Machado de Assis (Gutenberg) | PT | milhares de obras | Domínio público |
| Corpus Carolina | PT-BR | 823M tokens | **Licenças mistas, incluindo NC**: filtrar por licença |
| BrWaC | PT-BR | 2,68B tokens | Liberado "para pesquisa": **não usar em produção** sem checar |
| GigaVerbo | PT | 200B tokens | Checar a licença de cada subconjunto |
| PG-19 / Project Gutenberg | EN | ~2B palavras | Domínio público |
| WritingPrompts | EN | 303k histórias do Reddit | Uso comercial é zona cinzenta |
| TinyStories | EN | sintético | Útil para narrativa simples |
| PIPPA | EN | 26k sessões de roleplay | Apache 2.0, **com muito NSFW**: definir política |
| Sessões sintéticas de um professor com pesos abertos (ex.: Qwen3, Apache 2.0) | PT + EN | quanto quisermos | Evita restrições de ToS de APIs fechadas |
| Logs do próprio app (fase posterior) | PT + EN | — | Só com consentimento (**LGPD**) |

Meta de pré-treino: **~2–3B tokens**, com ~45% PT, ~45% EN e ~10% diálogo/roleplay, sendo pelo menos metade ficção. Isso passa bem do ótimo de Chinchilla (~20 tokens/parâmetro) para ~70M parâmetros.

## 7. Como avaliar

Proposta de benchmark próprio, **StoryCtx-PTEN**: 100–200 sessões longas (20k–150k tokens), metade em PT e metade em EN, sem sobreposição com o treino.

**Métricas**
1. **Custo efetivo por turno em dólares**, simulando o cache com os tempos reais entre turnos.
2. **QA de continuidade**: perguntas automáticas sobre fatos estabelecidos cedo ("onde está a espada?"), com resposta verificada.
3. **Qualidade da continuação**: juiz LLM pareado (com troca de posição) e taxa de contradições. O LitBench mostra que o melhor juiz pronto concorda só **73%** com humanos em escrita criativa, então é preciso calibrar com ~50 pares avaliados por pessoas.
4. **Métricas do helper**: F1 por campo do estado contra o professor, recall@k do seletor, latência em CPU.

**Baselines obrigatórios**

| Código | Condição |
|---|---|
| B0 | Histórico completo + cache |
| B1 | Só os últimos N turnos |
| B2 | Resumo contínuo feito pelo próprio LLM grande |
| B3 | LLMLingua-2 multilíngue pronto |
| B4 | Busca com embeddings prontos (multilingual-e5-small) + recência |
| B5 | Qwen3-0.6B afinado com os mesmos dados (verifica se vale treinar do zero) |

**Hipóteses**
- **H1**: estado + trechos selecionados + últimos turnos atingem pelo menos 90% da acurácia de continuidade do B0 usando no máximo 30% dos tokens.
- **H2**: em ficção, seleção extrativa supera poda de tokens.
- **H3**: com layout que preserva cache, o custo efetivo fica abaixo do B0 em sessões com mais de ~30k tokens.
- **H4**: em sessões muito longas (mais de 80k tokens), o helper **supera** o B0 em continuidade.

## Referências

- Pan et al. *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression.* Findings ACL 2024. https://arxiv.org/abs/2403.12968
- Jiang et al. *LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression.* ACL 2024. https://arxiv.org/abs/2310.06839
- Jiang et al. *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models.* EMNLP 2023.
- Li et al. *Compressing Context to Enhance Inference Efficiency of Large Language Models* (Selective Context). EMNLP 2023.
- Jha, Erdogan, Kim, Keutzer, Gholami. *Characterizing Prompt Compression Methods for Long Context Inference.* ES-FoMo II @ ICML 2024. https://arxiv.org/abs/2407.08892
- Peng et al. *BOOKMARKS: Efficient Active Storyline Memory for Role-playing.* 2026. https://arxiv.org/abs/2605.14169
- Yang et al. *Re3: Generating Longer Stories With Recursive Reprompting and Revision.* EMNLP 2022. https://aclanthology.org/2022.emnlp-main.296/
- Yang et al. *DOC: Improving Long Story Coherence With Detailed Outline Control.* ACL 2023. https://aclanthology.org/2023.acl-long.190/
- Packer et al. *MemGPT: Towards LLMs as Operating Systems.* 2023. https://arxiv.org/abs/2310.08560
- Wang et al. *Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models.* 2023. https://arxiv.org/abs/2308.15022
- Liu et al. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024.
- Eldan & Li. *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* 2023. https://arxiv.org/abs/2305.07759
- *TF3-RO-50M: Training Compact Romanian Language Models from Scratch on Synthetic Moral Microfiction.* 2026. https://arxiv.org/abs/2601.10410
- Hsieh et al. *Distilling Step-by-Step!* Findings ACL 2023. https://arxiv.org/abs/2305.02301
- BehnamGhader et al. *LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders.* 2024. https://arxiv.org/abs/2404.05961
- Nogueira et al. *Document Ranking with a Pretrained Sequence-to-Sequence Model* (monoT5). Findings EMNLP 2020.
- Chen & Varoquaux. *What is the Role of Small Models in the LLM Era: A Survey.* 2024. https://arxiv.org/abs/2409.06857
- Liu et al. *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases.* ICML 2024. https://arxiv.org/abs/2402.14905
- Leviathan et al. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.
- Mu et al. *Learning to Compress Prompts with Gist Tokens.* NeurIPS 2023. · Ge et al. *In-context Autoencoder for Context Compression.* ICLR 2024.
- Chen et al. *FrugalGPT.* 2023. · Ong et al. *RouteLLM.* 2024.
- Corrêa et al. *TeenyTinyLlama: open-source tiny language models trained in Brazilian Portuguese.* 2024. https://arxiv.org/abs/2401.16640
- Corrêa et al. *Tucano: Advancing Neural Text Generation for Portuguese.* https://nkluge-correa.github.io/Tucano/
- *LitBench: A Benchmark and Dataset for Reliable Evaluation of Creative Writing.* 2025. https://arxiv.org/abs/2507.00769
- Gosling et al. *PIPPA: A Partially Synthetic Conversational Dataset.* 2023. https://huggingface.co/datasets/PygmalionAI/PIPPA
- Fan et al. *Hierarchical Neural Story Generation* (WritingPrompts). ACL 2018. https://aclanthology.org/P18-1082/
- Rae et al. *Compressive Transformers for Long-Range Sequence Modelling* (PG-19). ICLR 2020.
- PleIAs. *Portuguese-PD.* https://huggingface.co/datasets/PleIAs/Portuguese-PD
- Crespo et al. *Carolina: a General Corpus of Contemporary Brazilian Portuguese.* 2023. https://sites.usp.br/corpuscarolina/
- Wagner Filho et al. *The brWaC Corpus: A New Open Resource for Brazilian Portuguese.* LREC 2018. https://aclanthology.org/L18-1686/
- Anthropic. *Prompt caching.* https://platform.claude.com/docs/en/build-with-claude/prompt-caching
