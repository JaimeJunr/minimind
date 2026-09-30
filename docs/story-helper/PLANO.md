# Plano: MiniMind como copiloto de contexto para LLMs em ficção (PT + EN)

> Status: **aprovado, ainda não iniciado.** A fundamentação está em [PESQUISA.md](./PESQUISA.md).

## Contexto

O objetivo é treinar um modelo pequeno da família MiniMind (~70–125M) que **não escreve a história**: quem escreve é um LLM de fronteira, ainda não definido. O modelo pequeno trabalha antes de cada chamada ao LLM grande para:

1. **economizar tokens**, montando um contexto enxuto em vez de mandar o histórico inteiro;
2. **melhorar o resultado**, entregando ao LLM grande o estado da história, os fatos antigos relevantes e menos ruído.

Restrições:
- usuários escrevem em português, e o produto quer o público internacional → **modelo bilíngue PT + EN**;
- LLM alvo indefinido → interface **só por texto**;
- desenvolvimento num notebook **só com CPU** → treino em **GPU alugada por hora**, e todo o resto local;
- produto próprio → o modelo roda como **microserviço** no backend.

## 1. Tarefas do modelo ("StoryHelper")

Um único modelo multitarefa. A tarefa vai na mensagem *system* do formato de chat já usado por `SFTDataset` ([`dataset/lm_dataset.py:61`](../../dataset/lm_dataset.py#L61)).

1. **`atualizar_estado`**: recebe o estado anterior e os turnos novos, e devolve um **patch** JSON curto (add/update/remove). O patch é mais curto que o estado inteiro, mais rápido de gerar em CPU e erra menos.
2. **`resumir_capitulo`**: quando uma cena ou capítulo fecha, gera notas factuais compactas. Essas notas são **anexadas e nunca reescritas**.
3. **`relevancia`**: recebe o pedido e um trecho antigo, e devolve "sim"/"não". A probabilidade do "sim" vira o score para escolher os top-k trechos originais.
4. **`intencao`**: classifica o pedido (continuar cena, reescrever, perguntar sobre a história, brainstorm) e decide quais blocos de contexto vão na chamada.

## 2. Desenho

### 2.1 Modelo
- Base: `MiniMindForCausalLM` ([`model/model_minimind.py:239`](../../model/model_minimind.py#L239)), densa, `hidden_size=768`, `num_hidden_layers=8`, vocabulário de 16k: **~71M parâmetros**.
- Ablação com 16 camadas (~125M). O MobileLLM indica que, nessa escala, profundidade rende mais que largura.
- Pré-treino com `max_seq_len=1024` e SFT das tarefas com 2048. YaRN já está implementado ([`model/model_minimind.py:62`](../../model/model_minimind.py#L62)).
- Inferência em CPU no serviço.

### 2.2 Layout do prompt enviado ao LLM grande (preserva o cache)

```
[1] sistema + guia de estilo + lore fixa            ← cache (muda raramente)
[2] notas de capítulos fechados (só anexadas)       ← cache incremental
[3] estado atual da história (curto)                ← muda por turno
[4] trechos antigos relevantes, verbatim (top-k)    ← muda por turno
[5] últimos N turnos verbatim (preservam o estilo)  ← janela deslizante
[6] pedido do usuário
```

O helper só entra em ação quando o histórico passa de um limiar (por exemplo, 16k tokens). Abaixo disso, manda-se o histórico inteiro com cache.

## 3. Fases

> **Local** = notebook com CPU. **Nuvem** = GPU alugada por hora.

### Fase 0 — Harness de avaliação e baselines (Local; ~1–2 semanas)
Responde, antes de treinar qualquer coisa, se existe ganho a buscar.
- Criar `story_helper/eval/`: gerador de sessões de teste, gerador de perguntas de continuidade, montador de contexto para cada baseline (B0–B4, ver [PESQUISA.md §7](./PESQUISA.md#7-como-avaliar)), cliente de LLM agnóstico (compatível com OpenAI e Anthropic), simulador de custo com cache e juiz pareado.
- Rodar primeiro em ~20 sessões com um LLM barato, depois no conjunto completo.
- **Portão A**: se B1/B4 já chegarem perto do B0 em continuidade, o escopo se estreita para o estado estruturado.

### Fase 1 — Dados e tokenizer (Local; ~1–2 semanas)
- `story_helper/prep/`: download, limpeza de OCR, identificação de idioma, deduplicação (MinHash), filtros de licença e de NSFW, e fatiamento de livros em janelas de ~1024 tokens. O fatiamento é necessário porque `PretrainDataset` ([`dataset/lm_dataset.py:40`](../../dataset/lm_dataset.py#L40)) trunca cada amostra em `max_length`. Saída no formato `{"text": ...}`.
- [`trainer/train_tokenizer.py`](../../trainer/train_tokenizer.py): trocar as constantes fixas (linhas 7–11) por argparse (vários arquivos, `--vocab_size`, `--out_dir`), manter os tokens especiais e o chat template, e trocar as amostras do `eval_tokenizer` por ficção em PT e EN.
- Deixar tokenizer e vocabulário configuráveis no treino. `init_model` ([`trainer/trainer_utils.py:121`](../../trainer/trainer_utils.py#L121)) já recebe `tokenizer_path`, e `MiniMindConfig` já lê `vocab_size` de kwargs ([`model/model_minimind.py:18`](../../model/model_minimind.py#L18)). Falta adicionar `--tokenizer_path` e `--vocab_size` em `train_pretrain.py`, `train_full_sft.py`, `train_lora.py` e `eval_llm.py`.
- **Portão B**: tokenizer com pelo menos ~3,5 caracteres/token em ficção PT e EN, medido com [`medir_tokenizer.py`](./medir_tokenizer.py) ou com o `eval_tokenizer`.

### Fase 2 — Pré-treino bilíngue (Nuvem; ~1 dia de relógio)
- `trainer/train_pretrain.py --from_weight none --tokenizer_path ... --vocab_size 16384 --max_seq_len 1024` sobre ~2–3B tokens. DDP, retomada por checkpoint e wandb/swanlab já existem.
- Antes de alugar a GPU, fazer um **smoke test em CPU** com modelo minúsculo (`--hidden_size 256 --num_hidden_layers 2`, algumas centenas de steps).
- Acompanhar a loss e o **BPB por idioma**.

### Fase 3 — Dados do professor (Nuvem; em paralelo à Fase 2; ~1 semana)
- `story_helper/teacher/`: LLM com pesos abertos (ex.: Qwen3, Apache 2.0) servido com vLLM na GPU alugada. Ele gera (a) sessões sintéticas de coautoria e roleplay em PT e EN e (b) rótulos das 4 tarefas sobre essas sessões e sobre trechos de obras de domínio público.
- Volume: ~20–50k exemplos por tarefa, no formato `{"conversations": [...]}`. Como todo exemplo traz o próprio *system*, o `pre_processing_chat` ([`dataset/lm_dataset.py:9`](../../dataset/lm_dataset.py#L9)) não injeta os prompts de sistema em chinês e não precisa mudar.
- Auditar manualmente ~200 rótulos por tarefa antes de treinar.

### Fase 4 — SFT multitarefa (Nuvem; horas)
- `trainer/train_full_sft.py --from_weight pretrain --max_seq_len 2048 --data_path story_tasks.jsonl`, que reusa `SFTDataset` e a máscara de loss só nas respostas do assistente.
- Variante: um adaptador LoRA por tarefa (`trainer/train_lora.py`, `model/model_lora.py`).
- **Portão C**: F1 do estado ≥ 0,8 contra o professor e recall@5 do seletor ≥ B4. Se não passar: testar 16 camadas ou MoE 198M (`--use_moe 1`), ou adotar o B5 naquela tarefa.

### Fase 5 — Serviço (Local; ~1 semana)
- `scripts/serve_story_helper.py` (FastAPI, no padrão de [`scripts/serve_openai_api.py`](../../scripts/serve_openai_api.py)), com endpoints `/state/update`, `/chapter/close`, `/context/build` (devolve os blocos 1–6 já ordenados) e `/rank`. Roda em CPU.
- `story_helper/context_builder.py`: implementa o layout da seção 2.2 e o limiar de ativação.

### Fase 6 — Avaliação final e relatório (Local + API; ~1–2 semanas)
- Rodar StoryCtx-PTEN com o helper contra B0–B5 e as ablações (70M vs 125M; vocabulário; com e sem estado; extrativo vs poda de tokens).
- Fazer a avaliação humana pequena para calibrar o juiz.
- Atualizar [PESQUISA.md](./PESQUISA.md) com os resultados.

### Futuro (fora do escopo inicial)
- GRPO (`trainer/train_grpo.py`) com recompensa de continuidade medida por um LLM proxy, penalizando o tamanho do contexto.
- Poda de tokens com variante bidirecional (LLM2Vec).
- Retreinar com logs reais, com consentimento.

**Duração estimada:** ~6–8 semanas de trabalho, com GPU só em blocos de horas.

## 4. Custo de nuvem

Preços de mercado em 2026 (RunPod e Vast.ai): RTX 4090 a **US$0,34–0,69/h**, A100 80GB a **US$0,50–1,60/h**, H100 a **US$0,90–3,50/h**. As ofertas "não verificadas" são as mais baratas e as menos confiáveis.

Referência de vazão: o README registra ~1,2 h numa 3090 para o `pretrain_t2t_mini`, o que dá algo em torno de 50k tokens/s para 64M parâmetros (estimativa).

| Etapa | Onde | Tempo estimado | Custo estimado |
|---|---|---|---|
| Limpeza de dados, tokenizer, harness | Notebook (CPU) | dias | US$0 |
| Pré-treino, ~2,5B tokens | 1× RTX 4090 | ~8–10 h | US$5–10 |
| Ablações e reinícios do pré-treino | 4090 | ~10–20 h | US$5–15 |
| Geração pelo professor (Qwen3 + vLLM) | 1× A100/H100 | ~10–20 h | US$20–60 |
| SFT multitarefa + ablações | 4090 | ~5–10 h | US$3–7 |
| Avaliação com LLM de fronteira (API) | API | — | US$30–150, dependendo do modelo |
| **Primeiro ciclo completo** | | | **~US$70–250** |

A 4090 tem o melhor custo-benefício para esse tamanho, porque modelos pequenos aproveitam mal a H100. Salve os checkpoints fora do pod, já que instâncias baratas podem ser interrompidas; a retomada via `--from_resume 1` já existe.

## 5. Riscos

- **Capacidade**: 70M pode ser pouco para extração bilíngue. Mitigação: patches curtos, ablação com 125M e o B5 como alternativa.
- **Cache**: a economia pode sumir em sessões curtas. Mitigação: limiar de ativação e medição dos tempos reais entre turnos.
- **Licenças**: filtrar dados NC e "só para pesquisa"; usar professor com pesos abertos.
- **Conteúdo**: definir a política de NSFW para roleplay antes de montar os dados.
- **Privacidade**: histórias dos usuários só entram no treino com consentimento (LGPD).

## 6. Verificação

1. Tokenizer: `python trainer/train_tokenizer.py --data ... --vocab_size 16384`, conferindo caracteres/token em PT e EN.
2. Pipeline: smoke test em CPU (`train_pretrain.py --device cpu --hidden_size 256 --num_hidden_layers 2 --max_seq_len 256`); a loss precisa cair.
3. Modelo base: `eval_llm.py --weight pretrain` gerando texto plausível em PT e EN, com BPB de validação por idioma.
4. Tarefas: script de avaliação com F1 do estado, recall@k e acurácia de intenção no conjunto separado.
5. Serviço: `curl` em `/context/build` com uma sessão longa, conferindo a ordem dos blocos e a contagem de tokens.
6. Fim a fim: harness da Fase 0 comparando o helper com B0–B5, com custo efetivo simulado.
