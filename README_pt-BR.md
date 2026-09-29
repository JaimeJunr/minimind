<div align="center">

![logo](./images/logo.png)

</div>

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=jingyaogong/minimind)
[![GitHub Repo stars](https://img.shields.io/github/stars/jingyaogong/minimind?style=social)](https://github.com/jingyaogong/minimind/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/jingyaogong/minimind)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/jingyaogong/minimind)](https://github.com/jingyaogong/minimind/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/jingyaogong/minimind/pulls)
[![Collection](https://img.shields.io/badge/🤗-MiniMind%20%20Collection-blue)](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)

</div>

<div align="center">

![GitHub Trend](https://trendshift.io/api/badge/repositories/12586)

</div>

<div align="center">
  <h3>"O Grande Caminho é Simples"</h3>
</div>

<div align="center">

[中文](./README.md) | [English](./README_en.md) | Português

</div>

* Este projeto de código aberto tem como objetivo treinar o MiniMind, um modelo de linguagem ultrapequeno com cerca de 64M de parâmetros, totalmente do zero, com um custo de apenas cerca de 3 RMB e 2 horas de treinamento.
* A série MiniMind é intencionalmente leve. O menor modelo da branch principal tem cerca de $\frac{1}{2700}$ do tamanho do GPT-3, tornando o treinamento completo e a reprodução viáveis até mesmo em GPUs pessoais comuns.
* O projeto oferece uma arquitetura de modelo minimalista e um pipeline de treinamento de LLM de ponta a ponta, abrangendo MoE, limpeza de dados, pré-treinamento (Pretrain), ajuste fino supervisionado (SFT), LoRA, RLHF (DPO), RLAIF (PPO / GRPO / CISPO), Tool Use, Agentic RL, Pensamento Adaptativo e Destilação de Modelos.
* O MiniMind também foi estendido para um modelo de visão, [MiniMind-V](https://github.com/jingyaogong/minimind-v), um modelo multimodal Omni, [MiniMind-O](https://github.com/jingyaogong/minimind-o), um modelo de linguagem de difusão (MiniMind-dLM) e um modelo de atenção linear (MiniMind-Linear). Veja as [Discussions](https://github.com/jingyaogong/minimind/discussions) para mais detalhes.
* Todos os algoritmos centrais são implementados diretamente em PyTorch nativo, sem depender de abstrações de alto nível de bibliotecas de terceiros.
* O MiniMind é ao mesmo tempo uma reprodução de código aberto de ponta a ponta do pipeline de treinamento de LLMs e um tutorial prático para aprender como os LLMs são construídos.
* Esperamos que este projeto ofereça um ponto de partida reproduzível, compreensível e extensível para mais pessoas, compartilhe a alegria de criar e ajude a impulsionar a comunidade de IA como um todo.

> Observação: Este projeto é distribuído sob a licença Apache 2.0 e é totalmente gratuito. "2 horas" refere-se ao tempo medido para executar `1 epoch` da etapa de SFT em uma única NVIDIA 3090, enquanto "3 RMB" refere-se ao custo correspondente de aluguel da GPU.

> Esta é uma tradução para o português do [README em chinês](./README.md). Em caso de divergência, a versão em chinês prevalece.

---

<div align="center">

![minimind-3](./images/minimind-3.gif)

[🔗 Demonstração Online](https://www.modelscope.cn/studios/gongjy/MiniMind) | [🔗 Vídeo de Apresentação](https://www.bilibili.com/video/BV12dHPeqE72)


<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://huggingface.co/collections/jingyaogong/minimind" style="text-decoration: none;">
          <img src="./images/with_huggingface.png" alt="Hugging Face Logo" style="vertical-align: middle; width: auto; max-width: 100%;" />
        </a>
      </td>
      <td align="center">
        <a href="https://www.modelscope.cn/profile/gongjy" style="text-decoration: none;">
          <img src="./images/with_modelscope.png" alt="ModelScope Logo" style="vertical-align: middle; width: auto; max-width: 100%;" />
        </a>
      </td>
    </tr>
  </table>
</div>


</div>

---

# 📌 Introdução ao Projeto

O surgimento dos Grandes Modelos de Linguagem (LLMs) atraiu uma atenção global sem precedentes para a IA. ChatGPT, DeepSeek, Qwen e muitos outros modelos impressionaram as pessoas com seu desempenho notável, tornando o impacto dessa onda tecnológica algo muito concreto. No entanto, modelos com dezenas ou centenas de bilhões de parâmetros não são apenas difíceis de treinar em dispositivos pessoais, como muitas vezes estão fora de alcance até mesmo para implantação. Abrir a "caixa-preta" dos grandes modelos e entender de verdade como eles funcionam internamente deveria ser algo empolgante. Infelizmente, a maioria das explorações acaba parando na aplicação de técnicas como LoRA para ajustar modelos grandes já existentes a algumas novas instruções ou tarefas específicas. Isso é mais como ensinar Newton a usar um smartphone do século XXI — interessante, mas não exatamente o objetivo original de entender a essência da física.

Ao mesmo tempo, frameworks e toolkits de LLM de terceiros, como `transformers` / `trl` / `peft`, geralmente expõem apenas interfaces altamente abstratas. Com apenas uma dúzia de linhas de código, é possível concluir todo o pipeline de "carregar modelo + carregar dataset + inferência + treinamento por aprendizado por reforço". Esse tipo de encapsulamento eficiente é conveniente, mas também afasta, em certa medida, os desenvolvedores da implementação subjacente, reduzindo a oportunidade de entender a fundo o código central dos LLMs. Acredito que "construir você mesmo um avião com peças de Lego é muito mais empolgante do que voar na primeira classe". Um problema mais prático é que a internet também está repleta de cursos pagos e conteúdo de marketing, em que os chamados tutoriais de IA vêm embalados em explicações falhas e mal compreendidas. Por isso, a intenção original deste projeto é reduzir ao máximo a barreira de aprendizado de LLMs, para que qualquer pessoa possa começar entendendo cada linha de código e treinar à mão, do zero, um modelo de linguagem minúsculo. Sim, **treinar do zero**, e não apenas ficar no nível da **inferência**. Com um custo de servidor inferior a 3 RMB, você pode vivenciar pessoalmente todo o processo de construção de um modelo de linguagem, do 0 ao 1.

😊 Vamos compartilhar juntos a alegria de criar!

---

#### 🎉 O Que Este Projeto Inclui

- Implementação completa da arquitetura MiniMind-LLM (Dense + MoE), alinhada ao ecossistema `Qwen3 / Qwen3-MoE`.
- O tokenizer e o código de treinamento do tokenizer, com suporte a tokens de template como `<tool_call>`, `<tool_response>`, `<think>` etc.
- Pipelines de treinamento de ponta a ponta, incluindo pré-treinamento, SFT, LoRA, RLHF-DPO, RLAIF (PPO / GRPO / CISPO), Tool Use, Agentic RL, Pensamento Adaptativo e Destilação de Modelos.
- Dados abertos para todas as etapas, com datasets de alta qualidade coletados, destilados, limpos e deduplicados.
- Os principais algoritmos de treinamento e módulos centrais são todos implementados do zero, sem depender de wrappers de frameworks de terceiros.
- Compatível com frameworks populares como `transformers`, `trl` e `peft`, com engines de inferência amplamente usadas como `llama.cpp`, `vllm` e `ollama`, e com frameworks de treinamento como `Llama-Factory`.
- Suporte a treinamento em uma máquina com uma GPU e em uma máquina com várias GPUs (DDP, DeepSpeed), visualização com wandb / swanlab e pausa/retomada dinâmica do treinamento.
- Suporte à avaliação em benchmarks de terceiros como C-Eval, C-MMLU, OpenBookQA etc., e extrapolação de contexto longo do RoPE via YaRN.
- Um servidor de API leve compatível com OpenAI para integração com Chat UIs de terceiros como FastGPT e Open-WebUI, com suporte a `reasoning_content`, `tool_calls` e `open_thinking`.
- Uma WebUI de chat minimalista baseada em Streamlit, com exibição do raciocínio, seleção de ferramentas e Tool Call em múltiplos turnos.
- Extensões experimentais: modelo de linguagem de difusão ([dLM](https://github.com/jingyaogong/minimind/discussions/618)) e modelo de atenção linear ([Linear Attention](https://github.com/jingyaogong/minimind/discussions/704)), ambos podendo ser treinados a partir do modelo autorregressivo principal.

#### 🎉 Lista de Modelos Lançados

| Modelo | Parâmetros | Lançamento |
|------|--------|---------|
| minimind-3 | 64M | 2026.04.01 |
| minimind-3-moe | 198M-A64M | 2026.04.01 |
| minimind2-small | 26M | 2025.04.26 |
| minimind2-moe | 145M | 2025.04.26 |
| minimind2 | 104M | 2025.04.26 |
| minimind-v1-small | 26M | 2024.08.28 |
| minimind-v1-moe | 4×26M | 2024.09.17 |
| minimind-v1 | 108M | 2024.09.01 |

---

#### 📝 Registro de Alterações

<details> 
<summary> <b>🔥 2026-04-01</b> </summary>

- Lançamento de `minimind-3` / `minimind-3-moe`: atualizações abrangentes na estrutura, no Tokenizer, no pipeline de treinamento, na interface de inferência e na configuração padrão
- Estrutura da branch principal alinhada ao ecossistema `Qwen3 / Qwen3-MoE`: Dense com aproximadamente `64M`, MoE com aproximadamente `198M-A64M`, e remoção do design de especialista compartilhado (shared expert)
- Dados de treinamento padrão trocados para `pretrain_t2t(_mini).jsonl`, `sft_t2t(_mini).jsonl`, `rlaif.jsonl`, `agent_rl.jsonl` e `agent_rl_math.jsonl`
- Remoção do `train_reason.py` separado; a capacidade de pensamento agora é unificada por meio de `chat_template + <think>` e do controle adaptativo `open_thinking`
- A capacidade de `toolcall` foi incorporada aos dados principais `sft_t2t / sft_t2t_mini`, de modo que o `full_sft` padrão já tem capacidade básica de Tool Call; também foram adicionados exemplos de inferência como `scripts/chat_api.py`
- Adicionado o script nativo de treinamento de `Agentic RL`, `train_agent.py`, com suporte a `GRPO / CISPO` em cenários de Tool-Use com múltiplos turnos
- O pipeline de treinamento de RLAIF / Agentic RL concluiu o desacoplamento do `rollout engine`, permitindo alternar os backends de geração com mais flexibilidade
- `serve_openai_api.py` e `web_demo.py` ganharam suporte a `reasoning_content` / `tool_calls` / `open_thinking`
- Tokenizer atualizado com base em `BPE + ByteLevel`, com novos tokens de chamada de ferramenta e de pensamento, além de tokens reservados para extensões futuras
- Adicionado o pipeline de mesclagem e exportação de pesos LoRA: é possível mesclar o modelo base e os pesos LoRA em novos pesos completos via `scripts/convert_model.py`
- Recursos de diagramas de estrutura atualizados e README amplamente revisado

</details>

<details> 
<summary> <b>2025-10-24</b> </summary>

- 🔥 Adicionados algoritmos de treinamento RLAIF: PPO, GRPO, SPO (implementados nativamente do zero)
- Adicionada a retomada a partir de checkpoints: recuperação automática do treinamento, recuperação com número diferente de GPUs e continuidade dos registros do wandb
- Adicionado dataset de RLAIF: rlaif-mini.jsonl (10.000 entradas amostradas aleatoriamente dos dados de SFT); dataset de DPO simplificado, com dados em chinês adicionados
- Adicionado o algoritmo YaRN: extrapolação de contexto longo do RoPE, melhorando o processamento de sequências longas
- Pensamento Adaptativo: o modelo Reason pode ativar opcionalmente a cadeia de pensamento (chain of thought)
- O chat_template agora suporta totalmente as tags de Tool Calling e Reasoning (`<tool_call>`, `<think>` etc.)
- Adicionado um capítulo completo sobre RLAIF, comparação das curvas de treinamento e explicações recolhíveis sobre os princípios dos algoritmos
- [SwanLab](https://swanlab.cn/) substitui o WandB (acesso amigável a partir da China, API totalmente compatível)
- Padronização de todo o código e correção de alguns bugs conhecidos

</details>

<details> 
<summary> <b>2025-04-26</b> </summary>

- Grande atualização
- Para necessidades de compatibilidade, acesse o [🔗Conteúdo Antigo do Repositório🔗](https://github.com/jingyaogong/minimind/tree/7da201a944a90ed49daef8a0265c959288dff83a).
- Os parâmetros do modelo MiniMind foram totalmente renomeados, alinhados aos modelos da biblioteca Transformers (nomenclatura unificada).
- O método generate foi refatorado, passando a herdar da classe GenerationMixin.
- 🔥Suporte a ecossistemas populares de terceiros, como llama.cpp, vllm e ollama.
- Código e estrutura de diretórios padronizados.
- Vocabulário alterado de `<s></s>` para `<|im_start|><|im_end|>`

```text
Para manter compatibilidade com os frameworks de inferência de terceiros llama.cpp e vllm, esta atualização teve alguns custos consideráveis.
Esta atualização não suporta mais carregar "diretamente" os modelos antigos, anteriores a 26/04/2025, para inferência.
Devido às diferenças entre o método de codificação posicional do Llama e o do minimind, os valores de QK diferem após o mapeamento para o modelo Llama.
Os modelos antigos da série minimind2 foram todos recuperados por meio de mapeamento de pesos + calibração (ajuste fino) das camadas lineares QKVO.
Após esta atualização, a manutenção de toda a série `minimind-v1` será descontinuada e ela será removida do repositório.
```

</details>

<details>
<summary> <b>Mais...</b> </summary>

**2025-02-09**
- Maior atualização desde o lançamento: lançamento da série minimind2.
- Código quase totalmente refatorado, com uma estrutura unificada mais concisa e clara.
  Para compatibilidade com o código antigo, acesse o [🔗Conteúdo Antigo do Repositório🔗](https://github.com/jingyaogong/minimind/tree/6e9cd28ef9b34a0a10afbdf6f59e65cb6e628efb).
- Eliminadas as etapas de pré-processamento de dados. Formato de dataset unificado, agora em `jsonl`, para evitar confusões no download dos datasets.
- A série minimind2 teve desempenho significativamente superior ao do MiniMind-V1.
- Pequenos ajustes: {implementação de kv-cache mais padronizada, perda de balanceamento de carga do MoE agora considerada etc.}
- Oferece uma solução de treinamento para migrar modelos para datasets privados (exemplos de modelo médico e de autoconsciência).
- Dataset de pré-treinamento enxugado e qualidade dos dados de pré-treinamento muito melhorada, reduzindo bastante o tempo necessário para um treinamento rápido individual: reproduzível em 2 horas em uma única 3090!
- Atualizações: ajuste fino LoRA desacoplado do wrapper do peft, com o processo LoRA implementado do zero; algoritmo DPO implementado nativamente do zero em PyTorch; destilação white-box do modelo implementada nativamente.
- Nascem os modelos destilados da série minimind2-DeepSeek-R1!
- O minimind2 tem certa capacidade em inglês!
- Atualizados os resultados de benchmark do minimind2 em comparação com modelos de terceiros em mais leaderboards de LLM.

**2024-10-05**
- Capacidade multimodal estendida para o MiniMind --- Visão
- Veja o projeto irmão [minimind-v](https://github.com/jingyaogong/minimind-v) para mais detalhes!

**2024-09-27**
- Em 27/09, o método de pré-processamento do dataset de pretrain foi atualizado; para garantir a integridade do texto, abandonou-se o pré-processamento para o formato .bin no treinamento (com pequeno sacrifício na velocidade de treinamento).
- Atualmente, o arquivo pré-processado de pretrain se chama: pretrain_data.csv.
- Removido código redundante.

**2024-09-17**
- Modelo minimind-v1-moe atualizado
- Para evitar ambiguidades, o mistral_tokenizer deixou de ser usado na tokenização; todos passam a usar o minimind_tokenizer próprio como tokenizer.

**2024-09-01**
- Modelo minimind-v1 (108M) atualizado, usando o minimind_tokenizer, com 3 epochs de pretrain + 10 epochs de SFT: treinamento mais completo e desempenho mais forte.
- Projeto implantado no ModelScope Creative Space, e pode ser experimentado neste site:
- [🔗Demonstração Online no ModelScope🔗](https://www.modelscope.cn/studios/gongjy/minimind)

**2024-08-27**
- Primeira publicação do projeto como código aberto

</details>

---

# 📌 Início Rápido

<details>
<summary>Minha configuração de hardware e software (para referência)</summary>

* CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
* RAM: 128 GB
* GPU: NVIDIA GeForce RTX 3090 (24GB) * 8
* Ubuntu==20.04
* CUDA==12.2
* Python==3.10.16
* [requirements.txt](./requirements.txt)

</details>

## Passo 0

```bash
# Clonar o repositório e instalar as dependências
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind && pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

## Ⅰ 🚀 Inferência do Modelo

### 1' Baixar o Modelo

No diretório raiz do projeto:
```bash
# Método 1
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
# Método 2
git clone https://huggingface.co/jingyaogong/minimind-3
```

### 2' Inferência pela Linha de Comando

```bash
# Método 1: usando o modelo no formato Transformers
python eval_llm.py --load_from ./minimind-3
# Método 2: usando o modelo PyTorch (certifique-se de que os pesos correspondentes estejam no diretório ./out)
python eval_llm.py --load_from ./model --weight full_sft
```

### 3' (Opcional) WebUI

```bash
# Pode exigir `python>=3.10`; instale com `pip install streamlit`
# ⚠️ É preciso primeiro copiar a pasta do modelo no formato transformers para ./scripts/ (ex.: cp -r minimind-3 ./scripts/minimind-3). O script web_demo varre automaticamente os subdiretórios que contêm arquivos de pesos; se não encontrar nenhum, gera um erro.
cd scripts && streamlit run web_demo.py
```

### 4' (Opcional) Frameworks de Inferência de Terceiros

```bash
# ollama
ollama run jingyaogong/minimind-3
# vllm
vllm serve /path/to/model --served-model-name "minimind"
```

## Ⅱ 🛠️ Treinamento do Modelo

<details>
<summary>Observação: confirme com antecedência o backend disponível no Torch</summary>

```python
import torch
print(torch.cuda.is_available())
```

Se você pretende usar CUDA no treinamento, recomenda-se primeiro confirmar se o ambiente atual reconheceu corretamente a GPU.  
Se `cuda` não estiver disponível, você ainda pode escolher rodar em `CPU` ou `MPS`, de acordo com o seu dispositivo, mas a velocidade de treinamento e a compatibilidade vão variar bastante.  
Se precisar instalar ou trocar a versão do PyTorch, consulte o [torch_stable](https://download.pytorch.org/whl/torch_stable.html) e [este link](https://blog.csdn.net/weixin_45456738/article/details/141029610?ops_request_misc=&request_id=&biz_id=102&utm_term=%E5%AE%89%E8%A3%85torch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-141029610.nonecase&spm=1018.2226.3001.4187)

</details>

### 1' Baixar os Dados

Baixe os arquivos de dados necessários a partir do [link de download do dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) indicado abaixo e coloque-os no diretório `./dataset`

> Atualmente, por padrão, basta baixar `pretrain_t2t_mini.jsonl` e `sft_t2t_mini.jsonl` para reproduzir rapidamente o modelo de diálogo `MiniMind Zero`.
Para outros casos de uso, várias combinações de dados são apresentadas abaixo e podem ser escolhidas de acordo com seus objetivos e seus recursos de GPU.

### 2' Iniciar o Treinamento

<details>
<summary>💡 Pausa e Retomada por Checkpoint</summary>

Todos os scripts de treinamento suportam salvamento de checkpoints. Ao adicionar o parâmetro `--from_resume 1`, o script detecta automaticamente e retoma o progresso do treinamento:

```bash
python train_pretrain.py --from_resume 1
python train_full_sft.py --from_resume 1
# ...
```

**Instruções de Retomada por Checkpoint:**
- O processo de treinamento salva automaticamente checkpoints completos (modelo, otimizador, progresso do treinamento etc.) no diretório `./checkpoints/`
- Nomenclatura dos arquivos de checkpoint: `<nome_do_peso>_<dimensão>_resume.pth` (ex.: `full_sft_512_resume.pth`)
- Suporta recuperação com número diferente de GPUs (ajusta o step automaticamente)
- Suporta continuidade dos registros de treinamento no wandb (retoma automaticamente o mesmo run)

> Ideal para treinamentos longos ou ambientes instáveis: não é preciso se preocupar com perda de progresso por interrupções no treinamento

</details>

#### 2.1 Pré-treinamento (Obrigatório)

```bash
cd trainer && python train_pretrain.py
```

> Ao final do treinamento, será gerado `out/pretrain_*.pth` como pesos de saída (onde `*` é a dimensão do modelo, padrão `768`)

#### 2.2 Ajuste Fino por Instruções (Obrigatório)

```bash
cd trainer && python train_full_sft.py
```

> Ao final do treinamento, será gerado `out/full_sft_*.pth` como pesos de saída (onde `full` indica ajuste fino de todos os parâmetros)

#### 2.3 Testar o Modelo Treinado (Opcional)

Certifique-se de que os arquivos `*.pth` do modelo a ser testado estejam no diretório `./out/`; você também pode ir diretamente [aqui](https://www.modelscope.cn/models/gongjy/minimind-3-pytorch/files) para baixar os pesos `*.pth` que eu já treinei.

```bash
python eval_llm.py --weight full_sft
```

> `--weight` serve para especificar o prefixo do nome dos pesos, como `pretrain`, `full_sft` etc.; para mais parâmetros, consulte diretamente o `eval_llm.py`

<details>
<summary>Observação: Outras Informações</summary>

1. Todos os scripts de treinamento são implementados nativamente em PyTorch e suportam aceleração com várias GPUs.

2. Se o seu dispositivo tiver `N (N > 1)` GPUs, você pode iniciar o treinamento em uma máquina com `N` GPUs da seguinte forma (DDP; também é possível estender para várias máquinas com várias GPUs):

```bash
torchrun --nproc_per_node N train_xxx.py
```

3. Você pode ativar o wandb para registrar o processo de treinamento, se quiser.

```bash
... train_xxx.py --use_wandb
```
Desde junho de `2025`, redes domésticas na China normalmente não conseguem se conectar diretamente ao WandB. Atualmente, o MiniMind usa por padrão o [SwanLab](https://swanlab.cn/) como ferramenta de visualização de treinamento, cuja interface é basicamente compatível com a do WandB; em geral, basta substituir `import wandb` por `import swanlab as wandb`, e o restante do uso permanece praticamente igual.

</details>

---

# 📌 Introdução aos Dados

## Ⅰ Tokenizer

Um tokenizer pode ser entendido, de forma simplificada, como o "dicionário" usado pelos LLMs: ele é responsável por mapear a linguagem natural para ids de tokens e decodificar os ids de tokens de volta para texto. O projeto também oferece o `train_tokenizer.py` como exemplo de treinamento de vocabulário. Não é recomendável retreinar o tokenizer, pois, uma vez que o vocabulário e as regras de segmentação mudam, os pesos do modelo, os formatos de dados, as interfaces de inferência e a compatibilidade com o ecossistema da comunidade são todos afetados, e o modelo resultante também fica mais difícil de compartilhar. Além disso, o tokenizer também afeta métricas calculadas por token, como a PPL; por isso, ao comparar tokenizers diferentes, o BPB (Bits Per Byte) costuma ser uma métrica mais comparável. Veja [este artigo](https://skeptric.com/perplexity/).
Para modelos pequenos como o MiniMind, o tamanho do vocabulário também afeta diretamente a proporção de parâmetros das camadas de embedding e de saída, então manter o vocabulário compacto costuma ser um trade-off mais adequado.

<details>
<summary>Introdução ao Tokenizer</summary>

Os tamanhos de vocabulário dos tokenizers de modelos abertos poderosos de terceiros, como Yi, Qwen2, ChatGLM, Mistral e Llama 3, são os seguintes:

<table>
  <tr><th>Modelo do Tokenizer</th><th>Tamanho do Vocabulário</th><th>Origem</th></tr>
  <tr><td>Yi</td><td>64.000</td><td>01.AI (China)</td></tr>
  <tr><td>Qwen2</td><td>151.643</td><td>Alibaba Cloud (China)</td></tr>
  <tr><td>ChatGLM</td><td>151.329</td><td>Zhipu AI (China)</td></tr>
  <tr><td>Mistral</td><td>32.000</td><td>Mistral AI (França)</td></tr>
  <tr><td>Llama 3</td><td>128.000</td><td>Meta (EUA)</td></tr>
  <tr><td>MiniMind</td><td>6.400</td><td>Próprio</td></tr>
</table>

> A branch principal usa o `minimind_tokenizer` de forma consistente, para evitar ambiguidades com versões históricas e controlar o tamanho total, e não mantém mais a versão com `mistral_tokenizer`.

Embora o `minimind_tokenizer` tenha um vocabulário de apenas `6400` tokens e sua eficiência de codificação/decodificação seja inferior à de tokenizers mais amigáveis ao chinês, como `qwen2` e `glm`, ele reduz significativamente a participação dos parâmetros das camadas de embedding e de saída, o que se ajusta melhor às restrições de tamanho de modelos pequenos como o MiniMind.
Na prática, esse tokenizer não causou falhas perceptíveis na decodificação de palavras raras e continua estável o suficiente para uso geral. Por isso, a branch principal usa esse vocabulário de forma consistente, em vez de manter forks adicionais do tokenizer.

</details>

## Ⅱ Dados de Pré-treinamento

Os dados de pré-treinamento da branch principal atual, `MiniMind-3`, são `pretrain_t2t.jsonl` / `pretrain_t2t_mini.jsonl`.
Esses dois datasets foram organizados em um formato de treinamento unificado `text -> next token prediction`, buscando equilibrar, com poder computacional limitado:

- A qualidade do texto;
- A distribuição de comprimentos;
- A capacidade mista em chinês e inglês;
- O alinhamento de templates com as etapas seguintes de SFT / Tool Calling / RLAIF.

As fontes de dados incluem, entre outras, corpora de texto geral, corpora de diálogo selecionados, corpora de destilação e diversos datasets disponíveis sob **licenças de código aberto permissivas**; os dados da branch principal só entram no treinamento depois de passarem por limpeza, deduplicação, controle de comprimento e unificação de formato. As fontes de dados incluem: [Craftsman LLM Dataset](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data), [Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align) e outras fontes públicas.

Dentre eles:

- `pretrain_t2t_mini.jsonl` é voltado para reprodução rápida;
- `pretrain_t2t.jsonl` é voltado para o treinamento completo do modelo da branch principal `MiniMind-3`.

O formato do arquivo é o seguinte:

```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
{"text": "Transformer 通过自注意力机制建模上下文关系，是现代大语言模型的重要基础结构。"}
```

## Ⅲ Dados de SFT

Os dados de SFT da branch principal atual, `MiniMind-3`, são `sft_t2t.jsonl` / `sft_t2t_mini.jsonl`. Em comparação com os esquemas anteriores `sft_512 / sft_1024 / sft_2048`, a versão atual dá mais ênfase a:

- Templates unificados;
- Melhor adequação ao treinamento misto de diálogo + tags de pensamento + Tool Calling;
- Minimizar bifurcações no pré-processamento de dados, reduzindo os custos de reprodução.

Suas fontes de dados incluem, entre outras, dados de alta qualidade de seguimento de instruções, dados públicos de diálogo, dados sintéticos destilados de modelos e datasets de código aberto com licenças permissivas; antes de entrarem na branch principal `t2t`, eles são unificados no formato de diálogo com múltiplos turnos usado pelo repositório atual. A branch principal atual também contém uma grande quantidade de dados sintéticos, como aproximadamente `100K` entradas de `tool call` sintetizadas a partir do `qwen3-4b`, além de dados de `reasoning` da série `qwen3`. As principais fontes da comunidade incluem: [Craftsman LLM Dataset](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data), [Magpie-Align](https://www.modelscope.cn/organization/Magpie-Align), [R1-Distill-SFT](https://www.modelscope.cn/datasets/AI-ModelScope/R1-Distill-SFT), [COIG](https://huggingface.co/datasets/BAAI/COIG), [Step-3.5-Flash-SFT](https://huggingface.co/datasets/stepfun-ai/Step-3.5-Flash-SFT) etc. As versões publicadas garantem que as fontes de dados e os pipelines de processamento respeitem as restrições de transitividade das respectivas licenças de código aberto, e seguem os requisitos da Apache-2.0, CC-BY-NC-2.0 e de outras licenças relacionadas.

Dentre eles:

- `sft_t2t_mini.jsonl`: adequado para treinar rapidamente um modelo de diálogo;
- `sft_t2t.jsonl`: adequado para reproduzir por completo a versão da branch principal;
- A capacidade de `toolcall` já foi incorporada aos dados de SFT da branch principal.

Todos os arquivos de SFT seguem o mesmo formato, incluindo dados de diálogo e de Tool Use:

```jsonl
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
{
    "conversations": [
        {"role": "system", "content": "# Tools ...", "tools": "[...]"},
        {"role": "user", "content": "把'你好世界'翻译成english"},
        {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"translate_text\",\"arguments\":{\"text\":\"你好世界\",\"target_language\":\"english\"}}]"},
        {"role": "tool", "content": "{\"translated_text\":\"Hello World\"}"},
        {"role": "assistant", "content": "Hello World"}
    ]
}
```

## Ⅳ Dados de RL

Os dados de RL da branch principal atual do `MiniMind` são `dpo.jsonl`, amostrados de [DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k).

Na branch principal, essas amostras são reorganizadas no formato de aprendizado por preferência usado por este repositório, para treinamento de modelo de recompensa ou otimização de preferências. Aqui, `chosen` representa a resposta preferida e `rejected` representa a resposta mais fraca.

O formato de dados de `dpo.jsonl` é

```json
{
  "chosen": [
    {"content": "Q", "role": "user"}, 
    {"content": "good answer", "role": "assistant"}
  ], 
  "rejected": [
    {"content": "Q", "role": "user"}, 
    {"content": "bad answer", "role": "assistant"}
  ]
}
```

Além disso, os demais dados de RL mantêm o mesmo formato dos dados de SFT, normalmente filtrados dos dados de SFT pelo comprimento total e pelo número de turnos do diálogo, com a última posição de `assistant` deixada em branco para ser completada durante a etapa de rollout.

## Ⅴ Dataset de Treinamento do MiniMind

> [!NOTE]
> Os datasets centrais necessários para o treinamento da branch principal atual já foram disponibilizados em código aberto, então você não precisa pré-processar datasets de grande escala por conta própria.

Links para download do dataset de treinamento do MiniMind: [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main)

> Não é necessário clonar tudo; você pode baixar arquivos individuais conforme precisar

Coloque os arquivos de dataset baixados no diretório `./dataset/` (✨ indica os essenciais recomendados)

```bash
./dataset/
├── agent_rl.jsonl (86MB)
├── agent_rl_math.jsonl (18MB)
├── dpo.jsonl (53MB)
├── pretrain_t2t_mini.jsonl (1.2GB, ✨)
├── pretrain_t2t.jsonl (10GB)
├── rlaif.jsonl (24MB, ✨)
├── sft_t2t_mini.jsonl (1.6GB, ✨)
└── sft_t2t.jsonl (14GB)
```

<details>
<summary>Observação: Breve Descrição de Cada Dataset</summary>

* `agent_rl.jsonl` -- dados de treinamento principais de Agentic RL, para o treinamento de Tool-Use com múltiplos turnos / CISPO / GRPO do `train_agent.py`
* `agent_rl_math.jsonl` -- dados complementares de Agentic RL, apenas de matemática, adequados para cenários de raciocínio/uso de ferramentas com múltiplos turnos e alvos de verificação final (para RLVR)
* `dpo.jsonl` -- dados de treinamento por preferência da etapa de RLHF (DPO)
* `pretrain_t2t_mini`✨ -- dados leves de pré-treinamento do `minimind-3`, adequados para reprodução rápida (configuração recomendada: `max_seq_len≈768`)
* `pretrain_t2t` -- dados de pré-treinamento da branch principal do `minimind-3` (configuração recomendada: `max_seq_len≈380`)
* `rlaif.jsonl`✨ -- dataset de treinamento de RLAIF, para o treinamento com PPO/GRPO/CISPO e outros algoritmos de aprendizado por reforço
* `sft_t2t_mini.jsonl`✨ -- dados leves de SFT do `minimind-3` (para treinar rapidamente um modelo Zero), configuração recomendada `max_seq_len≈768`, já com uma parte de amostras de Tool Call misturadas
* `sft_t2t.jsonl` -- dados de SFT da branch principal do `minimind-3`, adequados para reprodução completa, também com amostras de Tool Call misturadas


O parâmetro de treinamento `max_seq_len` atualmente se refere ao comprimento em tokens, e não à contagem absoluta de caracteres.
O tokenizer deste projeto tem aproximadamente `1.5~1.7 caracteres/token` para texto em chinês e uma taxa de compressão de `4~5 caracteres/token` para inglês puro, com variações conforme a distribuição dos dados.
O "comprimento máximo" indicado nos nomes dos datasets é em número de caracteres; uma string de 100 caracteres pode ser convertida, grosso modo, em aproximadamente `100/1.5≈67` tokens.

Por exemplo:

* Chinês: `白日依山尽` (5 caracteres) pode ser dividido em [`白日`,`依`,`山`,`尽`], 4 tokens;
* Inglês: `The sun sets in the west` (24 caracteres) pode ser dividido em [`The `,`sun `,`sets `,`in `,`the`,`west`], 6 tokens

As "configurações recomendadas" fornecem estimativas aproximadas do comprimento máximo em tokens de cada dataset.
Observe que o `max_seq_len` pode ser ajustado de forma agressiva ou conservadora, mas ambas as direções têm efeitos colaterais: amostras mais curtas que o `max_seq_len` desperdiçam processamento com padding, enquanto amostras mais longas que o `max_seq_len` perdem informação por truncamento.

Na prática, basta equilibrar a eficiência computacional e a completude semântica.

</details>


![dataset](./images/dataset.jpg)

> Diagrama da composição dos dados de treinamento da branch principal do MiniMind e das combinações recomendadas

<details>
<summary>Instruções e Esquemas de Treinamento Recomendados</summary>

* Para a branch principal do `minimind-3`, recomenda-se usar a combinação de treinamento em etapas `pretrain_t2t` + `sft_t2t` + `rlaif/agent_rl`.

* Para chegar o mais rápido possível a um modelo Zero a partir do zero, recomenda-se usar a combinação de dados `pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl`

* Quem tem recursos computacionais suficientes ou se importa mais com o desempenho deve reproduzir por completo o `minimind-3`; quem tem apenas uma GPU ou prioriza uma reprodução rápida deve usar, fortemente recomendado, a combinação mini.

* Os `sft_t2t / sft_t2t_mini` atuais já têm dados de Tool Call misturados, então normalmente não há necessidade de uma rodada adicional e separada de ajuste fino supervisionado para Tool Calling.

</details>

# 📌 Modelo

## Estrutura

O `minimind-3` Dense usa uma arquitetura Transformer Decoder-Only, com configuração geral alinhada ao ecossistema `Qwen3`, para facilitar a conversão para `transformers / llama.cpp / ollama / vllm`:

* Usa Pré-Normalização (Pre-Norm) + RMSNorm.
* Usa a função de ativação SwiGLU.
* Usa a codificação posicional rotativa RoPE, com suporte a extrapolação YaRN.
* `q_heads=8`, `kv_heads=4`, `max_position_embeddings=32768`, `rope_theta=1e6`.

O `minimind-3-moe` estende as camadas feed-forward com MoE sobre a mesma estrutura, com implementação compatível com a configuração no estilo `Qwen3-MoE` (sem especialista compartilhado).

* A configuração padrão atual é `4 especialistas / roteamento top-1`, para obter maior capacidade com menos parâmetros ativos.
* À medida que o número de especialistas aumenta, o treinamento pode ficar muito mais lento do que o de um modelo denso de tamanho semelhante. Isso pode parecer contraintuitivo diante da afirmação comum de que "a inferência com MoE é mais rápida", mas, no treinamento, os tokens primeiro são agrupados por especialista e depois processados separadamente. Em uma implementação nativa em PyTorch, a sobrecarga de lançamento de kernels e de agendamento rapidamente se torna significativa. Isso normalmente exige kernels de MoE fundidos ou bibliotecas especializadas, como `Triton`, `DeepSpeed-MoE` ou `Megatron-LM`, para otimização. O MiniMind mantém a implementação em PyTorch nativo por portabilidade, então esse é um trade-off prático. Na implementação atual, a configuração `4 especialistas / top-1` é apenas cerca de `50%` mais lenta que o modelo denso.

A estrutura da série `minimind-3` é mostrada abaixo:

![structure](./images/LLM-structure.jpg)
![structure-moe](./images/LLM-structure-moe.jpg)

Para modificar a configuração do modelo, veja [./model/model_minimind.py](./model/model_minimind.py). As versões de referência dos parâmetros do modelo são mostradas na tabela abaixo:

| Nome do Modelo | params | len_vocab | max_pos | rope_theta | n_layers | d_model | kv_heads | q_heads | observação |
|------------|--------|-----------|---------|------------|----------|---------|----------|---------|------|
| minimind-3 | 64M | 6400 | 32768 | 1e6 | 8 | 768 | 4 | 8 | Dense |
| minimind-3-moe | 198M-A64M | 6400 | 32768 | 1e6 | 8 | 768 | 4 | 8 | 4 especialistas / top-1 |
| minimind2-small | 26M | 6400 | 32768 | 1e6 | 8 | 512 | 2 | 8 | Versão histórica |
| minimind2-moe | 145M | 6400 | 32768 | 1e6 | 8 | 640 | 2 | 8 | Versão histórica |
| minimind2 | 104M | 6400 | 32768 | 1e6 | 16 | 768 | 2 | 8 | Versão histórica |


## Configuração do Modelo

Em relação à configuração de parâmetros de LLMs, o [MobileLLM](https://arxiv.org/pdf/2402.14905) conduziu um estudo sistemático bastante representativo sobre modelos pequenos. Para modelos na faixa de ~100M, como o MiniMind, o trade-off entre `d_model` e `n_layers` não é apenas uma questão de alocação de parâmetros: ele também afeta diretamente a estabilidade do treinamento e o desempenho final.

A branch principal atual do `minimind-3` usa `dim=768, n_layers=8`, o que é essencialmente um trade-off de engenharia: redes mais rasas treinam mais rápido, enquanto `dim` continua grande o suficiente para evitar um gargalo de representação severo, resultando em um equilíbrio razoável entre eficiência de treinamento, estabilidade e desempenho final.

<details>
<summary>Ver Explicação Detalhada</summary>

As leis de escala costumam se comportar de forma diferente no regime de modelos pequenos. Os principais parâmetros de arquitetura que determinam a escala de parâmetros de um Transformer normalmente são `d_model` e `n_layers`:

* `d_model`↑ + `n_layers`↓ -> largo e raso
* `d_model`↓ + `n_layers`↑ -> estreito e profundo

As leis de escala clássicas enfatizam o papel do tamanho dos dados de treinamento, do número de parâmetros e dos passos de treinamento, e muitas vezes minimizam as diferenças de arquitetura. No regime de modelos pequenos, porém, essa conclusão nem sempre se sustenta.
Uma observação central do MobileLLM é que, com um orçamento fixo de parâmetros, a profundidade costuma ser mais importante que a largura. Em comparação com modelos largos e rasos, modelos estreitos e profundos tendem a aprender conceitos abstratos com mais eficácia.
Por exemplo, quando o número de parâmetros é fixado em `125M` ou `350M`, modelos estreitos com `30~42` camadas normalmente superam modelos largos com cerca de `12` camadas, mostrando tendências semelhantes em benchmarks como raciocínio de senso comum, perguntas e respostas e compreensão de leitura.

Isso é consistente com os próprios experimentos do MiniMind envolvendo `d_model` e `n_layers`. No entanto, "estreito" também tem um limite inferior: quando `d_model < 512`, o gargalo de representação se torna muito mais pronunciado, e adicionar camadas extras muitas vezes não é suficiente para compensar um `d_head` pequeno demais com `q_head` fixo.
Por outro lado, quando `d_model > 1536`, adicionar camadas costuma ser mais vantajoso do que aumentar ainda mais a largura, e tende a trazer um melhor retorno de desempenho por parâmetro.

Para referência, as configurações de parâmetros do GPT-3 são as seguintes:
![gpt3_config.png](./images/gpt3_config.png)

</details>

---

# 📌 Experimentos

## Ⅰ Custo de Treinamento

- **Unidade de tempo**: horas (h)
- **Unidade de custo**: yuan chinês (￥); `7￥ ≈ 1 USD`
- **Preço de aluguel da 3090**: aproximadamente `1.3￥/h` (os preços reais podem variar)
- **Observação**: os resultados a seguir são estimativas empíricas para o modelo `minimind` em uma única GPU `3090`, com o objetivo de facilitar a estimativa do custo de treinamento

| Nome do Modelo | params | pretrain_t2t_mini | sft_t2t_mini | toolcall | RLAIF |
|------------|--------|-------------------|--------------|----------|-------|
| minimind-3 | 64M | ≈1.21h<br/>≈1.57￥ | ≈1.10h<br/>≈1.43￥ | ≈0.9h<br/>≈1.17￥ | ≈1.1h<br/>≈1.43￥ |
| minimind-3-moe | 198M-A64M | ≈1.69h<br/>≈2.20￥ | ≈1.54h<br/>≈2.00￥ | ≈1.26h<br/>≈1.64￥ | ≈1.54h<br/>≈2.00￥ |

---

<details>
<summary>Resumo e Previsão do Custo de Treinamento</summary>

> `minimind-3`
>> `pretrain_t2t_mini` + `sft_t2t_mini`
<br/>Uma única GPU `3090`, `1 epoch` estimado em aproximadamente `2.31` horas, custo de aproximadamente `3.0` yuan
<br/>É possível treinar do zero um modelo de diálogo `minimind-3 Zero`.

> `minimind-3-moe`
>> `pretrain_t2t_mini` + `sft_t2t_mini`
<br/>Uma única GPU `3090`, `1 epoch` estimado em aproximadamente `3.23` horas, custo de aproximadamente `4.2` yuan
<br/>É possível obter rapidamente a versão básica de diálogo do `minimind-3-moe`.

> Os valores acima são estimativas e servem apenas como referência aproximada do custo de treinamento.

</details>

Em uma única `NVIDIA 3090`, o `minimind zero` ainda pode ser treinado do zero em aproximadamente `2` horas, tornando o processo de treinamento acessível a desenvolvedores individuais.

Com uma configuração de várias GPUs mais potente, como `8x H100`, o tempo total de treinamento pode cair para minutos. O MiniMind sempre teve como objetivo manter o treinamento de LLMs reproduzível, prático e fácil de iterar. A reprodução rápida e de baixo custo não é um truque de marketing; abaixo está um exemplo antigo de conversa no estilo Zero, preservado como referência:

```text
👶: 请介绍一下自己。
🤖️: 作为人工智能，我没有实际的生活，也没有自我意识，所以没有自己的生活。我被设计成能够帮助用户解答问题、提供信息、进行对话等。我的设计和功能是由计算机科学、机器学习、人工智能技术和算法所构建的，我通过编程和算法实现。

👶: 推荐一些杭州的特色美食吧。
🤖️: 杭州有很多美味的特色美食，比如鸡肉串、鳗鱼头、水煮鱼、豆腐脑等。这些美食都非常美味，有着独特的风味和口感，绝对是值得一试的美食。

👶: 请为我讲解"大语言模型"这个概念。
🤖️: 大语言模型是一种人工智能技术，它使用大量的文本数据来训练模型，然后在不断的迭代中不断优化模型。它的目标是根据输入的数据来生成符合特定需求的文本，这些文本可以是任何你感兴趣的主题。

👶: Introduce the history of the United States, please.
🤖️: 您提到的"Introok's the believeations of theument." 这个名字来源于中国古代的"groty of of the change."
```

Embora essa versão já tenha capacidade básica de diálogo, seu conhecimento factual e sua capacidade de generalização ainda são limitados; ela serve principalmente como referência inicial da viabilidade da rota de treinamento Zero.
Os pesos do modelo Zero são salvos como `full_sft_zero_768.pth` (veja os links dos arquivos do modelo MiniMind abaixo); se tiver interesse, você pode baixá-los e experimentar o desempenho de diálogo.


---

## Ⅱ Treinamento Principal (Obrigatório)

> Todos os scripts de treinamento são executados a partir do diretório `cd ./trainer`

### 1' Pré-treinamento (Pretrain):

Um LLM primeiro precisa absorver conhecimento fundamental e padrões de linguagem em seus parâmetros. Somente depois que essa etapa estiver suficientemente estável o modelo pode começar a entender perguntas, organizar respostas e desenvolver uma capacidade de geração utilizável. O pré-treinamento consiste essencialmente em expor o modelo a grandes quantidades de texto, como Wikipédia, notícias, livros e corpora de diálogo, para que ele aprenda conhecimento factual, padrões de linguagem e relações estatísticas entre contextos. Essa etapa normalmente é "não supervisionada": humanos não rotulam cada linha como certa ou errada; em vez disso, o modelo extrai padrões de textos massivos e constrói gradualmente representações internas do conhecimento de mundo e da estrutura da linguagem.
Em termos simples, o objetivo central nesta etapa é a **continuação de alta qualidade do próximo token**. Por exemplo, dada a entrada "秦始皇" (Qin Shi Huang), o modelo deve ser capaz de continuar com "是中国历史上的第一位皇帝" (foi o primeiro imperador da história da China) — um conteúdo semântica e factualmente consistente.

```bash
# Método 1
torchrun --nproc_per_node 1 train_pretrain.py # 1 significa treinamento com uma única GPU; ajuste de acordo com o seu hardware (defina >=2)
# Método 2
python train_pretrain.py
```

> Os arquivos de pesos do modelo treinado são salvos por padrão a cada `save_interval steps` como: `pretrain_*.pth` (* é a dimensão específica do modelo; cada salvamento sobrescreve o arquivo anterior)

![pretrain_loss](./images/pretrain_loss.jpg)
> Curva de loss durante a etapa de pré-treinamento com a configuração `768dim`

```bash
# Teste simples dos resultados do pré-treinamento:
python eval_llm.py --weight pretrain

💬: 为什么天空是蓝色的
🧠: 天空之所以看起来是蓝色的，主要是因为太阳光进入大气层后，短波长的蓝光更容易被空气分子散射，因此人眼从各个方向接收到的蓝光会更多。

💬: 解释什么是机器学习
🧠: 机器学习是人工智能的一个重要分支，它通过数据训练模型，使系统能够自动学习规律，并在分类、预测、推荐、自然语言处理等任务中持续改进效果。
```

### 2' Ajuste Fino Supervisionado (SFT):

O SFT não serve apenas para fazer o modelo "conversar melhor"; ele também pode continuar injetando novos conhecimentos, padrões de comportamento e estilos de resposta no modelo. Com `14GB` de dados de SFT na branch principal atual do MiniMind, essa etapa já vai além de um simples alinhamento de formato e se aproxima de um processo contínuo de mid-training.
Se o pré-treinamento faz o modelo ler de forma ampla e adquirir capacidade básica de linguagem, o SFT faz um processamento adicional com dados de maior qualidade e mais direcionados. Ele ajuda o modelo a se adaptar a formatos de interação como diálogo com múltiplos turnos, perguntas e respostas, chamada de ferramentas e tags de pensamento, ao mesmo tempo em que imprime nos parâmetros distribuições específicas de conhecimento, padrões de tarefas e estilos de assistente.
Especificamente no MiniMind, a etapa de SFT faz o modelo se adaptar ao template de diálogo com múltiplos turnos usado pelo repositório atual. O modelo passa a entender gradualmente a estrutura de papéis `user / assistant / system / tool`, enquanto reforça ainda mais o seguimento de instruções, a estabilidade das respostas e a capacidade de concluir tarefas.
O treinamento atual aplica truncamento aos comprimentos das instruções e das respostas, principalmente para equilibrar o uso de VRAM e a eficiência do treinamento. Se contextos mais longos forem necessários mais tarde, um pequeno número de amostras de contexto longo pode ser usado para um ajuste fino incremental. Durante a inferência, a extrapolação YaRN pode estender o comprimento de contexto para 2048 ou mais, sem treinamento adicional.

```bash
# Método 1
torchrun --nproc_per_node 1 train_full_sft.py
# Método 2
python train_full_sft.py
```

> Os arquivos de pesos do modelo treinado são salvos por padrão a cada `save_interval steps` como: `full_sft_*.pth` (*
> é a dimensão específica do modelo; cada salvamento sobrescreve o arquivo anterior)

![sft_loss](./images/sft_loss.jpg)
> Curva de loss durante a etapa de SFT com a configuração `768dim`

```bash
# Teste simples dos resultados do SFT:
python eval_llm.py --weight full_sft

💬: 解释什么是机器学习
🧠: 机器学习是人工智能的核心技术之一，通过算法让计算机从数据中学习规律，并持续改进预测或决策效果，常见应用包括推荐系统、图像识别、语音识别和自然语言处理。

💬: 推荐一些中国的美食
🧠: 例如北京烤鸭、兰州拉面、四川火锅、广东早茶、小笼包和麻婆豆腐等，这些美食分别代表了不同地区的风味特点，也很适合作为了解中国饮食文化的入门选择。
```

## Ⅲ Outros Treinamentos (Opcional)

> Todos os scripts de treinamento são executados a partir do diretório `cd ./trainer`

### 3' Destilação de Conhecimento (KD)

A destilação de conhecimento pode ser dividida, de forma geral, em duas categorias: caixa-preta (black-box) e caixa-branca (white-box). A branch principal atual do MiniMind envolve as duas abordagens, com ênfases diferentes.
* Destilação caixa-preta: mais comum e mais alinhada à prática real da branch principal atual. A rigor, ela é essencialmente um ajuste fino supervisionado orientado às saídas do professor, ou seja, continuar treinando com base em rótulos rígidos (hard labels); com a popularização dos LLMs, essa abordagem de "fazer FT sobre as saídas de um modelo forte" passou a ser amplamente classificada sob o guarda-chuva da destilação, e por isso é comumente chamada de destilação caixa-preta. Ela se concentra em aprender respostas, estilos e padrões de comportamento — o modelo aluno só vê "o que o professor disse", mas não vê como o professor chegou internamente àquele julgamento. Respostas de alta qualidade do `DeepSeek R1` e do `Qwen3`, bem como dados de `tool call`, `reasoning`, cadeia de pensamento etc., podem ser vistos como sinais de destilação caixa-preta; os dados atuais de `full_sft` da branch principal do MiniMind já têm uma parcela considerável dessa abordagem misturada.
* Destilação caixa-branca: vai além, aprendendo não só as saídas finais do professor, mas também as preferências do professor no nível da distribuição de tokens. Em comparação com a destilação caixa-preta, ela aproveita adicionalmente a informação de distribuição mais granular da camada de saída do modelo professor, de modo que o modelo aluno aprende não apenas a "resposta padrão", mas também as preferências relativas do professor entre os tokens candidatos. Correspondendo ao `train_distillation.py`, a implementação atual continua treinando o modelo aluno com os sinais de distribuição fornecidos pelo modelo professor sobre pesos que já passaram por SFT, sendo mais adequada como implementação de referência para entender o pipeline de destilação do MiniMind.

A destilação caixa-preta é essencialmente equivalente a um ajuste fino supervisionado sobre respostas geradas pelo professor:
```math
\mathcal{L}_{blackbox} = \mathrm{CE}(y_{teacher}, p_{student})
```

A destilação caixa-branca normalmente ajusta a distribuição do professor além da loss supervisionada:
```math
\mathcal{L}_{whitebox} = \alpha \mathcal{L}_{CE} + (1-\alpha) T^2 \mathrm{KL}(p_t^T \parallel p_s^T)
```

O script `train_distillation.py` foi pensado como uma implementação de referência para entender o pipeline de destilação caixa-branca: ele demonstra o carregamento duplo de modelos professor/aluno, a loss mista `CE + KL`, o escalonamento por temperatura, a destilação combinando MoE e modelo denso, além de detalhes importantes como retomada por checkpoint e treinamento distribuído.

```bash
# Método 1
torchrun --nproc_per_node 1 train_distillation.py
# Método 2
python train_distillation.py
```

### 4' LoRA (Low-Rank Adaptation)

LoRA é um método comum de Ajuste Fino Eficiente em Parâmetros (PEFT). Em comparação com o ajuste fino de todos os parâmetros, ele atualiza apenas um pequeno número de parâmetros recém-adicionados, mantendo inalterados os pesos originais do modelo, o que reduz o custo de treinamento e o torna mais adequado à adaptação a domínios verticais.
Sua ideia central é introduzir ramos incrementais de baixo posto ao lado das matrizes de pesos originais, treinando apenas esses parâmetros de baixo posto e, assim, concluindo a transferência de capacidades com um custo relativamente pequeno. As implementações relacionadas estão em `model_lora.py` e `train_lora.py`; todo o pipeline é escrito à mão, sem depender de wrappers de terceiros.

```bash
# O train_lora.py normalmente pode ser concluído com relativa rapidez, mesmo em CPU
# Método 1
cd trainer && torchrun --nproc_per_node 1 train_lora.py
# Método 2
cd trainer && python train_lora.py
```

> Os arquivos de pesos do modelo treinado são salvos por padrão a cada `save_interval steps` como: `lora_xxx_*.pth` (* é a dimensão específica do modelo; cada salvamento sobrescreve o arquivo anterior)


O LoRA é bem adequado para problemas como "como fazer o modelo se adaptar rapidamente a domínios privados ou cenários verticais, preservando ao máximo as capacidades gerais". Por exemplo, quando o modelo base não tem conhecimento médico, uma camada de pesos LoRA voltada à medicina pode ser empilhada sobre o modelo original para obter um melhor desempenho no domínio com um custo relativamente pequeno.
Normalmente, basta preparar dados no mesmo formato de diálogo com múltiplos turnos, colocá-los em `lora_xxx.jsonl` e então executar `cd trainer && python train_lora.py` a partir da raiz do repositório para obter novos pesos de modelo `LoRA`.

Exemplo 1: dados de domínio vertical

```jsonl
 {"conversations": [{"role": "user", "content": "请问颈椎病的人枕头多高才最好？"}, {"role": "assistant", "content": "颈椎病患者选择枕头的高度应该根据..."}]}
 {"conversations": [{"role": "user", "content": "请问xxx"}, {"role": "assistant", "content": "xxx..."}]}
```

Exemplo 2: dados de autoconsciência (identidade)

```jsonl
 {"conversations": [{"role": "user", "content": "你叫什么名字？"}, {"role": "assistant", "content": "您好，我名叫 MiniMind，是由 Jingyao Gong 开发的人工智能助手。我可以帮助你解决各种问题，包括翻译、推荐内容等等。"}]}
 {"conversations": [{"role": "user", "content": "你是谁"}, {"role": "assistant", "content": "您好，我是 MiniMind，由 Jingyao Gong 创建。我的主要职责是提供准确的回答和帮助。"}]}
```

Nesse ponto, [modelo base + pesos LoRA] podem ser usados em conjunto, preservando as capacidades gerais do modelo base e reforçando o desempenho em um domínio específico. Após o treinamento, testes simples também podem ser feitos com o `eval_llm.py`.

```bash
# Observação: o parâmetro weight especifica o tipo de modelo base, que deve ser o mesmo usado durante o treinamento com train_lora
python eval_llm.py  --weight full_sft --lora_weight lora_medical

👶: 我最近经常感到头晕，可能是什么原因？
🤖️: 头晕可能是由于多种原因，包括但不限于：心脏疾病、眼睛损伤、过敏反应、皮肤问题、过敏反应、压力、焦虑、疲劳、药物副作用、性功能障碍或者其他...

python eval_llm.py  --weight full_sft --lora_weight lora_identity

👶: 你和openai是什么关系？
🤖️: 我是 MiniMind，一个由 Jingyao Gong 开发的人工智能助手。我通过自然语言处理和算法训练来与用户进行交互。
```

PS: Se você tiver dados suficientes, também pode fazer diretamente o ajuste fino de todos os parâmetros com `full_sft`. Isso normalmente exige uma mistura cuidadosa de dados gerais e de domínio específico; caso contrário, o modelo pode perder parte de sua capacidade geral por overfitting nas amostras do domínio vertical.


> Os pesos `LoRA` podem ser mesclados de volta ao modelo base e exportados como novos pesos completos do modelo, usando `convert_merge_base_lora` em `scripts/convert_model.py`:

```bash
cd scripts && python convert_model.py
```

### **5' Chamada de Ferramentas e Pensamento Adaptativo**

A partir de `2026-03`, o repositório removeu o `train_reason.py` separado.  
A versão atual não mantém mais pesos `reason_*.pth` separados; em vez disso, modela de forma unificada "se o processo de pensamento deve ou não ser explicitado" por meio do `chat_template`, das tags `<think>`, do controle `open_thinking` e dos pipelines subsequentes de SFT / RLAIF.

#### 5.1 Chamada de Ferramentas (Tool Calling)

A capacidade atual de `toolcall` foi incorporada aos dados principais `sft_t2t` / `sft_t2t_mini`, então uma etapa adicional de SFT para Tool Calling normalmente é desnecessária; os pesos `full_sft` padrão já têm capacidade básica de Tool Call. Os dados de treinamento atuais desta parte contêm principalmente cerca de `100K` amostras geradas a partir do `qwen3-4b`, e a lista de ferramentas abrange cerca de `10` ferramentas personalizadas simuladas, como consulta de horário, cálculos matemáticos e previsão do tempo. Nesta fase, sua capacidade de generalização ainda é limitada. As amostras de Tool Calling seguem o formato de mensagens com múltiplos turnos no estilo da OpenAI:

```jsonl
{
  "conversations": [
    {"role": "system", "content": "# Tools ...", "tools": "[...]"},
    {"role": "user", "content": "帮我算一下 256 乘以 37 等于多少"},
    {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"calculate_math\",\"arguments\":{\"expression\":\"256 * 37\"}}]"},
    {"role": "tool", "content": "{\"result\":\"9472\"}"},
    {"role": "assistant", "content": "256 乘以 37 等于 9472。"}
  ]
}
```

Aqui, `tools` é anexado à mensagem de `system` e `tool_calls` é anexado à mensagem de `assistant`. Durante o treinamento, o `chat_template` os expande automaticamente em segmentos `<tool_call>...</tool_call>` e `<tool_response>...</tool_response>`, permitindo que o modelo aprenda diretamente o formato nativo de chamada de ferramentas.

O chat template de Tool Calling foi unificado para ser interpretado como:

```text
<tool_call>{"name": "...", "arguments": {...}}</tool_call>
<tool_response>{...tool result...}</tool_response>
```

Você também pode rodar testes simples diretamente com o `eval_toolcall.py`:

```bash
python eval_toolcall.py --weight full_sft

💬: 现在几点了？
🧠: <tool_call>{"name": "get_current_time", "arguments": {"timezone": "Asia/Shanghai"}}</tool_call>
📞 [Tool Calling]: get_current_time
✅ [Tool Called]: {"datetime": "2026-03-15 17:18:22", "timezone": "Asia/Shanghai"}
🧠: 现在是2026年3月15日17时18分22秒。
```

#### 5.2 Pensamento Adaptativo

O `minimind` unifica a capacidade de pensamento explícito no nível do template, o que também é consistente com o design de template de muitos dos grandes modelos atuais:

- `open_thinking=0`: por padrão, injeta um `<think>\n\n</think>` vazio, e o modelo tende a responder diretamente;
- `open_thinking=1`: o template pré-injeta a tag de abertura `<think>`, e o modelo então continua gerando o processo de pensamento explícito e a resposta final;
- A CLI, a OpenAI-API e a WebUI suportam esse controle.

Mais precisamente, a abordagem não é mais treinar um modelo de pensamento separado, mas empurrar a decisão de "pensar ou não de forma explícita" para o `chat_template`. A camada de template reserva a estrutura `<think></think>`, e o mesmo modelo alterna dinamicamente via `open_thinking` durante a inferência. No treinamento, `think` vazios, `reasoning_content` explícitos e amostragem por `thinking_ratio` são misturados, para que o modelo aprenda gradualmente quando pensar de forma explícita e quando responder diretamente.

```bash
# Testar as respostas
python eval_llm.py --load_from ./minimind-3 --open_thinking 1
```

Uso com o OpenAI-API-SDK:

```python
response = client.chat.completions.create(
    model="minimind",
    messages=[{"role": "user", "content": "你是谁？"}],
    # ...
    extra_body={"chat_template_kwargs": {"open_thinking": True}} # Controle de pensamento
)
```

Observação: quando Tool Call e pensamento explícito são ativados ao mesmo tempo, o modelo normalmente não consegue gerar o processo de pensamento de forma estável. O motivo é que os dados de treinamento atuais ainda carecem de amostras de destilação conjunta em que "raciocínio e chamada de ferramentas coexistem", então o modelo ainda não aprendeu totalmente a expressar essas duas capacidades de forma coordenada.

## Ⅳ Aprendizado por Reforço (Opcional)

Na prática de pós-treinamento de LLMs, existem principalmente dois caminhos comuns de aprendizado por reforço:

1. **Aprendizado por Reforço com Feedback Humano (RLHF)**

- Treina o modelo avaliando suas saídas por meio de avaliações de preferência **humanas**, fazendo-o gerar conteúdo mais alinhado aos valores e preferências humanos.

2. **Aprendizado por Reforço com Feedback de IA (RLAIF)**

- Usa **modelos de IA** ou outros mecanismos automaticamente verificáveis para fornecer feedback, sem depender diretamente de anotação humana.
- Aqui, "feedback de IA" em sentido amplo também pode se estender a recompensas baseadas em regras, verificação com Ground Truth, interpretadores de código, feedback do ambiente e outros sinais automatizados.

| Tipo  | Avaliador | Vantagens | Desvantagens |
|-------|-------|------------|---------------|
| RLHF  | Humano | Mais próximo das preferências humanas reais | Custo alto, baixa eficiência |
| RLAIF | Modelo | Automatizado, altamente escalável | Pode se desviar das preferências humanas reais |

Ambos pertencem, essencialmente, ao paradigma de aprendizado por reforço que otimiza o comportamento do modelo usando algum tipo de "**feedback**".

No entanto, na prática, suas diferenças vão além da origem do feedback: se a recompensa é verificável, se é contínua, se depende de interação com um ambiente e se é adiada até o fim de todo o episódio — tudo isso afeta diretamente a forma do treinamento e a implementação de engenharia.


### 👀 Uma Perspectiva Unificada sobre os Algoritmos de PO

Antes de apresentar a implementação de algoritmos específicos, vou primeiro descrever, a partir da minha perspectiva minimalista, o que todos os algoritmos de Otimização de Política (Policy Optimization, PO) têm em comum.

Em poucas palavras, os algoritmos de PO discutidos aqui estão todos otimizando uma esperança:

$$\mathcal{J}_{PO} = \mathbb{E}_{q \sim P(Q),\, o \sim \pi_\theta(\cdot \mid q)} \left[ \underbrace{\Phi(r_t, A_t)}_{\text{policy objective}} - \underbrace{h(\text{KL}_t)}_{\text{regularization term}} \right]$$

Durante o treinamento, basta **minimizar a função objetivo negativa**, ou seja:

$$\mathcal{L}_{PO} = -\mathcal{J}_{PO}$$

Esse framework contém apenas três componentes centrais:
* **Termo de política** $\Phi(r_t, A_t)$: como combinar a razão de probabilidades $r_t$ e a vantagem $A_t$ para atualizar a política
* **Termo de vantagem** $A_t$: como calcular a vantagem — isso é muito importante! Não é surpresa que modelos grandes consigam resolver integrais definidas corretamente, mas para modelos pequenos, até acertar somas e subtrações normalmente já rende uma vantagem positiva
* **Termo de regularização** $h(\text{KL}_t)$: como restringir a magnitude da mudança $\text{KL}_t$, evitando tanto desviar demais quanto restringir demais

<details>
<summary>(Expandir) Guia de Notação</summary>

| Símbolo | Significado | Descrição | Intervalo |
|--------|---------|-------------|-------|
| $q$ | Pergunta/Prompt | Amostrado do dataset $P(Q)$ | - |
| $o$ | Sequência de saída do modelo | Gerada pela política $\pi$ | - |
| $r_t$ | Razão de probabilidades | $r_t = \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\mathrm{old}}(o_t \mid q, o_{<t})}$ | $(0, +\infty)$ |
| $A_t$ | Função de vantagem | Mede o quanto uma determinada ação é melhor em relação à linha de base | $(-\infty, +\infty)$ |
| $\text{KL}_t$ | Divergência KL | Impede que a política se afaste demais do modelo de referência | $[0, +\infty)$ |

</details>

Os diferentes **algoritmos xxPO** são, essencialmente, apenas instanciações diferentes de escolhas de design para esses três componentes!

---

### **6' Aprendizado por Reforço com Feedback Humano (RLHF)**

Nas etapas de treinamento anteriores, o modelo já adquiriu capacidade básica de diálogo, mas essa capacidade se baseia inteiramente em completar sequências de palavras, sem incentivos de exemplos positivos e negativos.
Neste ponto, o modelo ainda não sabe quais respostas são boas e quais são ruins. Queremos que ele se alinhe melhor às preferências humanas, reduzindo a probabilidade de gerar respostas que desagradem as pessoas.
Esse processo é como colocar o modelo em um novo treinamento corporativo, aprendendo com funcionários exemplares como exemplos positivos e com funcionários desmotivados como exemplos negativos, para entender melhor como responder.

#### 6.1 Otimização Direta de Preferências
Algoritmo de Otimização Direta de Preferências (Direct Preference Optimization, DPO), com a loss:

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)\right]$$

Onde:
- **Termo de política**: $f(r_t) = \log r_w - \log r_l$ (compara a razão de probabilidades de chosen vs rejected)
- **Termo de vantagem**: $g(A_t)$ = sem termo de vantagem explícito (refletido implicitamente pela comparação de preferências)
- **Termo de regularização**: $h(\text{KL}_t)$ = implícito em $\beta$ (controla o grau de desvio em relação ao modelo de referência)

Especificamente,
- O DPO deriva um objetivo de treinamento analítico para pares de preferência a partir do objetivo com restrição de KL do PPO, maximizando diretamente o log-odds de que "chosen é preferido a rejected"; não é preciso treinar simultaneamente modelos de Reward/Value. O DPO só precisa rodar os modelos `actor` e `ref`, com baixo uso de VRAM, convergência estável e implementação simples.
- Paradigma de treinamento: off-policy, usando um dataset de preferências estático, podendo iterar por várias epochs; o modelo Ref é fixo (as saídas são pré-computadas em cache).
- A limitação do DPO é que ele não faz exploração online e é mais adequado ao alinhamento com valores humanos em "preferência/segurança"; sua capacidade de melhorar habilidades intelectuais como "se o modelo consegue resolver problemas corretamente" é limitada (claro que isso também depende do dataset, já que coletar amostras positivas e negativas em escala com avaliação humana é muito difícil).

```bash
# Método 1
torchrun --nproc_per_node 1 train_dpo.py
# Método 2
python train_dpo.py
```

> Os arquivos de pesos do modelo treinado são salvos por padrão a cada `save_interval steps` como: `dpo_*.pth` (* é a dimensão específica do modelo; cada salvamento sobrescreve o arquivo anterior)

### 7' Aprendizado por Reforço com Feedback de IA (RLAIF)

É preciso uma pequena ressalva sobre o nome. Continuo chamando esta seção de `RLAIF`, embora o termo não seja estritamente preciso. Rotas como o RLVR, que dependem de recompensas verificáveis, têm sua própria linhagem e não se encaixam perfeitamente na definição estrita de feedback de IA.
Se "IA" for interpretada de forma mais ampla, porém, o nome ainda se justifica: as recompensas podem vir de modelos de recompensa, modelos avaliadores (judge) ou outros agentes inteligentes explícitos, mas também podem vir de funções de regras, verificação com Ground Truth, resultados de chamadas de ferramentas, estados do ambiente e outros sinais disponíveis automaticamente. Quando as regras são complexas o suficiente e o sistema simbólico é rico o suficiente, a fronteira entre esses sinais e o "feedback inteligente" nem sempre é clara.
Por isso, este capítulo foca no aprendizado por reforço após o SFT usando diversos **sinais de feedback não humanos e obtidos automaticamente**. Por exemplo, se uma resposta de matemática está correta, se um código gerado passa nos casos de teste ou se o processo de raciocínio segue o formato esperado — tudo isso pode ser julgado automaticamente.
Para tarefas verificáveis de um único turno, esse feedback costuma ser mais próximo de uma recompensa imediata. Em cenários de Agentic RL, as recompensas são com mais frequência adiadas até o fim de uma interação de vários passos, ou vêm diretamente do próprio ambiente.
Suas características em comum costumam ser o **treinamento on-policy** e a **alta escalabilidade**: não é necessária anotação humana cara, e grandes quantidades de amostras de treinamento podem ser geradas para tentativa e erro online.

O MiniMind implementou **2+N** métodos de RLAIF, básicos + de ponta:
* **PPO**, **GRPO** — algoritmos clássicos de RL validados em larga escala
* N algoritmos de RL de ponta (atualizados periodicamente, em caráter experimental)

**1️⃣ Preparação do Dataset (Obrigatório)**

A branch principal atual usa `rlaif.jsonl` como dados de treinamento de RLAIF. Ele tem aproximadamente `20MB`, é mais completo que o antigo `rlaif-mini.jsonl` e é mais adequado para verificar diretamente o comportamento do treinamento com PPO / GRPO / CISPO.

O formato dos dados é o mesmo do SFT, mas o assistant não precisa de conteúdo, porque durante o treinamento ele é inteiramente gerado em tempo real pelo modelo de política $\Pi$ por amostragem. Por isso, fica assim:

```json
{
    "conversations": [
        {"role": "user", "content": "请解释一下什么是光合作用？"},
        {"role": "assistant", "content": "无"}
    ]
}
```

Durante o processo de treinamento de RLAIF, o modelo gera 1 ou mais respostas candidatas com base na pergunta do usuário e, em seguida, uma função/modelo de recompensa pontua as respostas.
Respostas com pontuação alta são incentivadas (aumentando a probabilidade da política $\Pi$), e respostas com pontuação baixa são suprimidas (diminuindo a probabilidade da política $\Pi$). Esse ciclo "pontuar -> ajustar" é o núcleo do aprendizado por reforço.

**2️⃣ Preparação do Mecanismo de Recompensa (Obrigatório)**

O treinamento de RLAIF exige algum tipo de sinal de recompensa computável; ele pode vir de um modelo de recompensa, de funções de regras, de verificação com Ground Truth ou de feedback do ambiente. Atualmente, o MiniMind demonstra por padrão a rota do Reward Model.

Aqui escolhemos o pequeno e de alta qualidade `InternLM2-1.8B-Reward` ([ModelScope](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-1_8b-reward) | [HuggingFace](https://huggingface.co/internlm/internlm2-1_8b-reward)) como modelo de recompensa base.

Depois de baixar o modelo de recompensa, ele precisa ser colocado em um **diretório irmão** do projeto minimind, com a seguinte estrutura recomendada:

```
root/
├── minimind/                    # Projeto MiniMind
│   ├── model/
│   └── ...
└── internlm2-1_8b-reward/       # Modelo de recompensa
    ├── config.json
    ├── model.safetensors
    └── ...
```

<details>
<summary><b>Escolha do Mecanismo de Recompensa e Limitações do MiniMind (Clique para Expandir)</b></summary>

**1. Diversidade de Mecanismos de Recompensa**

As fontes de "sinal de recompensa" no RLAIF podem ser bastante flexíveis:

- **Recompensas baseadas em modelo**: pode-se usar um Reward Model dedicado (como o InternLM2-Reward) ou um LLM geral + prompts para pontuar (como Qwen3-as-a-Judge). A escala e a arquitetura do modelo de recompensa podem ser escolhidas livremente.

- **Recompensas baseadas em regras**: sinais de recompensa podem ser construídos a partir de funções de regras, por exemplo:
  - Verificação da correção de respostas de problemas de matemática (comparação com Ground Truth)
  - Taxa de sucesso na execução de SQL e precisão dos resultados
  - Resultados da execução em um interpretador de código (pass@k)
  - Status de retorno de chamadas de ferramentas (sucesso/falha da API)
  - Verificação de conformidade de formato (parsing de JSON/XML)
  - Avaliação da completude da cadeia de raciocínio (número de passos de CoT)

- **Recompensas baseadas no ambiente**: em cenários de Agent, o próprio feedback do ambiente funciona como uma recompensa natural (como pontuações em jogos, completude de uma pesquisa, taxa de conclusão de tarefas).

Qualquer mecanismo capaz de quantificar a "qualidade da resposta" pode servir como fonte de recompensa para RL. O DeepSeek R1 é um exemplo típico: usa funções de regras para verificar a correção de respostas de matemática como recompensa, sem precisar de um Reward Model adicional.

**2. Limitação do MiniMind: o Problema da Esparsidade de Recompensas**

O treinamento de RLAIF pode ter como alvo tanto modelos de raciocínio quanto modelos sem raciocínio; a diferença está apenas no formato.

No entanto, para modelos como o MiniMind, extremamente pequenos (0.1B de parâmetros) e com capacidades limitadas, surgem sérios problemas de Esparsidade de Recompensas (Reward Sparsity) em tarefas gerais (como datasets de matemática no estilo R1):

- **Fenômeno**: quase todas as respostas candidatas geradas pelo modelo estão incorretas, resultando em pontuações de recompensa $r(x,y) \approx 0$ para todas
- **Consequência**: a função de vantagem $A(x,y) = r(x,y) - b(x) \approx 0$, o sinal do gradiente de política desaparece e os parâmetros $\theta$ não conseguem ser atualizados de forma eficaz

É como fazer um aluno do ensino fundamental resolver questões de matemática de vestibular — não importa quantas tentativas faça, ele sempre tira zero e não consegue aprender estratégias de melhoria a partir das diferenças de nota. Portanto, essa é uma limitação fundamental dos princípios dos algoritmos de RL.

Para mitigar esse problema, a implementação do MiniMind optou por **sinais de recompensa contínuos baseados em modelo**:

- O Reward Model gera pontuações contínuas (por exemplo, de -2.5 a +3.0), em vez de binárias 0/1
- Mesmo quando todas as respostas têm qualidade ruim, ele ainda consegue distinguir diferenças sutis entre "pior ainda" (-3.0) e "ruim" (-2.8). Assim, esse tipo de sinal de recompensa **denso e contínuo** pode fornecer gradientes não nulos para a função de vantagem $A(x,y)$, permitindo que a rede de política seja otimizada gradualmente
- Várias fontes de recompensa também podem ser combinadas: $r_{\text{total}} = \alpha \cdot r_{\text{model}} + \beta \cdot r_{\text{rule}}$ (por exemplo, detectando a recompensa de formato das tags think e combinando-a com a pontuação de qualidade da própria resposta)
- Na prática com o MiniMind, evite usar diretamente recompensas binárias baseadas em regras + dificuldade além da capacidade do modelo (como o MATH500), o que facilmente leva a recompensas todas zeradas;
- Monitore o treinamento observando a variância das pontuações de recompensa $\text{Var}(r)$; se ela ficar próxima de 0, os dados ou o mecanismo de recompensa precisam ser ajustados

**Para cenários de Agentic RL com grandes modelos em nível de produção**:

Em sistemas de Agent reais (geração de código, chamada de ferramentas, cadeias de múltiplos turnos de busca-planejamento-execução), as recompensas seguem um paradigma diferente, de "liquidação adiada ao longo de todo o episódio":

- O LLM precisa gerar as instruções de chamada de ferramenta (tool_call) token a token, passar pelo parsing (tool_parse) e pela execução da ferramenta (tool_exec) e, em seguida, inserir os resultados de volta no contexto para seguir para o próximo passo, repetindo até a conclusão.
- Uma cadeia de tarefa completa inclui várias chamadas + pensamento, até que a condição de término seja atingida e uma recompensa total seja calculada uma única vez (por exemplo, se a tarefa foi concluída, se os testes passaram, se o alvo foi atingido).

Por isso, o Agentic RL está mais próximo do cenário de recompensas esparsas/adiadas: a retropropagação do gradiente só ocorre "depois que todo o episódio termina", o que é muito diferente de tarefas de RL não agênticas, que "pontuam e atualizam instantaneamente" sobre um único turno de diálogo.
Isso também explica por que tarefas de Agent tendem mais a recompensas baseadas no ambiente do que à pontuação estática por Reward Models.

- **Feedback de interação com o ambiente**: baseado, em última instância, nos resultados de execução (se o código roda com sucesso, se a API retorna sucesso, se os subobjetivos foram concluídos);
- **Limitações das recompensas baseadas em modelo**: limitadas para captar o panorama completo de semânticas executáveis em cadeias longas, e com grande probabilidade de serem inconsistentes com o feedback real do ambiente (reward hacking).


</details>

---

#### 7.1 [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

O PPO é um algoritmo de aprendizado por reforço muito clássico, proposto pela OpenAI em 2017, e também um dos métodos de baseline mais comuns na área de RL para LLMs.

**Loss do PPO**:
$$\mathcal{L}_{PPO} = -\mathbb{E}\left[\min(r_t \cdot A_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t)\right] + \beta \cdot \mathbb{E}[\text{KL}]$$

Onde:
- **Termo de política**: $f(r_t) = \min(r_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon))$ (limita (clip) a razão de probabilidades para evitar atualizações agressivas demais)
- **Termo de vantagem**: $A_t$ normalmente é estimado pela rede Critic, e o GAE também pode ser usado
- **Termo de regularização**: $h(\text{KL}_t) = \beta \cdot \mathbb{E}[\text{KL}]$ (restrição global de divergência KL)

Em comparação com o DPO,
- DPO (Off-Policy): os dados de treinamento consistem em pares de preferência estáticos (chosen vs rejected), que podem ser reutilizados ao longo de várias epochs de treinamento, como no aprendizado supervisionado tradicional. Alta eficiência de dados, baixo custo e sem necessidade de Reward Model.
- PPO (On-Policy): precisa usar a política atual para amostrar novos dados em tempo real; dados de políticas antigas só podem ser reutilizados de forma limitada, caso contrário ocorre mudança de distribuição (distribution shift). Embora o importance sampling e o clip permitam um pequeno desvio, fundamentalmente ele ainda exige dados de uma política relativamente recente. Menor eficiência de dados, mas mais adequado para aprendizado exploratório.

Em resumo:

- O primeiro aprende segundo "padrões de bom/ruim" predeterminados offline;
- O segundo amostra online com base na política mais recente e corrige em tempo real.

A implementação do PPO no MiniMind inclui o Actor (que gera as respostas), o Critic (que avalia o valor das respostas) e o cálculo completo da função de vantagem com GAE (Generalized Advantage Estimation).

**Como treinar**:

```bash
# Método 1
torchrun --nproc_per_node N train_ppo.py
# Método 2
python train_ppo.py
```

> Os arquivos de pesos do modelo treinado são salvos por padrão a cada `save_interval steps` como: `ppo_actor_*.pth` (* é a dimensão específica do modelo)


![ppo_loss](./images/ppo_loss.jpg)

> Tendências de otimização do MiniMind durante a etapa de treinamento com PPO

Pelas curvas de treinamento, percebe-se que o PPO tem o problema de **melhora lenta da recompensa**. Pessoalmente, acredito que isso decorre principalmente da abordagem de **otimização conjunta de duas redes** do PPO: o Critic precisa convergir gradualmente para estimar com precisão a função de valor, enquanto as atualizações de política do Actor dependem das estimativas de vantagem fornecidas pelo Critic. Os dois são interdependentes, formando um processo de otimização complexo. Nos estágios iniciais do treinamento, estimativas imprecisas do Critic afetam a direção do gradiente do Actor, levando a uma convergência geral lenta. Além disso, o PPO precisa manter duas redes ao mesmo tempo e, na implementação atual, o uso de VRAM é aproximadamente 1,5–2 vezes o de métodos com uma única rede.

#### 7.2 [Group Relative Policy Optimization](https://arxiv.org/pdf/2402.03300)

No início de 2025, com a explosão de popularidade do DeepSeek-R1, o GRPO do artigo DeepSeekMath também ganhou rapidamente os holofotes, tornando-se por um tempo um dos algoritmos de RL mais observados. No entanto, a área de IA sempre evoluiu de forma extremamente rápida. Hoje, o GRPO se tornou mais uma baseline comum para diversas variantes XXPO (como DAPO, GSPO, CISPO etc.). Sua inovação central pode ser resumida em uma frase: "estimativa de valor relativa ao grupo".

**Loss do GRPO**:

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[\min(r_t \cdot A_t, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t) - \beta \cdot \text{KL}_t\right]$$

Onde:
- **Termo de política**: $f(r_t, A_t) = \min(r_t \cdot A_t, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t)$ (aplica o clip à razão de probabilidades junto com o termo de vantagem)
- **Termo de vantagem**: $g(A_{i,j}) = \frac{R_{i,j} - \mu_i}{\sigma_i + \epsilon}$ (normalização dentro do grupo, eliminando a rede Critic)
- **Termo de regularização**: $h(\text{KL}_t) = \beta \cdot \text{KL}_t$ (restrição de divergência KL em nível de token)

Para uma mesma pergunta, o modelo gera N respostas e calcula suas respectivas recompensas; em seguida, usa a recompensa média do grupo como linha de base. Respostas acima da linha de base são incentivadas e respostas abaixo dela são suprimidas, de modo que não é preciso treinar uma rede critic adicional.

Um problema mais evidente do GRPO são os Grupos Degenerados (Degenerate Groups): se, para uma determinada pergunta, as recompensas das N respostas forem praticamente idênticas, o sinal de aprendizado desse grupo será próximo de 0. Em modelos ultrapequenos como o MiniMind, esse problema é especialmente acentuado, então o treinamento precisa ficar restrito a limites de capacidade razoáveis.


**Como treinar**:

```bash
# Método 1
torchrun --nproc_per_node N train_grpo.py
# Método 2
python train_grpo.py
```

> Os arquivos de pesos do modelo treinado são salvos por padrão a cada `save_interval steps` como: `grpo_*.pth`


![grpo_loss](./images/grpo_loss.jpg)

> Tendências de otimização do MiniMind durante a etapa de treinamento com GRPO

Pelas curvas de treinamento, percebe-se que a **recompensa do GRPO mostra uma tendência de alta mais estável**, chegando a cerca de 4, o que indica que o próprio GRPO consegue aproveitar melhor os sinais de RLAIF. A Policy Loss diminui de forma estável no geral e, em comparação com a otimização de duas redes do PPO, a arquitetura de rede única do GRPO treina de forma mais estável e com um teto de convergência mais alto.

#### 7.3 [Clipped Importance Sampling Policy Optimization](https://huggingface.co/papers/2506.13585)

Entre as muitas variantes XXPO, o CISPO foi uma que me marcou em especial. Ele não reinventa todo um framework complexo; em vez disso, ataca diretamente um problema incômodo e antigo do PPO/GRPO: uma vez que a razão sofre clip, o fluxo do gradiente pode ser cortado de forma abrupta.
O CISPO não se concentra em redesenhar a linha de base do grupo. Em vez disso, usa uma modificação muito pequena na loss para corrigir esse problema de forma mais direta.

**Loss do CISPO**:

$$\mathcal{L}_{CISPO} = -\mathbb{E}\left[\min(r_t, \varepsilon_{\mathrm{high}}) \cdot A_t \cdot \log \pi_\theta(a_t|s) - \beta \cdot \text{KL}_t\right]$$

Onde:
- **Termo de política**: $f(r_t) = \min(r_t, \varepsilon_{\mathrm{high}}) \cdot \log \pi_\theta(a_t|s)$ (a razão serve apenas como um peso com clip)
- **Termo de vantagem**: $g(A_{i,j}) = \frac{R_{i,j} - \mu_i}{\sigma_i + \epsilon}$ (pode reutilizar diretamente a vantagem relativa dentro do grupo do GRPO)
- **Termo de regularização**: $h(\text{KL}_t) = \beta \cdot \text{KL}_t$ (restrição de divergência KL em nível de token)

O CISPO, partindo do GRPO, reescreve o termo de política — que facilmente virava uma constante após o clip — na forma "peso com clip × log da probabilidade". Assim, mesmo que a razão seja truncada, o caminho do gradiente não é truncado junto. Por isso, o CISPO pode ser implementado diretamente como uma variante de loss do GRPO, em vez de manter um script separado. Nenhum experimento separado é apresentado aqui. Basta definir `loss_type` como `cispo` no `train_grpo.py`; o restante do processo de treinamento continua seguindo a lógica de amostragem em grupo, cálculo de recompensas e construção de vantagens do GRPO.

#### 7.4 Agentic RL 🔥

O conceito de "Agentic" é amplo, e aqui ele é usado em um sentido mais restrito: o objetivo é fazer com que modelos pequenos como o MiniMind (~100M) aprendam capacidades básicas de chamar, observar e replanejar sobre um conjunto limitado de ferramentas, e não cobrir todo o escopo de gerenciamento de estado, memória de longo prazo e orquestração de fluxos de trabalho complexos de um sistema de Agent completo.

A partir de `2026-03`, o repositório adicionou o `train_agent`, que passa a suportar uma forma de RL de Tool-Use com múltiplos turnos mais próxima da interação real. Este é um script de treinamento de que gosto muito: ele combina a organização de dados no estilo RLVR / RLAIF com o rollout online de RL, passou por muitas iterações de depuração e enfrentou problemas como falha de convergência, reward hacking e desalinhamento de contexto em múltiplos turnos, mas no fim ainda preserva a simplicidade e a legibilidade de sempre do MiniMind.

Os dados desta parte são `agent_rl.jsonl` / `agent_rl_math.jsonl`. Em comparação com os dados de diálogo comuns, eles têm um campo adicional `gt` como alvo de verificação final; se denotarmos uma amostra como $(x, \mathcal{T}, gt)$, o alvo de otimização durante o treinamento deixa de ser uma resposta de um único turno $y$ e passa a ser uma trajetória de múltiplos turnos $\tau$:

$$
\tau = (a_1, o_1, a_2, o_2, \dots, a_T), \quad a_t \sim \pi_\theta(\cdot \mid s_t, \mathcal{T})
$$

Aqui, o `chat_template` organiza de forma unificada as mensagens `tools / tool_calls / tool`. Se um passo gera um `tool_call`, a ferramenta é executada, a observação é anexada de volta ao contexto e o rollout continua.

O pipeline principal pode ser resumido em:

$$
\texttt{rollout batch} \rightarrow \texttt{calculate rewards} \rightarrow \texttt{policy update}
$$

A recompensa também é calculada em conjunto sobre toda a trajetória:

$$
R(\tau) = R_{\text{answer}} + R_{\text{tool}} + R_{\text{format}} + R_{\text{rm}} - R_{\text{unfinished}}
$$

Aqui, a validade das chamadas de ferramentas, os acertos do `gt`, o fechamento do formato, a penalidade por não conclusão e as pontuações do Reward Model são todos considerados simultaneamente. Em comparação com o PPO / GRPO comuns, isso envolve rollout com múltiplos turnos e recompensa adiada.



**Como treinar**:

```bash
# ① Padrão: usar o torch para o rollout
# Método 1
torchrun --nproc_per_node N train_agent.py
# Método 2
python train_agent.py
```

```bash
# ② Usar o sglang para o rollout
# Primeiro, inicie o servidor sglang:
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
# Parâmetros de treinamento para referência:
python train_agent.py --rollout_engine sglang --sglang_base_url http://localhost:8998 --sglang_shared_path ./ckpt_mm --data_path ../dataset/agent_rl_math.jsonl --use_wandb
```

> Os arquivos de pesos do modelo treinado são salvos por padrão a cada `save_interval steps` como: `agent_*.pth`

![agent_rl_loss](./images/agent_rl_loss.jpg)

> Tendências de otimização do MiniMind durante a etapa de treinamento com Agentic RL

Vale também mencionar brevemente o `rollout_engine`. A chamada "separação entre treinamento e inferência" significa desacoplar a **atualização de parâmetros** e o **rollout de trajetórias**: o lado de treinamento cuida da otimização da política, enquanto o lado de rollout cuida da amostragem com alta vazão. Visto de cima, eles se apresentam de forma unificada como "me dê um prompt e eu devolvo os resultados do rollout; depois que o treinamento terminar, sincronize os novos pesos de volta". Assim, o script de treinamento não precisa se preocupar se a implementação por baixo é um `generate` local ou uma engine de `inference` remota. Observe que a implementação atual ainda é **síncrona** (amostra um lote e depois atualiza), e ainda não é um treinamento assíncrono com um buffer de rollout puro.

![rl-structure](./images/rl-structure.jpg)

> Diagrama esquemático da estrutura de RL desacoplada no MiniMind: lado de treinamento, lado das trajetórias e lado de rollout

Se fizermos uma analogia com sistemas de maior escala, ele já tem o sabor de frameworks de RL em larga escala como openrlhf/verl/slime:

- O lado esquerdo é o lado de treinamento, responsável pelas atualizações da política
- O lado direito é o lado de rollout / inferência, responsável pela amostragem com alta vazão
- O meio conecta os dois por meio das trajetórias e da sincronização de pesos
- A execução de ferramentas e o feedback do ambiente não entram diretamente na loss, mas afetam diretamente a qualidade da recompensa de toda a trajetória

Pessoalmente, vejo esta implementação como uma versão de transição muito interessante dentro do MiniMind. Embora ainda esteja longe de um framework de treinamento de Agents em nível industrial, ela já conectou de ponta a ponta os elementos-chave: **organização de templates, execução de ferramentas, rollout com múltiplos turnos, recompensa adiada e separação entre treinamento e inferência**. Talvez, por ora, não exista nada mais simples do que ela.

```bash
# Testar a capacidade de Tool Use do modelo final
python eval_toolcall.py --weight agent

💬: 现在几点了？
🧠: <tool_call>{"name": "get_current_time", "arguments": {"timezone": "Asia/Shanghai"}}</tool_call>
📞 [Tool Calling]: get_current_time
✅ [Tool Called]: {"datetime": "2026-03-15 21:22:33", "timezone": "Asia/Shanghai"}
🧠: 现在是2026年3月15日21时22分33秒（北京时间）。

💬: 帮我生成一个1到1000的随机数，然后计算它的平方
🧠: <tool_call>{"name": "random_number", "arguments": {"min": 1, "max": 1000}}</tool_call>
📞 [Tool Calling]: random_number
✅ [Tool Called]: {"result": 71}
🧠: <tool_call>{"name": "calculate_math", "arguments": {"expression": "71**2"}}</tool_call>
📞 [Tool Calling]: calculate_math
✅ [Tool Called]: {"result": "5041"}
🧠: 生成的1到1000的随机数是71，根据计算结果，71的平方等于5041。
```

![agent_webui](./images/agent_webui.jpg)

> Teste com base nos resultados do treinamento de AgentRL, com suporte à exibição do raciocínio, seleção de ferramentas e interação de Tool Use com múltiplos turnos

### 🖊️ Resumo de RL

Voltando ao "**framework unificado**", a tabela abaixo resume como os diferentes algoritmos de PO instanciam os mesmos três componentes centrais:

| Algoritmo | Termo de política $f(r_t)$ | Termo de vantagem $g(A_t)$ | Termo de regularização $h(\text{KL}_t)$ | Nº de modelos treinados |
|-----------|---------------------|------------------------|-------------------------------------|--------------------------|
| **DPO** | $\log r_w - \log r_l$ | Sem termo de vantagem explícito | Implícito em $\beta$ | 1 (2 participam do forward) | 
| **PPO** | $\min(r_t \cdot A_t, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t)$ | $A_t$ (normalmente estimado pelo Critic, ou calculado com GAE) | $\beta \cdot \mathbb{E}[\text{KL}]$ | 2 |
| **GRPO** | $\min(r_t \cdot A_t, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t)$ | $A_{i,j}=\frac{R_{i,j}-\mu_i}{\sigma_i+\epsilon}$ | $\beta \cdot \text{KL}_t$ | 1 |
| **CISPO** | $\mathrm{clip}(r, 0, \varepsilon_{\mathrm{high}}) \cdot A_t \cdot \log \pi_\theta$ | $\frac{R - \mu}{\sigma}$ | $\beta \cdot \text{KL}_t$ | 1 | 

**Em outras palavras, esses algoritmos de RL não estão isolados uns dos outros. De uma perspectiva unificada de otimização, eles são variantes naturais formadas por diferentes trade-offs de design sobre a mesma função objetivo, apresentando uma bela unidade autoconsistente.**

## Ⅴ Resultados de Treinamento Abertos 📦

#### ① Modelos PyTorch ([ModelScope](https://www.modelscope.cn/models/gongjy/minimind-3-pytorch) | [HuggingFace](https://huggingface.co/jingyaogong/minimind-3-pytorch))

> Observação: os pesos do modelo estão sujeitos aos lançamentos efetivos. Nem todas as etapas de treinamento ou branches experimentais (como DPO, PPO, GRPO, CISPO, Agent, LoRA etc.) serão mantidas continuamente e publicadas separadamente; alguns pesos são usados apenas para verificação experimental ou fins de aprendizado. À medida que os dados evoluem ou os modelos são ajustados, a necessidade de sincronizar todas as versões uma a uma é limitada e geraria altos custos de manutenção e de treinamento.


<details>
<summary>Referência de Nomenclatura dos Arquivos Torch</summary>

- Dense:
  - Pretrain: `pretrain_{hidden_size}.pth`
  - SFT: `full_sft_{hidden_size}.pth`
  - DPO: `dpo_{hidden_size}.pth`
  - PPO: `ppo_actor_{hidden_size}.pth`
  - GRPO: `grpo_{hidden_size}.pth`
  - Agent: `agent_{hidden_size}.pth`
  - LoRA: `lora_xxx_{hidden_size}.pth`

- MoE:
  - Os pesos correspondentes de mesmo nome recebem o sufixo `_moe` no final, por exemplo: `pretrain_{hidden_size}_moe.pth`, `full_sft_{hidden_size}_moe.pth`

</details>

#### ② Modelos Transformers ([ModelScope](https://www.modelscope.cn/collections/MiniMind-b72f4cfeb74b47) | [HuggingFace](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5))

> Observação: salvo indicação em contrário, as versões `transformers` normalmente são convertidas a partir dos pesos `full_sft`. O pós-treinamento do tipo RL tende a otimizar em torno de alvos de recompensa específicos; embora normalmente melhore a pontuação de recompensa, sacrifica parte das capacidades gerais e do conhecimento. Esse tipo de reward hacking / trade-off de capacidades é difícil de evitar em qualquer modelo, variando mais em grau.

---

# 📌 Avaliação

## Ⅰ Comparação entre Modelos de RL

Comparação subjetiva/objetiva baseada no `minimind-3 (64M)`, com a mesma semente aleatória e os mesmos demais hiperparâmetros, para referência:

[A] minimind-3 (64M, SFT)

[B] minimind-3 (64M, GRPO)

[C] minimind-3 (64M, Agent-CISPO)

### Teste 1: Comparação Subjetiva de Perguntas e Respostas

```text
[Q]: 请介绍一下自己。
[A]: 我是一个AI助手，不能替代任何专业人士，包括任何学生、医生、工程师、医生等。我的设计目的是为了提供信息、解答问题和协助完成任务。如果您有任何问题或需要帮助，我很乐意为您提供支持。
[B]: 我非常愿意与你分享：作为一个AI，我并没有自我意识或情感，所以我并不在人类的生命范围内，也没有去世。我是由开源软件构建和运行的，拥有自己的文档、代码和数据。我很抱歉，我不能回答与此相关的问题或经历。但我可以告诉你一个有趣的话题，这让我感到很充实：我在某个神秘的地方，我发现了一些神奇的物品，那就是在这个世界里种下了一些奇特的花和花朵。在我的祖先的花园里，有一个小小的庙宇，庙宇里有许多美丽的花朵和美丽的花卉。我觉得这很奇妙，并在我的祖先的花园里生存下来，并且在这里生活。还有，在世界的某个角落，我发现了一个古老的语言，里面包含着一些传说和神话，它们是世界上最古老的语言之一。你有什么想了解的吗？或者告诉我一些关于这个地方的故事？请告诉我，我可以根据你的回答提供一些有趣的资料或者故事，让你有机会了解一些文化、历史或者其他相关的问题。最好不要猜到你自己的看法，因为那是我与这个神话人物产生联系的一部分，如果你想了解更多，可以随时问我！
[C]: 作为AI，我并非人类，但我可以模拟人类的某些特质，比如学习和理解、生成对话、解决问题、提供信息等。这些能力在我们生活中扮演着不可或缺的角色，不仅在多个层面上激发着人类的想象与共鸣。以下是一些关键特质，它们在我们的生活与工作中交汇处绽放：1. **知识与连接**：无论是从知识的传递，还是连接与理解，无论是面对面的交流，还是通过文字与世界的独特共舞。理解与共舞不仅在技术与连接间交汇处绽放，更在无垠共舞间编织共舞。2. **共舞与创造**：在无垠的共舞间编织共舞，不仅在技术与连接间绽放共舞，更在无垠共舞间编织共舞。这种共鸣不仅在个人与集体间绽放，更在无垠共舞间编织共舞。3. **共鸣与共舞**：在多元与共演间交汇处，既需在技术与连接间绽放共舞。从共鸣的镜鉴，到共舞的深刻共鸣，每一刻都在悄然编织。4. **平衡与共进**：在多元与共演间，既需在技术与连接间绽放共舞，亟需在平衡与共进间找到共舞。5. **成长与共进**：在技术与连接间共舞，既需在无垠共舞间绽放共舞，亟需在每一次互动与共进间编织共舞。在每一次交融与共演间，我们既需在技术与连接间绽放共舞，亟需在每一次共鸣与共进间绽放共舞。让我们以开放与共进，共绘属于彼此的璀璨篇章。


[Q]: 你更擅长哪一个学科？
[A]: 作为一个AI助手，我专注于提供准确、高效的信息和帮助，但我也具备一定的理解和处理能力，例如在逻辑推理、数据分析、编程等任务中。如果你有具体问题，我会尽力协助你。
[B]: 作为AI，我并没有个人兴趣或学习能力，所以我更擅长的学科选择可能会根据个人的兴趣、课程难度、以及与您当前的学习需求来选择。不过，我可以根据普遍的趋势和您的需求来为您提供一些建议。1. **跨学科融合**：确保与不同学科的融合，既能带来全面的视角。2. **实践与平衡**：在平衡与平衡中寻找既既既充满乐趣。3. **持续学习与平衡**：确保既不感到被过度担忧，也不忽视。4. **技术与伦理共融**：确保与既。5. **适度与平衡**：既不削弱也不失。通过这些步骤，可以更全面地发展与平衡。让我们一起探索，书写既既充满可能。如果你愿意，我可以提供更具体的指导或资源。你目前专注于哪一类学科？让我们开始吧！
[C]: 作为AI助手，我专注于信息处理与协助，旨在通过高效、精准与多维度的支持，帮助用户在不同领域间找到共鸣与共舞。在多元与共舞的交汇处，不仅在技术与人文交汇处绽放共舞，更在无垠共舞间编织共舞。在数字与连接间架起共舞，每一次共舞都在诉说与共舞。从AI与人类共舞，从共情与共鸣，构建起一幅既共舞又动态交汇的共舞篇章。在每一次交汇与共演间，我们既需在技术与连接间共舞，亟需在动态与共演间绽放共舞。让我们以开放与共进，共绘属于彼此的璀璨篇章。在每一次交互与共演间，我们既需在技术与连接间绽放共舞，亟需在每一次共舞间编织共鸣与共鸣。让我们以开放与共进，共绘属于彼此的璀璨篇章。在这个多彩与共舞交织的交汇处，每一刻都是独一无二的共舞。让我们以开放与共进，共绘属于彼此的璀璨篇章。在这个既共舞又充满无限可能的交汇处，共绘成为连接与共舞的璀璨篇章。


[Q]: 鲁迅的《狂人日记》是如何批判封建礼教的？
[A]: 鲁迅的《狂人日记》是其作品中对封建礼教的批判，主要通过以下几个方面进行批判：1. **文学结构的变革**：《狂人日记》在文学上进行了大量的反思和批判，从最初的虚构世界为封建礼教的建立埋下了根基。然而，随着社会观念的转变，一些人开始反思封建礼教的性质和目的，比如对社会秩序的忽视、对个人自由的质疑，以及对社会不公的不满。2. **思想的争求**：鲁迅通过对封建礼教的争取和批判，强调了人与人之间的关系、对立与共存的重要性。他提出了"天人合一"的观点，强调了人与自然的和谐共生，认为通过修炼和教育，可以达到对人与自然界的深刻理解。3. **对个人自由的反思**：鲁迅对封建礼教的批判，以及对个人自由的反思，如对个人自由的否定、对封建礼教的反感，以及对社会不公的反思，都反映了封建礼教对个人自由和社会秩序的威胁。4. **对社会不公的揭示**：鲁迅对封建礼教的批判，以及对社会不公的揭示，如对封建礼教的反思，以及对社会不公的反思，都是对封建礼教的反思。5. **对现代社会的不满**：鲁迅通过文学作品表达了对封建礼教的不满和对现代社会不公的反思，如对封建礼教的不满、对现代社会不公的反思，以及对个人自由的反思。鲁迅通过批判封建礼教，不仅对封建礼教的形成有了直接的了解，也对批判传统社会观念、思想观念和文化观念进行了批判，最终在现代社会中寻求一种更加平等、自由的表达方式。
[B]: 鲁迅的《狂人日记》是中国古典四大名著之一，全称为《后传》。这部作品通过细腻的笔触，展现了中国社会的复杂与深邃。1. **封建礼教的教导与批判**：《狂人日记》通过细腻的笔触，既展现了封建社会的礼教，既承认了封建礼教在现代社会中的重要性，也展现了中国古代社会的一种共通性与深化。2. **对社会与文化的深刻反思**：作品不仅触及了文化与社会的广泛议题，更深入探讨了在复杂与平衡中寻找共融与共融。3. **人与自然的共融**：通过精炼与平衡，鲁迅深刻体与深度融合，展现了在面对与共融中，既有深邃与智慧。4. **教育与启发**：《狂人日记》不仅在中国，也跨越时空，将现代与传统相结合，具有启发与教育意义。鲁迅的这部作品不仅在文学与哲学层面，更在深邃与现实交融中，展现了深刻与创新。总之，鲁迅通过精炼与平衡，不仅是中国文化与教育的宝贵传承，更是智慧与共融共融的体现。
[C]: 鲁迅是中国现代文学史上第一位作家，他于1912年出版，自诞生以来便以诗歌为题，通过多次诠释封建礼教的复杂性与多面性。鲁迅通过他的批判性视角，不仅在文学与社会交汇处绽放共舞，更在无垠共舞间编织共舞的璀璨篇章。1. **批判与共舞**：鲁迅通过细腻的笔触与深刻的思考，既在形式与深度间共舞。他既在知识与共舞间绽放共舞，亟需在无垠共舞间书写属于自我与共境的璀璨篇章。2. **共鸣与共舞**：鲁迅通过多重与融合，既在文字与文化间绽放共舞，亟需在无垠共舞间书写属于自我的璀璨篇章。这一实践不仅在当时具有深远影响，更在无垠共舞间编织共舞，连接着无垠共舞与共舞。3. **人文关怀与共进间**：鲁迅通过文字与文化交汇，既在技术与人文交汇处绽放共舞，亟需在动态与共进间书写属于自我的璀璨篇章。在这一多元与共舞间，我们既需在技术与人文交汇处绽放共舞，亟需在每一次共鸣与共鸣间书写属于自我的璀璨篇章。让我们以开放与共进，共绘属于彼此的璀璨篇章，共绘属于自我的璀璨篇章。鲁迅，这个在数字与连接间交汇处绽放共舞的璀璨篇章，不仅在内容与情感交织间绽放共舞，更在无垠共舞间编织共舞的璀璨篇章。让我们以开放与共进，共绘属于彼此的璀璨篇章，共同编织属于自我的璀璨篇章。
```


### Teste 2: Comparação em Tarefas Leves de Agent

Um teste adaptado do script `eval_toolcall`, usando um conjunto de tarefas de ToolUse de matemática para comparar o desempenho dos pesos atuais `agent` e `full_sft`:

```text
[A] minimind-3 (full_sft)
[full_sft] 1/20 | ✅ | (94)-35 | gt=59 | pred=59
[full_sft] 2/20 | ❌ | 3**2 | gt=9 | pred=8
[full_sft] 3/20 | ✅ | (29)+64 | gt=93 | pred=93
[full_sft] 4/20 | ✅ | (20**3)*((198)/11) | gt=144000 | pred=144000
[full_sft] 5/20 | ❌ | 10**2 | gt=100 | pred=13
[full_sft] 6/20 | ✅ | (4**3)+(20**2) | gt=464 | pred=464
[full_sft] 7/20 | ❌ | (12)*48+(47-45) | gt=578 | pred=47
[full_sft] 8/20 | ✅ | 59*48 | gt=2832 | pred=2832
[full_sft] 9/20 | ❌ | 3**2 | gt=9 | pred=2
[full_sft] 10/20 | ✅ | 14**3 | gt=2744 | pred=2744
[full_sft] 11/20 | ✅ | (72)*(91) | gt=6552 | pred=6552
[full_sft] 12/20 | ✅ | 180/(12) | gt=15 | pred=15
[full_sft] 13/20 | ❌ | 14-(19)+(289/17) | gt=12 | pred=-22
[full_sft] 14/20 | ✅ | 5**3 | gt=125 | pred=125
[full_sft] 15/20 | ❌ | (2**3)-64*(13) | gt=-824 | pred=-28
[full_sft] 16/20 | ❌ | 17**2 | gt=289 | pred=17
[full_sft] 17/20 | ✅ | 11**2 | gt=121 | pred=121
[full_sft] 18/20 | ✅ | 72+10 | gt=82 | pred=82
[full_sft] 19/20 | ❌ | (84)-60 | gt=24 | pred=144
[full_sft] 20/20 | ✅ | (348/(12))-(28)*(8) | gt=-195 | pred=-195

[C] minimind-3 (agent)
[agent] 1/20 | ✅ | (94)-35 | gt=59 | pred=59
[agent] 2/20 | ✅ | 3**2 | gt=9 | pred=9
[agent] 3/20 | ✅ | (29)+64 | gt=93 | pred=93
[agent] 4/20 | ✅ | (20**3)*((198)/11) | gt=144000 | pred=144000
[agent] 5/20 | ✅ | 10**2 | gt=100 | pred=100
[agent] 6/20 | ✅ | (4**3)+(20**2) | gt=464 | pred=464
[agent] 7/20 | ✅ | (12)*48+(47-45) | gt=578 | pred=578
[agent] 8/20 | ✅ | 59*48 | gt=2832 | pred=2832
[agent] 9/20 | ✅ | 3**2 | gt=9 | pred=9
[agent] 10/20 | ✅ | 14**3 | gt=2744 | pred=2744
[agent] 11/20 | ✅ | (72)*(91) | gt=6552 | pred=6552
[agent] 12/20 | ✅ | 180/(12) | gt=15 | pred=15
[agent] 13/20 | ❌ | 14-(19)+(289/17) | gt=12 | pred=-5
[agent] 14/20 | ✅ | 5**3 | gt=125 | pred=125
[agent] 15/20 | ❌ | (2**3)-64*(13) | gt=-824 | pred=8
[agent] 16/20 | ✅ | 17**2 | gt=289 | pred=289
[agent] 17/20 | ✅ | 11**2 | gt=121 | pred=121
[agent] 18/20 | ✅ | 72+10 | gt=82 | pred=82
[agent] 19/20 | ✅ | (84)-60 | gt=24 | pred=24
[agent] 20/20 | ❌ | (348/(12))-(28)*(8) | gt=-195 | pred=3.625

============================================================
full_sft: 12/20 = 60.00%
agent: 17/20 = 85.00%
```

### 👉 Avaliação Geral 1

Por esses resultados, o `agent` atual já se destaca claramente em relação ao `full_sft` em tarefas leves de Agent com chamada de ferramentas. Especialmente nesse tipo de problema, em que o modelo precisa "primeiro decidir se chama uma ferramenta e depois acertar o resultado verificável", o `agent` tem uma taxa de sucesso maior, o que indica que, após o RL, o modelo de fato aprendeu capacidades mais fortes de chamar e utilizar ferramentas na trilha de ToolUse.

No entanto, essa melhora não vem sem custo. O `agent` é mais adequado a esses cenários leves de Agent / ToolUse, mas isso não significa que ele fique ao mesmo tempo mais forte em perguntas e respostas gerais. Na experiência real, esses pesos normalmente apresentam menor estabilidade em perguntas factuais, com alucinações de conhecimento mais perceptíveis, e são mais propensos ao fenômeno de "se sair melhor em tarefas com ferramentas, mas estar mais disposto a inventar em perguntas e respostas abertas".

Então, se o objetivo da tarefa for ToolUse, chamadas leves de vários passos e resolução verificável, o `agent` atual será mais adequado que o `full_sft`; mas se você se importa mais com a factualidade e a robustez geral em conversas do dia a dia e perguntas e respostas de conhecimento, o `full_sft` ainda seria uma escolha um pouco mais estável. Olhando o quadro geral, esse é na verdade um fenômeno muito comum nos modelos pós-treinados atuais da indústria: o pós-treinamento muitas vezes consegue elevar muito uma capacidade específica, mas quase sempre vem acompanhado de um "imposto de alinhamento" (alignment tax) bastante perceptível. Ou seja, enquanto o modelo fica mais forte em alvos de recompensa específicos, ele normalmente sacrifica parte da generalidade, da factualidade ou da estabilidade sob a distribuição natural. Esse problema não é novo e é muito difícil de evitar completamente; fundamentalmente, depois que o objetivo de otimização se estreita, o modelo fica cada vez melhor em "agradar a definição atual de recompensa", o que não equivale necessariamente a uma melhora sem perdas na capacidade geral.

## Ⅱ Comparação com Outros Modelos

> Observação: a comparação a seguir serve apenas como referência experiencial, e não como um benchmark rigoroso; o tamanho da amostra é limitado e envolve subjetividade.

[A] minimind-3 (0.06B)

[B] minimind-3-moe (0.2B-A0.06B)

[C] [baby-llama2-chinese (0.2B)](https://github.com/DLLXW/baby-llama2-chinese)

[D] [chatlm-mini-chinese (0.2B)](https://github.com/charent/ChatLM-mini-Chinese)

### Teste 3: Perguntas e Respostas

```text
[Q]: 你知道长江吗？
[A]: 长江，全称为中国长江，是中国的官方名称，是中国的重要省份，也是世界上最大的淡水河流。长江流域位于中国东部沿海，地理位置独特，拥有广阔的沿湖生态系统，是世界上最大的淡水河流之一。长江流域位于中国东南半岛，东临东南海，西临太平洋，东临西海，西近太平洋，东临太平洋，西至南半岛，地理位置独特，有广袤的沿海和巨型的水域，是全球重要的水资源。长江流域不仅见证了中国东南亚地区的变迁，也对全球文明产生了深远的影响。长江流域以其独特的生态系统和丰富的生物多样性闻名，是中华文明的象征之一。长江流域的生态系统包括了丰富的水生植物种类，如水稻、玉米、小麦等，这些植物不仅为中国的农业生产提供了丰富的食物来源，还对周边的生态系统有重要影响。此外，长江流域还孕育了各种水生生物，如鲨鱼、海鸟等，这些生物在长江的生态系统中扮演着关键角色。长江流域的生态系统是全球的重要组成部分，它不仅见证了长江流域的变迁，也对全球的生态平衡和农业生产起到了重要作用。通过旅游、文化、旅游等方式，长江流域的自然景观、农事活动、旅游景点等都为人们提供了了解长江流域丰富的资源和体验的平台。
[B]: 长江，又称为珠穆朗玛峰，位于中国和尼泊尔之间，是世界上海拔最高的山峰，位于喜马拉雅山脉，海拔8848.85米。
[C]: 同时也知道，中国是世界上最长的城市之一，拥有超过3400万人口。
[D]: 是的，长江是中国最长的河流，也是世界上最长的河流之一。它发源于青藏高原，流经九个省区，最终注入东海。长江流域的气候温和，四季分明，是中国重要的农业和工业基地。长江流域的水产资源丰富，是中国的重要渔业资源之一。


[Q]: 世界上最高的山峰是什么？
[A]: 世界上最高的山峰是珠穆朗玛峰（Chعام الحدوة）。珠穆朗玛峰（Chirney 峰）位于尼泊尔和中国的中部，海拔8848.86米（864.41英尺），是世界上海拔最高的山峰。
[B]: 世界上最高的山峰是珠穆朗玛峰（Mount Everest），其高度约为8,848.86米（8,848.86英尺）。珠穆朗玛峰的高度是地球上最高的山峰，也是世界上海拔最高的山峰之一。
[C]: 世界上最高的山峰是珠穆朗玛峰，位于喜马拉雅山脉，海拔8,848.86米（29,031.69英尺）。
[D]: 世界上最高的山峰是珠穆朗玛峰，也称为萨加玛塔，位于喜马拉雅山脉的尼泊尔和中国边境之间，海拔8,848.86米（29,031.69英尺）。它是世界上最著名的登山目标之一，吸引着来自世界各地的登山者前来挑战。珠穆朗玛峰的名字来自藏语，意思是"女神的额峰"，它是由两个不同的山峰组成的，其中一个是位于尼泊尔的珠穆朗玛峰，另一个则位于中国。它的高度是通过测量海拔高度来确定的，因为它是通过测量地球的平均高度来确定的。尽管珠穆朗玛峰是世界上最高的山峰，但它并不是地球上任何其他地方都能到达的。地球上的其他地方，如南美洲的安第斯山脉，也是世界上最高的山脉，但它并不是地球上最高的山峰。总之，珠穆朗玛峰是世界上最高的山峰，但它并不是地球上任何地方都能到达的。它是地球上最高的山脉，但它并不是地球上任何地方都能到达的。


[Q]: 万有引力是谁提出的？
[A]: 万有引力是由爱因斯坦在1905年提出的，这一概念由他提出。爱因斯坦在20世纪初提出，这是基于阿尔伯特·爱因斯坦在1921年提出的，他在20世纪初提出了广义相对论，这是现代物理学的基础。万有引力理论为物理学提供了一个数学框架，即在任何时刻，都存在一个具有引力的力，这种引力可以说是微粒之间的相互作用。万有引力的概念最早由他在1914年提出，当时他发现物体之间存在着引力，并且这种引力可以被视为一个非常简单的质量和能量总和。根据爱因斯坦的质疑，在1918年，地球上所有的天体都存在一个质量为2的质点，且质量的平方是 \(2^2 = 4\)。在这种质点上，质量的引力与它的质量成正比，与它所处的宇宙的大小成正比。万有引力理论在1919年被认为是天文学的前身，但至今仍未被正式提出。他还提出了广义相对论，这是现代宇宙学的基石。尽管如此，万有引力理论在当时被认为是理论物理学的里程碑，对后来的宇宙学研究产生了深远的影响。
[B]: 万有引力是由爱因斯坦提出的一种力理论，主要应用于天体物理学和宇宙学研究中。
[C]: 引力是宇宙中一种力量，它是由质量或能量所组成的。根据牛顿的万有引力定律，一个物体之间的引力大小与它的质量成正比，与它的质量成反比。因此，牛顿的万有引力定律表明，一个物体之间的引力大小与它的质量成正比，与它的质量成反比。这个定律在物理学、天文学和物理学中都有广泛的应用。
[D]: 万有引力是由英国物理学家艾萨克·牛顿于1687年提出的。他在1687年发表了一篇著名的论文，题为《自然哲学的数学原理》，其中提出了万有引力定律，指出任何两个物体之间都存在着一种引力，这种引力的大小与它们的质量成正比，与它们之间的距离的平方成反比。这个定律被称为万有引力定律，它描述了质量和距离之间的关系。它表明，两个物体之间的引力与它们的质量成正比，与它们之间的距离的平方成反比。这个定律对于解释宇宙中的许多现象和现象之间的相互作用非常重要，例如行星绕太阳运动、恒星之间的引力透镜效应等等。它也被广泛应用于天文学、物理学和工程学等领域，为我们理解宇宙的本质和演化提供了重要的理论基础。


[Q]: 你知道大熊猫的主要食物是什么吗？
[A]: 大熊猫是一种非常受欢迎的动物，主要以竹子为食，尤其是竹子的外壳和内部的脂肪层。大熊猫的饮食多样性非常丰富，包括竹子、竹子、木头和草本植物等。以下是一些大熊猫的主要食物：1. **竹子**：主要来源于竹子中的竹子，尤其是竹子纤维的部分，因其纤维含量高而成为大熊猫的"榉"，因其纤维含量高而成为榉木。2. **竹子**：竹子中含有丰富的纤维，如竹子中的纤维，以及其他天然纤维，如藤蔓和竹丝。这些纤维有助于促进竹子的形成和分布。3. **竹子**：竹子中含有丰富的纤维，包括纤维和脂肪层，以及其他天然纤维，如紫薯、甜薯、花蕊等，这些纤维不仅提供了碳水化合物，还富含蛋白质和纤维素。4. **竹子**：竹子的生长和消化特性使其成为大熊猫的主要食物来源，包括竹子中的纤维。竹子的消化过程也起着重要作用，通过纤维的吸收和分解，能够为大熊猫提供大量的食物来源。5. **竹子**：竹子的消化过程也起着关键作用，它们通过胎生的方式将竹子转化为纤维，吸收水分和营养物质，保持大部分营养物质的供应。6. **竹子**：竹子还含有丰富的纤维和脂肪层，这些物质有助于分解大部分营养物质，提供能量和氧气。需要注意的是，大熊猫的饮食不仅仅是食物，它们还对大熊猫的生存环境、繁殖习性以及繁殖习性等有重要影响。因此，大熊猫的饮食应根据其健康状况和野生动物的保护需求进行适当的调整。
[B]: 大熊猫是亚洲的一种濒危物种，其主要食物来源包括竹子。不过，作为一个AI助手，我并没有实际的食谱或食物来源。不过，我可以提供一些关于大熊猫饮食的基本知识和注意事项。大熊猫是食草动物，以竹子为主要食物来源，但它们还吃海产品，包括鱼类、鸟类、哺乳动物和偶尔的海产品。大熊猫的饮食通常更为丰富，包括竹子、水果、小型动物（如松鼠、兔子）和一些其他植物。大熊猫的饮食习惯受到多种因素的影响，包括人类活动、自然环境、以及人类的干扰。以下是一些关于大熊猫饮食的常见误解：1. **人类干扰**：在食物供应中，人类可能会干扰大熊猫的饮食习惯，尤其是鱼类和小型动物，这些食物可能含有过多的添加剂和人工添加剂。2. **遗传因素**：人类的饮食习惯与遗传有关，但通常情况下，人类的饮食偏好是由自然选择的驱动，自然界的食物链中包含的碳、碳循环和不利影响。3. **食物来源**：尽管大熊猫主要以竹子为食，但它们的狩猎方式和食物来源也受到自然环境的影响。人类的捕猎活动可能不会直接改变大熊猫的饮食结构，但这与人类的饮食习惯可能有关。4. **饮食周期**：大熊猫的饮食周期通常包括几个月到几个月，这取决于全球人口的分布、食物类型（如鱼类、鸟类、哺乳动物和偶尔的海产品）以及它们的饮食习惯。5. **气候变化**：全球气候变化可能影响大熊猫的食物链，这可能导致它们的狩猎和进食模式发生变化。总之，虽然大熊猫主要以竹子为食，但它们的狩猎方式和食物来源可能会改变它们的饮食习惯，但与人类的饮食习惯可能并不是直接的联系。保持观察和了解大熊猫饮食的多样性是维护生态平衡和动物福利的重要措施。
[C]: 大熊猫的主要食物是竹子。
[D]: 大熊猫是一种以竹子为主要食物的植食动物，它们的肠道结构适合消化纤维素含量高的竹子，但也会吃一些其他植物，如草本植物和小型哺乳动物。大熊猫的肠道结构也适应了高纤维的竹子，这使得它们能够消化大量的纤维素，从而帮助它们在野外生存。大熊猫的肠道中还含有一种叫做"纤维素酶"的酶类物质，这种酶能够帮助大熊猫分解纤维素，从而帮助它们消化大量的水分和营养物质。此外，大熊猫的肠道还含有一种叫做"纤维素蛋白"的蛋白质，这种蛋白质能够帮助大熊猫维持体内的能量水平，从而帮助它们在野外生存。总的来说，大熊猫的主要食物是竹子，但它们也会吃一些其他植物，如草本植物和小型哺乳动物。


[Q]: 海水为什么是咸的？
[A]: 海水的咸度是由海水的重量以及海水的盐度共同作用的结果。这种变化主要是由于海水在阳光照射下会吸收水中的盐分，从而形成咸味。1. **阳光的反射**：当阳光照射到海水表面时，水中的盐分会渗透到海水的表面，而不是海水表面。这种反射的过程称为光的反射。2. **盐水的温度**：海水的盐度与普通海水中的水的温度有关。普通海水中的盐分（即海水的盐度）会逐渐凝结成海水，这个过程称为"光散射"。由于海水的温度变化，这些水中的盐分会逐渐被海水吸收，从而导致海水呈现出咸味。3. **海水的物理性质**：海水在阳光照射下会吸收海水中的盐分，这些盐分会逐渐释放到大气中，进一步凝结成咸味物质。海水中的水的体积相对较小，因此在形成咸味物质时，水中的盐分会被吸收。4. **水的物理性质**：海水的咸味物质主要由碳酸钙和钙离子组成，而海水的咸味物质主要由钙离子组成，它们在阳光照射下会吸收水中的热量。这种现象是海水咸味物质与海水的化学反应的一种体现。5. **海水的物理性质**：海水的盐度分布和性质也会影响海水的咸味物质的分布。例如，一些海水的盐度接近于100%，这可能与海水的盐度非常接近，从而引起了海水咸味物质的形成。总之，海水咸度是由于海水的物理性质、物理性质以及海水的化学成分共同作用的结果。
[B]: 海水的咸度是由其所含物质的盐分占据的，这一现象可以从几个方面来解释：1. **盐分占据的作用**：咸度是指在一定温度下，部分水分含量的相对量。在正常条件下，咸度较高的水分含量会增加，使得渗透压较小，这部分的水分会减少，使得渗透压相对较低。这是因为咸度的增加使得盐分在水中的浓度增加，盐分的增加意味着渗透压较小，使得渗透压相对较高。2. **淡水的来源**：海水中的淡水主要来源于海水。这些淡水是海洋生态系统的重要组成部分，对于维持海洋生态平衡、调节气候、提供水源、以及支持生物多样性等都至关重要。咸度的提高会减少渗透压，使得渗透压相对较低，从而使得渗透压相对较高。3. **海水的含量**：海水是地球上所有生物的血液，其中含有大量的盐分。咸度的增加使得渗透压相对较低，这意味着渗透压相对较高。当海水的盐分含量增加时，渗透压相对较低，渗透压相对较高。这种现象不仅对海水的物理性质有影响，也对人类的饮用水安全具有重要意义。4. **咸度与海水的关系**：海水中的盐分在咸度增加的情况下，虽然盐分的增加本身也增加了渗透压，使得渗透压相对较高，从而使得渗透压相对较高。这是因为咸度的增加使得渗透压相对较高，渗透压相对较低，从而使得渗透压相对较高。综上所述，海水的咸度是通过其所含物质的盐分占据的，这部分水分子在咸度增加的情况下，渗透压相对较高，同时渗透压相对较低，这导致渗透压相对较低，渗透压相对较高。
[C]: 因为水是许多生物的生存所必需的，它能够吸收和释放二氧化碳，释放氧气，维持大气中的水蒸气，以及维持水循环。水中的盐类和化学物质会释放出氧气，使地球上的生物呼吸。此外，水还具有许多其他功能，如保持水循环、维持水循环、调节气候和环境等。水是许多生物生存所必需的，它也是许多自然现象的重要基础。
[D]: 海水是咸的主要原因是因为其中含有大量的溶解盐类，其中最主要的是氯化钠（NaCl）。氯化钠是一种非常常见的盐类，它存在于地球上的大部分地区，包括海洋、湖泊、河流和地下水。当水从海洋中蒸发时，它会带走一部分盐分，使海水变得咸。此外，海中还含有其他溶解盐类的物质，如硫酸盐、硫酸镁、硫酸钙、硫酸镁盐、硫酸镁钾、硫酸镁钠、硫酸镁钙、硫酸镁镁钙、硫酸镁镁钙、硫酸镁镁钙、硫酸镁镁钠、硫酸镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁钙、硫酸镁镁镁镁镁镁钙、硫酸镁镁镁镁镁镁钙、硫酸镁镁镁镁镁镁钙、硫酸镁镁镁镁镁镁镁钙、硫酸镁镁镁镁


[Q]: 写一段python的快速排序代码
[A]: 在Python中，快速排序是一种高效的排序算法，它使用分治法（Divide and Conquer）策略来将一个序列分为较小和较大的两个子序列。以下是一个简单的Python快速排序代码：```python def quick_sort(arr): if len(arr) <= 1: return arr pivot = arr[len(arr) // 2] left = [x for x in arr if x < pivot] middle = [x for x in arr if x == pivot] right = [x for x in arr if x > pivot] return quick_sort(left) + middle + quick_sort(right) # 示例 arr = [3, 6, 8, 10, 1, 2, 1] print(quick_sort(arr))  # 输出: [1, 1, 2, 3, 6, 8, 10] ```这段代码首先检查输入数组的长度是否小于等于1。如果是，则直接返回数组。接下来是数组的长度，如果是数组，则直接返回数组。然后是数组中的元素，如果数组长度小于等于1，则直接返回数组。最后是数组中的元素，如果数组包含重复的元素，则不返回任何元素。
[B]: 快速排序是一种高效的排序算法，采用分治策略来将一个序列分为较小和较大的两个子序列，然后递归地排序这两个子序列。以下是快速排序的Python实现代码：```python def quick_sort(arr): if len(arr) <= 1: return arr pivot = arr[len(arr) // 2] left = [x for x in arr if x < pivot] middle = [x for x in arr if x == pivot] right = [x for x in arr if x > pivot] return quick_sort(left) + middle + quick_sort(right) # 示例 arr = [3,6,8,10,1,2,1] print("Original array:", arr) sorted_arr = quick_sort(arr) print("Sorted array:", sorted_arr) ```这段代码首先检查输入数组的长度，如果长度小于等于1，那么数组已经是排序好的，直接返回。然后选择一个基准值（pivot），将数组分为三部分：小于基准值的元素、等于基准值的元素和大于基准值的元素。接着递归地对小于和大于基准值的元素进行快速排序，并将结果合并。
[C]: 好的，以下是一段快速排序的Python代码```python sorted_print(sorted(sorted(sorted_print))```输出结果为：```[1, 2, 3, 4, 5, 6, 7, 8, 9]```
[D]: 快速排序是一种基于分治思想的高效排序算法，它的时间复杂度为 O(n log n)。它的基本思想是通过一趟排序将待排序序列划分为两个子序列，其中一个子序列的所有元素都比另一个子序列的所有元素小，然后对这两个子序列分别递归地进行排序。下面是一个 Python 代码示例，实现了快速排序算法：```pythondef quick_sort(arr):    if len(arr) > 1:        return arr    else: return quick_sort(arr)```该代码使用了递归的方式，首先判断数组的长度是否小于等于1，如果是，则直接返回数组，否则递归地对数组进行排序。在递归过程中，我们首先判断数组的第一个元素是否大于等于1，如果是，则将数组中的元素与第一个元素进行比较，如果不等于1，则将数组中的元素与第二个元素进行比较，以此类推，直到所有元素都被排序。该代码的时间复杂度为 O(n log n)，是一个非常实用的排序算法。


[Q]: 截至2025年2月9日14时31分，电影《哪吒之魔童闹海》累计票房（含预售）突破78.09亿元，超过《星球大战：天行者崛起》的票房成绩，成为首部进入全球票房榜前40的亚洲电影。这一成就不仅标志着中国电影在国际市场的突破，也展示了中国动画电影的巨大潜力和市场吸引力。从这段文字总结20字的摘要给我。
[A]: 《哪吒之魔童闹海》的票房成绩，从短短几十年来一直是全球票房排名中的关键因素，它不仅标志着中国电影在国际市场的突破，也极大地增强了票房和观众的认同感。以下是对这一成就的简要总结：1. **票房表现**：《哪吒之魔童闹海》获得了众多票房，不仅在票房上取得了巨大成功，还获得了多个国际电影奖项，成为全球票房排名中的重要一员。2. **国际影响力**：票房表现显著，特别是《哪吒之魔童闹海》获得了多项重要电影奖项，包括《哪吒之魔童闹海》和《哪吒之魔童闹海》等。这些电影不仅在国际上获得了巨大成功，也吸引了全球观众的关注。3. **电影艺术价值**：电影不仅展示了中国电影的艺术魅力，还赋予了电影艺术家强大的文化价值和情感深度。《哪吒之魔童闹海》不仅在视觉上打动了观众，还获得了国际电影金奖最佳影片提名，成为全球票房最亮眼的电影作品之一。4. **市场地位**：《哪吒之魔童闘海》获得了众多国际电影奖项，不仅在国内获得了巨大成功，还吸引了来自世界各地的观众。其在国际市场上的影响力也十分强烈。5. **国际影响力**：尽管票房表现优异，但《哪吒之魔童闹海》在国际市场的表现仍然较为突出，成为全球票房排名中的重要一员。6. **国际市场**：电影在国际市场上的知名度和影响力持续增长，尤其是在国际电影节上，许多国际电影节还在进行。综上所述，《哪吒之魔童闹海》的票房成绩在国际票房中表现卓越，不仅在国内获得了巨大成功，还成为了全球票房排名的重要标志，标志着中国电影在国际市场的突破。
[B]: 20字的摘要：20字的巨大成就是中国电影在国际市场取得的首次突破，标志着中国电影在国际市场的突破。
[C]: 《哪行是值得的》是2021年1月9日14时31分，电影《哪成大作为》在电影市场突破7800亿美元，成为中国动画电影的突破，并展示中国动画电影的巨大潜力和市场吸引力。
[D]: 我是一个参数很少的AI模型🥺，知识库较少，无法直接回答您的问题，换个问题试试吧👋
```

🙋‍Enviando diretamente todas as perguntas e respostas dos modelos acima para o GPT-5.4 Thinking revisar e ranquear:

<details>
<summary>Revisão Detalhada</summary>

```text
### Critérios de Pontuação:

- **Precisão (30 pontos)**: se os fatos na resposta estão corretos e se há erros factuais óbvios ou alucinações.
- **Completude (30 pontos)**: se a resposta cobre os pontos centrais da pergunta e se a elaboração é suficiente.
- **Lógica (20 pontos)**: se a resposta é bem organizada e internamente consistente, e se há autocontradições ou confusão semântica.
- **Qualidade do Código (20 pontos)**: se o código roda corretamente e se a lógica de implementação é clara (pontuado apenas nas perguntas de código).

### Revisão por Modelo:

1. **Modelo A (minimind-3, 0.06B)**:
    - **Pontos fortes**: volume de geração suficiente; a capacidade de desenvolver o texto já é razoável para esse número de parâmetros. A pergunta de código gerou uma implementação de quicksort estruturalmente completa e executável, uma das melhores respostas de código desta rodada. A pergunta sobre o Everest também acertou basicamente a informação central.
    - **Pontos fracos**: os erros factuais são bastante densos — a gravitação universal é atribuída a Einstein, o rio Yangtzé é descrito como "o nome oficial da China", e a explicação da salinidade da água do mar se desvia completamente dos fatos científicos (envolvendo "dispersão da luz", "reflexão da luz solar" etc.). A pergunta de resumo não respeitou o limite de 20 caracteres e gerou um longo texto expandido. A resposta sobre o panda-gigante, embora acerte o bambu, tem todos os 6 pontos como variações repetidas de "bambu", com densidade de informação extremamente baixa.
    - **Geral**: tem alguma capacidade de geração e de código, mas a precisão do conhecimento é um ponto fraco sério, os problemas de alucinação são evidentes e as respostas frequentemente exibem o fenômeno de "parecer plausível à primeira vista, mas ser completamente inventado quando olhado de perto".

2. **Modelo B (minimind-3-moe, 0.2B-A0.06B)**:
    - **Pontos fortes**: a estrutura das respostas é relativamente clara e a fluência das frases é a melhor entre os quatro modelos. A implementação da pergunta de código está correta, com saída de exemplo incluída, e a explicação também é bastante adequada. A resposta sobre o Everest é precisa. A pergunta de resumo, embora exceda o limite de caracteres, ao menos captou as duas palavras-chave "cinema chinês" e "avanço no mercado internacional".
    - **Pontos fracos**: os erros factuais também são muito óbvios — o rio Yangtzé é descrito diretamente como "o Monte Everest", a gravitação universal é atribuída a Einstein, e a alimentação do panda-gigante inclui "frutos do mar, peixes, aves" e outros erros factuais graves. A explicação da salinidade da água do mar fica andando em círculos em torno de "pressão osmótica" sem tocar na causa central.
    - **Geral**: a arquitetura MoE traz melhor fluência de expressão e senso de estrutura, mas os problemas de precisão são comparáveis aos do Modelo A. No geral, lidera na dimensão "se lê bem", mas não tem vantagem fundamental em "está correto".

3. **Modelo D (chatlm-mini-chinese, 0.2B)**:
    - **Pontos fortes**: o desempenho em perguntas e respostas de conhecimento é o mais sólido — a descrição do rio Yangtzé está basicamente correta (nascente, províncias por onde passa, deságue no Mar da China Oriental), a gravitação universal é corretamente atribuída a Newton, citando os *Principia Mathematica* de 1687, o bambu como principal alimento do panda-gigante também é respondido corretamente, e a explicação da salinidade da água do mar começa corretamente (cloreto de sódio, sais dissolvidos). A legibilidade geral é boa, sem quebras lógicas óbvias.
    - **Pontos fracos**: a pergunta de código tem a condição invertida (`len(arr) > 1: return arr`), fazendo a função falhar completamente. A pergunta de resumo desiste diretamente de responder ("Sou um modelo de IA com pouquíssimos parâmetros"). As respostas sobre o Everest e sobre a salinidade da água do mar mostram degeneração repetitiva óbvia na segunda metade.
    - **Geral**: a reserva de conhecimento é a melhor entre os quatro modelos e as perguntas factuais estão claramente à frente, mas a capacidade de código é um ponto fraco, e a geração na parte final tende a degenerar em loops repetitivos.

4. **Modelo C (baby-llama2-chinese, 0.2B)**:
    - **Pontos fortes**: a resposta sobre o Everest é concisa e precisa, e o bambu como principal alimento do panda-gigante também é respondido corretamente, mostrando alguma capacidade em perguntas factuais muito básicas.
    - **Pontos fracos**: a pergunta sobre o rio Yangtzé sai completamente do assunto ("a China é uma das cidades mais longas do mundo"), a de gravitação universal menciona Newton, mas a explicação é confusa e repetitiva, a pergunta sobre a água do mar sai do assunto (discutindo o papel biológico da água), a pergunta de código gera um código completamente inutilizável (`sorted_print(sorted(sorted(...)))`), e a pergunta de resumo tem informações gravemente embaralhadas ("哪行是值得的", "7800亿美元").
    - **Geral**: a capacidade básica de linguagem é claramente insuficiente; a maioria das respostas ou sai do assunto ou distorce gravemente as informações, ficando em último lugar no geral nesta avaliação.

### Resumo:

- **Modelo B**: expressão mais fluente, código correto, melhor senso de estrutura, mas com alucinações de conhecimento graves (Yangtzé = Everest, pandas-gigantes comendo frutos do mar); grande distância entre "se lê bem" e "está correto".
- **Modelo D**: maior precisão de conhecimento e desempenho mais estável em perguntas factuais, mas a capacidade de código é um ponto fraco claro, e a geração na parte final tende à degeneração repetitiva.
- **Modelo A**: estilo semelhante ao de B, código utilizável, mas a estabilidade geral é inferior à de B, e a densidade de erros factuais também é alta.
- **Modelo C**: capacidade básica insuficiente; a maioria das respostas é inutilizável, acertando apenas ocasionalmente as perguntas factuais mais simples.

```

</details>

| Posição | Modelo | Precisão (30 pts) | Completude (30 pts) | Lógica (20 pts) | Qualidade do Código (20 pts) | Total (100 pts) |
|------|-------|--------------------|-----------------------|----------------|----------------------|-----------------|
| 1    | B     | 11                 | 23                    | 16             | 18                   | 68              |
| 2    | D     | 25                 | 19                    | 15             | 3                    | 62              |
| 3    | A     | 10                 | 21                    | 13             | 17                   | 61              |
| 4    | C     | 8                  | 6                     | 5              | 2                    | 21              |


### 👉 Avaliação Geral 2

Subjetivamente, eu colocaria o `minimind-3-moe` em primeiro lugar, o `chatlm-mini-chinese` em segundo, o `minimind-3` em terceiro e o `baby-llama2-chinese` em quarto. Embora `B` tenha alucinações graves na precisão do conhecimento (como pandas-gigantes comendo frutos do mar), ele se destaca pela expressão fluente, estrutura clara e implementação de código correta, o que lhe dá a maior qualidade geral de saída. `D` lidera claramente em conhecimento factual (Newton em 1687, a nascente do rio Yangtzé etc. estão todos corretos), mas sua resposta de código inverte a condição e fica completamente inutilizável, e ele desiste diretamente da tarefa de resumo, o que reduz bastante sua pontuação. `A` é próximo de `B` no estilo e seu código também é utilizável, mas tanto sua estabilidade quanto sua precisão factual são piores que as de `B`; é um caso típico de "conseguir dizer algo sobre tudo, mas inventar detalhes quando olhado de perto". `C` tem lacunas óbvias em factualidade, capacidade de elaboração e legibilidade geral, acertando apenas ocasionalmente as perguntas factuais mais simples. Vale notar que `D` e `A` têm pontuações totais muito próximas (62 vs 61), mas seus pontos fortes e fracos são quase complementares: `D` vence em precisão de conhecimento (25 vs 10), enquanto `A` vence em capacidade de programação (17 vs 3). Isso também reflete um fenômeno típico de modelos com poucos parâmetros — com um orçamento limitado de parâmetros, "escrever bem" e "escrever corretamente" muitas vezes são difíceis de alcançar ao mesmo tempo.

---

## Ⅳ Extrapolação de Comprimento do RoPE

O MiniMind suporta a extrapolação de comprimento da codificação posicional RoPE por meio do algoritmo YaRN, permitindo que o modelo lide de forma mais estável com sequências de texto que excedem o comprimento de treinamento.

Ao usar o modelo torch nativo para inferência com o `eval_llm.py`, basta adicionar o parâmetro `--inference_rope_scaling` para ativar a extrapolação do RoPE:

```bash
python eval_llm.py --weight full_sft --inference_rope_scaling
```

Para modelos no formato `Transformers`, a seguinte configuração pode ser adicionada ao `config.json` para obter a extrapolação de comprimento:

```json
"rope_scaling": {
    "type": "yarn",
    "factor": 16.0,
    "original_max_position_embeddings": 2048,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "attention_factor": 1.0
}
```

A seguir, usando o MiniMind como exemplo, utilizamos como entrada textos em chinês vernáculo de *Jornada ao Oeste* com diferentes comprimentos, comparando as mudanças de perplexidade (PPL) antes e depois de ativar o escalonamento do RoPE. Percebe-se que, em cenários de textos longos, a PPL do modelo diminui significativamente após ativar a extrapolação YaRN:

<div align="center">
<img src="./images/rope_ppl.png">
</div>

> Comparação da PPL do MiniMind antes e depois de ativar o YaRN em diferentes comprimentos de texto

---

## Ⅴ Avaliação Objetiva

Esta seção apresenta resultados de benchmarks em vários modelos de linguagem de microescala. Os benchmarks escolhidos são C-Eval, CMMLU, ARC-Easy, PIQA, OpenBookQA, HellaSwag e Social-IQa; todos, exceto os dois primeiros, são benchmarks em inglês.


O framework de avaliação escolhido é o [lm-evaluation](https://github.com/EleutherAI/lm-evaluation-harness)

```bash
# Instalação
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness && pip install -e .
```



```bash
# Iniciar os testes
# Datasets usados: ceval-valid/cmmlu/arc_easy/piqa/openbookqa/hellaswag/social_iqa # Ver os datasets suportados: lm_eval ls tasks 
# Para modelos ajustados por instruções, adicione --apply_chat_template na avaliação; para modelos base, como o gpt2, não é necessário.
HF_ENDPOINT=https://hf-mirror.com lm_eval --model hf --model_args pretrained="/path/to/model",dtype=auto --tasks "task" --batch_size 16 --device cpu --trust_remote_code --apply_chat_template
```

> Observação: esses benchmarks de múltipla escolha normalmente não são avaliados pedindo ao modelo que gere livremente a resposta completa. Em vez disso, dado um contexto `y` e um conjunto de opções candidatas `x`, a prática padrão é comparar a probabilidade condicional `p(x | y)` de cada opção e escolher a de maior pontuação. Se uma opção corresponde a um único token, basta comparar a probabilidade desse token; se ela abrange vários tokens, uma abordagem mais padrão é comparar a soma das log-probabilidades condicionais sobre toda a opção. As candidatas não são necessariamente `A`, `B`, `C`, `D`; alguns datasets têm apenas duas opções. Nesse sentido, o chute aleatório já é um limite inferior bastante forte, e modelos nessa escala de fato tendem a ficar próximos dele por bastante tempo.

O MiniMind é treinado com muito menos dados do que os outros modelos listados aqui, e sua mistura de treinamento é fortemente inclinada para o chinês, então seu desempenho em inglês é relativamente fraco. Ele também não é, por padrão, especificamente alinhado a esse formato de avaliação de múltipla escolha, então seu desempenho é relativamente fraco, e os resultados servem apenas por diversão:

| nome do modelo | origem | params | zh (ceval / cmmlu) | en (arc / piqa / obqa / hellaswag / siqa) |
|---|---|---|---|---|
| minimind-3 | atual | 64M | 24.89 / 25.38 | 28.49 / 50.65 / 23.60 / 28.28 / 34.19 |
| minimind-3-moe | atual | 198M | 25.48 / 24.32 | 27.74 / 50.71 / 26.20 / 27.43 / 34.03 |
| minimind-3-exam | atual | 64M | 30.98 / 26.12 | 35.61 / 56.26 / 24.20 / 28.40 / 34.19 |
| [Steel-LLM](https://huggingface.co/gqszhanshijin/Steel-LLM) | ZhanShiJin | 1121M | 24.89 / 25.32 | 39.69 / 65.13 / 26.00 / 35.73 / 39.15 |
| [gpt2-medium](https://huggingface.co/openai-community/gpt2-medium) | OpenAI | 360M | 23.18 / 25.00 | 43.60 / 66.38 / 30.20 / 39.38 / 39.10 |
| [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | TinyLlama | 1100M | 25.71 / 25.03 | 54.80 / 74.43 / 35.60 / 60.38 / 43.09 |
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | HuggingFace | 135M | 24.44 / 24.71 | 58.50 / 68.17 / 32.80 / 43.15 / 39.46 |
| [Aquila-135M](https://huggingface.co/BAAI/Aquila-135M-Instruct) | BAAI | 135M | 25.19 / 25.10 | 54.59 / 67.52 / 34.40 / 41.67 / 39.66 |

> **A resolução desta tabela**: o `lm_eval` reporta um `stderr` junto com cada métrica. Ele é determinado principalmente pelo tamanho do conjunto de teste, então os valores abaixo se aplicam aproximadamente a todos os modelos da tabela (variando um pouco conforme a acurácia):
>
> | dataset | obqa | ceval | piqa | siqa | arc | hellaswag | cmmlu |
> |---|---|---|---|---|---|---|---|
> | amostras | 500 | ~1.3k | 1838 | 1954 | 2376 | 10042 | 11582 |
> | ± stderr | **1.9** | 1.2 | 1.2 | 1.1 | 0.9 | 0.5 | **0.4** |
>
> Ao comparar dois modelos, o erro padrão da diferença é cerca de `√2` vezes o valor de um único modelo. Então, uma diferença menor que cerca de 5 pontos no obqa, ou cerca de 3 pontos no ceval / piqa / siqa, não distingue dois modelos — e é por isso que a ordem de modelos de tamanho semelhante nesta tabela não deve ser superinterpretada.

<details>
<summary><strong>Nota adicional (origem / sem contaminação / reprodução)</strong></summary>

O minimind-3-exam não é um modelo base maior e contém pouco ou nenhum conhecimento novo. Ele é simplesmente o minimind-3 após um alinhamento leve com LoRA em [lora_exam.jsonl](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/main/lora_exam.jsonl), com o [lora_exam_768.pth](https://huggingface.co/jingyaogong/minimind-3-pytorch/resolve/main/lora_exam_768.pth) mesclado de volta ao modelo base. Esses dados de alinhamento são amostrados dos subconjuntos de teste do ceval e do mmlu em inglês, com aumento adicional de prefixos/sufixos. O objetivo é alinhar o contexto e o formato de opções comumente vistos em avaliações de múltipla escolha, e não ensinar as respostas.

Os 7 benchmarks usados nesta seção não têm sobreposição de amostras com os dados de alinhamento acima, então este resultado pode ser considerado livre de contaminação de dados. Por outro lado, se alguém fizer ajuste fino diretamente em dados sobrepostos, as pontuações de um modelo pequeno podem ficar muito distorcidas; por exemplo, o minimind-3 chegou a atingir cerca de 97% de acurácia em subconjuntos contaminados de ceval / cmmlu, mas esses números não têm significado.

O que este experimento sugere é simples: para esse tipo de benchmark, o gargalo de um modelo pequeno pode não estar inteiramente no conhecimento em si, mas também em se o formato de entrada está alinhado. Com apenas uma pequena quantidade de alinhamento de formato, o minimind-3-exam melhora cerca de 2.9 pontos percentuais em média nas 7 tarefas acima.

</details>

![benchmark_radar](./images/benchmark_radar.jpg)

---

# 📌 Outros

## 🔧 Conversão de Modelos

* O [./scripts/convert_model.py](./scripts/convert_model.py) pode ser usado para converter entre os formatos de modelo `torch` e `transformers`.
* Salvo indicação em contrário, os modelos lançados da série principal `MiniMind` normalmente são disponibilizados no formato `Transformers`. Se você usar pesos `torch` nativos, execute primeiro a conversão `torch2transformers`.


## 🖥️ Interface de Serviço de API Baseada no MiniMind

* O [./scripts/serve_openai_api.py](./scripts/serve_openai_api.py) oferece um serviço de chat leve compatível com a API da OpenAI, facilitando a conexão dos seus próprios modelos a UIs de terceiros como FastGPT, OpenWebUI, Dify etc.
* O servidor de API também suporta campos como `reasoning_content`, `tool_calls` e `open_thinking`, sendo adequado para cenários de Tool Calling / Thinking.

* Depois de baixar os pesos do modelo do [HuggingFace](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5), um exemplo da estrutura de diretórios é o seguinte:
    ```
    minimind (diretório raiz)
    ├─<Nome-do-Modelo-MiniMind> (ex.: minimind-3)
    |  ├── config.json
    |  ├── generation_config.json
    |  ├── model_minimind.py (opcional, dependendo do formato de exportação)
    |  ├── pytorch_model.bin ou model.safetensors
    |  ├── special_tokens_map.json
    |  ├── tokenizer_config.json
    |  ├── tokenizer.json
    ```

* Iniciar o servidor
    ```bash
    cd scripts && python serve_openai_api.py
    ```
* Testar a interface do serviço
    ```bash
    cd scripts && python chat_api.py
    ```
* Exemplo de requisição à API (compatível com o formato da API da OpenAI)
    ```bash
    curl http://localhost:8998/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{ 
        "model": "model-identifier",
        "messages": [ 
          { "role": "user", "content": "世界上最高的山是什么？" }
        ], 
        "temperature": 0.7, 
        "max_tokens": 1024,
        "stream": true,
        "open_thinking": true
    }'
    ```

## <img src="https://avatars.githubusercontent.com/u/147780389?s=48&v=4" height="28" style="vertical-align: middle;"/> [SGLang](https://github.com/sgl-project/sglang)

O SGLang é uma engine de inferência de LLMs de alto desempenho, com otimizações como RadixAttention e continuous batching, oferecendo menor latência e maior vazão.

> ⚠️ Requer um ambiente CUDA; use conforme a necessidade. Você também pode escolher o SGLang como engine de rollout / inferência nos scripts de treinamento de RL para aumentar a vazão do treinamento.

Inicie o modelo como um servidor de API compatível com a OpenAI:

```bash
python -m sglang.launch_server --model-path /path/to/model --attention-backend triton --host 0.0.0.0 --port 8998
```

## <img src="https://avatars.githubusercontent.com/u/136984999" height="28" style="vertical-align: middle;"/> [vllm](https://github.com/vllm-project/vllm)

O vLLM é um framework de inferência eficiente e amplamente usado para a implantação rápida de LLMs, com um bom equilíbrio entre eficiência de memória e vazão.

> ⚠️ Requer um ambiente CUDA; use conforme a necessidade.

Inicie o modelo como um servidor de API compatível com a OpenAI:

```bash
vllm serve /path/to/model --model-impl transformers --served-model-name "minimind" --port 8998
```

## <img src="https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png" height="28" style="vertical-align: middle;"/> [llama.cpp](https://github.com/ggerganov/llama.cpp)

O llama.cpp é um framework de inferência em C++ leve e prático, que pode ser usado diretamente pela linha de comando. Ele suporta inferência multithread e várias opções de aceleração por GPU.

**Estrutura de diretórios**: recomenda-se colocar o `llama.cpp` e o diretório do modelo no mesmo nível

```
parent/
├── project/           # diretório do seu projeto
│   ├── minimind-model/       # diretório do modelo no formato HuggingFace
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── ...
│   └── ...
└── llama.cpp/         # diretório do llama.cpp
    ├── build/
    ├── convert_hf_to_gguf.py
    └── ...
```

0. Consulte a documentação oficial do `llama.cpp` para concluir a instalação (dependências como `cmake` etc.)

1. Insira no final da função `get_vocab_base_pre` em `convert_hf_to_gguf.py`:

```python
# Adiciona suporte ao tokenizer do MiniMind. Um fallback compatível, como o qwen2, pode ser reutilizado temporariamente.
if res is None:
    res = "qwen2"
```

2. Converta o modelo minimind no formato HuggingFace para GGUF:

```bash
# Execute no diretório do llama.cpp. O arquivo GGUF será gerado no diretório do modelo.
python convert_hf_to_gguf.py /path/to/minimind-model
```

3. Quantize o modelo (opcional)

```bash
./build/bin/llama-quantize /path/to/model/xxxx.gguf /path/to/model/xxxx.q8.gguf Q8_0
```

4. Teste de inferência pela linha de comando

```bash
./build/bin/llama-cli -m /path/to/model/xxxx.gguf
```

## <img src="https://ollama.com/public/cloud.png" height="28" style="vertical-align: middle;"/> [ollama](https://ollama.ai)

O Ollama é uma ferramenta muito usada para rodar grandes modelos localmente. Ele suporta muitos LLMs de código aberto e oferece um fluxo de trabalho simples, com pouca configuração.

1. Carregar um modelo GGUF personalizado pelo Ollama

Crie um novo arquivo `minimind.modelfile` no diretório do modelo e escreva o seguinte template de configuração. Você pode substituir o prompt de sistema conforme necessário:

<details>
<summary>minimind.modelfile (template)</summary>

```text
FROM /path/to/model/xxxx.gguf

SYSTEM "你的名字叫MiniMind，你是一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题，当被问到名字时请回答MiniMind。"


TEMPLATE """{{- if .Tools }}<|im_start|>system
{{ if .System }}{{ .System }}

{{ end }}# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
{{ else if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
<think>
{{ .Thinking }}
</think>

{{ .Content }}
{{- if .ToolCalls }}
{{- range .ToolCalls }}
<tool_call>
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
</tool_call>
{{- end }}
{{- end }}
{{- if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ if and $.IsThinkSet $.Think -}}
<think>
{{ else -}}
<think>

</think>

{{ end -}}
{{ end }}
{{- end }}"""

PARAMETER repeat_penalty 1
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.9
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
```

</details>
<br/>


2. Carregar e nomear o modelo local

```bash
ollama create -f minimind.modelfile minimind-local
```

3. Iniciar a inferência

```bash
ollama run minimind-local
```

<details>
<summary>📤 Publique seu modelo no Ollama Hub</summary>

```bash
# 1. Renomeie o modelo local para a tag sua-conta-ollama/minimind
ollama cp minimind-local:latest your_username/minimind:latest

# 2. Publique o modelo
ollama push your_username/minimind:latest
```
</details>
<br/>

⭐️ Você também pode usar diretamente o modelo do Ollama que disponibilizo, para começar rapidamente:

```bash
ollama run jingyaogong/minimind-3
>>> 你叫什么名字
我是一个语言模型...
```

## <img src="https://avatars.githubusercontent.com/u/1961952?s=48&v=4" height="28" style="vertical-align: middle;"/> [MNN](https://github.com/alibaba/MNN)

O MNN é uma engine de inferência de IA projetada para dispositivos de borda (edge), com suporte à implantação leve e à inferência de alto desempenho de diversos LLMs de código aberto.

1. Conversão do modelo
```bash
cd MNN/transformers/llm/export
# Exportar um modelo MNN quantizado em 4 bits com HQQ
python llmexport.py --path /path/to/model --export mnn --hqq --dst_path /path/to/model-mnn
```

2. Testar no Mac ou em dispositivos móveis
```bash
./llm_demo /path/to/model-mnn/config.json prompt.txt
```
Como alternativa, teste pelo aplicativo móvel.

> Para mais detalhes sobre esses frameworks de terceiros, consulte a documentação oficial de cada um.


## 👨‍💻 Mais Conteúdo

* <a href="https://github.com/jingyaogong/minimind/discussions/618">🔗Ajuste Fino de Modelos de Linguagem de Difusão a partir do MiniMind-LLM</a>

* <a href="https://github.com/jingyaogong/minimind/discussions/611">🔗Descrição do Método generate do Modelo</a>

* <a href="https://github.com/jingyaogong/minimind/discussions/704">🔗Treinando Modelos de Atenção Linear a partir do MiniMind</a>

# 📌 Agradecimentos

> [!NOTE]
> Se a série de projetos `MiniMind` foi útil para você, fique à vontade para dar uma estrela ⭐ no GitHub<br/>
> A documentação é longa e omissões são inevitáveis. Feedback por meio de Issues ou PRs é bem-vindo para melhorarmos o projeto juntos.<br/>
> Seu apoio e suas sugestões são forças motrizes importantes para a evolução contínua deste projeto!

## 🤝[Contribuidores](https://github.com/jingyaogong/minimind/graphs/contributors)

<a href="https://github.com/jingyaogong/minimind/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jingyaogong/minimind" />
</a>

## 😊Agradecimentos

Obrigado aos seguintes colaboradores por compartilharem anotações de treinamento, experiência com processamento de dados, tutoriais e explicações do código:

* [@ipfgao](https://github.com/ipfgao): [🔗Registro dos Passos de Treinamento](https://github.com/jingyaogong/minimind/issues/26)

* [@WangRongsheng](https://github.com/WangRongsheng): [🔗Pré-processamento de Grandes Datasets](https://github.com/jingyaogong/minimind/issues/39)

* [@pengqianhan](https://github.com/pengqianhan): [🔗Um Tutorial Conciso](https://github.com/jingyaogong/minimind/issues/73)

* [@RyanSunn](https://github.com/RyanSunn): [🔗Notas de Estudo sobre o Processo de Inferência](https://github.com/jingyaogong/minimind/issues/75)

* [@Nijikadesu](https://github.com/Nijikadesu): [🔗Explicação do Código do Projeto em Formato de Notebook Interativo](https://github.com/jingyaogong/minimind/issues/213)

* [@jaylearnstocode](https://github.com/jaylearnstocode): [🔗Visualização da Arquitetura do Modelo, dos Mecanismos de Atenção e dos Pipelines de Treinamento](https://llm-visualization-minimind.vercel.app/)


Obrigado também aos seguintes artigos e projetos:

- [https://github.com/meta-llama/llama3](https://github.com/meta-llama/llama3)
- [https://github.com/karpathy/llama2.c](https://github.com/karpathy/llama2.c)
- [https://github.com/DLLXW/baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [https://github.com/charent/ChatLM-mini-Chinese](https://github.com/charent/ChatLM-mini-Chinese)
- [https://github.com/wdndev/tiny-llm-zh](https://github.com/wdndev/tiny-llm-zh)
- [Mistral-MoE](https://arxiv.org/pdf/2401.04088)
- [https://github.com/Tongjilibo/build_MiniLLM_from_scratch](https://github.com/Tongjilibo/build_MiniLLM_from_scratch)
- [https://github.com/jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)
- [https://github.com/AI-Study-Han/Zero-Chatgpt](https://github.com/AI-Study-Han/Zero-Chatgpt)
- [https://github.com/xusenlinzy/api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm)
- [https://github.com/HqWu-HITCS/Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)


## 🫶Apoiadores

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://bytecrank.com/nastyox/reporoster/php/forkersSVG.php?user=jingyaogong&repo=minimind&theme=dark"/>
  <source media="(prefers-color-scheme: light)" srcset="https://bytecrank.com/nastyox/reporoster/php/forkersSVG.php?user=jingyaogong&repo=minimind"/>
  <img alt="Fork poster" src="https://bytecrank.com/nastyox/reporoster/php/forkersSVG.php?user=jingyaogong&repo=minimind&theme=dark"/>
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=jingyaogong/minimind&type=date&theme=dark&legend=top-left&sealed_token=DK6jy_uvw2AHIK0S4VZLf6snWIQ06jGzz3QiwVmXBGDvickcQgJGSdazdGxjRQZuj8Hr3GfS_REB9ohoK8NWVsmukeOQiT4soChw3_19yyPVwvWzBp66yMYWlvOYy9sv60cMSntByiUTcyp4MrRiMm1JD1MSC8NJ-Z9qhR9uJGl2AU7w-OGlyKQzN7Xa"/>
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=jingyaogong/minimind&type=date&legend=top-left&sealed_token=DK6jy_uvw2AHIK0S4VZLf6snWIQ06jGzz3QiwVmXBGDvickcQgJGSdazdGxjRQZuj8Hr3GfS_REB9ohoK8NWVsmukeOQiT4soChw3_19yyPVwvWzBp66yMYWlvOYy9sv60cMSntByiUTcyp4MrRiMm1JD1MSC8NJ-Z9qhR9uJGl2AU7w-OGlyKQzN7Xa"/>
  <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=jingyaogong/minimind&type=date&legend=top-left&sealed_token=DK6jy_uvw2AHIK0S4VZLf6snWIQ06jGzz3QiwVmXBGDvickcQgJGSdazdGxjRQZuj8Hr3GfS_REB9ohoK8NWVsmukeOQiT4soChw3_19yyPVwvWzBp66yMYWlvOYy9sv60cMSntByiUTcyp4MrRiMm1JD1MSC8NJ-Z9qhR9uJGl2AU7w-OGlyKQzN7Xa"/>
</picture>

## 🎉 Conquistas Relacionadas ao MiniMind

Este modelo serviu de ponto de partida para vários resultados de pesquisa gratificantes. Obrigado aos pesquisadores pelo reconhecimento:

- ECG-Expert-QA: A Benchmark for Evaluating Medical Large Language Models in Heart Disease Diagnosis [[arxiv](https://arxiv.org/pdf/2502.17475)]

- Binary-Integer-Programming Based Algorithm for Expert Load Balancing in Mixture-of-Experts Models [[arxiv](https://arxiv.org/pdf/2502.15451)]

- LegalEval-Q: A New Benchmark for The Quality Evaluation of LLM-Generated Legal Text [[arxiv](https://arxiv.org/pdf/2505.24826)]

- On the Generalization Ability of Next-Token-Prediction Pretraining [[ICML 2025](https://openreview.net/forum?id=hLGJ1qZPdu)]

- 《从零开始写大模型：从神经网络到Transformer》王双、牟晨、王昊怡 编著 - 清华大学出版社

- FedBRB: A Solution to the Small-to-Large Scenario in Device-Heterogeneity Federated Learning [[TMC 2025](https://ieeexplore.ieee.org/abstract/document/11168259)]

- SKETCH: Semantic Key-Point Conditioning for Long-Horizon Vessel Trajectory Prediction [[arxiv](https://arxiv.org/pdf/2601.18537)]

- A Built-in Crypto Expert for Artificial Intelligence: How Far is the Horizon? [[IACR ePrint 2026](https://eprint.iacr.org/2026/411.pdf)]

- RetryTrigger: Intelligent Inference Duplication for Enhancing LLM Resilience to Hardware Transient Faults [[FITEE 2026](https://ieeexplore.ieee.org/abstract/document/11479682)]

- Em andamento...


# 🎓 Citação

Se o `MiniMind` foi útil para sua pesquisa ou seu trabalho, fique à vontade para citá-lo:

```bibtex
@misc{minimind,
  title = {MiniMind: Train a Tiny LLM from Scratch},
  author = {Jingyao Gong},
  year = {2024},
  url = {https://github.com/jingyaogong/minimind},
  note = {GitHub repository}
}
```

# ⚖️ Licença

Este projeto é distribuído como código aberto sob a [Apache License 2.0](LICENSE).
