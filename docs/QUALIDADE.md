# Qualidade de código e CI

Todo push na `master` e todo PR passam pelas verificações abaixo. Um job vermelho bloqueia o merge; os itens marcados como *informativo* só geram aviso e relatório.

## O que roda

| Workflow / job | Ferramenta | O que garante | Bloqueia? |
|---|---|---|---|
| `CI` › Lint, formatação e complexidade | `ruff check` | Bugs comuns (pyflakes, bugbear), imports, nomes, sintaxe moderna, simplificações | Sim |
| | `ruff check` (C901, PLR091x) | **Complexidade**: no máximo 10 de complexidade ciclomática, 12 ramificações, 50 instruções, 5 argumentos e 6 retornos por função | Sim |
| | `ruff format --check` | Formatação padronizada (linhas de até 120 caracteres) | Sim |
| | `radon cc` | Relatório das funções mais complexas do repositório, no resumo do job | Informativo |
| `CI` › Segurança do código e das dependências | `bandit` | Padrões inseguros no código (execução de shell, `eval`, `torch.load` sem proteção etc.), severidade e confiança médias ou maiores | Sim, só para achados **novos** |
| | `pip-audit` | Vulnerabilidades conhecidas (CVE/OSV) nas versões fixadas em `requirements*.txt` | Informativo |
| `CI` › Segredos no histórico | `gitleaks` | Nenhuma chave, token ou senha em todo o histórico do git | Sim |
| `CI` › Testes de fumaça (CPU) | `pytest` | O modelo roda em CPU (forward, loss, KV cache, geração, MoE, LoRA), o tokenizer e o chat template funcionam | Sim |
| `CodeQL` | GitHub CodeQL (`security-and-quality`) | Análise de fluxo de dados para segurança e qualidade; alertas na aba **Security** e nos PRs | Pelos alertas novos no PR |
| Dependabot | — | PRs semanais atualizando dependências Python e as actions | — |

O `CI` também roda toda segunda-feira, para pegar vulnerabilidades recém-publicadas mesmo sem commits novos.

## Código herdado do upstream

O código em `model/`, `trainer/`, `dataset/`, `scripts/` e `eval_llm.py` veio do [jingyaogong/minimind](https://github.com/jingyaogong/minimind). Reformatá-lo inteiro criaria conflitos em todo merge futuro com o upstream. A política é uma **catraca** (ratchet): o que já existe não pode piorar, e o que é novo segue o padrão completo.

- **Ruff**: em `pyproject.toml`, `per-file-ignores` desliga nesses caminhos só as regras que eles já violam. Todas as outras continuam valendo. Quem corrigir um arquivo herdado deve remover o código correspondente da lista.
- **Formatação**: os caminhos herdados e os exemplos de código dos READMEs ficam fora do `ruff format`.
- **Bandit**: os 29 achados existentes estão registrados em `.github/bandit-baseline.json`. O CI só falha com achados que não estão nessa lista. Para regenerar a lista depois de corrigir algo:
  ```bash
  bandit -q -r . -x ./.git,./.venv,./venv,./tests -ll -ii -f json -o .github/bandit-baseline.json
  ```
- **Código novo** (por exemplo, o futuro `story_helper/`, `tests/` e `docs/`) segue todas as regras, inclusive os limites de complexidade.

## Situação atual (setembro de 2026)

- **Complexidade**: 20 funções herdadas passam de pelo menos um limite (41 violações no total). Pelo radon, as piores são `rl_train_epoch` (35) e `calculate_rewards` (30) em `trainer/train_agent.py`, e `grpo_train_epoch` (25) em `trainer/train_grpo.py`. O relatório do radon no CI mostra a lista completa.
- **Bandit**: 29 achados herdados. São 18 × B615 (download do Hugging Face sem fixar a revisão; na maioria, falso positivo para caminhos locais), 10 × B614 (`torch.load` sem `weights_only=True`, que é arriscado ao carregar pesos de terceiros) e 1 × B104 (servidor escutando em `0.0.0.0`).
- **Dependências**: o `requirements.txt` herdado tem **71 vulnerabilidades conhecidas em 10 pacotes**, entre eles `transformers`, `jinja2`, `nltk`, `ujson`, `flask-cors` e `streamlit`. Por isso o `pip-audit` ainda é informativo. Atualizar pacotes como `transformers` (4.x → 5.x) exige validar o treino, então cada atualização deve vir num PR próprio (o Dependabot abre esses PRs), passando pelos testes de fumaça.

## Rodar localmente

```bash
pip install -r requirements-dev.txt            # ruff, bandit, pip-audit, radon
ruff check . && ruff format --check .           # lint, complexidade e formatação
ruff check --fix . && ruff format .             # corrige o que for automático
bandit -q -r . -x ./.git,./.venv,./venv,./tests -ll -ii -b .github/bandit-baseline.json
pip-audit -r requirements.txt --no-deps --disable-pip

pip install -r requirements-test.txt            # torch (CPU), transformers, pytest
pytest
```

## Configuração no GitHub (uma vez)

1. **Ativar o Actions no fork**: em forks, o GitHub desativa os workflows até o dono ativá-los na aba **Actions**.
2. **Alertas do Dependabot**: *Settings › Code security › Dependabot alerts* e *Dependabot security updates*.
3. **Proteção da `master`** (recomendado): *Settings › Branches* ou *Rules*, exigindo os checks do `CI` e do `CodeQL` antes do merge.
4. Se o **CodeQL "default setup"** estiver ligado em *Code security*, desligue-o, porque ele conflita com o workflow `codeql.yml`.

## Versões fixadas

As ferramentas estão fixadas em `requirements-dev.txt` e `requirements-test.txt`, e as actions por SHA de commit, para proteger a cadeia de dependências. O Dependabot atualiza essas versões. A exceção é o `gitleaks`, fixado por versão e checksum SHA-256 no próprio `ci.yml`: ele precisa ser atualizado à mão.
