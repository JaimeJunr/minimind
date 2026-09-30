# StoryHelper — estudo

Estudo e plano para transformar o MiniMind num modelo pequeno (PT + EN) que ajuda LLMs de fronteira em escrita de ficção. Ele mantém a memória da história e monta contextos enxutos, para gastar menos tokens e obter narrativas mais consistentes.

**Status:** estudo concluído e plano aprovado; implementação ainda não iniciada.

| Arquivo | Conteúdo |
|---|---|
| [PESQUISA.md](./PESQUISA.md) | Revisão da literatura, economia com prompt caching, idioma e tokenizer, dados e licenças, metodologia de avaliação, hipóteses e referências |
| [PLANO.md](./PLANO.md) | Tarefas do modelo, desenho do sistema, fases com os arquivos a mudar, custos de nuvem, riscos e verificação |
| [medir_tokenizer.py](./medir_tokenizer.py) | Script sem dependências que mede caracteres/token do tokenizer em PT e EN (`python docs/story-helper/medir_tokenizer.py`) |
