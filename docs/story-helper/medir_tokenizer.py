"""Mede quantos caracteres por token o tokenizer do MiniMind rende em PT e EN.

Não depende de `tokenizers`/`transformers`: reimplementa o BPE byte-level a partir de
model/tokenizer.json. A pré-tokenização usa uma regex aproximada da do GPT-2, então os
números absolutos são estimativas; a comparação entre idiomas é o que importa.

Uso (na raiz do repositório):
    python docs/story-helper/medir_tokenizer.py
    python docs/story-helper/medir_tokenizer.py --tokenizer caminho/para/tokenizer.json
"""

import argparse
import json
import re

AMOSTRAS = {
    "pt": (
        "Era uma noite fria em Ouro Preto quando Helena, a jovem aprendiz de alquimista, encontrou a carta "
        "escondida no fundo do baú do avô. As palavras, escritas com tinta desbotada, falavam de um pacto "
        "antigo com os guardiões da serra."
    ),
    "en": (
        "It was a cold night in Ouro Preto when Helena, the young alchemist apprentice, found the letter hidden "
        "at the bottom of her grandfather's chest. The words, written in faded ink, spoke of an ancient pact "
        "with the guardians of the mountains."
    ),
}

PADRAO = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")


def bytes_para_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs), strict=True))


def carregar(caminho):
    with open(caminho, encoding="utf-8") as f:
        tj = json.load(f)
    merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in tj["model"]["merges"]]
    return {m: i for i, m in enumerate(merges)}, len(tj["model"]["vocab"])


def bpe(palavra, rank):
    partes = list(palavra)
    while len(partes) > 1:
        r, i = min((rank.get((partes[i], partes[i + 1]), float("inf")), i) for i in range(len(partes) - 1))
        if r == float("inf"):
            break
        partes = partes[:i] + [partes[i] + partes[i + 1]] + partes[i + 2 :]
    return partes


def tokenizar(texto, rank, b2u):
    tokens = []
    for pedaco in PADRAO.findall(texto):
        tokens += bpe("".join(b2u[b] for b in pedaco.encode("utf-8")), rank)
    return tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="model/tokenizer.json")
    args = parser.parse_args()
    rank, tamanho_vocab = carregar(args.tokenizer)
    b2u = bytes_para_unicode()
    print(f"vocabulário: {tamanho_vocab} tokens")
    for idioma, texto in AMOSTRAS.items():
        n = len(tokenizar(texto, rank, b2u))
        print(f"{idioma}: {len(texto)} caracteres, {n} tokens, {len(texto) / n:.2f} caracteres/token")
