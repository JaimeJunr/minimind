"""Testes do tokenizer do modelo e do script de medição em docs/story-helper."""

import importlib.util
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def load_medir_tokenizer():
    spec = importlib.util.spec_from_file_location("medir_tokenizer", ROOT / "docs/story-helper/medir_tokenizer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chat_template_ends_with_assistant_turn():
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    messages = [{"role": "user", "content": "Olá!"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert prompt.startswith("<|im_start|>user\nOlá!<|im_end|>\n")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_tokenizer_round_trips_portuguese():
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model")
    text = "Ação, coração e pão: a história começou em Ouro Preto."
    ids = tokenizer(text, add_special_tokens=False).input_ids
    assert tokenizer.decode(ids) == text


def test_medir_tokenizer_bpe_round_trips_bytes():
    medir = load_medir_tokenizer()
    rank, vocab_size = medir.carregar(ROOT / "model/tokenizer.json")
    b2u = medir.bytes_para_unicode()
    u2b = {char: byte for byte, char in b2u.items()}
    assert vocab_size == 6400
    for text in medir.AMOSTRAS.values():
        tokens = medir.tokenizar(text, rank, b2u)
        assert bytes(u2b[char] for char in "".join(tokens)).decode("utf-8") == text


def test_medir_tokenizer_shows_portuguese_is_less_efficient():
    medir = load_medir_tokenizer()
    rank, _ = medir.carregar(ROOT / "model/tokenizer.json")
    b2u = medir.bytes_para_unicode()
    ratio = {lang: len(text) / len(medir.tokenizar(text, rank, b2u)) for lang, text in medir.AMOSTRAS.items()}
    assert ratio["pt"] < ratio["en"]
