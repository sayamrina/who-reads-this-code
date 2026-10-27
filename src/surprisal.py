"""Token surprisal for source code from a small open code language model.

A "token" here is a whitespace-separated word of the program, as in the EMIP Toolkit. Its surprisal is the sum of
the model's surprisal (in bits) over the sub-word pieces that make up the word, given all preceding code.
"""
import math
import re
from dataclasses import dataclass
from functools import lru_cache

MODEL = "Qwen/Qwen2.5-Coder-0.5B"


@dataclass
class TokenSurprisal:
    line: int  # 1-based line number in the source file
    index: int  # 0-based position of the word within its line
    text: str
    start: int  # character offset in the source
    end: int
    bits: float


@lru_cache(maxsize=1)
def _load(model_name: str = MODEL):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()
    return tok, model


def words(source: str) -> list[tuple[int, int, int, int, str]]:
    """(line, index, start, end, text) for every whitespace-separated word, lines numbered from 1."""
    out, offset = [], 0
    for n, line in enumerate(source.split("\n"), start=1):
        for i, m in enumerate(re.finditer(r"\S+", line)):
            out.append((n, i, offset + m.start(), offset + m.end(), m.group()))
        offset += len(line) + 1
    return out


def piece_surprisal(source: str, model_name: str = MODEL) -> list[tuple[int, int, float]]:
    """(start, end, bits) for every model token of the source; the first token gets no surprisal."""
    import torch

    tok, model = _load(model_name)
    enc = tok(source, return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"]
    with torch.no_grad():
        logits = model(ids).logits[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    offsets = enc["offset_mapping"][0].tolist()
    out = []
    for i in range(1, ids.shape[1]):
        lp = logp[i - 1, ids[0, i]].item()
        out.append((offsets[i][0], offsets[i][1], -lp / math.log(2)))
    return out


def word_surprisal(source: str, model_name: str = MODEL) -> list[TokenSurprisal]:
    """Sum the surprisal of all model pieces whose characters fall inside each word."""
    pieces = piece_surprisal(source, model_name)
    result = []
    for line, index, start, end, text in words(source):
        bits = sum(b for s, e, b in pieces if s < end and e > start and source[max(s, start):min(e, end)].strip())
        result.append(TokenSurprisal(line, index, text, start, end, round(bits, 3)))
    return result
