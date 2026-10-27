"""Study 3: does model size change how well code-LLM perplexity tracks the human confusion response?

Uses the public replication package of Abdelsalam et al. (2025) (stimuli with human AOIs, per-snippet mean FRP amplitude at Fz,
400–800 ms, from Bergum et al., 2026) as data. We recompute AOI perplexity with our own code for additional model sizes and
first replicate one of their models to check that our pipeline gives the same numbers.

Usage: python -I src/aoc/perplexity_scaling.py <path to llm-perplexity-AoC> <hf model id> [bf16]
"""
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path(__file__).resolve().parents[2] / "results" / "aoc"
TAG = re.compile(r"\[aoi=(\w+)\]|\[/aoi\]")


def parse(code: str) -> tuple[str, dict[str, list[tuple[int, int]]]]:
    """Strip [aoi=X]…[/aoi] tags; return clean code and AOI spans (character offsets in the clean code) per type."""
    out, regions, stack, pos = [], {}, [], 0
    for m in TAG.finditer(code):
        out.append(code[pos:m.start()])
        clean_pos = sum(len(x) for x in out)
        if m.group(1):
            stack.append((m.group(1), clean_pos))
        elif stack:
            kind, start = stack.pop()
            regions.setdefault(kind, []).append((start, clean_pos))
        pos = m.end()
    out.append(code[pos:])
    return "".join(out), regions


def optimal_aoi(base: str) -> dict[str, str]:
    with open(f"{base}/code_snippets_with_aois/aoi_sizes.csv", encoding="utf-8") as f:
        return {r["Snippet"]: r["NO"] for r in csv.DictReader(f, delimiter=";") if r["OPTIMAL"] == "True"}


def token_logprobs(code: str, tok, model) -> list[tuple[int, int, float]]:
    """(start, end, log prob) per token; like the original pipeline, a start token is prepended (BOS, else EOS)."""
    enc = tok(code, add_special_tokens=False, return_offsets_mapping=True)
    start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ids = torch.tensor([[start_id] + enc["input_ids"]])
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    return [(s, e, logp[i, ids[0, i + 1]].item()) for i, (s, e) in enumerate(enc["offset_mapping"])]


def aoi_stats(tokens: list[tuple[int, int, float]], spans: list[tuple[int, int]]) -> dict:
    inside = [lp for s, e, lp in tokens if any(s < b and e > a for a, b in spans)]
    if not inside:
        return {}
    ppl = [math.exp(-lp) for lp in inside]
    return {"mean_aoi_ppl": sum(ppl) / len(ppl), "max_aoi_ppl": max(ppl), "overall_aoi_ppl": math.exp(-sum(inside) / len(inside)),
            "n_tokens": len(inside)}


def main() -> None:
    base, model_id = sys.argv[1], sys.argv[2]
    dtype = torch.bfloat16 if len(sys.argv) > 3 and sys.argv[3] == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.eval()
    best = optimal_aoi(base)
    rows = []
    for path in sorted(glob.glob(f"{base}/code_snippets_with_aois/v*/*.java")):
        name = os.path.basename(path)[:-5]
        if name not in best:
            continue
        code, regions = parse(Path(path).read_text(encoding="utf-8"))
        spans = regions.get(best[name], [])
        stats = aoi_stats(token_logprobs(code, tok, model), spans)
        if stats:
            rows.append({"file": name, "aoi_type": best[name], **stats})
    OUT.mkdir(parents=True, exist_ok=True)
    short = model_id.split("/")[-1]
    json.dump(rows, open(OUT / f"ppl_{short}.json", "w"), indent=1)
    print(short, len(rows), "files")


if __name__ == "__main__":
    main()
