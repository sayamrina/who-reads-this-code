"""Study 2: profile-conditioned code explanations and a counterfactual test of their faithfulness.

Two systems explain the same program to a developer described by a profile (constructs known or unknown):

* prompt-only: the profile is written into the prompt and the LLM writes the explanation directly;
* RSA-style selection: the LLM writes one note per (line, construct) once, independently of any profile; a speaker
  then chooses, line by line, the set of notes that maximises the probability that a literal listener built from the
  profile understands the line, minus a cost for length (Goodman and Frank, 2016).

Faithfulness test: flip one construct from unknown to known. A faithful system changes only the lines that use that
construct. We count changed targeted lines and changed untargeted lines over all single-construct flips, and compare
with the change rate of the prompt-only system when nothing is changed (its run-to-run noise).
"""
import difflib
import itertools
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import constructs  # noqa: E402
import emip  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODEL = "gpt-4o-mini"
PROGRAMS = {"Rectangle.java": "Rectangle", "Vehicle.java": "Vehicle"}
# Baseline profile: a beginner who knows basic statements but not object orientation.
KNOWN = {"arithmetic", "print", "return", "class"}
ALPHA, COST_PER_WORD, P_UNDERSTOOD, P_MISSED = 4.0, 0.01, 0.95, 0.15
SIMILAR = 0.6  # two notes on a line count as unchanged when their text similarity is at least this


def client():
    from openai import OpenAI

    return OpenAI(timeout=60, max_retries=2)


def chat_json(system: str, user: str, seed: int = 0) -> dict:
    r = client().chat.completions.create(model=MODEL, temperature=0, seed=seed, response_format={"type": "json_object"},
                                         messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return json.loads(r.choices[0].message.content)


def numbered(source: str) -> str:
    return "\n".join(f"{n:>2}  {line}" for n, line in enumerate(constructs.program_lines(source), start=1))


# --- system B: RSA-style selection over a profile-independent pool of notes -----------------------------------------

POOL_SYSTEM = """You explain Java code to learners. For each requested (line, construct) pair, write ONE short note
(max. 25 words) that explains that construct as it is used on that line. Refer to the actual names in the line.
Return JSON: {"notes": [{"line": int, "construct": str, "text": str}]}"""


def note_pool(source: str, lc: dict[int, list[str]]) -> list[dict]:
    pairs = [{"line": n, "construct": c, "meaning": constructs.NAMES[c]} for n, cs in lc.items() for c in cs]
    data = chat_json(POOL_SYSTEM, f"Code:\n{numbered(source)}\n\nPairs:\n{json.dumps(pairs)}")
    valid = {(p["line"], p["construct"]) for p in pairs}
    return [n for n in data.get("notes", []) if (n.get("line"), n.get("construct")) in valid]


def listener(line_cs: list[str], explained: set[str], known: set[str]) -> float:
    """Literal listener: probability that this developer understands the line given the notes they read."""
    p = 1.0
    for c in line_cs:
        p *= P_UNDERSTOOD if (c in known or c in explained) else P_MISSED
    return p


def rsa_select(lc: dict[int, list[str]], pool: list[dict], known: set[str]) -> dict[int, list[dict]]:
    """For each line, choose the subset of notes with the highest speaker utility, alpha·log L0 − cost."""
    by_line: dict[int, list[dict]] = {}
    for n in pool:
        by_line.setdefault(n["line"], []).append(n)
    chosen = {}
    for line, cs in lc.items():
        notes = by_line.get(line, [])
        best, best_u = [], None
        for r in range(len(notes) + 1):
            for subset in itertools.combinations(notes, r):
                explained = {n["construct"] for n in subset}
                words = sum(len(n["text"].split()) for n in subset)
                u = ALPHA * math.log(listener(cs, explained, known)) - COST_PER_WORD * ALPHA * words
                if best_u is None or u > best_u + 1e-9:
                    best, best_u = list(subset), u
        if best:
            chosen[line] = best
    return chosen


# --- system A: prompt-only ------------------------------------------------------------------------------------------

PROMPT_SYSTEM = """You explain Java code to one specific developer. You are given what they already know and do not
know. Explain only what this developer needs, line by line, and skip lines that need no explanation for them.
Every note must name the line it is about. Max. 25 words per note.
Return JSON: {"notes": [{"line": int, "text": str}]}"""


def prompt_only(source: str, known: set[str], all_cs: set[str], seed: int = 0) -> dict[int, list[dict]]:
    profile = {"knows": sorted(constructs.NAMES[c] for c in all_cs & known),
               "does not know": sorted(constructs.NAMES[c] for c in all_cs - known)}
    data = chat_json(PROMPT_SYSTEM, f"Developer profile: {json.dumps(profile)}\n\nCode:\n{numbered(source)}", seed=seed)
    out: dict[int, list[dict]] = {}
    for n in data.get("notes", []):
        if isinstance(n.get("line"), int):
            out.setdefault(n["line"], []).append(n)
    return out


# --- comparison -----------------------------------------------------------------------------------------------------

def line_text(expl: dict[int, list[dict]], line: int) -> str:
    return " ".join(n["text"] for n in expl.get(line, []))


def changed_lines(a: dict, b: dict, lines: list[int]) -> set[int]:
    out = set()
    for line in lines:
        ta, tb = line_text(a, line), line_text(b, line)
        if bool(ta) != bool(tb) or (ta and difflib.SequenceMatcher(None, ta, tb).ratio() < SIMILAR):
            out.add(line)
    return out


def grounded(expl: dict[int, list[dict]], n_lines: int) -> float:
    notes = [n for ns in expl.values() for n in ns]
    return sum(1 for n in notes if 1 <= n["line"] <= n_lines) / max(len(notes), 1)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set")
    report = {"settings": {"model": MODEL, "alpha": ALPHA, "cost_per_word": COST_PER_WORD, "p_understood": P_UNDERSTOOD,
                           "p_missed": P_MISSED, "similarity_threshold": SIMILAR, "baseline_known": sorted(KNOWN)},
              "programs": {}}
    totals = {k: {"targeted": 0, "targeted_changed": 0, "untargeted": 0, "untargeted_changed": 0} for k in ("prompt", "rsa")}
    noise = {"lines": 0, "changed": 0}
    examples, pools = {}, {}
    for file, cls in PROGRAMS.items():
        source = (emip.PROGRAMS / file).read_text(encoding="utf-8", errors="ignore")
        lc = constructs.line_constructs(source, cls)
        lines = list(lc)
        all_cs = {c for cs in lc.values() for c in cs}
        pool = note_pool(source, lc)
        pools[file] = pool
        base_rsa = rsa_select(lc, pool, KNOWN)
        base_prompt = prompt_only(source, KNOWN, all_cs)
        for seed in (1, 2):  # run-to-run noise of the prompt-only system with an unchanged profile
            again = prompt_only(source, KNOWN, all_cs, seed=seed)
            noise["lines"] += len(lines)
            noise["changed"] += len(changed_lines(base_prompt, again, lines))
        flips = []
        for c in sorted(all_cs - KNOWN):
            known = KNOWN | {c}
            target = {n for n, cs in lc.items() if c in cs}
            untarget = set(lines) - target
            res = {"construct": c, "targeted_lines": sorted(target)}
            for name, before, after in (("rsa", base_rsa, rsa_select(lc, pool, known)),
                                        ("prompt", base_prompt, prompt_only(source, known, all_cs))):
                ch = changed_lines(before, after, lines)
                totals[name]["targeted"] += len(target)
                totals[name]["targeted_changed"] += len(ch & target)
                totals[name]["untargeted"] += len(untarget)
                totals[name]["untargeted_changed"] += len(ch & untarget)
                res[name] = {"changed_targeted": sorted(ch & target), "changed_untargeted": sorted(ch & untarget)}
            flips.append(res)
        report["programs"][file] = {"lines": len(lines), "constructs": {str(k): v for k, v in lc.items()},
                                    "pool_notes": len(pool), "rsa_lines_explained": len(base_rsa),
                                    "prompt_lines_explained": len(base_prompt),
                                    "grounded_prompt": round(grounded(base_prompt, len(lines)), 3), "flips": flips}
        examples[file] = {"rsa": {str(k): [n["text"] for n in v] for k, v in base_rsa.items()},
                          "prompt": {str(k): [n["text"] for n in v] for k, v in base_prompt.items()}}
    report["summary"] = {
        name: {"targeted_change_rate": round(t["targeted_changed"] / t["targeted"], 3),
               "untargeted_change_rate": round(t["untargeted_changed"] / t["untargeted"], 3), **t}
        for name, t in totals.items()}
    report["summary"]["prompt_noise_change_rate"] = round(noise["changed"] / noise["lines"], 3)
    json.dump(report, open(RESULTS / "study2.json", "w"), indent=1)
    json.dump(examples, open(RESULTS / "study2_examples.json", "w"), indent=1, ensure_ascii=False)
    json.dump(pools, open(RESULTS / "note_pool.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
