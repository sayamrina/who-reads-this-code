"""Build the token-level and trial-level tables for all Java trials in EMIP.

Output: results/tokens.csv (one row per participant × trial × token), results/trials.csv (one row per trial) and
results/exclusions.json (how many trials were dropped and why).
"""
import csv
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import emip  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
JAVA = {"rectangle_java", "rectangle_java2", "vehicle_java", "vehicle_java2"}
MAX_LOSS = 0.30
MIN_ON_CODE = 0.50
MIN_FIXATIONS = 30
EXPERTISE = {"none": 0, "low": 1, "medium": 2, "high": 3}


def years(value: str) -> float | None:
    try:
        return float(str(value).replace("+", ""))
    except ValueError:
        return None


def one_participant(pid: str) -> tuple[list[dict], list[dict], list[tuple[str, str]]]:
    path = emip.ROOT / "rawdata" / f"{pid}_rawdata.tsv"
    tokens, trials, dropped = [], [], []
    for stim, samples in emip.read_trials(path).items():
        if stim not in JAVA:
            continue
        boxes, _ = emip.aois(stim)
        src = emip.source_tokens(stim)
        fix = emip.fixations(samples)
        ft = [(f, emip.hit(f, boxes)) for f in fix]
        loss = emip.tracking_loss(samples)
        on = sum(1 for _, t in ft if t is not None) / max(len(ft), 1)
        if loss > MAX_LOSS:
            dropped.append((stim, "tracking loss"))
            continue
        if len(fix) < MIN_FIXATIONS:
            dropped.append((stim, "too few fixations"))
            continue
        if on < MIN_ON_CODE:
            dropped.append((stim, "gaze mostly off the code"))
            continue
        feats = emip.strategy_features(ft, [s[0] for s in src])
        trials.append({"participant": pid, "stimulus": stim, "program": stim.split("_")[0], "loss": round(loss, 3),
                       "duration_s": round((samples[-1][0] - samples[0][0]) / 1000, 1), **feats})
        for m in emip.token_measures(ft, len(src)):
            line, pos, text = src[m["token"]]
            tokens.append({"participant": pid, "stimulus": stim, "program": stim.split("_")[0], "token": m["token"],
                           "text": text, "line": line, "pos": pos, **m})
    return tokens, trials, dropped


def main() -> None:
    meta = {r["id"]: r for r in csv.DictReader(open(emip.ROOT / "emip_metadata.csv", encoding="utf-8", errors="ignore"))}
    pids = sorted((p for p, r in meta.items() if r["experiment_language"] == "Java"), key=int)
    surprisal = json.load(open(RESULTS / "surprisal.json"))
    bits = {"rectangle": [w["bits"] for w in surprisal["Rectangle.java"]], "vehicle": [w["bits"] for w in surprisal["Vehicle.java"]]}
    all_tokens, all_trials, reasons = [], [], Counter()
    with ProcessPoolExecutor(6) as ex:
        for tokens, trials, dropped in ex.map(one_participant, pids):
            all_tokens += tokens
            all_trials += trials
            reasons.update(r for _, r in dropped)
    freq = Counter()
    for name in ("Rectangle.java", "Vehicle.java"):
        freq.update(w["text"] for w in surprisal[name])
    for row in all_tokens:
        m = meta[row["participant"]]
        row["surprisal"] = bits[row["program"]][row["token"]]
        row["length"] = len(row["text"])
        row["log_freq"] = round(math.log(freq[row["text"]]), 3)
        row["expertise"] = EXPERTISE.get(m["expertise_programming"], None)
        row["years"] = years(m["time_programming"])
    for row in all_trials:
        m = meta[row["participant"]]
        row["expertise"] = EXPERTISE.get(m["expertise_programming"], None)
        row["years"] = years(m["time_programming"])
        row["correct"] = m[f"correct_{row['program']}"]
    for name, rows in (("tokens.csv", all_tokens), ("trials.csv", all_trials)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(RESULTS / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)
    summary = {"participants_java": len(pids), "trials_kept": len(all_trials),
               "participants_kept": len({t["participant"] for t in all_trials}), "dropped": dict(reasons),
               "token_rows": len(all_tokens)}
    json.dump(summary, open(RESULTS / "exclusions.json", "w"), indent=1)
    print(summary)


if __name__ == "__main__":
    main()
