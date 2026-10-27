"""Study 5: can gaze plus perplexity tell confusing from clean code better than perplexity alone?

Snippet-level features: log maximum AOI perplexity (Qwen2.5-Coder-32B, from the replication package) and log mean first-pass
dwell time in the AOI across readers. Leave-one-pair-out logistic regression; area under the ROC curve and accuracy.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

RESULTS = Path(__file__).resolve().parents[2] / "results" / "aoc"
FEATURES = {"perplexity": ["log_ppl"], "gaze": ["log_dwell"], "perplexity + gaze": ["log_ppl", "log_dwell"]}


def snippets() -> pd.DataFrame:
    t = pd.read_csv(RESULTS / "aoc_trials.csv", dtype={"participant": str})
    s = t.groupby("file").agg(confusing=("confusing", "first"), pair=("pair", "first"), ppl=("ppl_Qwen2.5-Coder-32B", "first"),
                              dwell=("first_pass_s", "mean"), readers=("participant", "nunique")).reset_index()
    s["log_ppl"], s["log_dwell"] = np.log(s["ppl"]), np.log(s["dwell"])
    return s.dropna()


def loo_probs(s: pd.DataFrame, cols: list[str]) -> np.ndarray:
    prob = pd.Series(index=s.index, dtype=float)
    for pair in s["pair"].unique():
        test = s["pair"] == pair
        sc = StandardScaler().fit(s.loc[~test, cols])
        clf = LogisticRegression().fit(sc.transform(s.loc[~test, cols]), s.loc[~test, "confusing"])
        prob[test] = clf.predict_proba(sc.transform(s.loc[test, cols]))[:, 1]
    return prob.to_numpy()


def loo_pairs(s: pd.DataFrame, cols: list[str]) -> tuple[float, float]:
    prob = loo_probs(s, cols)
    return roc_auc_score(s["confusing"], prob), accuracy_score(s["confusing"], prob > 0.5)


def paired_choice(s: pd.DataFrame, col: str) -> float:
    """Share of pairs in which the confusing version has the higher value (no fitting at all)."""
    wide = s.pivot_table(index="pair", columns="confusing", values=col)
    return float((wide[1] > wide[0]).mean())


def main() -> None:
    s = snippets()
    out = {"n_snippets": int(len(s)), "n_pairs": int(s["pair"].nunique())}
    for name, cols in FEATURES.items():
        auc, acc = loo_pairs(s, cols)
        out[name] = {"auc": round(auc, 3), "accuracy": round(acc, 3)}
    out["paired_choice"] = {"perplexity": round(paired_choice(s, "log_ppl"), 3), "gaze": round(paired_choice(s, "log_dwell"), 3)}
    # Bootstrap over pairs for the AUC difference: predictions are fixed from the leave-one-pair-out fits, pairs are resampled
    probs = {name: loo_probs(s, cols) for name, cols in FEATURES.items()}
    rng = np.random.default_rng(0)
    pairs = s["pair"].unique()
    diffs = []
    for _ in range(2000):
        pick = rng.choice(pairs, size=len(pairs), replace=True)
        idx = np.concatenate([np.flatnonzero(s["pair"].to_numpy() == p) for p in pick])
        y = s["confusing"].to_numpy()[idx]
        diffs.append(roc_auc_score(y, probs["perplexity + gaze"][idx]) - roc_auc_score(y, probs["perplexity"][idx]))
    out["auc_gain_both_vs_perplexity"] = {"mean": round(float(np.mean(diffs)), 3), "ci95": [round(float(np.percentile(diffs, 2.5)), 3),
                                                                                           round(float(np.percentile(diffs, 97.5)), 3)]}
    json.dump(out, open(RESULTS / "study5.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
