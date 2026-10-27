"""Study 3: alignment of AOI perplexity with the human confusion response across model sizes.

For every model: Spearman correlation of maximum AOI perplexity with the snippet-mean late frontal positivity (Fz, 400–800 ms)
for confusing snippets (their RQ2) and over all 144 snippets, and the gain in fit of a single-trial mixed model when AOI
surprisal is added (likelihood-ratio statistic, ML fits).
Usage: python -I src/aoc/study3.py <aoc_frp data dir>
"""
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

RESULTS = Path(__file__).resolve().parents[2] / "results" / "aoc"
SIZE = {"Qwen2.5-Coder-0.5B": 0.5, "Qwen2.5-Coder-1.5B": 1.5, "Qwen2.5-Coder-3B": 3, "Qwen2.5-Coder-32B": 32,
        "deepseek-coder-1.3b-base": 1.3, "deepseek-coder-6.7b-base": 6.7, "Llama-3.2-3B": 3, "Yi-Coder-9B": 9,
        "Artigenz-Coder-DS-6.7B": 6.7}
SOURCE = {"Qwen2.5-Coder-0.5B": "ours", "Qwen2.5-Coder-1.5B": "ours", "Qwen2.5-Coder-3B": "ours"}


def perplexities(base: str) -> dict[str, pd.Series]:
    out = {}
    for d in glob.glob(f"{base}/llm-perplexity-AoC/model_results/*"):
        name = Path(d).name
        s = pd.read_csv(f"{d}/pairwise_analysis/pairwise_comparisons/perplexity_stats.csv")
        out[name] = s.set_index("File")["Max AOI Perplexity"]
    for f in glob.glob(str(RESULTS / "ppl_Qwen2.5-Coder-*.json")):
        name = Path(f).stem[4:]
        out[name] = pd.DataFrame(json.load(open(f))).set_index("file")["max_aoi_ppl"]
    return out


def snippet_eeg(base: str) -> pd.Series:
    pairwise = f"{base}/llm-perplexity-AoC/model_results/Qwen2.5-Coder-32B/pairwise_analysis/pairwise_comparisons"
    d = pd.read_csv(f"{pairwise}/eeg_perplexity_analysis.csv")
    return pd.concat([d.set_index("clean_file")["clean_mean_eeg"], d.set_index("obf_file")["obf_mean_eeg"]])


def lr_gain(trials: pd.DataFrame, ppl: pd.Series) -> float:
    d = trials.copy()
    d["s"] = np.log2(d["file"].map(ppl))
    d = d.dropna(subset=["s"])
    d["s"] = (d["s"] - d["s"].mean()) / d["s"].std()
    d["all"] = 1
    vc = {"participant": "0 + C(participant)", "pair": "0 + C(pair)"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m0 = smf.mixedlm("lfp_fz ~ confusing", d, groups=d["all"], re_formula="0", vc_formula=vc).fit(reml=False, method="lbfgs")
        m1 = smf.mixedlm("lfp_fz ~ confusing + s", d, groups=d["all"], re_formula="0", vc_formula=vc).fit(reml=False, method="lbfgs")
    return round(2 * (m1.llf - m0.llf), 3)


def main() -> None:
    base = sys.argv[1]
    eeg = snippet_eeg(base)
    trials = pd.read_csv(RESULTS / "aoc_trials.csv", dtype={"participant": str})
    rows = []
    for name, ppl in sorted(perplexities(base).items(), key=lambda kv: SIZE.get(kv[0], 0)):
        df = pd.DataFrame({"ppl": ppl}).join(eeg.rename("eeg"), how="inner")
        obf = df[df.index.str.contains("-obf-")]
        r_obf, r_all = spearmanr(obf["ppl"], obf["eeg"]), spearmanr(df["ppl"], df["eeg"])
        rows.append({"model": name, "size_b": SIZE.get(name), "family": name.split("-")[0], "source": SOURCE.get(name, "published"),
                     "rho_confusing": round(float(r_obf.statistic), 3), "p_confusing": float(r_obf.pvalue),
                     "rho_all": round(float(r_all.statistic), 3), "p_all": float(r_all.pvalue), "n_snippets": int(len(df)),
                     "lr_single_trial": lr_gain(trials, ppl)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "study3_scaling.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
