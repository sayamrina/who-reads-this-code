"""Build a single-trial table from the published FRP epochs of Bergum et al. (2026).

One row per participant × snippet: mean amplitude (µV) in three windows, first-pass dwell time in the AOI, years of programming
experience, and AOI perplexity from the replication package of Abdelsalam et al. (2025) and from our own model runs.
Usage: python -I src/aoc/build_trials.py <aoc_frp data dir>
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "results" / "aoc"
WINDOWS = {"lfp_fz": (["Fz"], 0.4, 0.8), "n400": (["Cz", "CPz", "Pz"], 0.3, 0.5), "p600": (["CPz", "Pz", "P3", "P4"], 0.6, 0.9)}
YEARS = {"<1 year": 0.5, "1-3 years": 2, "3-5 years": 4, "5-10 years": 7.5, ">10 years": 12, "10+ years": 12}


def participants(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    rows = []
    for _, r in raw.iloc[2:].iterrows():
        if pd.isna(r[0]):
            continue
        rows.append({"participant": str(r[0]).zfill(3), "years_label": str(r[7]).strip(), "recent_activity": str(r[8]).strip(),
                     "profession": str(r[6]).strip()})
    df = pd.DataFrame(rows)
    df["years"] = df["years_label"].map(YEARS)
    return df


def amplitudes(fif: str) -> dict:
    ep = mne.read_epochs(fif, verbose="error")
    data = ep.get_data(copy=True)[0] * 1e6  # volts to microvolts
    out = {}
    for name, (chs, t0, t1) in WINDOWS.items():
        idx = [ep.ch_names.index(c) for c in chs]
        mask = (ep.times >= t0) & (ep.times <= t1)
        out[name] = float(data[idx][:, mask].mean())
    return out


def main() -> None:
    base = sys.argv[1]
    rows = []
    for fif in sorted(glob.glob(f"{base}/results/results/EEG_FRP/epochs/*/faois_-300_1000_all_epoch_*_epo.fif")):
        m = re.search(r"_epoch_(\d{3})_(.+)_epo\.fif$", os.path.basename(fif))
        pid, snippet = m.group(1), m.group(2)
        rows.append({"participant": pid, "file": snippet, "confusing": int("-obf-" in snippet),
                     "pair": snippet.replace("-obf-", "-").replace("-clean-", "-"), **amplitudes(fif)})
    df = pd.DataFrame(rows)
    reading = pd.read_csv(f"{base}/results/results/EEG_FRP/fixation_analysis/all_trial_reading metrics.csv", sep=";",
                          dtype={"Participant": str})
    reading = reading.rename(columns={"Participant": "participant", "Snippet": "file", "first_pass_aoi": "first_pass_s"})
    reading["participant"] = reading["participant"].str.zfill(3)
    df = df.merge(reading, on=["participant", "file"], how="left")
    df = df.merge(participants(f"{base}/figure_data/figure data/Data Table2 participant information.xlsx"), on="participant", how="left")
    for f in glob.glob(str(OUT / "ppl_*.json")):
        model = Path(f).stem[4:]
        ppl = pd.DataFrame(json.load(open(f)))[["file", "max_aoi_ppl"]].rename(columns={"max_aoi_ppl": f"ppl_{model}"})
        df = df.merge(ppl, on="file", how="left")
    pairwise = f"{base}/llm-perplexity-AoC/model_results/Qwen2.5-Coder-32B/pairwise_analysis/pairwise_comparisons"
    theirs = pd.read_csv(f"{pairwise}/perplexity_stats.csv")
    df = df.merge(theirs[["File", "Max AOI Perplexity"]].rename(columns={"File": "file", "Max AOI Perplexity": "ppl_Qwen2.5-Coder-32B"}),
                  on="file", how="left")
    df.to_csv(OUT / "aoc_trials.csv", index=False)
    print(len(df), "trials,", df["participant"].nunique(), "participants,", df["file"].nunique(), "snippets")
    print(df[["lfp_fz", "n400", "p600", "first_pass_s", "years"]].describe().round(2).to_string())
    print("missing years:", df["years"].isna().sum(), "| labels:", sorted(df["years_label"].dropna().unique()))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
