"""Read the per-model RQ2 correlation tables from the llm-perplexity-AoC replication package (data only)."""
import glob
import sys

import pandas as pd

base = sys.argv[1]
eeg = None
for m in sorted(glob.glob(f"{base}/model_results/*")):
    name = m.split("/")[-1]
    c = pd.read_csv(f"{m}/pairwise_analysis/scatter_plots/correlation_statistics.csv")
    d = pd.read_csv(f"{m}/pairwise_analysis/pairwise_comparisons/eeg_perplexity_analysis.csv")
    e = d[["pair_id", "clean_mean_eeg", "obf_mean_eeg"]].sort_values("pair_id").reset_index(drop=True)
    if eeg is None:
        eeg = e
    same = (eeg.merge(e, on="pair_id").eval("abs(obf_mean_eeg_x - obf_mean_eeg_y) < 1e-15")).mean()
    print(name, "pairs", len(d), "eeg identical:", round(same, 3))
    print(c[["comparison_type", "eeg_metric", "perplexity_metric", "spearman_r", "spearman_p"]].to_string(index=False))
