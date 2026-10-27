"""Compare our AOI perplexities with the published ones for the same model (replication check)."""
import json
import sys

import pandas as pd
from scipy.stats import pearsonr, spearmanr

ours = pd.DataFrame(json.load(open(sys.argv[1])))
theirs = pd.read_csv(sys.argv[2])
m = ours.merge(theirs, left_on="file", right_on="File")
print("matched files:", len(m), "of", len(ours), "ours and", len(theirs), "theirs")
for a, b in (("max_aoi_ppl", "Max AOI Perplexity"), ("mean_aoi_ppl", "Mean AOI Perplexity"), ("overall_aoi_ppl", "Overall AOI Perplexity")):
    print(a, "pearson", round(pearsonr(m[a], m[b]).statistic, 4), "spearman", round(spearmanr(m[a], m[b]).statistic, 4),
          "median ratio", round((m[a] / m[b]).median(), 4))
