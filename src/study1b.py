"""Study 1b (exploratory): interpretable reader types from reading-strategy features.

k-means on standardised per-trial features, k chosen by silhouette among 2–5; each cluster is described by its
centroid in plain language, and cluster membership is compared with self-rated expertise and answer correctness.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RESULTS = Path(__file__).resolve().parents[1] / "results"
FEATURES = ["mean_fix_ms", "regression_rate", "next_line_rate", "line_regression_rate", "coverage"]
LABEL = {"mean_fix_ms": ("longer fixations", "shorter fixations"),
         "regression_rate": ("many regressions", "few regressions"),
         "next_line_rate": ("reads line by line", "jumps between distant lines"),
         "line_regression_rate": ("often goes back to earlier lines", "rarely goes back"),
         "coverage": ("reads most of the code", "reads selectively")}


def describe(z_centroid: pd.Series) -> str:
    """The two most distinctive features of a cluster, in words."""
    top = z_centroid.abs().sort_values(ascending=False).index[:2]
    return "; ".join(LABEL[f][0] if z_centroid[f] > 0 else LABEL[f][1] for f in top)


def main() -> None:
    df = pd.read_csv(RESULTS / "trials.csv").dropna(subset=FEATURES + ["expertise"])
    X = StandardScaler().fit_transform(df[FEATURES])
    scores = {k: float(silhouette_score(X, KMeans(k, n_init=20, random_state=0).fit_predict(X))) for k in range(2, 6)}
    k = max(scores, key=scores.get)
    km = KMeans(k, n_init=20, random_state=0).fit(X)
    df["cluster"] = km.labels_
    z = pd.DataFrame(km.cluster_centers_, columns=FEATURES)
    raw = df.groupby("cluster")[FEATURES].mean()
    clusters = []
    for c in range(k):
        sub = df[df["cluster"] == c]
        clusters.append({"cluster": c, "n_trials": int(len(sub)), "description": describe(z.loc[c]),
                         "mean_expertise": round(float(sub["expertise"].mean()), 2),
                         "share_high_expertise": round(float((sub["expertise"] >= 3).mean()), 2),
                         "share_correct": round(float(sub["correct"].mean()), 2),
                         "centroid": {f: round(float(raw.loc[c, f]), 3) for f in FEATURES}})
    groups = [df.loc[df["cluster"] == c, "expertise"] for c in range(k)]
    kw = kruskal(*groups)
    chi = chi2_contingency(pd.crosstab(df["cluster"], df["correct"]))
    out = {"silhouette": {str(a): round(b, 3) for a, b in scores.items()}, "k": k, "clusters": clusters,
           "expertise_by_cluster_kruskal": {"H": round(float(kw.statistic), 3), "p": float(kw.pvalue)},
           "correct_by_cluster_chi2": {"chi2": round(float(chi[0]), 3), "p": float(chi[1])},
           "spearman_features_vs_years": {f: round(float(df[[f, "years"]].corr(method="spearman").iloc[0, 1]), 3) for f in FEATURES}}
    df[["participant", "stimulus", "cluster"]].to_csv(RESULTS / "study1b_clusters.csv", index=False)
    json.dump(out, open(RESULTS / "study1b.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    np.random.seed(0)
    main()
