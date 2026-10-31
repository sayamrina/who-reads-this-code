"""Study 4: single-trial FRPs and first-pass dwell time against AOI surprisal, and their dependence on programming experience.

Linear mixed-effects models with crossed random intercepts for participants and snippet pairs, as planned in the
study plan addendum.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RESULTS = Path(__file__).resolve().parents[2] / "results" / "aoc"
MODEL = "ppl_Qwen2.5-Coder-32B"


def prepare(model_col: str = MODEL) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "aoc_trials.csv", dtype={"participant": str}).dropna(subset=[model_col])
    df["surprisal"] = np.log2(df[model_col])  # max AOI perplexity in bits
    df["surprisal_z"] = (df["surprisal"] - df["surprisal"].mean()) / df["surprisal"].std()
    df["years_c"] = df["years"] - df["years"].mean()
    df["log_first_pass"] = np.log(df["first_pass_s"])
    return df


def fit(df: pd.DataFrame, dv: str, rhs: str) -> dict:
    d = df.dropna(subset=[dv]).copy()
    # Crossed random intercepts for participants and snippet pairs: one constant group, both as variance components
    d["all"] = 1
    model = smf.mixedlm(f"{dv} ~ {rhs}", d, groups=d["all"], re_formula="0",
                        vc_formula={"participant": "0 + C(participant)", "pair": "0 + C(pair)"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = model.fit(reml=True, method="lbfgs")
    keep = [k for k in r.params.index if k not in ("Intercept",) and not k.endswith(" Var") and k != "Group Var"]
    return {"n": int(len(d)), "participants": int(d["participant"].nunique()), "converged": bool(r.converged),
            "coef": {k: {"b": round(float(r.params[k]), 4), "se": round(float(r.bse[k]), 4), "p": float(r.pvalues[k])} for k in keep}}


def by_item(df: pd.DataFrame, dv: str) -> dict:
    """Check that does not depend on the random-effects structure: snippet means against snippet surprisal."""
    items = df.dropna(subset=[dv]).groupby("file").agg(y=(dv, "mean"), surprisal_z=("surprisal_z", "first"),
                                                       confusing=("confusing", "first")).reset_index()
    r = smf.ols("y ~ surprisal_z + confusing", items).fit()
    return {"n_snippets": int(len(items)), "b": round(float(r.params["surprisal_z"]), 4), "se": round(float(r.bse["surprisal_z"]), 4),
            "p": float(r.pvalues["surprisal_z"])}


def by_experience(df: pd.DataFrame, dv: str) -> list[dict]:
    """Surprisal slope per experience group, for the figure."""
    out = []
    for label, d in df.groupby("years_label"):
        if d["participant"].nunique() >= 2:
            r = fit(d, dv, "surprisal_z + confusing")
            out.append({"years_label": label, "participants": r["participants"], "n": r["n"], **r["coef"]["surprisal_z"]})
    return out


def main() -> None:
    df = prepare()
    out = {
        "H6_lfp_surprisal": fit(df, "lfp_fz", "surprisal_z"),
        "H6_lfp_surprisal_and_condition": fit(df, "lfp_fz", "surprisal_z + confusing"),
        "H7_lfp_experience": fit(df, "lfp_fz", "surprisal_z * years_c"),
        "H8_n400": fit(df, "n400", "surprisal_z * years_c"),
        "H8_p600": fit(df, "p600", "surprisal_z * years_c"),
        "H9_first_pass": fit(df, "log_first_pass", "surprisal_z * years_c"),
        "H9_first_pass_and_condition": fit(df, "log_first_pass", "surprisal_z * years_c + confusing"),
        "check_lfp_by_item": by_item(df, "lfp_fz"),
        "check_first_pass_by_item": by_item(df, "log_first_pass"),
        "lfp_slope_by_experience": by_experience(df, "lfp_fz"),
        "first_pass_slope_by_experience": by_experience(df, "log_first_pass"),
        "participants_by_experience": df.groupby("years_label")["participant"].nunique().to_dict(),
    }
    json.dump(out, open(RESULTS / "study4.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k.startswith("H")}, indent=1))
    print(json.dumps(out["participants_by_experience"]))


if __name__ == "__main__":
    main()
