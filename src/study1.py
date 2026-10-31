"""Study 1: does code-LLM surprisal predict reading times on code tokens, and does expertise change the effect?

Linear mixed-effects models on log reading times with crossed random intercepts for participants and token items, as
planned in STUDY_PLAN.md. A by-item regression on token means is reported as a check. Skipping is modelled with a logistic
regression with participant-clustered standard errors.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

RESULTS = Path(__file__).resolve().parents[1] / "results"
CONTROLS = "length_z + line_z + pos_z + log_freq_z"


def prepare() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "tokens.csv")
    df = df[df["expertise"].notna()].copy()
    df["item"] = df["program"] + ":" + df["token"].astype(str)
    for col in ("surprisal", "length", "line", "pos", "log_freq"):
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std()
    df["expertise_c"] = df["expertise"] - df["expertise"].mean()
    df["high_expertise"] = (df["expertise"] >= 3).astype(int)
    return df


def mixed(df: pd.DataFrame, dv: str, formula_rhs: str) -> dict:
    data = df[df[dv] > 0].copy()
    data["y"] = np.log(data[dv])
    # Crossed random intercepts for participants and token items: one constant group, both as variance components
    data["all"] = 1
    model = smf.mixedlm(f"y ~ {formula_rhs}", data, groups=data["all"], re_formula="0",
                        vc_formula={"participant": "0 + C(participant)", "item": "0 + C(item)"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(reml=True, method="lbfgs")
    coefs = {k: {"b": round(float(fit.params[k]), 4), "se": round(float(fit.bse[k]), 4), "p": float(fit.pvalues[k])}
             for k in fit.params.index if k in ("surprisal_z", "surprisal_z:expertise_c", "expertise_c", "length_z")}
    # effect in ms per bit at the mean reading time: exp(b / sd_surprisal) on the geometric mean
    sd = df["surprisal"].std()
    gm = float(np.exp(data["y"].mean()))
    coefs["ms_per_bit_at_mean"] = round(gm * (np.exp(fit.params["surprisal_z"] / sd) - 1), 2)
    return {"n_obs": int(len(data)), "n_participants": int(data["participant"].nunique()), "converged": bool(fit.converged),
            "coefficients": coefs}


def skipping(df: pd.DataFrame) -> dict:
    model = smf.glm(f"skipped ~ surprisal_z * expertise_c + {CONTROLS}", df, family=sm.families.Binomial())
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["participant"])[0]})
    keys = ("surprisal_z", "surprisal_z:expertise_c", "expertise_c")
    return {"n_obs": int(len(df)), "coefficients": {k: {"b": round(float(fit.params[k]), 4), "se": round(float(fit.bse[k]), 4),
                                                        "p": float(fit.pvalues[k])} for k in keys}}


def by_item(df: pd.DataFrame) -> dict:
    """Check that does not depend on the random-effects structure: regress mean log total time per token on its predictors."""
    d = df[df["total_ms"] > 0].copy()
    d["y"] = np.log(d["total_ms"])
    items = d.groupby("item").agg(y=("y", "mean"), surprisal_z=("surprisal_z", "first"), length_z=("length_z", "first"),
                                  line_z=("line_z", "first"), pos_z=("pos_z", "first"), log_freq_z=("log_freq_z", "first"),
                                  n=("y", "size")).reset_index()
    r = smf.ols(f"y ~ surprisal_z + {CONTROLS}", items).fit()
    return {"n_items": int(len(items)), "b": round(float(r.params["surprisal_z"]), 4), "se": round(float(r.bse["surprisal_z"]), 4),
            "p": float(r.pvalues["surprisal_z"])}


def binned(df: pd.DataFrame) -> pd.DataFrame:
    """Mean total reading time by surprisal bin and expertise group, for the figure."""
    d = df[df["total_ms"] > 0].copy()
    d["bin"] = pd.qcut(d["surprisal"].rank(method="first"), 5, labels=False)
    d["group"] = np.where(d["expertise"] >= 3, "high expertise", "none to medium")
    return d.groupby(["group", "bin"]).agg(surprisal=("surprisal", "mean"), total_ms=("total_ms", "mean"),
                                          gaze_ms=("gaze_ms", lambda s: s[s > 0].mean()), n=("total_ms", "size")).reset_index()


def main() -> None:
    df = prepare()
    out = {
        "correlation_surprisal_length": round(float(df[["surprisal", "length"]].drop_duplicates().corr().iloc[0, 1]), 3),
        "H1_gaze_duration": mixed(df, "gaze_ms", f"surprisal_z + {CONTROLS}"),
        "H1_total_time": mixed(df, "total_ms", f"surprisal_z + {CONTROLS}"),
        "H2_gaze_duration": mixed(df, "gaze_ms", f"surprisal_z * expertise_c + {CONTROLS}"),
        "H2_total_time": mixed(df, "total_ms", f"surprisal_z * expertise_c + {CONTROLS}"),
        "H3_skipping": skipping(df),
        "check_H1_total_by_item": by_item(df),
        "robustness_H1_total_by_program": {prog: mixed(df[df["program"] == prog], "total_ms", f"surprisal_z + {CONTROLS}")
                                           for prog in ("rectangle", "vehicle")},
    }
    binned(df).to_csv(RESULTS / "study1_bins.csv", index=False)
    json.dump(out, open(RESULTS / "study1.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
