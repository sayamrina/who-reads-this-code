"""Study 1: does code-LLM surprisal predict reading times on code tokens, and does expertise change the effect?

Linear mixed-effects models on log reading times with a random intercept per participant and a variance component
for the token item (crossed random effects), as planned in PREREGISTRATION.md. Skipping is modelled with a logistic
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
    model = smf.mixedlm(f"y ~ {formula_rhs}", data, groups=data["participant"], vc_formula={"item": "0 + C(item)"})
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
        "robustness_H1_total_by_program": {prog: mixed(df[df["program"] == prog], "total_ms", f"surprisal_z + {CONTROLS}")
                                           for prog in ("rectangle", "vehicle")},
    }
    binned(df).to_csv(RESULTS / "study1_bins.csv", index=False)
    json.dump(out, open(RESULTS / "study1.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
