"""Study 6 (exploratory): an RSA listener learned from human reading data instead of hand-set values.

For each programming construct, compare how long lines that contain it are read by lower- and by higher-expertise EMIP readers,
after removing what token length, position, frequency and surprisal already explain. A construct whose lines cost novices
extra time is treated as likely unknown to a novice. This gives, per construct and expertise level, a probability that the
line is understood without help, which replaces the fixed listener of Study 2.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
import constructs  # noqa: E402
import emip  # noqa: E402
import study2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROGRAMS = {"rectangle": ("Rectangle.java", "Rectangle"), "vehicle": ("Vehicle.java", "Vehicle")}


def line_residuals() -> pd.DataFrame:
    """Residual log total reading time per participant × line, after token-level controls."""
    df = pd.read_csv(RESULTS / "tokens.csv")
    df = df[(df["total_ms"] > 0) & df["expertise"].notna()].copy()
    df["y"] = np.log(df["total_ms"])
    fit = smf.ols("y ~ length + line + pos + log_freq + surprisal", df).fit()
    df["resid"] = fit.resid
    return df.groupby(["participant", "program", "line", "expertise"], as_index=False)["resid"].sum()


def construct_costs(lines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for program, (file, cls) in PROGRAMS.items():
        source = (emip.PROGRAMS / file).read_text(encoding="utf-8", errors="ignore")
        for n, cs in constructs.line_constructs(source, cls).items():
            for c in cs:
                rows.append({"program": program, "line": n - 1, "construct": c})  # tokens.csv numbers lines from 0
    m = lines.merge(pd.DataFrame(rows), on=["program", "line"])
    m["novice"] = (m["expertise"] <= 1).astype(int)  # none or low
    out = []
    for c, d in m.groupby("construct"):
        nov, exp = d.loc[d["novice"] == 1, "resid"], d.loc[d["novice"] == 0, "resid"]
        if len(nov) < 20 or len(exp) < 20:
            continue
        out.append({"construct": c, "novice_extra": float(nov.mean() - exp.mean()), "n_novice": int(len(nov)), "n_other": int(len(exp))})
    return pd.DataFrame(out).sort_values("novice_extra", ascending=False)


def learned_known(costs: pd.DataFrame, level: str) -> dict[str, float]:
    """P(understood without a note) per construct: logistic in the novice extra cost, centred on the median construct."""
    z = (costs["novice_extra"] - costs["novice_extra"].median()) / costs["novice_extra"].std()
    if level == "novice":
        p = 1 / (1 + np.exp(1.5 * z))  # costly constructs are probably unknown to a novice
    else:
        p = 0.9 + 0.05 / (1 + np.exp(z))  # experienced readers know nearly everything
    return dict(zip(costs["construct"], p.round(3), strict=True))


def select(lc: dict, pool: list[dict], p_known: dict[str, float]) -> dict:
    """RSA-style selection with a learned listener: expected comprehension instead of a known/unknown switch."""
    def listener(cs, explained):
        prob = 1.0
        for c in cs:
            known = p_known.get(c, 0.5)
            prob *= study2.P_UNDERSTOOD if c in explained else known * study2.P_UNDERSTOOD + (1 - known) * study2.P_MISSED
        return prob
    by_line = {}
    for n in pool:
        by_line.setdefault(n["line"], []).append(n)
    chosen = {}
    for line, cs in lc.items():
        notes = by_line.get(line, [])
        best, best_u = [], None
        for mask in range(1 << len(notes)):
            subset = [notes[i] for i in range(len(notes)) if mask >> i & 1]
            words = sum(len(n["text"].split()) for n in subset)
            u = study2.ALPHA * math.log(listener(cs, {n["construct"] for n in subset})) - study2.COST_PER_WORD * study2.ALPHA * words
            if best_u is None or u > best_u + 1e-9:
                best, best_u = subset, u
        if best:
            chosen[line] = best
    return chosen


def main() -> None:
    costs = construct_costs(line_residuals())
    pools = json.load(open(RESULTS / "note_pool.json"))
    out = {"construct_costs": costs.round(4).to_dict("records"), "programs": {}}
    for level in ("novice", "experienced"):
        out[f"p_known_{level}"] = learned_known(costs, level)
    for file, cls in PROGRAMS.values():
        source = (emip.PROGRAMS / file).read_text(encoding="utf-8", errors="ignore")
        lc = constructs.line_constructs(source, cls)
        res = {}
        for level in ("novice", "experienced"):
            ch = select(lc, pools[file], out[f"p_known_{level}"])
            res[level] = {"lines_explained": sorted(ch), "notes": sum(len(v) for v in ch.values()),
                          "constructs_explained": sorted({n["construct"] for v in ch.values() for n in v})}
        # faithfulness: raise one construct's learned probability to "known" and count changed lines elsewhere
        base = select(lc, pools[file], out["p_known_novice"])
        tgt_changed = untgt_changed = tgt = untgt = 0
        for c in sorted({c for cs in lc.values() for c in cs}):
            pk = dict(out["p_known_novice"])
            pk[c] = 1.0
            after = select(lc, pools[file], pk)
            ch = study2.changed_lines(base, after, list(lc))
            target = {n for n, cs in lc.items() if c in cs}
            tgt += len(target)
            untgt += len(lc) - len(target)
            tgt_changed += len(ch & target)
            untgt_changed += len(ch - target)
        res["faithfulness"] = {"targeted_change_rate": round(tgt_changed / tgt, 3),
                               "untargeted_change_rate": round(untgt_changed / untgt, 3)}
        out["programs"][file] = res
    json.dump(out, open(RESULTS / "study6.json", "w"), indent=1)
    print(costs.round(3).to_string(index=False))
    print(json.dumps(out["programs"], indent=1))


if __name__ == "__main__":
    main()
