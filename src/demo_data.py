"""Export a few real EMIP trials (fixations mapped to tokens) and the stimulus images for the demo app.

EMIP is licensed CC-BY 4.0 (Bednarik et al., 2020); the images and gaze data are redistributed with attribution.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import emip  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trials = pd.read_csv(ROOT / "results" / "trials.csv").merge(pd.read_csv(ROOT / "results" / "study1b_clusters.csv"),
                                                                 on=["participant", "stimulus"])
    good = trials[(trials["on_code_share"] > 0.85) & (trials["loss"] < 0.1) & (trials["n_fixations"].between(120, 400))]
    picks = []
    for program in ("rectangle", "vehicle"):
        for cluster in sorted(good["cluster"].unique()):
            d = good[(good["program"] == program) & (good["cluster"] == cluster)].sort_values("on_code_share", ascending=False)
            picks += d.head(2).to_dict("records")
    export = []
    for t in picks:
        samples = emip.read_trials(emip.ROOT / "rawdata" / f"{t['participant']}_rawdata.tsv")[t["stimulus"]]
        boxes, lines = emip.aois(t["stimulus"])
        fix = emip.fixations(samples)
        t0 = fix[0].start_ms if fix else 0
        export.append({"participant": str(t["participant"]), "stimulus": t["stimulus"], "program": t["program"],
                       "cluster": int(t["cluster"]), "expertise": int(t["expertise"]), "correct": int(t["correct"]),
                       "fixations": [{"t": round(f.start_ms - t0), "d": round(f.duration_ms), "x": round(f.x, 1), "y": round(f.y, 1),
                                      "token": emip.hit(f, boxes), "line": emip.hit(f, lines, 15)} for f in fix]})
        shutil.copy(emip.ROOT / "stimuli" / f"{t['stimulus']}.jpg", OUT / f"{t['stimulus']}.jpg")
        shutil.copy(emip.ROOT / "stimuli" / f"{t['stimulus']}.csv", OUT / f"{t['stimulus']}.csv")
    for prog in ("Rectangle.java", "Vehicle.java"):
        shutil.copy(emip.PROGRAMS / prog, OUT / prog)
    json.dump(export, open(OUT / "trials.json", "w"))
    shutil.copy(ROOT / "results" / "note_pool.json", OUT / "note_pool.json")
    print(len(export), "trials exported:", [(e["participant"], e["stimulus"], e["cluster"]) for e in export])


if __name__ == "__main__":
    main()
