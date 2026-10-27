"""Compute token surprisal for the two Java stimuli and write results/surprisal.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import emip  # noqa: E402
import surprisal  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

if __name__ == "__main__":
    out = {}
    for program in ("Rectangle.java", "Vehicle.java"):
        source = (emip.PROGRAMS / program).read_text(encoding="utf-8", errors="ignore")
        out[program] = [w.__dict__ for w in surprisal.word_surprisal(source)]
        print(program, len(out[program]), "tokens")
    json.dump(out, open(RESULTS / "surprisal.json", "w"), indent=1)
