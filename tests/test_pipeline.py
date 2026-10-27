"""Tests for the parts of the pipeline whose correctness the results depend on (no data download or model needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import constructs  # noqa: E402
import emip  # noqa: E402
import surprisal  # noqa: E402
from study2 import changed_lines, listener, rsa_select  # noqa: E402


def samples(points, start=0.0, step=4.0):
    """points: list of (x, y, n_samples); returns 250 Hz samples."""
    out, t = [], start
    for x, y, n in points:
        for _ in range(n):
            out.append((t, x, y))
            t += step
    return out


def test_fixation_detection_splits_on_movement_and_tracking_loss():
    s = samples([(100, 100, 30), (400, 100, 30)])  # two 116 ms fixations
    s += [(s[-1][0] + 4, float("nan"), float("nan"))] + samples([(400, 300, 10)], start=s[-1][0] + 8)  # 36 ms: too short
    fix = emip.fixations(s)
    assert [round(f.x) for f in fix] == [100, 400]
    assert all(emip.MIN_FIXATION_MS <= f.duration_ms <= emip.MAX_FIXATION_MS for f in fix)


def test_hit_prefers_the_nearest_box_within_the_margin():
    boxes = [emip.Box(0, 0, 0, 50, 20), emip.Box(1, 55, 0, 100, 20)]
    assert emip.hit(emip.Fixation(0, 100, 53, 10), boxes) == 1
    assert emip.hit(emip.Fixation(0, 100, 300, 300), boxes) is None


def test_first_pass_gaze_ignores_tokens_entered_after_reading_further():
    f = lambda d: emip.Fixation(0, d, 0, 0)  # noqa: E731
    # token 0 (100+50 ms), then 2 (200 ms), then back to 1 (80 ms): token 1 was entered after token 2, so no first pass
    seq = [(f(100), 0), (f(50), 0), (f(200), 2), (f(80), 1), (f(120), 0)]
    m = {r["token"]: r for r in emip.token_measures(seq, 3)}
    assert m[0]["gaze_ms"] == 150 and m[0]["total_ms"] == 270
    assert m[1]["gaze_ms"] == 0 and m[1]["total_ms"] == 80 and m[1]["skipped"] == 1
    assert m[2]["gaze_ms"] == 200


def test_words_match_whitespace_tokens_and_offsets():
    src = "int x = 1 ;\n\n  return x ;"
    ws = surprisal.words(src)
    assert [w[4] for w in ws] == ["int", "x", "=", "1", ";", "return", "x", ";"]
    assert all(src[s:e] == t for _, _, s, e, t in ws)
    assert ws[5][0] == 3  # blank lines keep their line number


def test_construct_rules_on_stimulus_lines():
    assert constructs.detect("this.x1 = x1 ;", "Rectangle") == ["this_field"]
    assert "constructor" in constructs.detect("public Rectangle ( int x1 , int y1 ) {", "Rectangle")
    assert set(constructs.detect("public static void main ( String [ ] args ) {", "Rectangle")) == {"main", "array"}


def test_rsa_selection_is_local_when_one_construct_flips():
    lc = {1: ["this_field"], 2: ["new"], 3: ["this_field", "arithmetic"], 4: []}
    pool = [{"line": 1, "construct": "this_field", "text": "this refers to the object"},
            {"line": 2, "construct": "new", "text": "new creates an object"},
            {"line": 3, "construct": "this_field", "text": "reads the field of this object"},
            {"line": 3, "construct": "arithmetic", "text": "subtracts two numbers"}]
    before = rsa_select(lc, pool, {"arithmetic"})
    after = rsa_select(lc, pool, {"arithmetic", "this_field"})
    assert set(before) == {1, 2, 3} and set(after) == {2}
    assert changed_lines(before, after, list(lc)) == {1, 3}
    assert listener(["new"], set(), {"new"}) > listener(["new"], set(), set())
