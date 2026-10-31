"""Who Reads This Code? Replay real eye movements on code and see how an assistant adapts its help.

Gaze data: EMIP (Bednarik et al., 2020), CC-BY 4.0. Notes: written once per (line, construct) by a language model and
selected for a developer profile by an RSA-style speaker (see src/study2.py). The demo makes no model calls.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
import constructs  # noqa: E402
from study2 import KNOWN, rsa_select  # noqa: E402

DATA = HERE / "data"
SOURCE = {"rectangle": ("Rectangle.java", "Rectangle"), "vehicle": ("Vehicle.java", "Vehicle")}
TYPE_NAME = {0: "reads line by line, rarely goes back", 1: "jumps between lines, often goes back"}
EXPERTISE = {0: "none", 1: "low", 2: "medium", 3: "high"}
GAZE, FLAG, INK = (194, 24, 91), (242, 183, 5), (23, 34, 43)

st.set_page_config(page_title="Who Reads This Code?", page_icon="👁️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, label, button, input { font-family: 'Atkinson Hyperlegible', sans-serif !important; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMainBlockContainer"] { padding-top: 3.2rem; max-width: 1280px; }
h1 { font-weight: 700; letter-spacing: -0.01em; color: #17222B; font-size: 2.1rem !important; margin-bottom: 0 !important; }
.lede { color: #3d4b57; font-size: 1.05rem; max-width: 64ch; margin: .3rem 0 1.2rem; }
.note { border-left: 4px solid #0F766E; background: #ffffff; padding: .55rem .8rem; margin: .45rem 0; border-radius: 0 6px 6px 0; }
.note b { color: #0F766E; }
.why { color: #3d4b57; font-size: .92rem; margin: .15rem 0 .1rem; }
.source { background: #ffffff; border: 1px solid #d5dde3; border-radius: 8px; padding: .7rem .95rem; margin: 0 0 1.1rem;
  font-size: .95rem; color: #2c3a45; max-width: 100%; }
.source p { margin: 0 0 .35rem; }
.source p:last-child { margin: 0; }
.source b { color: #17222B; }
.flagline { display: inline-block; background: #F2B705; color: #17222B; font-weight: 700; padding: 0 .4rem; border-radius: 3px; }
.project-popup { display: flex; align-items: center; gap: 12px; width: fit-content; max-width: 100%; margin: 0 auto 18px;
  padding: 8px 10px 8px 14px; background: #fff; color: #17222B; font-size: 14px; border: 1px solid #d5dde3;
  border-left: 4px solid #C2185B; border-radius: 10px; }
.project-popup a { padding: 6px 12px; background: #17222B; color: #fff !important; font-weight: 700;
  text-decoration: none !important; border-radius: 7px; }
@media (max-width: 640px) { .project-popup { flex-wrap: wrap; } }
</style>
<aside class="project-popup" aria-label="Portfolio project">
  <span>This app was developed by Amrina as part of a portfolio.</span>
  <a href="https://sayamrina.github.io/" target="_blank" rel="noopener">View the portfolio</a>
</aside>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    trials = json.load(open(DATA / "trials.json"))
    pools = json.load(open(DATA / "note_pool.json"))
    return trials, pools


@st.cache_data
def boxes(stimulus: str):
    tok, lines = [], []
    with open(DATA / f"{stimulus}.csv") as f:
        for r in csv.DictReader(f):
            b = tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            (tok if r["level"] == "element" else lines).append(b)
    return tok, lines


def line_stats(fixations: list[dict], until_ms: float) -> dict[int, dict]:
    """Dwell time and number of returns per line (1-based), using fixations that started before until_ms."""
    stats: dict[int, dict] = {}
    prev = None
    for f in fixations:
        if f["t"] > until_ms or f["line"] is None:
            continue
        line = f["line"] + 1
        s = stats.setdefault(line, {"dwell": 0.0, "visits": 0})
        s["dwell"] += f["d"]
        if line != prev:
            s["visits"] += 1
        prev = line
    return stats


def flagged(stats: dict[int, dict]) -> dict[int, str]:
    """Lines that look effortful for this reader: at least twice their own median dwell, or twice their median returns."""
    if len(stats) < 3:
        return {}
    med = statistics.median(s["dwell"] for s in stats.values())
    med_visits = statistics.median(s["visits"] for s in stats.values())
    out = {}
    for line, s in stats.items():
        reasons = []
        if s["dwell"] >= 2 * med:
            reasons.append(f"{s['dwell'] / 1000:.1f} s here, {s['dwell'] / med:.1f}× this reader’s typical line")
        if s["visits"] >= max(4, 2 * med_visits):
            reasons.append(f"came back {s['visits'] - 1} times")
        if reasons:
            out[line] = "; ".join(reasons)
    return out


def draw(stimulus: str, fixations: list[dict], until_ms: float, flags: dict[int, str]) -> Image.Image:
    img = Image.open(DATA / f"{stimulus}.jpg").convert("RGB")
    tok, lines = boxes(stimulus)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for line in flags:
        x1, y1, x2, y2 = lines[line - 1]
        d.rectangle((x1 - 8, y1 - 4, x2 + 8, y2 + 4), fill=FLAG + (95,))
    shown = [f for f in fixations if f["t"] <= until_ms]
    for a, b in zip(shown, shown[1:], strict=False):
        d.line((a["x"], a["y"], b["x"], b["y"]), fill=GAZE + (90,), width=2)
    for i, f in enumerate(shown):
        r = max(4, (f["d"] ** 0.5) * 0.9)
        alpha = 200 if i >= len(shown) - 1 else 120
        d.ellipse((f["x"] - r, f["y"] - r, f["x"] + r, f["y"] + r), fill=GAZE + (alpha // 2,), outline=GAZE + (alpha,), width=2)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    xs = [b[0] for b in lines] + [b[2] for b in lines]
    ys = [b[1] for b in lines] + [b[3] for b in lines]
    return img.crop((max(min(xs) - 60, 0), max(min(ys) - 50, 0), min(max(xs) + 80, img.width), min(max(ys) + 50, img.height)))


trials, pools = load()
labels = {i: f"Reader {t['participant']}, {t['program'].title()}: {TYPE_NAME[t['cluster']]} (expertise {EXPERTISE[t['expertise']]})"
          for i, t in enumerate(trials)}

with st.sidebar:
    st.markdown("### Reader")
    idx = st.selectbox("Recorded reading session", list(labels), index=2, format_func=labels.get,
                       help="Real sessions from the EMIP eye-tracking dataset; reader numbers are the dataset’s participant IDs.")
    trial = trials[idx]
    file, cls = SOURCE[trial["program"]]
    source = (DATA / file).read_text(encoding="utf-8", errors="ignore")
    lc = constructs.line_constructs(source, cls)
    present = sorted({c for cs in lc.values() for c in cs})
    st.markdown("### What this developer already knows")
    known = {c for c in present if st.checkbox(constructs.NAMES[c], value=c in KNOWN, key=f"k_{trial['program']}_{c}")}
    st.caption("Untick a construct and the assistant starts explaining it, only on the lines that use it.")

st.title("Who reads this code?")
st.markdown('<p class="lede">Replay how a real programmer’s eyes moved through a short Java program, and see which lines an '
            'assistant would explain to them, based on where they struggled and what they already know.</p>',
            unsafe_allow_html=True)
st.markdown("""
<div class="source" role="note" aria-label="Data and method">
<p><b>Data.</b> Every session shown here is a real recording from EMIP, the Eye Movements in Programming dataset
(Bednarik et al., 2020; 216 programmers, SMI RED250 eye tracker at 250 Hz, CC-BY 4.0,
<a href="https://osf.io/53kts" target="_blank" rel="noopener">osf.io/53kts</a>). The code is the dataset’s own Java stimulus.</p>
<p><b>Method.</b> Fixations are detected with a dispersion threshold (I-DT: 25 px, 80–1,200 ms) and mapped to the
dataset’s token and line boxes. A line is flagged when this reader looks at it at least twice as long as at their median
line, or returns to it unusually often. The notes were written once per line and construct by gpt-4o-mini, without any
profile; for the profile on the left, an RSA-style speaker picks which notes to show. The demo itself makes no model calls.</p>
</div>
""", unsafe_allow_html=True)

total = trial["fixations"][-1]["t"] + trial["fixations"][-1]["d"]
until = st.slider("Reading time", 0.0, total / 1000, total / 1000, 0.5, format="%.1f s", key=f"t_{idx}") * 1000
stats = line_stats(trial["fixations"], until)
flags = flagged(stats)

left, right = st.columns([1.55, 1])
with left:
    st.image(draw(trial["stimulus"], trial["fixations"], until, flags), width="stretch")
    n = sum(1 for f in trial["fixations"] if f["t"] <= until)
    st.caption(f"{n} fixations so far. Circle size shows how long the eyes rested; yellow lines look effortful for this reader. "
               f"Answer to the comprehension question: {'correct' if trial['correct'] else 'incorrect'}.")
with right:
    st.markdown("#### What the assistant noticed")
    chosen = rsa_select(lc, pools[file], known)
    lines_src = constructs.program_lines(source)
    if not flags:
        st.info("Nothing stands out yet. Move the slider further to let the reader look at more of the code.")
    for line in sorted(flags):
        notes = chosen.get(line, [])
        st.markdown(f'<span class="flagline">Line {line}</span> <code>{lines_src[line - 1][:60]}</code>', unsafe_allow_html=True)
        st.markdown(f'<p class="why">Why flagged: {flags[line]}.</p>', unsafe_allow_html=True)
        if notes:
            for nt in notes:
                st.markdown(f'<div class="note"><b>{constructs.NAMES[nt["construct"]].capitalize()}.</b> {nt["text"]}</div>',
                            unsafe_allow_html=True)
        else:
            unknown = [constructs.NAMES[c] for c in lc.get(line, []) if c not in known]
            st.markdown('<div class="note">No note: everything on this line is marked as known, so the assistant stays quiet '
                        'instead of repeating what the developer knows.</div>' if not unknown else
                        f'<div class="note">No note available for {", ".join(unknown)}.</div>', unsafe_allow_html=True)

with st.expander("What the studies found"):
    st.markdown("""
**Data.** EMIP, the Eye Movements in Programming dataset (Bednarik et al., 2020): 216 programmers reading two Java programs.
This analysis keeps 351 trials from 185 Java readers after planned exclusions.

**Study 1.** Tokens that a code language model finds less predictable (higher surprisal) are read longer: about 5 ms per bit
of total reading time (p = .004), after controlling for length, position and frequency; the effect is not robust in by-token
and per-program checks. It did not differ by expertise, and surprisal did not predict skipping.

**Study 1b.** Reading strategies form two types: line by line, or jumping back and forth. The non-linear type is more common
among more experienced readers (p = .03), in line with Busjahn et al. (2015).

**Study 2.** When a developer profile is pasted into the prompt, flipping one construct to “known” also changed the
explanation of 13 % of unrelated lines. Selecting notes per line with an RSA-style speaker changed no unrelated lines.

**Part B, open EEG data of Bergum et al. (2026).** AOI surprisal predicts the single-trial late frontal positivity
(p < .001) beyond the confusing/clean label, and first-pass dwell time, but not N400 or P600 amplitudes; experience does not
change the effect; and code models from 0.5B to 32B align about equally well (Qwen2.5-Coder-1.5B matches 32B).
Gaze adds nothing to perplexity for telling confusing from clean snippets (AUC .72 for both). An RSA listener learned
from EMIP reading times gives novices 26 notes and experienced readers 5 for the same program, and changes no unrelated lines.

| Open question in the Saarland studies | Answer here |
|---|---|
| Q1 Individual differences | The surprisal effect does not depend on experience |
| Q2 Single trials and eye tracking | Surprisal predicts single-trial EEG beyond the label, and dwell time |
| Q3 Other EEG components | No N400 or P600 effect |
| Q4 Model size | 1.5B matches 32B |
| Q5 Assistance not built | Faithful RSA selection and this demo; a user study is still missing |
| Q6 Hand-set RSA listeners | A listener learned from reading times stays faithful |

**In short:** code-model predictability tracks *where* programmers struggle, but not *which* programmer struggles more.
Personal differences have to be modelled, not assumed, and adaptation has to be built into how content is selected,
not requested in a prompt.

**Flagging rule used here.** A line is flagged when the reader’s dwell time is at least twice their own median line
dwell, or when they came back to it at least twice as often as to their median line
(and at least three times). This is a simple, readable rule, not a validated detector.

Full report, code and study plan: [github.com/sayamrina/who-reads-this-code](https://github.com/sayamrina/who-reads-this-code)
""")
