"""Read EMIP raw gaze data, detect fixations and map them to the token AOIs of each code stimulus.

EMIP ships, for every stimulus image, a CSV of areas of interest: one box per line and one box per whitespace-separated
token ("element"), in the same order as the tokens of the source file. That lets us align gaze with code tokens
without guessing positions from the image.
"""
import csv
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "emip" / "emip_dataset"
PROGRAMS = ROOT / "EMIP_DataCollection_Materials" / "emip_stimulus_programs"
SOURCE = {"rectangle_java": "Rectangle.java", "rectangle_java2": "Rectangle.java", "vehicle_java": "Vehicle.java",
          "vehicle_java2": "Vehicle.java", "rectangle_python": "rectangle.py", "vehicle_python": "vehicle.py",
          "rectangle_scala": "Rectangle.scala", "vehicle_scala": "Vehicle.scala"}
CODE_STIMULI = set(SOURCE)

# Fixation detection (dispersion threshold, I-DT; Salvucci and Goldberg, 2000)
MAX_DISPERSION_PX = 25
MIN_FIXATION_MS = 80
MAX_FIXATION_MS = 1200
HIT_MARGIN_PX = 10  # tolerance around a token box, roughly half a character


@dataclass
class Box:
    index: int  # token index in reading order (0-based), or line index for line boxes
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Fixation:
    start_ms: float
    duration_ms: float
    x: float
    y: float


def aois(stimulus: str) -> tuple[list[Box], list[Box]]:
    """(token boxes, line boxes) for a stimulus, from the CSV that ships with EMIP."""
    tokens, lines = [], []
    with open(ROOT / "stimuli" / f"{stimulus}.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            box = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
            if row["level"] == "element":
                tokens.append(Box(len(tokens), *box))
            elif row["level"] == "line":
                lines.append(Box(len(lines), *box))
    return tokens, lines


def source_tokens(stimulus: str) -> list[tuple[int, int, str]]:
    """(line number among non-empty lines, position in line, text) for every token, in AOI order."""
    text = (PROGRAMS / SOURCE[stimulus]).read_text(encoding="utf-8", errors="ignore")
    out, line_no = [], 0
    for line in text.split("\n"):
        parts = line.split()
        if not parts:
            continue
        out.extend((line_no, i, t) for i, t in enumerate(parts))
        line_no += 1
    return out


def read_trials(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    """Gaze samples (time in ms, x, y) per code stimulus shown to one participant.

    The screen showing a stimulus starts with a message naming the image and ends at the next message.
    Gaze is the mean of the valid eyes' point of regard; samples where both eyes are lost are dropped.
    """
    trials: dict[str, list] = {}
    current = None
    header = None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("##"):
                continue
            cols = line.rstrip("\n").split("\t")
            if header is None:
                header = {name: i for i, name in enumerate(cols)}
                ix = [header[k] for k in ("L POR X [px]", "L POR Y [px]", "R POR X [px]", "R POR Y [px]")]
                continue
            if len(cols) < 4:
                continue
            if cols[1] == "MSG":
                m = re.search(r"Message: (\w+)\.jpg", line)
                name = m.group(1) if m else None
                current = name if name in CODE_STIMULI else None
                if current:
                    trials.setdefault(current, [])
                continue
            if current is None or cols[1] != "SMP":
                continue
            try:
                lx, ly, rx, ry = (float(cols[i]) for i in ix)
            except (ValueError, IndexError):
                continue
            eyes = [(x, y) for x, y in ((lx, ly), (rx, ry)) if x > 0 and y > 0]
            t = float(cols[0]) / 1000.0  # microseconds to ms
            if eyes:
                trials[current].append((t, sum(e[0] for e in eyes) / len(eyes), sum(e[1] for e in eyes) / len(eyes)))
            else:
                trials[current].append((t, float("nan"), float("nan")))
    return trials


def fixations(samples: list[tuple[float, float, float]]) -> list[Fixation]:
    """Dispersion-threshold fixation detection; gaps from tracking loss end a fixation."""
    out: list[Fixation] = []
    window: list[tuple[float, float, float]] = []

    def flush():
        if len(window) >= 2:
            dur = window[-1][0] - window[0][0]
            if MIN_FIXATION_MS <= dur <= MAX_FIXATION_MS:
                xs, ys = [w[1] for w in window], [w[2] for w in window]
                out.append(Fixation(window[0][0], dur, sum(xs) / len(xs), sum(ys) / len(ys)))

    for s in samples:
        if s[1] != s[1]:  # NaN: tracking loss
            flush()
            window = []
            continue
        candidate = window + [s]
        xs, ys = [c[1] for c in candidate], [c[2] for c in candidate]
        if (max(xs) - min(xs)) + (max(ys) - min(ys)) <= MAX_DISPERSION_PX:
            window = candidate
        else:
            flush()
            window = [s]
    flush()
    return out


def hit(fix: Fixation, boxes: list[Box], margin: float = HIT_MARGIN_PX) -> int | None:
    """Index of the box that contains the fixation (within a margin); the nearest box wins if several do."""
    best, best_d = None, None
    for b in boxes:
        if b.x1 - margin <= fix.x <= b.x2 + margin and b.y1 - margin <= fix.y <= b.y2 + margin:
            cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
            d = (fix.x - cx) ** 2 + (fix.y - cy) ** 2
            if best is None or d < best_d:
                best, best_d = b.index, d
    return best


def tracking_loss(samples: list[tuple[float, float, float]]) -> float:
    return sum(1 for s in samples if s[1] != s[1]) / max(len(samples), 1)


def token_measures(fix_tokens: list[tuple[Fixation, int | None]], n_tokens: int) -> list[dict]:
    """First-pass gaze duration, total reading time and skipping for every token.

    First-pass gaze duration: the summed duration of the first run of consecutive fixations on a token, counted only
    if the token was entered before any token further along in reading order was fixated (as in text reading).
    """
    total = [0.0] * n_tokens
    first_pass = [0.0] * n_tokens
    count = [0] * n_tokens
    furthest = -1
    i = 0
    seq = [(f, t) for f, t in fix_tokens if t is not None]
    while i < len(seq):
        f, t = seq[i]
        j, run = i, 0.0
        while j < len(seq) and seq[j][1] == t:
            run += seq[j][0].duration_ms
            j += 1
        if count[t] == 0 and t > furthest:
            first_pass[t] = run
        for k in range(i, j):
            total[t] += seq[k][0].duration_ms
            count[t] += 1
        furthest = max(furthest, t)
        i = j
    return [{"token": k, "total_ms": total[k], "gaze_ms": first_pass[k], "fixations": count[k],
             "skipped": int(first_pass[k] == 0)} for k in range(n_tokens)]


def strategy_features(fix_tokens: list[tuple[Fixation, int | None]], token_lines: list[int]) -> dict:
    """Reading-strategy features of one trial (Busjahn et al., 2015): linearity, regressions, coverage."""
    seq = [t for _, t in fix_tokens if t is not None]
    if len(seq) < 10:
        return {}
    moves = list(zip(seq, seq[1:], strict=False))
    line_moves = [(token_lines[a], token_lines[b]) for a, b in moves if token_lines[a] != token_lines[b]]
    durations = [f.duration_ms for f, t in fix_tokens if t is not None]
    return {
        "n_fixations": len(seq),
        "mean_fix_ms": sum(durations) / len(durations),
        "regression_rate": sum(1 for a, b in moves if b < a) / len(moves),
        "next_line_rate": sum(1 for a, b in line_moves if b == a + 1) / max(len(line_moves), 1),
        "line_regression_rate": sum(1 for a, b in line_moves if b < a) / max(len(line_moves), 1),
        "coverage": len(set(seq)) / len(token_lines),
        "on_code_share": len(seq) / len(fix_tokens),
    }
