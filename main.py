"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun
gta6-twitch-ccu-forecast-01).

Model: locally-weighted log-linear regression on the comparables.
For each comparable launch we take its release-day hourly curve and, for
each forecast hour h = 2..24, fit

    log(viewers_h) = a_h + b_h * log(viewers_1)

by weighted least squares, where a comparable's weight is a Gaussian
kernel on the log distance between its hour-1 count and the target's
hour-1 count (bandwidth 1.6 log units).  The kernel makes the fit local:
a small launch is predicted mostly from small launches, a huge launch
mostly from huge launches.  The slope b_h is near 1 for the early hours
(hour 2-3 track hour 1) and falls toward ~0.1 by the evening, because
the evening peak is set by the game's total audience while hour 1 only
sees the timezones that have unlocked by 00:00 UTC.

Quantiles come from the weighted residual spread: q_tau =
exp(mu_h + z_tau * sd_h * 0.8), z the standard normal quantiles, the 0.8
a calibration factor that keeps the 90% band honest on the practice
cases.  A game whose hour-1 count is far outside the comparables still
gets a sane forecast: the kernel concentrates on the nearest launches
and the slope extrapolates their level.

Deterministic: pure stdlib, no network, no randomness, sorted inputs,
fixed rounding.  Falls back to a persistence forecast with the house
spread when a case has too few usable comparables for an hour.

Usage: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves the repo root off sys.path; nothing to import
# from here, but keep the line harmless for local runs.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Gaussian kernel bandwidth (log units) for the local fit, and the
# calibration multiplier on the residual spread.
BANDWIDTH = 1.6
SPREAD_CAL = 0.8
# Minimum comparables needed to trust a local fit for an hour.
MIN_FIT = 6
# Fallback spread (house baseline's) when the local fit is unavailable.
FALLBACK_SIGMA = 0.15


def round2(x):
    """Round to 2 decimals, half up, returning an int for whole numbers."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def release_curve(comp):
    """{hour: viewers} for a comparable's UTC release day."""
    day = comp["day_start"][:10]
    out = {}
    for pt in comp.get("hourly") or []:
        frm = pt.get("from", "")
        if frm.startswith(day):
            try:
                h = int(frm[11:13]) + 1
            except (ValueError, IndexError):
                continue
            v = pt.get("viewers")
            if isinstance(v, (int, float)) and v > 0:
                out[h] = float(v)
    return out


def weighted_ols(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, sd)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    neff = sw * sw / sum(w * w for w in ws)
    if neff > 2:
        var = sum(w * r * r for w, r in zip(ws, resid)) / max(1.0, neff - 2)
    else:
        var = sum(r * r for r in resid) / max(1, len(resid) - 2)
    return a, b, math.sqrt(max(var, 0.01))


def hour_fit(comps, p1, h):
    """Local regression for one hour. Returns (mu, sd) in log space, or None
    when too few comparables cover the hour. The slope is capped to [0, 1.2]:
    a fit that inverts the size relation or explodes for a huge hour-1 count
    is noise, not signal, and the cap keeps the final case sane."""
    lp = math.log(max(1.0, p1))
    xs, ys, ws = [], [], []
    for cur in comps:
        v1 = cur.get(1)
        vh = cur.get(h)
        if v1 is None or vh is None or v1 <= 0 or vh <= 0:
            continue
        xs.append(math.log(v1))
        ys.append(math.log(vh))
        d = math.log(v1) - lp
        ws.append(math.exp(-(d * d) / (2.0 * BANDWIDTH * BANDWIDTH)))
    if len(xs) < MIN_FIT:
        return None
    fit = weighted_ols(xs, ys, ws)
    if fit is None:
        return None
    a, b, sd = fit
    b = min(max(b, 0.0), 1.2)
    return a + b * lp, sd


def forecast_hours(comps, p1):
    """Five quantiles per hour. The log-spread is smoothed over a 3-hour
    window first: one hour's noisy spread estimate must not make that
    hour's band alone too narrow or too wide."""
    mus, sds = {}, {}
    for h in range(2, 25):
        fit = hour_fit(comps, p1, h)
        if fit is None:
            mus[h] = math.log(max(1.0, p1))
            sds[h] = FALLBACK_SIGMA * math.sqrt(h - 1) / SPREAD_CAL
        else:
            mus[h], sds[h] = fit
    smooth = {}
    for h in range(2, 25):
        vals = [math.log(sds[hh]) for hh in (h - 1, h, h + 1) if hh in sds]
        smooth[h] = math.exp(sum(vals) / len(vals))
    out = {}
    for h in range(2, 25):
        qs = [math.exp(mus[h] + z * smooth[h] * SPREAD_CAL) for z in Z]
        qs = [round2(min(max(q, 0.01), 1e9)) for q in qs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        out[h] = qs
    return out


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    comps = []
    for comp in case.get("comparables") or []:
        cur = release_curve(comp)
        if 1 in cur:
            comps.append(cur)
    hours = []
    qs_by_hour = forecast_hours(comps, p1)
    for row in case["forecast"]:
        h = row["hour"]
        hours.append({"hour": h, "q": qs_by_hour[h]})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
