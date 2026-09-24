#!/usr/bin/env python3
"""
ecmwf_mos.py -- station MOS built from free ECMWF IFS open data.

Pipeline
  probe    Live-check the ECMWF archive and IEM ASOS endpoints. Run this first.
  scan     Show when each configured parameter appears in the archive.
  fetch    Pull IFS HRES point values for the stations over a date range.
           Only the needed GRIB fields are downloaded, via byte ranges from
           the .index files. Only station values are stored on disk.
  obs      Pull ASOS/METAR obs from the Iowa Environmental Mesonet (IEM).
  train    Fit a ridge-regression MOS per station x cycle x lead x predictand.
           Prints held-out verification: raw IFS vs MOS, MAE and bias.
  predict  Apply the MOS to the latest complete run. Writes a CSV and a
           MAV-style HTML bulletin. Needs only mos_models.json (no sklearn),
           so it can run from GitHub Actions.

Predictands: 2 m temperature (F), 2 m dewpoint (F), 10 m wind speed (kt).

Data licence: ECMWF open data is CC-BY-4.0. Attribute ECMWF on any product.

Install
  pip install requests numpy pandas scikit-learn joblib eccodes

Typical run
  python ecmwf_mos.py probe
  python ecmwf_mos.py scan
  python ecmwf_mos.py fetch --start 2024-03-01 --end 2026-09-20
  python ecmwf_mos.py obs   --start 2024-03-01 --end 2026-09-24
  python ecmwf_mos.py train
  python ecmwf_mos.py predict
"""
from __future__ import annotations

import argparse
import html
import concurrent.futures as cf
import datetime as dt
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_URLS = {
    "aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",   # full archive
    "google": "https://storage.googleapis.com/ecmwf-open-data",
    "ecmwf": "https://data.ecmwf.int/forecasts",                       # recent days only
}

# IEM uses 3-letter IDs for US stations.
STATIONS = {
    "XMR": (28.4675, -80.5667),   # CCSFS Skid Strip
    "TTS": (28.6150, -80.6945),   # NASA Shuttle Landing Facility
    "COF": (28.2349, -80.6101),   # Patrick SFB
    "MLB": (28.1028, -80.6453),   # Melbourne Intl
}

SFC_PARAMS = ["2t", "2d", "10u", "10v", "msl", "tcc", "tp", "10fg"]
PL_PARAMS = [("t", 850), ("r", 850), ("r", 700)]

# MAV layout: 3-hourly to 60 h, then 66 and 72 h (21 columns).
DEFAULT_STEPS = "6-60/3,66,72"
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
OBS_VARS = ["tmpf", "dwpf", "sknt", "gust"]

# predictand -> raw model column, used for the raw-vs-MOS comparison
TARGETS = {"tmpf": "t2m", "dwpf": "d2m", "sknt": "ws10"}

KT_PER_MS = 1.943844

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "ecmwf_mos.py (station MOS research)"


def log(msg: str) -> None:
    print(f"[{dt.datetime.utcnow():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
def http_get(url, *, headers=None, params=None, timeout=60, tries=6):
    """GET with retry/backoff. Returns Response, or None on 404/403 (absent).
    429s honour Retry-After and back off harder (IEM rate-limits)."""
    for k in range(tries):
        wait = min(2 ** k * 2, 60)
        try:
            r = SESSION.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code in (403, 404):
                return None
            if r.status_code in (200, 206):
                return r
            if r.status_code == 429:
                ra = r.headers.get("Retry-After", "")
                wait = int(ra) if ra.isdigit() else min(10 * 2 ** k, 120)
                raise requests.HTTPError("HTTP 429")
            if r.status_code in (500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if k == tries - 1:
                raise
            log(f"  retry {k + 1}/{tries - 1} in {wait}s ({e}) {url.split('?')[0]}")
            time.sleep(wait)
    return None


# --------------------------------------------------------------------------
# ECMWF open data: URLs, index, byte ranges
# --------------------------------------------------------------------------
def run_url(base: str, run: dt.datetime, step: int, ext: str) -> str:
    # 00/12 UTC HRES lives under stream "oper" both before and after the
    # 2026-05-12 (IFS 50r1) restructure; 06/18 moved from "scda" to "oper".
    # This script only uses 00/12 so the path is stable across the archive.
    return (f"{base}/{run:%Y%m%d}/{run:%H}z/ifs/0p25/oper/"
            f"{run:%Y%m%d%H}0000-{step}h-oper-fc.{ext}")


def fetch_index(base, run, step):
    r = http_get(run_url(base, run, step, "index"))
    if r is None:
        return None
    return [json.loads(line) for line in r.text.splitlines() if line.strip()]


def field_name(entry) -> str | None:
    """Map an index entry to our variable name, or None if not wanted."""
    p, lt = entry.get("param"), entry.get("levtype")
    if lt == "sfc" and p in SFC_PARAMS:
        return p
    if lt == "pl":
        try:
            lev = int(entry.get("levelist"))
        except (TypeError, ValueError):
            return None
        if (p, lev) in PL_PARAMS:
            return f"{p}{lev}"
    return None


def get_field_bytes(base, run, step, entry) -> bytes:
    o, n = int(entry["_offset"]), int(entry["_length"])
    r = http_get(run_url(base, run, step, "grib2"),
                 headers={"Range": f"bytes={o}-{o + n - 1}"}, timeout=120)
    if r is None:
        raise RuntimeError("GRIB byte range not found")
    return r.content


# --------------------------------------------------------------------------
# GRIB decoding + bilinear interpolation to stations
# --------------------------------------------------------------------------
def point_values(msg: bytes, stations=STATIONS) -> dict[str, float]:
    import eccodes

    h = eccodes.codes_new_from_message(msg)
    try:
        ni = eccodes.codes_get(h, "Ni")
        nj = eccodes.codes_get(h, "Nj")
        lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        di = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
        dj = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
        jpos = eccodes.codes_get(h, "jScansPositively")
        vals = eccodes.codes_get_values(h).astype(float)
        if eccodes.codes_get(h, "bitmapPresent"):
            vals[vals == eccodes.codes_get(h, "missingValue")] = np.nan
    finally:
        eccodes.codes_release(h)

    grid = vals.reshape(nj, ni)
    out = {}
    for sid, (lat, lon) in stations.items():
        x = ((lon - lon0) % 360.0) / di
        y = (lat0 - lat) / dj if jpos == 0 else (lat - lat0) / dj
        i0, j0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - i0, y - j0
        i1, j1 = (i0 + 1) % ni, min(j0 + 1, nj - 1)
        out[sid] = float((1 - fx) * (1 - fy) * grid[j0, i0] + fx * (1 - fy) * grid[j0, i1]
                         + (1 - fx) * fy * grid[j1, i0] + fx * fy * grid[j1, i1])
    return out


# --------------------------------------------------------------------------
# fetch: one run -> one CSV of station point values
# --------------------------------------------------------------------------
def parse_steps(spec: str) -> list[int]:
    steps = []
    for part in spec.split(","):
        if "-" in part:
            rng, _, inc = part.partition("/")
            a, b = map(int, rng.split("-"))
            steps += list(range(a, b + 1, int(inc or 3)))
        else:
            steps.append(int(part))
    return sorted(set(steps))


def fetch_run(base, run, steps, model_dir: Path, workers=8, force=False) -> Path | None:
    out = model_dir / f"{run:%Y%m%d%H}.csv"
    miss = model_dir / f"{run:%Y%m%d%H}.missing"
    if out.exists() and not force:
        return out
    if miss.exists() and not force:
        return None

    tasks, seen_missing = [], set()
    for step in steps:
        idx = fetch_index(base, run, step)
        if idx is None:
            log(f"  {run:%Y%m%d%H} +{step}h: no index")
            continue
        found = {}
        for e in idx:
            name = field_name(e)
            if name and name not in found:
                found[name] = e
        wanted = SFC_PARAMS + [f"{p}{lev}" for p, lev in PL_PARAMS]
        seen_missing |= {w for w in wanted if w not in found}
        tasks += [(step, name, e) for name, e in found.items()]

    if not tasks:
        # Only mark as permanently missing once the run is old enough that
        # it should have been published.
        if dt.datetime.utcnow() - run > dt.timedelta(days=2):
            miss.touch()
        return None
    if seen_missing:
        log(f"  {run:%Y%m%d%H}: not in index for some steps: {sorted(seen_missing)}")

    def work(t):
        step, name, e = t
        vals = point_values(get_field_bytes(base, run, step, e))
        return [(run.strftime("%Y-%m-%d %H:%M"), step, sid, name, v)
                for sid, v in vals.items()]

    rows = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(work, tasks):
            rows += res

    df = pd.DataFrame(rows, columns=["run", "step", "station", "var", "value"])
    tmp = out.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    return out


def iter_runs(start: dt.date, end: dt.date, cycles):
    d = start
    while d <= end:
        for c in cycles:
            yield dt.datetime(d.year, d.month, d.day, int(c))
        d += dt.timedelta(days=1)


def cmd_fetch(a):
    model_dir = Path(a.data_dir) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    base, steps = BASE_URLS[a.source], parse_steps(a.steps)
    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    if start < dt.date(2024, 3, 1):
        log("WARNING: 0.25 deg IFS under /ifs/ paths starts around early 2024; "
            "earlier dates will likely come back missing.")
    runs = list(iter_runs(start, end, a.cycles))
    log(f"{len(runs)} runs x {len(steps)} steps from {a.source}")
    t0 = time.time()
    for k, run in enumerate(runs, 1):
        try:
            p = fetch_run(base, run, steps, model_dir, a.workers, a.force)
            status = "ok" if p else "missing"
        except Exception as e:  # keep going; rerun later fills the gap
            status = f"ERROR {e}"
        rate = (time.time() - t0) / k
        log(f"{k}/{len(runs)} {run:%Y%m%d%H} {status}  "
            f"(~{rate * (len(runs) - k) / 3600:.1f} h left)")


# --------------------------------------------------------------------------
# obs: IEM ASOS
# --------------------------------------------------------------------------
def fetch_iem(sid, start: dt.date, end: dt.date) -> pd.DataFrame:
    params = {
        "station": sid, "data": OBS_VARS, "tz": "Etc/UTC", "format": "onlycomma",
        "latlon": "no", "missing": "M", "trace": "T", "direct": "no",
        "report_type": ["3", "4"],  # routine + specials
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
    }
    r = http_get(IEM_URL, params=params, timeout=300)
    if r is None or not r.text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(r.text), na_values=["M", "T"])
    df["valid"] = pd.to_datetime(df["valid"])
    return df


def cmd_obs(a):
    obs_dir = Path(a.data_dir) / "obs"
    obs_dir.mkdir(parents=True, exist_ok=True)
    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    for sid in STATIONS:
        chunks, s = [], start
        while s <= end:  # one-year chunks, sequential, to be polite to IEM
            e = min(dt.date(s.year, 12, 31), end) + dt.timedelta(days=1)
            log(f"IEM {sid} {s} -> {e}")
            chunks.append(fetch_iem(sid, s, e))
            s = dt.date(s.year + 1, 1, 1)
            time.sleep(5)
        df = pd.concat([c for c in chunks if not c.empty] or [pd.DataFrame()])
        path = obs_dir / f"{sid}.csv"
        if path.exists() and not df.empty:
            old = pd.read_csv(path, parse_dates=["valid"])
            df = pd.concat([old, df])
        if not df.empty:
            df = df.drop_duplicates(["station", "valid"]).sort_values("valid")
            df.to_csv(path, index=False)
        log(f"{sid}: {len(df)} obs saved")


# --------------------------------------------------------------------------
# Building the training table
# --------------------------------------------------------------------------
def load_model(model_dir: Path, runs=None) -> pd.DataFrame:
    files = sorted(model_dir.glob("*.csv"))
    if runs is not None:
        names = {f"{r:%Y%m%d%H}.csv" for r in runs}
        files = [f for f in files if f.name in names]
    if not files:
        return pd.DataFrame()
    long = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    long["run"] = pd.to_datetime(long["run"], format="ISO8601")
    wide = long.pivot_table(index=["run", "step", "station"], columns="var",
                            values="value").reset_index()
    wide.columns.name = None
    return derive(wide)


def derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station", "run", "step"]).copy()
    k2f = lambda k: (k - 273.15) * 1.8 + 32.0
    if "2t" in df:  df["t2m"] = k2f(df["2t"])
    if "2d" in df:  df["d2m"] = k2f(df["2d"])
    if "10u" in df and "10v" in df:
        df["u10"], df["v10"] = df["10u"] * KT_PER_MS, df["10v"] * KT_PER_MS
        df["ws10"] = np.hypot(df["u10"], df["v10"])
    if "10fg" in df: df["gust10"] = df["10fg"] * KT_PER_MS
    if "msl" in df:  df["mslp"] = df["msl"] / 100.0
    if "t850" in df: df["t850c"] = df["t850"] - 273.15
    if "tp" in df:
        # tp is accumulated from t=0; convert to accumulation since the
        # previous fetched step (first step: since t=0), in mm.
        tp = df["tp"] * 1000.0
        df["tp_step"] = tp - tp.groupby([df["station"], df["run"]]).shift(1).fillna(0.0)
        df["tp_step"] = df["tp_step"].clip(lower=0)
    df["cycle"] = df["run"].dt.hour
    df["valid"] = df["run"] + pd.to_timedelta(df["step"], unit="h")
    doy = df["valid"].dt.dayofyear
    df["doy_s"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_c"] = np.cos(2 * np.pi * doy / 365.25)
    return df


FEATURES = ["t2m", "d2m", "u10", "v10", "ws10", "gust10", "mslp", "tcc",
            "tp_step", "t850c", "r850", "r700", "doy_s", "doy_c"]


def load_obs(obs_dir: Path) -> pd.DataFrame:
    files = sorted(obs_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    obs = pd.concat((pd.read_csv(f, parse_dates=["valid"]) for f in files))
    return obs.rename(columns={"valid": "obs_time"})


def pair(model: pd.DataFrame, obs: pd.DataFrame, tol_min=30) -> pd.DataFrame:
    """Match each forecast valid time to the nearest ob within +/- tol."""
    parts = []
    for sid, m in model.groupby("station"):
        o = obs[obs["station"] == sid].sort_values("obs_time")
        if o.empty:
            continue
        m = m.sort_values("valid")
        parts.append(pd.merge_asof(
            m, o[["obs_time"] + OBS_VARS], left_on="valid", right_on="obs_time",
            direction="nearest", tolerance=pd.Timedelta(minutes=tol_min)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------
def cmd_train(a):
    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    root = Path(a.data_dir)
    model = load_model(root / "model")
    obs = load_obs(root / "obs")
    if model.empty or obs.empty:
        sys.exit("Need both fetched model data and obs. Run fetch and obs first.")
    df = pair(model, obs)
    feats = [f for f in FEATURES if f in df.columns]
    log(f"paired rows: {len(df):,}  features: {feats}")

    # A param that only exists for part of the archive would otherwise
    # knock out every earlier row. Keep a feature only if it is present in
    # >= --min-coverage of the rows for that group; drop it otherwise.
    models, verif, dropped = {}, [], {}
    for (sid, cyc, step), g in df.groupby(["station", "cycle", "step"]):
        for tgt, raw in TARGETS.items():
            g0 = g.dropna(subset=[tgt, raw])
            if g0.empty:
                continue
            fg = [f for f in feats if g0[f].notna().mean() >= a.min_coverage]
            for f in set(feats) - set(fg):
                dropped[f] = dropped.get(f, 0) + 1
            gg = g0.dropna(subset=fg).sort_values("run")
            if len(gg) < a.min_samples:
                continue
            n_test = max(1, int(len(gg) * a.test_frac))
            tr, te = gg.iloc[:-n_test], gg.iloc[-n_test:]
            pipe = make_pipeline(StandardScaler(), Ridge(alpha=a.alpha))
            pipe.fit(tr[fg], tr[tgt])
            pred = pipe.predict(te[fg])
            if tgt == "sknt":
                pred = np.clip(pred, 0, None)
            verif.append({
                "station": sid, "cycle": cyc, "step": step, "target": tgt,
                "n_train": len(tr), "n_test": len(te),
                "mae_raw": np.mean(np.abs(te[raw] - te[tgt])),
                "mae_mos": np.mean(np.abs(pred - te[tgt])),
                "bias_raw": np.mean(te[raw] - te[tgt]),
                "bias_mos": np.mean(pred - te[tgt]),
            })
            # final model: refit on all data
            final = make_pipeline(StandardScaler(), Ridge(alpha=a.alpha))
            final.fit(gg[fg], gg[tgt])
            models[(sid, int(cyc), int(step), tgt)] = (final, fg)

    if dropped:
        log(f"features dropped for low coverage (count of models): {dropped}")
    if not models:
        sys.exit("No groups met --min-samples. Fetch more history or lower it.")
    joblib.dump({"models": models,
                 "trained": dt.datetime.utcnow().isoformat()},
                root / "mos_models.joblib")
    export_json(models, root / "mos_models.json")
    v = pd.DataFrame(verif)
    v.to_csv(root / "verification.csv", index=False)

    summ = (v.groupby(["target", "station"])[["mae_raw", "mae_mos", "bias_raw", "bias_mos"]]
             .mean().round(2))
    summ["mae_gain_%"] = (100 * (1 - summ["mae_mos"] / summ["mae_raw"])).round(1)
    print("\nHeld-out verification (last "
          f"{int(a.test_frac * 100)}% of runs, chronological), mean over leads:")
    print(summ.to_string())
    log(f"saved {len(models)} models -> {root / 'mos_models.joblib'} and mos_models.json")


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------
def latest_run(base, first_step) -> dt.datetime | None:
    now = dt.datetime.utcnow()
    t = now.replace(minute=0, second=0, microsecond=0,
                    hour=12 if now.hour >= 12 else 0)
    for _ in range(6):
        if fetch_index(base, t, first_step) is not None:
            return t
        t -= dt.timedelta(hours=12)
    return None


def export_json(models, path: Path) -> None:
    """Fold StandardScaler+Ridge into plain linear coefficients so predict
    needs no scikit-learn (and no pickle-version coupling)."""
    out = {"trained": dt.datetime.utcnow().isoformat(), "models": {}}
    for (sid, cyc, step, tgt), (pipe, fg) in models.items():
        sc, rg = pipe.steps[0][1], pipe.steps[1][1]
        coef = rg.coef_ / sc.scale_
        icpt = float(rg.intercept_ - np.sum(rg.coef_ * sc.mean_ / sc.scale_))
        out["models"][f"{sid}|{cyc}|{step}|{tgt}"] = {
            "features": fg, "coef": [float(c) for c in coef], "intercept": icpt}
    path.write_text(json.dumps(out))


class LinearMOS:
    def __init__(self, d):
        self.fg, self.coef, self.icpt = d["features"], np.array(d["coef"]), d["intercept"]

    def predict(self, X):
        return self.icpt + X[self.fg].to_numpy(float) @ self.coef


def load_models(path: Path) -> dict:
    d = json.loads(path.read_text())
    models = {}
    for k, v in d["models"].items():
        sid, cyc, step, tgt = k.split("|")
        m = LinearMOS(v)
        models[(sid, int(cyc), int(step), tgt)] = (m, m.fg)
    return models


def latest_complete_run(base, steps, cycles) -> dt.datetime | None:
    """Newest run in a trained cycle whose first AND last step are published."""
    now = dt.datetime.utcnow()
    t = now.replace(minute=0, second=0, microsecond=0,
                    hour=12 if now.hour >= 12 else 0)
    for _ in range(8):
        if (t.hour in cycles and fetch_index(base, t, steps[-1]) is not None
                and fetch_index(base, t, steps[0]) is not None):
            return t
        t -= dt.timedelta(hours=12)
    return None


def cmd_predict(a):
    root = Path(a.data_dir)
    mpath = Path(a.models) if a.models else root / "mos_models.json"
    models = load_models(mpath)
    cycles = sorted({k[1] for k in models})
    stations = a.stations or list(STATIONS)
    base, steps = BASE_URLS[a.source], parse_steps(a.steps)

    run = (dt.datetime.strptime(a.run, "%Y%m%d%H") if a.run
           else latest_complete_run(base, steps, cycles))
    if run is None:
        sys.exit(f"No complete run found for trained cycles {cycles}.")
    stamp = Path(a.html).with_name("run.txt") if a.html else None
    if stamp and stamp.exists() and stamp.read_text().strip() == f"{run:%Y%m%d%H}" and not a.run:
        log(f"{run:%Y%m%d%HZ} already published; nothing to do")
        return
    log(f"run {run:%Y-%m-%d %HZ}")
    (root / "model").mkdir(parents=True, exist_ok=True)
    fetch_run(base, run, steps, root / "model", a.workers, force=True)
    df = load_model(root / "model", runs=[run])

    rows = []
    for _, r in df.iterrows():
        rec = {"station": r["station"], "valid": r["valid"], "step": int(r["step"])}
        for tgt, raw in TARGETS.items():
            rec[f"{tgt}_raw"] = round(float(r.get(raw, np.nan)), 1)
            entry = models.get((r["station"], int(r["cycle"]), int(r["step"]), tgt))
            val = np.nan
            if entry is not None:
                pipe, fg = entry
                X = pd.DataFrame([[float(r.get(f, np.nan)) for f in fg]], columns=fg)
                if not X.isna().any(axis=None):
                    val = round(float(pipe.predict(X)[0]), 1)
            rec[f"{tgt}_mos"] = val
        u, v = r.get("u10", np.nan), r.get("v10", np.nan)
        rec["wdir_raw"] = (float((270.0 - math.degrees(math.atan2(v, u))) % 360.0)
                           if pd.notna(u) and pd.notna(v) else np.nan)
        rec["tcc_raw"] = float(r["tcc"]) if "tcc" in r and pd.notna(r["tcc"]) else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    out = out[out["station"].isin(stations)]
    out["sknt_mos"] = out["sknt_mos"].clip(lower=0)
    out["dwpf_mos"] = np.minimum(out["dwpf_mos"], out["tmpf_mos"])
    out = out.sort_values(["station", "step"])
    text = "\n\n".join(mos_bulletin(sid, run, out[out["station"] == sid])
                         for sid in stations if (out["station"] == sid).any())
    print("\n" + text + "\n")
    if a.html:
        hpath = Path(a.html)
        hpath.parent.mkdir(parents=True, exist_ok=True)
        path = hpath.with_name("mos_latest.csv")
        stamp.write_text(f"{run:%Y%m%d%H}\n")
    else:
        hpath = root / f"mos_{run:%Y%m%d%H}.html"
        path = root / f"mos_{run:%Y%m%d%H}.csv"
    out.to_csv(path, index=False)
    hpath.write_text(bulletin_html(text, run))
    log(f"saved {path}")
    log(f"saved {hpath}")


# --------------------------------------------------------------------------
# MAV-style text bulletin + HTML
# --------------------------------------------------------------------------
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUNE",
          "JULY", "AUG", "SEPT", "OCT", "NOV", "DEC"]


def _date_label(d) -> str:
    return "/" + MONTHS[d.month - 1].ljust(4) + f"{d.day:>3}"


def _cld(f) -> str:
    if pd.isna(f):
        return ""
    o = int(round(float(f) * 8))
    return "CL" if o == 0 else "FW" if o <= 2 else "SC" if o <= 4 else "BK" if o <= 7 else "OV"


def mos_bulletin(sid, run, g) -> str:
    """Fixed-width block in the MDL MAV layout: 5-char row label, 3-char columns."""
    g = g.sort_values("step").reset_index(drop=True)
    valid = pd.to_datetime(g["valid"])
    n = len(g)
    width = 5 + 3 * n

    def row(label, cells):
        return " " + label.ljust(4) + "".join(f"{c:>3}" for c in cells)

    def num(v, pad2=False):
        if pd.isna(v):
            return ""
        v = int(round(float(v)))
        return f"{v:02d}" if pad2 else str(v)

    # DT line: run date at col 4, then each day label starts on the first
    # digit of that day's 00Z column (same as MAV).
    dt_line = list(" DT " + " " * (width - 4 + 8))
    for i, ch in enumerate(_date_label(run)):
        dt_line[4 + i] = ch
    for k, t in enumerate(valid):
        if t.hour == 0:
            pos = 6 + 3 * k
            for i, ch in enumerate(_date_label(t)):
                if pos + i < len(dt_line):
                    dt_line[pos + i] = ch
    dt_line = "".join(dt_line).rstrip()

    wsp = [num(v, True) for v in g["sknt_mos"]]
    wdr = []
    for d, w in zip(g["wdir_raw"], g["sknt_mos"]):
        if pd.isna(d) or pd.isna(w):
            wdr.append("")
        elif int(round(w)) == 0:
            wdr.append("00")
        else:
            wdr.append(f"{(int(round(d / 10.0)) % 36) or 36:02d}")

    head = (f" K{sid:<3}   ECMWF MOS GUIDANCE  {run.month:>2}/{run.day:02d}/{run.year}"
            f"  {run.hour:02d}00 UTC")
    lines = [head, dt_line,
             row("HR", [f"{t.hour:02d}" for t in valid]),
             row("TMP", [num(v) for v in g["tmpf_mos"]]),
             row("DPT", [num(v) for v in g["dwpf_mos"]])]
    if g["tcc_raw"].notna().any():
        lines.append(row("CLD", [_cld(v) for v in g["tcc_raw"]]))
    if any(wdr):
        lines.append(row("WDR", wdr))
    lines.append(row("WSP", wsp))
    return "\n".join(lines)


def bulletin_html(text: str, run) -> str:
    notes = ("TMP/DPT/WSP: ECMWF IFS MOS (ridge regression vs ASOS).  "
             "CLD/WDR: raw IFS, not statistically corrected.\n"
             "Source: ECMWF open data, CC-BY-4.0.")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ECMWF MOS {run:%Y%m%d %H}Z</title>
<style>
  body {{ background:#fff; color:#000; margin:8px; }}
  pre  {{ font-family:"Courier New",Courier,monospace; font-size:13px;
          line-height:1.15; margin:0; overflow-x:auto; }}
  .note {{ color:#555; margin-top:1.2em; }}
</style></head>
<body><pre>{html.escape(text)}</pre>
<pre class="note">{html.escape(notes)}</pre></body></html>
"""


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------
def cmd_probe(a):
    base, ok = BASE_URLS[a.source], True
    print(f"== ECMWF open data ({a.source}: {base})")

    run = latest_run(base, 6)
    if run is None:
        print("  FAIL: no 00/12Z run index found in the last 3 days")
        ok = False
    else:
        idx = fetch_index(base, run, 6)
        sfc = sorted({e["param"] for e in idx if e.get("levtype") == "sfc"})
        pl = sorted({e["param"] for e in idx if e.get("levtype") == "pl"})
        lev = sorted({int(e["levelist"]) for e in idx if e.get("levtype") == "pl"})
        print(f"  latest run {run:%Y%m%d%HZ}: {len(idx)} fields at +6h")
        print(f"  sfc params: {sfc}")
        print(f"  pl params:  {pl}  levels: {lev}")
        missing = ([p for p in SFC_PARAMS if p not in sfc]
                   + [f"{p}{l}" for p, l in PL_PARAMS if p not in pl or l not in lev])
        print(f"  configured-but-absent: {missing or 'none'}")
        e2t = next((e for e in idx if e.get("param") == "2t"), None)
        if e2t:
            b = get_field_bytes(base, run, 6, e2t)
            vals = point_values(b)
            print(f"  2t +6h byte-range download: {len(b) / 1e6:.2f} MB")
            for sid, v in vals.items():
                print(f"    {sid}: {(v - 273.15) * 1.8 + 32:.1f} F")

    early = dt.datetime(2024, 3, 1, 0)
    idx = fetch_index(base, early, 6)
    print(f"  archive start check {early:%Y%m%d%HZ}: "
          f"{'present, ' + str(len(idx)) + ' fields' if idx else 'NOT FOUND'}")
    if not idx:
        print("   -> find the earliest available date before fetching history")

    print("\n== IEM ASOS")
    end = dt.date.today() + dt.timedelta(days=1)
    for sid in STATIONS:
        try:
            df = fetch_iem(sid, end - dt.timedelta(days=2), end)
            if df.empty:
                print(f"  {sid}: no data")
                ok = False
            else:
                last = df.iloc[-1]
                print(f"  {sid}: {len(df)} obs, last {last['valid']:%Y-%m-%d %H:%MZ} "
                      f"T={last['tmpf']} Td={last['dwpf']} wind={last['sknt']} kt")
        except Exception as e:
            print(f"  {sid}: FAIL {e}")
            ok = False
        time.sleep(5)

    print("\nPROBE", "PASSED" if ok else "HAD FAILURES")
    sys.exit(0 if ok else 1)


# --------------------------------------------------------------------------
# scan: when does each configured param appear in the archive?
# --------------------------------------------------------------------------
def cmd_scan(a):
    base = BASE_URLS[a.source]
    start = dt.date.fromisoformat(a.start)
    end = dt.date.fromisoformat(a.end) if a.end else dt.date.today() - dt.timedelta(days=1)
    wanted = SFC_PARAMS + [f"{p}{l}" for p, l in PL_PARAMS]
    rows, d = [], start
    while d <= end:
        run = dt.datetime(d.year, d.month, d.day, 0)
        idx = fetch_index(base, run, 6)
        if idx is None:
            rows.append({"run": f"{run:%Y-%m-%d}", "n": 0, **{w: "-" for w in wanted}})
        else:
            have = {field_name(e) for e in idx} - {None}
            rows.append({"run": f"{run:%Y-%m-%d}", "n": len(idx),
                         **{w: ("Y" if w in have else ".") for w in wanted}})
        d += dt.timedelta(days=a.every_days)
    df = pd.DataFrame(rows)
    print(f"Configured params at +6h, 00Z runs every {a.every_days} days "
          "(Y present, . absent, - no index):\n")
    print(df.to_string(index=False))
    print("\nFirst run with each param present:")
    for w in wanted:
        hit = df.loc[df[w] == "Y", "run"]
        print(f"  {w:6s} {hit.iloc[0] if len(hit) else 'never'}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="mos_data")
    ap.add_argument("--source", default="aws", choices=BASE_URLS)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe")

    s = sub.add_parser("scan")
    s.add_argument("--start", default="2024-03-01")
    s.add_argument("--end", default=None)
    s.add_argument("--every-days", type=int, default=14)

    f = sub.add_parser("fetch")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--cycles", nargs="+", default=["00"], choices=["00", "12"])
    f.add_argument("--steps", default=DEFAULT_STEPS)
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--force", action="store_true")

    o = sub.add_parser("obs")
    o.add_argument("--start", required=True)
    o.add_argument("--end", required=True)

    t = sub.add_parser("train")
    t.add_argument("--alpha", type=float, default=1.0)
    t.add_argument("--min-samples", type=int, default=150)
    t.add_argument("--test-frac", type=float, default=0.2)
    t.add_argument("--min-coverage", type=float, default=0.9,
                   help="keep a feature only if present in this fraction of rows")

    p = sub.add_parser("predict")
    p.add_argument("--run", help="YYYYMMDDHH; default latest 00/12Z")
    p.add_argument("--steps", default=DEFAULT_STEPS)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--models", help="path to mos_models.json (default: DATA_DIR)")
    p.add_argument("--stations", nargs="+", choices=list(STATIONS),
                   help="stations to show, in bulletin order")
    p.add_argument("--html", help="write the bulletin here (e.g. docs/index.html); "
                   "also writes run.txt + mos_latest.csv beside it and skips "
                   "runs already published")

    a = ap.parse_args()
    {"probe": cmd_probe, "scan": cmd_scan, "fetch": cmd_fetch, "obs": cmd_obs,
     "train": cmd_train, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
