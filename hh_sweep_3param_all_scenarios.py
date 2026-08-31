#!/usr/bin/env python3
"""
Three-parameter search: alpha_n.A + beta_h.A + beta_m.C  ->  all six Kow targets
================================================================================

WHY THESE THREE
---------------
beta_h.A + beta_m.C already reach four of the six conditions (Treat 5/10/15 and
Wash 10). Only Wash 5 (I_Na +35 %, I_K 0 %) and Wash 15 (I_Na +12.5 %, I_K +15 %)
are missing, and both need I_K pushed *up* relative to I_Na — a direction that pair
cannot produce, because it moves the two currents in strict anti-correlation.

Screening the 5,343 unique combinations already simulated shows what each constant
does on its own, in (dI_Na, dI_K) space:

    alpha_m.A   I_Na -55..+31 %, I_K  -0.6..0.0 %   pure sodium lever
    beta_n.A    I_Na  -0.5..+0.6 %, I_K -6..+7 %    pure potassium lever
    beta_m.C    I_Na  -3.7..+12.8 %, I_K -1.4..+0.9 %
    beta_h.B    I_Na  -2.1..+27 %,  I_K -21..+1.8 %
    alpha_n.A   I_Na -22..+100 %,  I_K -100..+66 %  strong, reaches I_K > 0
    beta_h.A    I_Na -130..+91 %,  I_K -84..+168 %  strongest, reaches far

alpha_n.A is the only constant besides beta_h.A that can drive I_K substantially
positive, and adding it to the existing pair was the best of all twenty possible
triples when screened with an additive surrogate (6/6 targets predicted reachable).
It is also the cheapest: two complete 2-D faces of this cube — beta_h.A x beta_m.C
(3,581 points) and alpha_n.A x beta_m.C (441 points) — have already been simulated,
so they are reused rather than re-run. Keeping both of the original constants means
every match already found is preserved by setting alpha_n.A = 0.

HOW THE SEARCH SAVES TIME
-------------------------
A blind 3-D grid at 10 % steps would be 21^3 = 9,261 simulations. Instead:

  Stage 0  Warm start — every previously simulated combination that uses only these
           three constants is imported and never re-run.
  Stage 1  An additive surrogate is built from the 1-D axis data (effects of each
           constant alone) and evaluated on a fine 41x41x41 grid in memory, which
           costs nothing. Only the top candidates per target are actually simulated.
  Stage 2  A 3-D compass search refines the best real result for each target,
           shrinking the step until it converges.

That is roughly 1,000 new simulations instead of 9,261, and it starts from points
the surrogate already believes are close.

The surrogate is only a seeding device: additivity was verified against the existing
2-D grids and is good for the weak constants (median error 0.1-0.5 points) but poor
for beta_h.A x beta_m.C (median 2.4, 90th percentile 44). Every number reported is
from a real simulation.

OUTPUT
------
    all_results_3param.csv   every combination, with both currents, latency,
                             spontaneous rate, silent flag, and a score, per-current
                             error and match flag against each of the six targets
    best_per_target.csv      the closest match found for each condition
    search_log.txt           settings, progress and the verdict

Usage
-----
    python hh_sweep_3param_all_scenarios.py                 # full run
    python hh_sweep_3param_all_scenarios.py --quick         # fewer candidates
    python hh_sweep_3param_all_scenarios.py --jobs 8
    python hh_sweep_3param_all_scenarios.py --candidates 80 --refine-rounds 4
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import time

import numpy as np

# ── The three swept constants ────────────────────────────────────────────────
SWEPT = [("alpha_n", "A"), ("beta_h", "A"), ("beta_m", "C")]
NAMES = ["alpha_n.A", "beta_h.A", "beta_m.C"]

TARGETS = {
    "treatment_5min":  {"INa": 30.0, "IK": -27.5},
    "treatment_10min": {"INa": 60.0, "IK": -33.0},
    "treatment_15min": {"INa": 55.0, "IK": -40.0},
    "wash_5min":       {"INa": 35.0, "IK": 0.0},
    "wash_10min":      {"INa": 15.0, "IK": 5.0},
    "wash_15min":      {"INa": 12.5, "IK": 15.0},
}
ORDER = list(TARGETS)

BASE = {
    "alpha_m": {"A": 0.32,  "B": 4.0,  "C": 13.0},
    "beta_m":  {"A": 0.28,  "B": 5.0,  "C": 40.0},
    "alpha_h": {"A": 0.128, "B": 17.0, "C": 18.0},
    "beta_h":  {"A": 40.0,  "B": 5.0},
    "alpha_n": {"A": 0.032, "B": 5.0,  "C": 15.0},
    "beta_n":  {"A": 0.5,   "B": 10.0, "C": 40.0},
}

PLOT_START_MS = 200
RUN_MS = 500
BASE_LATENCY = 243.73
BASE_SPONT = 13.3
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAM_RE = re.compile(r"^(?:alpha|beta)_[mhn]\.[A-C] %$")
_BASELINE = {}


# ── Model (verbatim from files-6/hh_grid_sweep_treatment_10min.py) ───────────

def _brian():
    from brian2 import (prefs, Equations, NeuronGroup, StateMonitor, SpikeMonitor,
                        start_scope, run, defaultclock,
                        ms, mV, uA, cm, ufarad, siemens, msiemens)
    prefs.codegen.target = 'numpy'
    g = globals()
    g.update(dict(Equations=Equations, NeuronGroup=NeuronGroup,
                  StateMonitor=StateMonitor, SpikeMonitor=SpikeMonitor,
                  start_scope=start_scope, run=run, defaultclock=defaultclock,
                  ms=ms, mV=mV, uA=uA, cm=cm))
    g.update(dict(Cm=1 * ufarad * cm**-2, gl=5e-5 * siemens * cm**-2,
                  El=-60 * mV, EK=-90 * mV, ENa=50 * mV,
                  g_na=100 * msiemens * cm**-2, g_kd=30 * msiemens * cm**-2,
                  VT=-63 * mV))
    defaultclock.dt = 0.01 * ms


EQ_GLOBAL = '''
        dv/dt = ( gl*(El-v) - INa - IK + I_ext) / Cm : volt
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        {am}
        {bm}
        {ah}
        {bh}
        {an}
        {bn}
        INa = g_na * (m**3) * h * (v - ENa)                             : amp/meter**2
        IK  = g_kd * (n**4)     * (v -  EK)                             : amp/meter**2
        I_ext   : amp/meter**2
    '''


def build_eq_line(eq_name, p):
    if eq_name == "alpha_m":
        return (f"alpha_m = {p['A']:.6g}*(mV**-1)*{p['B']:.6g}*mV/"
                f"exprel(({p['C']:.6g}*mV-v+VT)/({p['B']:.6g}*mV))/ms : Hz")
    if eq_name == "beta_m":
        return (f"beta_m  = {p['A']:.6g}*(mV**-1)*{p['B']:.6g}*mV/"
                f"exprel((v-VT-{p['C']:.6g}*mV)/({p['B']:.6g}*mV))/ms  : Hz")
    if eq_name == "alpha_h":
        return (f"alpha_h = {p['A']:.6g}*exp(({p['B']:.6g}*mV-v+VT)/({p['C']:.6g}*mV))/ms : Hz")
    if eq_name == "beta_h":
        return (f"beta_h  = 4./(1+exp(({p['A']:.6g}*mV-v+VT)/({p['B']:.6g}*mV)))/ms : Hz")
    if eq_name == "alpha_n":
        return (f"alpha_n = ({p['A']:.6g}/mV)*{p['B']:.6g}*mV/"
                f"exprel(({p['C']:.6g}*mV-v+VT)/({p['B']:.6g}*mV))/ms : Hz")
    if eq_name == "beta_n":
        return (f"beta_n  = {p['A']:.6g}*exp(({p['B']:.6g}*mV-v+VT)/({p['C']:.6g}*mV))/ms : Hz")


def eq_lines(pcts):
    params = {eq: dict(v) for eq, v in BASE.items()}
    for (eq_name, key), pct in zip(SWEPT, pcts):
        params[eq_name][key] = BASE[eq_name][key] * (1 + pct / 100.0)
    L = {n: build_eq_line(n, params[n]) for n in BASE}
    return (L["alpha_m"], L["beta_m"], L["alpha_h"], L["beta_h"], L["alpha_n"], L["beta_n"])


def run_timecourse(pcts, seed=42):
    """500 ms autonomous run; returns INa_min, IK_max (uA/cm^2), latency, spike count."""
    am, bm, ah, bh, an, bn = eq_lines(pcts)
    start_scope()
    eqs = Equations(EQ_GLOBAL.format(am=am, bm=bm, ah=ah, bh=bh, an=an, bn=bn))
    P = NeuronGroup(1, model=eqs, threshold='v>-20*mV', refractory=3*ms, method='rk4')
    np.random.seed(seed)
    P.v = 'El + (randn() * 5 - 5)*mV'
    P.I_ext = 0 * uA * cm**-2
    tr = StateMonitor(P, ['v', 'm', 'n', 'h'], record=[0])
    sp = SpikeMonitor(P)
    run(RUN_MS * ms)

    t = tr.t / ms
    mask = t >= PLOT_START_MS
    v_u = tr[0].v
    m, n, h = tr[0].m, tr[0].n, tr[0].h
    INa = (g_na * (m**3) * h * (v_u - ENa)) / (uA * cm**-2)
    IK = (g_kd * (n**4) * (v_u - EK)) / (uA * cm**-2)
    v_mV = v_u / mV
    tm, vm = t[mask], v_mV[mask]
    above = vm > -20.0
    latency = float(tm[int(np.argmax(above))]) if above.any() else None
    spikes = np.asarray(sp.t / ms)
    return (float(np.min(INa[mask])), float(np.max(IK[mask])), latency,
            int(np.sum(spikes >= PLOT_START_MS)))


# ── Evaluation ───────────────────────────────────────────────────────────────

def score_all(ina_pct, ik_pct):
    out = {}
    for k, t in TARGETS.items():
        ena, ek = ina_pct - t["INa"], ik_pct - t["IK"]
        out[k] = (float(np.hypot(ena, ek)), ena, ek)
    return out


def evaluate(pcts):
    pcts = tuple(round(float(x), 4) for x in pcts)
    rec = {"pct": pcts, "source": "simulated", "error": ""}
    try:
        INa, IK, lat, nsp = run_timecourse(pcts)
        rec.update(INa_uA=INa, IK_uA=IK, latency=lat, n_spikes=nsp,
                   spont_hz=nsp / ((RUN_MS - PLOT_START_MS) / 1000.0))
        rec["INa_pct"] = (INa - _BASELINE["INa"]) / abs(_BASELINE["INa"]) * 100
        rec["IK_pct"] = (IK - _BASELINE["IK"]) / abs(_BASELINE["IK"]) * 100
        rec["silent"] = (lat is None) or (nsp == 0)
        rec["scores"] = score_all(rec["INa_pct"], rec["IK_pct"])
    except Exception as exc:
        rec.update(error=str(exc).splitlines()[0][:200], silent=True,
                   INa_uA=None, IK_uA=None, latency=None, n_spikes=None,
                   spont_hz=None, INa_pct=None, IK_pct=None,
                   scores={k: (np.inf, np.inf, np.inf) for k in TARGETS})
    return rec


def _init_worker():
    _brian()
    INa, IK, lat, nsp = run_timecourse((0.0, 0.0, 0.0))
    _BASELINE.update(INa=INa, IK=IK, latency=lat, n_spikes=nsp)


def _worker(p):
    if not _BASELINE:
        _init_worker()
    return evaluate(p)


def evaluate_many(points, jobs, seen, log, tag=""):
    todo = [p for p in (tuple(round(float(x), 4) for x in q) for q in points)
            if p not in seen]
    if not todo:
        log(f"    {tag}: all {len(points)} already known")
        return []
    out, t0 = [], time.time()
    if jobs > 1:
        import multiprocessing as mp
        with mp.Pool(jobs, initializer=_init_worker) as pool:
            for i, rec in enumerate(pool.imap_unordered(_worker, todo, chunksize=4), 1):
                seen[rec["pct"]] = rec
                out.append(rec)
                if i % 50 == 0 or i == len(todo):
                    el = time.time() - t0
                    log(f"    {tag} {i}/{len(todo)}  ({el:.0f}s, {el/i:.2f}s each, "
                        f"~{el/i*(len(todo)-i)/60:.1f} min left)")
    else:
        for i, p in enumerate(todo, 1):
            rec = evaluate(p)
            seen[p] = rec
            out.append(rec)
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                log(f"    {tag} {i}/{len(todo)}  ({el:.0f}s, {el/i:.2f}s each)")
    return out


def best_for(seen, target):
    pool = [r for r in seen.values()
            if not r["silent"] and np.isfinite(r["scores"][target][0])]
    return min(pool, key=lambda r: r["scores"][target][0]) if pool else None


# ── Stage 0: warm start ──────────────────────────────────────────────────────

def load_warm(paths, log):
    """Import every previous simulation whose perturbation uses only our three
    constants (other constants must be at baseline)."""
    got = {}
    for path in paths:
        try:
            rows = list(csv.DictReader(open(path, newline="", errors="replace")))
        except OSError:
            continue
        if not rows:
            continue
        pcols = [c for c in rows[0] if PARAM_RE.match(c or "")]
        if not pcols:
            continue
        n_ok = 0
        for r in rows:
            def g(c):
                s = (r.get(c) or "").strip().replace("+", "")
                if s in ("", "n/a", "None", "nan"):
                    return None
                try:
                    return float(s)
                except ValueError:
                    return None
            pert = {}
            outside = False
            for c in pcols:
                v = g(c)
                if v is None or abs(v) < 1e-9:
                    continue
                nm = c.replace(" %", "")
                if nm not in NAMES:
                    outside = True
                    break
                pert[nm] = round(v, 4)
            if outside:
                continue
            ina, ik = g("INa (uA/cm^2)"), g("IK (uA/cm^2)")
            if ina is None or ik is None:
                continue
            lat = g("latency (ms)")
            spont = g("spontaneous rate (Hz)")
            key = tuple(round(pert.get(n, 0.0), 4) for n in NAMES)
            if key in got:
                continue
            got[key] = dict(pct=key, source="reused", error="",
                            INa_uA=ina, IK_uA=ik, latency=lat,
                            spont_hz=spont,
                            n_spikes=None if spont is None
                            else int(round(spont * (RUN_MS - PLOT_START_MS) / 1000.0)),
                            silent=(lat is None))
            n_ok += 1
        if n_ok:
            log(f"    {n_ok:>5} combinations reused from {os.path.basename(path)}")
    return got


def finalise_warm(got, seen, log):
    if not got:
        return
    ref = got.get((0.0, 0.0, 0.0))
    if ref:
        d1 = abs(ref["INa_uA"] - _BASELINE["INa"]) / abs(_BASELINE["INa"]) * 100
        d2 = abs(ref["IK_uA"] - _BASELINE["IK"]) / abs(_BASELINE["IK"]) * 100
        log(f"    baseline cross-check: reused (0,0,0) differs by "
            f"{d1:.4f}% (INa) / {d2:.4f}% (IK)")
        if max(d1, d2) > 1.0:
            log("    ! WARNING: reused data does not match this model — "
                "re-run with --no-warm-start")
    for k, r in got.items():
        r["INa_pct"] = (r["INa_uA"] - _BASELINE["INa"]) / abs(_BASELINE["INa"]) * 100
        r["IK_pct"] = (r["IK_uA"] - _BASELINE["IK"]) / abs(_BASELINE["IK"]) * 100
        r["scores"] = score_all(r["INa_pct"], r["IK_pct"])
        seen.setdefault(k, r)


# ── Stage 1: additive surrogate ──────────────────────────────────────────────

def build_surrogate(seen, log):
    """1-D effect of each constant on its own, linearly interpolated."""
    axes = {}
    for i, nm in enumerate(NAMES):
        pts = []
        for r in seen.values():
            if r["silent"] or r.get("INa_pct") is None:
                continue
            p = r["pct"]
            if abs(p[i]) > 1e-9 and all(abs(p[j]) < 1e-9 for j in range(3) if j != i):
                pts.append((p[i], r["INa_pct"], r["IK_pct"]))
        pts.append((0.0, 0.0, 0.0))
        pts = sorted(set(pts))
        axes[nm] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]),
                    np.array([p[2] for p in pts]))
        log(f"    {nm}: {len(pts)} points on its own axis "
            f"({axes[nm][0].min():+g}%..{axes[nm][0].max():+g}%)")
    return axes


def surrogate_predict(axes, grids):
    """Additive prediction of (INa%, IK%) on a mesh of the three constants."""
    ina = np.zeros([len(g) for g in grids])
    ik = np.zeros_like(ina)
    for i, nm in enumerate(NAMES):
        x, a, b = axes[nm]
        shape = [1, 1, 1]
        shape[i] = len(grids[i])
        ina += np.interp(grids[i], x, a).reshape(shape)
        ik += np.interp(grids[i], x, b).reshape(shape)
    return ina, ik


def surrogate_candidates(axes, lo, hi, step, n_per_target, seen, log, only=None):
    grids = [np.arange(lo, hi + step * .5, step) for _ in range(3)]
    ina, ik = surrogate_predict(axes, grids)
    picks = {}
    for k, t in ({only: TARGETS[only]} if only else TARGETS).items():
        d = np.hypot(ina - t["INa"], ik - t["IK"])
        flat = np.argsort(d, axis=None)[:n_per_target * 4]
        chosen, taken = [], []
        for f in flat:
            i, j, m = np.unravel_index(f, d.shape)
            p = (float(grids[0][i]), float(grids[1][j]), float(grids[2][m]))
            # spread the picks out so they don't all sit in one spot
            if any(max(abs(np.array(p) - np.array(q))) < step * 1.5 for q in taken):
                continue
            taken.append(p)
            chosen.append(p)
            if len(chosen) >= n_per_target:
                break
        picks[k] = chosen
        log(f"    {k:<16} surrogate best {d.min():5.2f}  -> {len(chosen)} candidates")
    return picks


# ── Stage 2: 3-D compass refinement ──────────────────────────────────────────

def compass(start, seen, target, jobs, log, bounds, step0=8.0, step_min=0.1):
    cur = tuple(round(float(x), 4) for x in start)
    evaluate_many([cur], jobs, seen, log, tag="seed")
    best = seen[cur]
    step = step0
    while step >= step_min:
        probes = []
        for i in range(3):
            for s in (+1, -1):
                q = list(best["pct"])
                q[i] = float(np.clip(q[i] + s * step, *bounds))
                probes.append(tuple(q))
        for s in (+1, -1):                       # a couple of diagonal moves
            q = [float(np.clip(v + s * step, *bounds)) for v in best["pct"]]
            probes.append(tuple(q))
        evaluate_many(probes, jobs, seen, log, tag=f"refine±{step:g}%")
        cand = [seen[p] for p in (tuple(round(float(x), 4) for x in q) for q in probes)
                if p in seen and not seen[p]["silent"]]
        if cand:
            c = min(cand, key=lambda r: r["scores"][target][0])
            if c["scores"][target][0] < best["scores"][target][0] - 1e-9:
                best = c
                continue
        step /= 2.0
    return best


# ── Output ───────────────────────────────────────────────────────────────────

FIELDS = ([f"{n} %" for n in NAMES]
          + ["INa (uA/cm^2)", "INa % change", "IK (uA/cm^2)", "IK % change",
             "latency (ms)", "latency shift (ms)", "spikes in window",
             "spontaneous rate (Hz)", "rate / baseline", "silent", "source"]
          + [f"{k}_{n}" for n in ORDER for k in ("score", "errINa", "errIK", "match")]
          + ["targets matched", "best score", "closest target", "error"])


def write_csv(seen, path, tolerance):
    def f(v, spec=".4f"):
        return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
            else format(v, spec)
    rows = list(seen.values())

    def key(r):
        b = min(r["scores"][k][0] for k in ORDER)
        anym = any(not r["silent"] and abs(r["scores"][k][1]) <= tolerance
                   and abs(r["scores"][k][2]) <= tolerance for k in ORDER)
        return (0 if anym else 1, b)
    rows.sort(key=key)

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for r in rows:
            nm = 0
            tail = []
            for k in ORDER:
                sc, ena, ek = r["scores"][k]
                ok = (not r["silent"]) and abs(ena) <= tolerance and abs(ek) <= tolerance
                nm += ok
                tail += [f(sc, ".3f"), f(ena, "+.3f"), f(ek, "+.3f"), "YES" if ok else ""]
            best_k = min(ORDER, key=lambda k: r["scores"][k][0])
            rate = r.get("spont_hz")
            w.writerow(
                [f(r["pct"][0]), f(r["pct"][1]), f(r["pct"][2]),
                 f(r.get("INa_uA"), ".6g"), f(r.get("INa_pct"), "+.3f"),
                 f(r.get("IK_uA"), ".6g"), f(r.get("IK_pct"), "+.3f"),
                 f(r.get("latency"), ".2f"),
                 "" if r.get("latency") is None else f(r["latency"] - BASE_LATENCY, "+.2f"),
                 "" if r.get("n_spikes") is None else str(r["n_spikes"]),
                 f(rate, ".1f"), "" if rate is None else f(rate / BASE_SPONT, ".3f"),
                 "yes" if r["silent"] else "no", r.get("source", "simulated")]
                + tail
                + [str(nm), f(r["scores"][best_k][0], ".3f"), best_k, r.get("error", "")])
    return path


def write_best(seen, path, tolerance, log):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "INa target %", "IK target %"]
                   + [f"{n} %" for n in NAMES]
                   + ["INa achieved %", "IK achieved %", "INa error", "IK error", "score",
                      "latency (ms)", "latency shift (ms)", "spontaneous rate (Hz)",
                      "rate / baseline", "source", "match"])
        for k in ORDER:
            b = best_for(seen, k)
            if b is None:
                continue
            sc, ena, ek = b["scores"][k]
            ok = abs(ena) <= tolerance and abs(ek) <= tolerance
            rate = b.get("spont_hz")
            w.writerow([k, TARGETS[k]["INa"], TARGETS[k]["IK"],
                        f"{b['pct'][0]:+.4f}", f"{b['pct'][1]:+.4f}", f"{b['pct'][2]:+.4f}",
                        f"{b['INa_pct']:+.3f}", f"{b['IK_pct']:+.3f}",
                        f"{ena:+.3f}", f"{ek:+.3f}", f"{sc:.3f}",
                        "" if b.get("latency") is None else f"{b['latency']:.2f}",
                        "" if b.get("latency") is None
                        else f"{b['latency'] - BASE_LATENCY:+.2f}",
                        "" if rate is None else f"{rate:.1f}",
                        "" if rate is None else f"{rate / BASE_SPONT:.3f}",
                        b.get("source", "simulated"), "YES" if ok else "no"])
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description="3-parameter search over "
                                             "alpha_n.A + beta_h.A + beta_m.C.")
    ap.add_argument("--output-root", default=os.path.join(SCRIPT_DIR, "sweep_3param"))
    ap.add_argument("--range", nargs=2, type=float, default=[-100.0, 150.0],
                    metavar=("LO", "HI"))
    ap.add_argument("--surrogate-step", type=float, default=2.5,
                    help="resolution of the in-memory surrogate grid (default 2.5%%)")
    ap.add_argument("--candidates", type=int, default=45,
                    help="simulated candidates per target from the surrogate (default 45)")
    ap.add_argument("--refine-rounds", type=int, default=3,
                    help="how many top candidates per target get compass-refined")
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="15 candidates, 1 refinement")
    ap.add_argument("--warm-start", action="append", default=None, metavar="CSV")
    ap.add_argument("--no-warm-start", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        args.candidates, args.refine_rounds = 15, 1

    os.makedirs(args.output_root, exist_ok=True)
    logf = open(os.path.join(args.output_root, "search_log.txt"), "w")

    def log(m=""):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    log("=" * 92)
    log("  3-parameter search:  " + "  +  ".join(NAMES))
    log(f"  range {args.range[0]:g}%..{args.range[1]:g}%   "
        f"surrogate step {args.surrogate_step:g}%   "
        f"{args.candidates} candidates/target   tolerance ±{args.tolerance:g} pts")
    log("=" * 92)

    _brian()
    _init_worker()
    log(f"\nbaseline: INa {_BASELINE['INa']:.4f}  IK {_BASELINE['IK']:.4f} uA/cm^2   "
        f"latency {_BASELINE['latency']:.2f} ms   "
        f"{_BASELINE['n_spikes']} spikes in the window")

    seen = {}
    if not args.no_warm_start:
        patterns = ("scenario_plots_sweepSERVER/all_results_*.csv",
                    "scenario_plots_sweep/all_results_*.csv",
                    "wash_from_bh_bm/all_results_bh_bm.csv",
                    "sweep_3param/all_results_3param.csv")
        paths = (args.warm_start or
                 sorted({p for d in (SCRIPT_DIR, os.getcwd())
                         for pat in patterns
                         for p in glob.glob(os.path.join(d, pat))}))
        log("\n[0] warm start — reusing everything already simulated in this subspace")
        if not paths:
            log("    no previous sweeps found next to the script or in the current folder")
        finalise_warm(load_warm(paths, log), seen, log)
        log(f"    {len(seen)} combinations available for free")

    # Which targets does the reused data already solve? Spend the simulation budget
    # on the ones it does not -- this is where the compute saving actually comes from.
    def solved(k):
        b = best_for(seen, k)
        if b is None:
            return False
        _, ena, ek = b["scores"][k]
        return abs(ena) <= args.tolerance and abs(ek) <= args.tolerance

    already = [k for k in ORDER if solved(k)]
    todo = [k for k in ORDER if k not in already]
    log(f"\n    already matched by reused data : "
        f"{', '.join(already) if already else '(none)'}")
    log(f"    still to solve                 : {', '.join(todo) if todo else '(none)'}")

    log("\n[1] additive surrogate built from the 1-D axes")
    axes = build_surrogate(seen, log)
    lo, hi = args.range
    budget = {k: (args.candidates if k in todo
                  else max(6, args.candidates // 5)) for k in ORDER}
    picks = {}
    for k in ORDER:
        picks[k] = surrogate_candidates(axes, lo, hi, args.surrogate_step,
                                        budget[k], seen, log, only=k)[k]

    log("\n[2] simulating the surrogate's candidates "
        f"({sum(len(v) for v in picks.values())} in total; "
        f"{sum(len(picks[k]) for k in todo)} of them for the unsolved targets)")
    for k in todo + already:                     # unsolved targets first
        evaluate_many(picks[k], args.jobs, seen, log, tag=k)
        b = best_for(seen, k)
        if b:
            log(f"    -> {k}: best real score so far {b['scores'][k][0]:.2f}")

    log("\n[3] compass refinement of the best candidates")
    for k in ORDER:
        b = best_for(seen, k)
        if b is None:
            continue
        if b["scores"][k][0] <= args.tolerance / 2:
            log(f"    {k}: already at {b['scores'][k][0]:.2f} — skipping refinement")
            continue
        pool = sorted((r for r in seen.values()
                       if not r["silent"] and np.isfinite(r["scores"][k][0])),
                      key=lambda r: r["scores"][k][0])[:args.refine_rounds]
        for st in pool:
            res = compass(st["pct"], seen, k, args.jobs, log, (lo, hi))
            log(f"    {k}: {st['scores'][k][0]:.2f} -> {res['scores'][k][0]:.2f} at "
                + ", ".join(f"{n} {v:+.3f}%" for n, v in zip(NAMES, res["pct"])))
            if res["scores"][k][0] <= args.tolerance / 2:
                break

    log("\n" + "=" * 92)
    log(f"  RESULT — {len(seen)} combinations "
        f"({sum(1 for r in seen.values() if r.get('source') == 'reused')} reused, "
        f"{sum(1 for r in seen.values() if r.get('source') == 'simulated')} newly simulated)")
    log("=" * 92)
    n_ok = 0
    for k in ORDER:
        b = best_for(seen, k)
        t = TARGETS[k]
        log(f"\n  {k}   target INa {t['INa']:+g}%  IK {t['IK']:+g}%")
        if b is None:
            log("    no spiking candidate found")
            continue
        sc, ena, ek = b["scores"][k]
        ok = abs(ena) <= args.tolerance and abs(ek) <= args.tolerance
        n_ok += ok
        log("    " + ",  ".join(f"{n} {v:+.3f}%" for n, v in zip(NAMES, b["pct"])))
        log(f"      achieved INa {b['INa_pct']:+.3f}%   IK {b['IK_pct']:+.3f}%   "
            f"(errors {ena:+.2f} / {ek:+.2f},  score {sc:.3f})")
        rate = b.get("spont_hz")
        lat_s = "n/a" if b.get("latency") is None else f"{b['latency']:.2f} ms"
        rate_s = ("n/a" if rate is None
                  else f"{rate:.1f} Hz = {rate / BASE_SPONT:.2f}x baseline")
        log(f"      latency {lat_s}   rate {rate_s}")
        log(f"      --> MATCH: {'YES' if ok else 'no'}   [{b.get('source')}]")
    log(f"\n  {n_ok} of 6 conditions matched within ±{args.tolerance:g} points on both "
        f"currents, with a firing neuron.")
    n_sil = sum(1 for r in seen.values() if r["silent"])
    log(f"  {n_sil} of {len(seen)} combinations were silent and excluded from selection.")

    p1 = write_csv(seen, os.path.join(args.output_root, "all_results_3param.csv"),
                   args.tolerance)
    p2 = write_best(seen, os.path.join(args.output_root, "best_per_target.csv"),
                    args.tolerance, log)
    log(f"\n  written: {p1}\n           {p2}\n")
    logf.close()


if __name__ == "__main__":
    main()
