"""
Brute-force grid sweep for ALL 6 treatment/wash scenarios
============================================================
Runs on YOUR machine with Brian2 installed (`pip install brian2`).

Extends the original 10*15-min sweep script to cover every scenario in
hh_scenarios&visualise.py, not just treatment_10min/15min:

    treatment_5min  : params (alpha_n.A, alpha_m.A)   target INa +30.0%   IK -27.5%
    treatment_10min : params (beta_h.A,  beta_m.C)    target INa +60.0%   IK -33.0%
    treatment_15min : params (alpha_n.A, beta_m.C)    target INa +55.0%   IK -40.0%
    wash_5min       : params (alpha_m.A)  <- ONLY ONE PARAMETER, 1-D sweep
                                                       target INa +35.0%   IK  +0.0%
    wash_10min      : params (beta_n.A,  beta_h.B)    target INa +15.0%   IK  +5.0%
    wash_15min      : params (alpha_n.A, beta_h.B)    target INa +12.5%   IK +15.0%

For each scenario this tries every combination of its tunable rate-equation
parameter(s) over a percentage range (default -100% to +100%), running a
full Brian2 simulation for each combination and recording, for EVERY
permutation tried (not just the ones that hit target):

    INa_min   (uA/cm^2)   + % change vs baseline   (most-negative INa, t >= 200 ms)
    IK_max    (uA/cm^2)   + % change vs baseline   (most-positive  IK, t >= 200 ms)
    latency   (ms)   + shift (ms) + % change vs baseline
                        (first crossing of v > -20 mV within t >= 200 ms, from
                         the same single autonomous I_ext=0 run used for INa/IK)
    firing rate (Hz) + % change vs baseline
                        (mean firing rate across a full 30-point f-I curve,
                         0-25 uA/cm^2, 200 ms settle + 1200 ms count window --
                         i.e. exactly like run_fi_curve() in the visualise
                         script, run fresh for every single combination)

Only INa/IK are compared against the scenario's target (tolerance box) to
decide "matches" -- latency and firing rate are recorded for every
permutation for your own inspection, not used to filter/score.

AUTO-RETRY / AUTO-ZOOM: exactly as in the original 10*15 script -- if the
first (coarse) pass finds zero combinations inside the INa/IK tolerance
box, the script zooms in around the best-scoring point, halves/thirds the
step size, and re-runs -- up to --max-rounds times.

NOTE ON RUNTIME: every single combination now runs TWO simulations instead
of one -- the 500 ms autonomous timecourse (INa/IK/latency) AND a full
f-I curve (30 parallel neurons, 200 ms + 1200 ms). That's roughly 2-4x
slower per point than the original script. A 21x21 grid (step=10, two
parameters) is 441 combinations per scenario -- budget accordingly, and
use --step to go coarser first, or --scenario to run one at a time.

Every combination ever tested is kept; at the end you get, per scenario:
  scenario_plots_sweep/all_results_{scenario}.csv   <- every combo, CSV (Excel/Sheets)
  scenario_plots_sweep/all_results_{scenario}.txt   <- every combo, plain text
  scenario_plots_sweep/matches_{scenario}.csv       <- combos within INa/IK tolerance (best first)

Usage:
    python hh_grid_sweep_all6.py                          # all 6 scenarios, step=10, -100..100, auto-zoom
    python hh_grid_sweep_all6.py --step 5                  # finer coarse grid (slower)
    python hh_grid_sweep_all6.py --scenario wash_10min     # just one scenario
    python hh_grid_sweep_all6.py --tolerance 3             # tighter match window
"""

from brian2 import *
import numpy as np
import os
import csv
import itertools
import argparse

prefs.codegen.target = 'numpy'

# ── Fixed biophysical parameters ─────────────────────────────────────────────
# NOTE: these are all expressed as *densities* (per cm^2), the classic HH
# convention, NOT scaled by a patch area into absolute conductances/capacitance
# (matches hh_combined19_jul_area_currrent_.py). Conductance (mS/cm^2) x
# voltage (mV) comes out dimensionally as current density in uA/cm^2
# (mS/cm^2 * mV = 1e-6 A/cm^2 = 1 uA/cm^2), so INa, IK, and I_ext below are
# all current *densities* in uA/cm^2, not absolute nA.
Cm   = 1    * ufarad   * cm**-2
gl   = 5e-5 * siemens  * cm**-2
El   = -60 * mV;  EK = -90 * mV;  ENa = 50 * mV
g_na = 100  * msiemens * cm**-2
g_kd = 30   * msiemens * cm**-2
VT   = -63 * mV

PLOT_START_MS = 200
TIME_STEP = 0.01 * ms
defaultclock.dt = TIME_STEP
SOLVER = 'rk4'

# f-I curve settings -- identical to run_fi_curve() in hh_combined19_jul_area_currrent_.py
# Current *density* sweep (uA/cm^2), classic HH units. 0-25 uA/cm^2 spans the
# same physiological stimulation range the old 0-5 nA sweep did.
FI_I_VALUES_UA_CM2 = np.linspace(0, 25, 30)  # uA/cm^2
FI_SETTLE_MS     = PLOT_START_MS           # drop leading transient
FI_DURATION_MS   = 1200                    # spike-counting window

BASE = {
    "alpha_m": {"A": 0.32,  "B": 4.0,  "C": 13.0},
    "beta_m":  {"A": 0.28,  "B": 5.0,  "C": 40.0},
    "alpha_h": {"A": 0.128, "B": 17.0, "C": 18.0},
    "beta_h":  {"A": 40.0,  "B": 5.0},
    "alpha_n": {"A": 0.032, "B": 5.0,  "C": 15.0},
    "beta_n":  {"A": 0.5,   "B": 10.0, "C": 40.0},
}

# scenario -> the tunable (eq_name, param_key) pairs that get swept; every
# other parameter stays at BASE. wash_5min has only ONE tunable parameter,
# so it gets a 1-D sweep instead of a 2-D grid -- everything below handles
# both cases generically (N params -> N-dimensional grid).
SCENARIO_PARAMS = {
    "treatment_5min":  [("alpha_n", "A"), ("alpha_m", "A")],
    "treatment_10min": [("beta_h",  "A"), ("beta_m",  "C")],
    "treatment_15min": [("alpha_n", "A"), ("beta_m",  "C")],
    "wash_5min":       [("alpha_m", "A")],
    "wash_10min":      [("beta_n",  "A"), ("beta_h",  "B")],
    "wash_15min":      [("alpha_n", "A"), ("beta_h",  "B")],
}

TARGETS = {
    "treatment_5min":  {"INa": 30.0,  "IK": -27.5},
    "treatment_10min": {"INa": 60.0,  "IK": -33.0},
    "treatment_15min": {"INa": 55.0,  "IK": -40.0},
    "wash_5min":       {"INa": 35.0,  "IK":   0.0},
    "wash_10min":      {"INa": 15.0,  "IK":   5.0},
    "wash_15min":      {"INa": 12.5,  "IK":  15.0},
}

eq_global = '''
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
    elif eq_name == "beta_m":
        return (f"beta_m  = {p['A']:.6g}*(mV**-1)*{p['B']:.6g}*mV/"
                f"exprel((v-VT-{p['C']:.6g}*mV)/({p['B']:.6g}*mV))/ms  : Hz")
    elif eq_name == "alpha_h":
        return (f"alpha_h = {p['A']:.6g}*exp(({p['B']:.6g}*mV-v+VT)/({p['C']:.6g}*mV))/ms : Hz")
    elif eq_name == "beta_h":
        return (f"beta_h  = 4./(1+exp(({p['A']:.6g}*mV-v+VT)/({p['B']:.6g}*mV)))/ms : Hz")
    elif eq_name == "alpha_n":
        return (f"alpha_n = ({p['A']:.6g}/mV)*{p['B']:.6g}*mV/"
                f"exprel(({p['C']:.6g}*mV-v+VT)/({p['B']:.6g}*mV))/ms : Hz")
    elif eq_name == "beta_n":
        return (f"beta_n  = {p['A']:.6g}*exp(({p['B']:.6g}*mV-v+VT)/({p['C']:.6g}*mV))/ms : Hz")


def params_with_overrides(overrides):
    params = {eq: dict(vals) for eq, vals in BASE.items()}
    for eq_name, param_key, pct in overrides:
        params[eq_name][param_key] = BASE[eq_name][param_key] * (1 + pct / 100.0)
    return params


def eq_lines_from_params(params):
    lines = {n: build_eq_line(n, params[n]) for n in BASE}
    return (lines["alpha_m"], lines["beta_m"], lines["alpha_h"], lines["beta_h"],
            lines["alpha_n"], lines["beta_n"])


def run_timecourse_metrics(overrides, seed=42):
    """
    500 ms autonomous (I_ext=0) run.
    Returns (INa_min [uA/cm^2], IK_max [uA/cm^2], latency_ms [or None]) measured
    exactly the way the combined-overlay script does: extremes and first spike
    (v > -20 mV) over t >= 200 ms.
    """
    am, bm, ah, bh, an, bn = eq_lines_from_params(params_with_overrides(overrides))
    start_scope()
    eqs = Equations(eq_global.format(am=am, bm=bm, ah=ah, bh=bh, an=an, bn=bn))
    P = NeuronGroup(1, model=eqs, threshold='v>-20*mV', refractory=3*ms, method=SOLVER)
    np.random.seed(seed)
    P.v = 'El + (randn() * 5 - 5)*mV'
    P.I_ext = 0 * uA * cm**-2
    tr = StateMonitor(P, ['v', 'm', 'n', 'h'], record=[0])
    run(0.5 * second)
    t = tr.t / ms
    mask = t >= PLOT_START_MS
    v_units = tr[0].v
    m = tr[0].m; n = tr[0].n; h = tr[0].h
    INa = (g_na * (m**3) * h * (v_units - ENa)) / (uA * cm**-2)
    IK  = (g_kd * (n**4)     * (v_units -  EK)) / (uA * cm**-2)
    INa_min = float(np.min(INa[mask]))
    IK_max  = float(np.max(IK[mask]))

    v_mV = v_units / mV
    t_m = t[mask]; v_m = v_mV[mask]
    above = v_m > -20.0
    latency_ms = float(t_m[int(np.argmax(above))]) if above.any() else None

    return INa_min, IK_max, latency_ms


def run_fi_curve_metrics(overrides):
    """
    Full 30-point f-I curve (0-25 uA/cm^2), 200 ms settle + 1200 ms spike-count
    window -- identical to run_fi_curve() in hh_combined19_jul_area_currrent_.py.
    Returns mean firing rate (Hz) across the 30 current densities.
    """
    am, bm, ah, bh, an, bn = eq_lines_from_params(params_with_overrides(overrides))
    I_values = FI_I_VALUES_UA_CM2 * uA * cm**-2

    start_scope()
    eqs_fi = Equations(eq_global.format(am=am, bm=bm, ah=ah, bh=bh, an=an, bn=bn))
    P = NeuronGroup(len(I_values), model=eqs_fi, threshold='v>-20*mV',
                    refractory=3*ms, method=SOLVER)
    P.v = El; P.I_ext = I_values
    run(FI_SETTLE_MS * ms)
    sp = SpikeMonitor(P)
    run(FI_DURATION_MS * ms)

    firing_rates = sp.count / (FI_DURATION_MS * 1e-3)  # Hz
    return float(np.mean(firing_rates))


def pct_delta(value, baseline):
    if value is None or baseline in (None, 0):
        return None
    return (value - baseline) / abs(baseline) * 100.0


def eval_combo(param_pair, pct_values, baselines, target):
    overrides = [(eq, pk, pct) for (eq, pk), pct in zip(param_pair, pct_values)]
    try:
        INa_min, IK_max, latency_ms = run_timecourse_metrics(overrides)
        fr_mean = run_fi_curve_metrics(overrides)

        ina_pct = pct_delta(INa_min, baselines["INa"])
        ik_pct  = pct_delta(IK_max,  baselines["IK"])
        lat_shift = (None if latency_ms is None or baselines["latency"] is None
                     else latency_ms - baselines["latency"])
        lat_pct = pct_delta(latency_ms, baselines["latency"])
        fr_pct  = pct_delta(fr_mean, baselines["FR"])

        ina_err = np.inf if ina_pct is None else ina_pct - target["INa"]
        ik_err  = np.inf if ik_pct  is None else ik_pct  - target["IK"]
        score = (ina_err ** 2 + ik_err ** 2) ** 0.5

        return dict(pct=tuple(pct_values), INa_uAcm2=INa_min, IK_uAcm2=IK_max,
                    INa_pct=ina_pct, IK_pct=ik_pct, INa_err=ina_err, IK_err=ik_err,
                    score=score, latency_ms=latency_ms, latency_shift_ms=lat_shift,
                    latency_pct=lat_pct, FR_Hz=fr_mean, FR_pct=fr_pct, error="")
    except Exception as exc:
        return dict(pct=tuple(pct_values), INa_uAcm2=None, IK_uAcm2=None,
                    INa_pct=None, IK_pct=None, INa_err=None, IK_err=None,
                    score=np.inf, latency_ms=None, latency_shift_ms=None,
                    latency_pct=None, FR_Hz=None, FR_pct=None, error=str(exc))


def fmt(v, spec="+.3f", suffix=""):
    return f"{v:{spec}}{suffix}" if v is not None and np.isfinite(v) else "n/a"


def sweep_scenario(scenario_key, baselines, lo, hi, step,
                    tolerance, max_rounds, output_root):
    param_pair = SCENARIO_PARAMS[scenario_key]
    target = TARGETS[scenario_key]
    n_params = len(param_pair)
    param_labels = [f"{eq}.{pk}" for eq, pk in param_pair]

    print(f"\n{'='*70}")
    print(f"  {scenario_key}  ({', '.join(param_labels)})"
          f"   target: INa {target['INa']:+.1f}%  IK {target['IK']:+.1f}%")
    print(f"{'='*70}")

    all_results = []
    seen = set()

    def run_grid(centers, half_range, step):
        axes = []
        for c in centers:
            vals = np.arange(max(lo, c - half_range), min(hi, c + half_range) + 1e-9, step)
            axes.append(vals)
        new = []
        for combo in itertools.product(*axes):
            key = tuple(round(float(x), 3) for x in combo)
            if key in seen:
                continue
            seen.add(key)
            r = eval_combo(param_pair, combo, baselines, target)
            all_results.append(r)
            new.append(r)
        return new

    # ── Round 1: full brute-force grid across the entire requested range ────
    print(f"\n  round 1 (full grid): {', '.join(param_labels)} from {lo}% to {hi}%, step {step}%")
    centers0 = [(lo + hi) / 2.0] * n_params
    round_results = run_grid(centers0, (hi - lo) / 2 + step, step)
    n_axis = int((hi - lo) / step) + 1
    grid_desc = "x".join([str(n_axis)] * n_params)
    print(f"    tested {len(round_results)} combinations ({grid_desc} grid)")

    def matches_within_tolerance(results):
        return [r for r in results if r["INa_pct"] is not None
                and abs(r["INa_err"]) <= tolerance and abs(r["IK_err"]) <= tolerance]

    matches = matches_within_tolerance(all_results)
    round_num = 1
    cur_step = step

    # ── Auto-retry / zoom: if nothing matched, zoom into the best point and
    #    re-run with a finer step, repeat up to max_rounds. ─────────────────
    while not matches and round_num < max_rounds:
        best = min(all_results, key=lambda r: r["score"])
        cur_step = max(cur_step / 3.0, 0.25)
        half_range = step  # zoom window shrinks each round but stays centered on best
        round_num += 1
        best_desc = "  ".join(f"{lbl}={v:+.2f}%" for lbl, v in zip(param_labels, best["pct"]))
        print(f"\n  no match yet (best score so far: {best['score']:.2f} pts, at {best_desc})")
        print(f"  round {round_num} (auto-zoom): centering on that point, "
              f"+/-{half_range}% window, step {cur_step:.2f}%")
        new_results = run_grid(list(best["pct"]), half_range, cur_step)
        print(f"    tested {len(new_results)} new combinations")
        matches = matches_within_tolerance(all_results)
        step = cur_step

    # ── Save CSV (every combination, all metrics) ───────────────────────────
    os.makedirs(output_root, exist_ok=True)
    all_results_sorted = sorted(all_results, key=lambda r: r["score"])

    csv_path = os.path.join(output_root, f"all_results_{scenario_key}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = [f"{lbl} %" for lbl in param_labels] + [
            "INa (uA/cm^2)", "INa % change", "IK (uA/cm^2)", "IK % change",
            "INa error vs target", "IK error vs target", "score",
            "latency (ms)", "latency shift (ms)", "latency % change",
            "firing rate (Hz)", "firing rate % change", "error",
        ]
        w.writerow(header)
        for r in all_results_sorted:
            row = list(r["pct"]) + [
                fmt(r["INa_uAcm2"], ".6g"), fmt(r["INa_pct"]),
                fmt(r["IK_uAcm2"], ".6g"), fmt(r["IK_pct"]),
                fmt(r["INa_err"]), fmt(r["IK_err"]),
                fmt(r["score"], ".4g") if np.isfinite(r["score"]) else "",
                fmt(r["latency_ms"], ".3f"), fmt(r["latency_shift_ms"], ".3f"),
                fmt(r["latency_pct"]),
                fmt(r["FR_Hz"], ".4g"), fmt(r["FR_pct"]),
                r["error"],
            ]
            w.writerow(row)
    print(f"\n  saved {csv_path}  ({len(all_results)} combinations tested total)")

    # ── Save plain-text log of every permutation ────────────────────────────
    txt_path = os.path.join(output_root, f"all_results_{scenario_key}.txt")
    with open(txt_path, "w") as f:
        f.write(f"All permutations tested — {scenario_key}\n")
        f.write(f"  parameters : {', '.join(param_labels)}  (range {lo}% .. {hi}%)\n")
        f.write(f"  target     : INa {target['INa']:+.1f}%   IK {target['IK']:+.1f}%\n")
        f.write(f"  baseline   : INa_min={baselines['INa']:.6g} uA/cm^2   "
                f"IK_max={baselines['IK']:.6g} uA/cm^2   "
                f"latency={fmt(baselines['latency'], '.3f')} ms   "
                f"FR={baselines['FR']:.6g} Hz\n")
        f.write(f"  sorted by score (closest to target first), {len(all_results_sorted)} total\n")
        f.write("=" * 100 + "\n\n")
        for r in all_results_sorted:
            combo_desc = "  ".join(f"{lbl}={v:+.2f}%" for lbl, v in zip(param_labels, r["pct"]))
            f.write(f"{combo_desc}\n")
            if r["error"]:
                f.write(f"    FAILED: {r['error']}\n\n")
                continue
            f.write(f"    INa       : {fmt(r['INa_uAcm2'], '.6g')} uA/cm^2   "
                    f"({fmt(r['INa_pct'])} % from baseline)\n")
            f.write(f"    IK        : {fmt(r['IK_uAcm2'], '.6g')} uA/cm^2   "
                    f"({fmt(r['IK_pct'])} % from baseline)\n")
            f.write(f"    latency   : {fmt(r['latency_ms'], '.3f')} ms   "
                    f"(shift {fmt(r['latency_shift_ms'], '.3f')} ms, "
                    f"{fmt(r['latency_pct'])} % from baseline)\n")
            f.write(f"    fire rate : {fmt(r['FR_Hz'], '.4g')} Hz   "
                    f"({fmt(r['FR_pct'])} % from baseline)\n")
            f.write(f"    score (INa/IK vs target): {fmt(r['score'], '.4g')}\n\n")
    print(f"  saved {txt_path}")

    # ── Save matches-only CSV ────────────────────────────────────────────────
    match_path = os.path.join(output_root, f"matches_{scenario_key}.csv")
    with open(match_path, "w", newline="") as f:
        w = csv.writer(f)
        header = [f"{lbl} %" for lbl in param_labels] + [
            "INa % change", "IK % change", "score",
            "latency (ms)", "latency % change", "firing rate (Hz)", "firing rate % change",
        ]
        w.writerow(header)
        for r in sorted(matches, key=lambda r: r["score"]):
            row = list(r["pct"]) + [
                fmt(r["INa_pct"]), fmt(r["IK_pct"]), fmt(r["score"], ".4g"),
                fmt(r["latency_ms"], ".3f"), fmt(r["latency_pct"]),
                fmt(r["FR_Hz"], ".4g"), fmt(r["FR_pct"]),
            ]
            w.writerow(row)
    print(f"  saved {match_path}  ({len(matches)} combinations within +/-{tolerance} pts)")

    if matches:
        print(f"\n  BEST MATCHES for {scenario_key}:")
        for r in sorted(matches, key=lambda r: r["score"])[:8]:
            combo_desc = "  ".join(f"{lbl}={v:+.2f}%" for lbl, v in zip(param_labels, r["pct"]))
            print(f"    {combo_desc}   -> INa {r['INa_pct']:+.2f}%   IK {r['IK_pct']:+.2f}%"
                  f"   latency {fmt(r['latency_ms'], '.2f')} ms   FR {fmt(r['FR_Hz'], '.3g')} Hz"
                  f"   score={r['score']:.3f}")
    else:
        best = min(all_results, key=lambda r: r["score"])
        combo_desc = "  ".join(f"{lbl}={v:+.2f}%" for lbl, v in zip(param_labels, best["pct"]))
        print(f"\n  No combination reached +/-{tolerance} pts after {round_num} rounds.")
        print(f"  Closest found: {combo_desc}"
              f"  -> INa {fmt(best['INa_pct'])}%  IK {fmt(best['IK_pct'])}%  (score {fmt(best['score'], '.3f')})")
        print(f"  Try: --lo/--hi to widen the range beyond [{lo}, {hi}], "
              f"or --max-rounds to allow more auto-zoom passes.")

    return all_results, matches


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=list(SCENARIO_PARAMS.keys()) + ["all"],
                     default="all")
    ap.add_argument("--lo", type=float, default=-100.0, help="Lower bound of the %% sweep range.")
    ap.add_argument("--hi", type=float, default=100.0, help="Upper bound of the %% sweep range.")
    ap.add_argument("--step", type=float, default=10.0, help="Coarse grid step in %%.")
    ap.add_argument("--tolerance", type=float, default=5.0,
                     help="Match window: +/- this many percentage points on BOTH INa and IK.")
    ap.add_argument("--max-rounds", type=int, default=4,
                     help="Max auto-zoom rounds if the coarse grid finds no match.")
    ap.add_argument("--output-root", default="scenario_plots_sweep")
    args = ap.parse_args()

    print("Running baseline (no overrides)...")
    base_INa, base_IK, base_latency = run_timecourse_metrics([])
    base_FR = run_fi_curve_metrics([])
    baselines = dict(INa=base_INa, IK=base_IK, latency=base_latency, FR=base_FR)
    print(f"  baseline INa_min   = {base_INa:.6g} uA/cm^2")
    print(f"  baseline IK_max    = {base_IK:.6g} uA/cm^2")
    print(f"  baseline latency   = {fmt(base_latency, '.3f')} ms")
    print(f"  baseline fire rate = {base_FR:.6g} Hz  (mean over 0-25 uA/cm^2 f-I curve)")

    scenarios = list(SCENARIO_PARAMS.keys()) if args.scenario == "all" else [args.scenario]
    for key in scenarios:
        sweep_scenario(key, baselines, args.lo, args.hi, args.step,
                        args.tolerance, args.max_rounds, args.output_root)

    print(f"\nAll done. Results in: {os.path.abspath(args.output_root)}")


if __name__ == "__main__":
    main()
