"""
Brute-force grid sweep for treatment_10min / treatment_15min
==============================================================
Runs on YOUR machine with Brian2 installed (`pip install brian2`).

For each of the two scenarios, this tries every combination of its two
tunable rate-equation parameters over a percentage range (default -100%
to +100%), running a full Brian2 simulation for each combination and
checking whether the resulting INa / IK % change (measured exactly the
way extremes_summary_heatmap.png measures it: most-negative INa and
most-positive IK over t >= 200 ms, relative to baseline) lands within
+/-TOLERANCE points of your targets:

    treatment_10min : INa +60%   IK -33%      (params: beta_h.A, beta_m.C)
    treatment_15min : INa +55%   IK -40%      (params: alpha_n.A, beta_m.C)

AUTO-RETRY: if the first (coarse) pass finds zero combinations inside the
tolerance box, the script automatically zooms in around the best-scoring
point found so far, halves the step size, and re-runs — up to
--max-rounds times — instead of making you re-invoke it by hand.

Every combination ever tested is kept; at the end you get, per scenario:
  scenario_plots_sweep/all_results_{scenario}.csv       <- every combo tried
  scenario_plots_sweep/matches_{scenario}.csv           <- combos within tolerance (best first)

Usage:
    python hh_grid_sweep.py                       # default: step=10, -100..100, auto-zoom
    python hh_grid_sweep.py --step 5               # finer coarse grid (slower, ~9x more sims)
    python hh_grid_sweep.py --scenario treatment_10min   # just one scenario
    python hh_grid_sweep.py --tolerance 3           # tighter match window

Runtime note: each Brian2 simulation is 500 ms of biological time at
dt=0.01 ms (50,000 RK4 steps) for a single neuron — typically well under
a second on a laptop, but a full -100..+100 grid at step=10 is 21x21=441
sims per scenario, so budget a few minutes; step=5 (41x41=1681) is more
like 15-30 min per scenario. Coarser first, then let the auto-zoom refine.
"""

from brian2 import *
import numpy as np
import os
import csv
import argparse

prefs.codegen.target = 'numpy'

# ── Fixed biophysical parameters (identical to the original script) ─────────
area = 20000 * umetre**2
Cm   = (1    * ufarad   * cm**-2) * area
gl   = (5e-5 * siemens  * cm**-2) * area
El   = -60 * mV;  EK = -90 * mV;  ENa = 50 * mV
g_na = (100  * msiemens * cm**-2) * area
g_kd = (30   * msiemens * cm**-2) * area
VT   = -63 * mV

PLOT_START_MS = 200
TIME_STEP = 0.01 * ms
defaultclock.dt = TIME_STEP
SOLVER = 'rk4'

BASE = {
    "alpha_m": {"A": 0.32,  "B": 4.0,  "C": 13.0},
    "beta_m":  {"A": 0.28,  "B": 5.0,  "C": 40.0},
    "alpha_h": {"A": 0.128, "B": 17.0, "C": 18.0},
    "beta_h":  {"A": 40.0,  "B": 5.0},
    "alpha_n": {"A": 0.032, "B": 5.0,  "C": 15.0},
    "beta_n":  {"A": 0.5,   "B": 10.0, "C": 40.0},
}

# scenario -> the two (eq_name, param_key) pairs that get swept; every other
# parameter stays at BASE.
SCENARIO_PARAMS = {
    "treatment_10min": [("beta_h", "A"), ("beta_m", "C")],
    "treatment_15min": [("alpha_n", "A"), ("beta_m", "C")],
}

TARGETS = {
    "treatment_10min": {"INa": 60.0, "IK": -33.0},
    "treatment_15min": {"INa": 55.0, "IK": -40.0},
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
        INa = g_na * (m**3) * h * (v - ENa)                             : amp
        IK  = g_kd * (n**4)     * (v -  EK)                             : amp
        I_ext   : amp
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


def timecourse_extremes(overrides, seed=42):
    """Runs one 500 ms Brian2 simulation and returns (INa_min, IK_max) over t >= 200 ms."""
    am, bm, ah, bh, an, bn = eq_lines_from_params(params_with_overrides(overrides))
    start_scope()
    eqs = Equations(eq_global.format(am=am, bm=bm, ah=ah, bh=bh, an=an, bn=bn))
    P = NeuronGroup(1, model=eqs, threshold='v>-20*mV', refractory=3*ms, method=SOLVER)
    np.random.seed(seed)
    P.v = 'El + (randn() * 5 - 5)*mV'
    P.I_ext = 0 * nA
    tr = StateMonitor(P, ['v', 'm', 'n', 'h'], record=[0])
    run(0.5 * second)
    t = tr.t / ms
    mask = t >= PLOT_START_MS
    v_units = tr[0].v
    m = tr[0].m; n = tr[0].n; h = tr[0].h
    INa = (g_na * (m**3) * h * (v_units - ENa)) / nA
    IK  = (g_kd * (n**4)     * (v_units -  EK)) / nA
    return float(np.min(INa[mask])), float(np.max(IK[mask]))


def pct_delta(value, baseline):
    return (value - baseline) / abs(baseline) * 100.0


def eval_combo(param_pair, a_pct, b_pct, baseline_INa, baseline_IK, target):
    (eqA, pkA), (eqB, pkB) = param_pair
    overrides = [(eqA, pkA, a_pct), (eqB, pkB, b_pct)]
    try:
        INa_min, IK_max = timecourse_extremes(overrides)
        ina_pct = pct_delta(INa_min, baseline_INa)
        ik_pct = pct_delta(IK_max, baseline_IK)
        ina_err = ina_pct - target["INa"]
        ik_err = ik_pct - target["IK"]
        score = (ina_err ** 2 + ik_err ** 2) ** 0.5
        return dict(a=a_pct, b=b_pct, INa_pct=ina_pct, IK_pct=ik_pct,
                    INa_err=ina_err, IK_err=ik_err, score=score, error="")
    except Exception as exc:
        return dict(a=a_pct, b=b_pct, INa_pct=None, IK_pct=None,
                    INa_err=None, IK_err=None, score=np.inf, error=str(exc))


def sweep_scenario(scenario_key, baseline_INa, baseline_IK, lo, hi, step,
                    tolerance, max_rounds, output_root):
    param_pair = SCENARIO_PARAMS[scenario_key]
    target = TARGETS[scenario_key]
    (eqA, pkA), (eqB, pkB) = param_pair

    print(f"\n{'='*70}")
    print(f"  {scenario_key}  ({eqA}.{pkA}, {eqB}.{pkB})"
          f"   target: INa {target['INa']:+.1f}%  IK {target['IK']:+.1f}%")
    print(f"{'='*70}")

    all_results = []
    seen = set()

    def run_grid(center_a, center_b, half_range, step):
        a_vals = np.arange(max(lo, center_a - half_range), min(hi, center_a + half_range) + 1e-9, step)
        b_vals = np.arange(max(lo, center_b - half_range), min(hi, center_b + half_range) + 1e-9, step)
        new = []
        for a in a_vals:
            for b in b_vals:
                key = (round(float(a), 3), round(float(b), 3))
                if key in seen:
                    continue
                seen.add(key)
                r = eval_combo(param_pair, float(a), float(b), baseline_INa, baseline_IK, target)
                all_results.append(r)
                new.append(r)
        return new

    # ── Round 1: full brute-force grid across the entire requested range ────
    print(f"\n  round 1 (full grid): {eqA}.{pkA} and {eqB}.{pkB} from {lo}% to {hi}%, step {step}%")
    round_results = run_grid((lo + hi) / 2, (lo + hi) / 2, (hi - lo) / 2 + step, step)
    print(f"    tested {len(round_results)} combinations "
          f"({int((hi-lo)/step)+1}x{int((hi-lo)/step)+1} grid)")

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
        print(f"\n  no match yet (best score so far: {best['score']:.2f} pts, "
              f"at {eqA}.{pkA}={best['a']:+.2f}%  {eqB}.{pkB}={best['b']:+.2f}%)")
        print(f"  round {round_num} (auto-zoom): centering on that point, "
              f"+/-{half_range}% window, step {cur_step:.2f}%")
        new_results = run_grid(best["a"], best["b"], half_range, cur_step)
        print(f"    tested {len(new_results)} new combinations")
        matches = matches_within_tolerance(all_results)
        step = cur_step

    # ── Save everything ──────────────────────────────────────────────────
    os.makedirs(output_root, exist_ok=True)
    all_path = os.path.join(output_root, f"all_results_{scenario_key}.csv")
    with open(all_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"{eqA}.{pkA} %", f"{eqB}.{pkB} %", "INa % change", "IK % change",
                    "INa error vs target", "IK error vs target", "score", "error"])
        for r in sorted(all_results, key=lambda r: r["score"]):
            w.writerow([r["a"], r["b"],
                        "" if r["INa_pct"] is None else f"{r['INa_pct']:.3f}",
                        "" if r["IK_pct"] is None else f"{r['IK_pct']:.3f}",
                        "" if r["INa_err"] is None else f"{r['INa_err']:+.3f}",
                        "" if r["IK_err"] is None else f"{r['IK_err']:+.3f}",
                        f"{r['score']:.4g}" if np.isfinite(r["score"]) else "",
                        r["error"]])
    print(f"\n  saved {all_path}  ({len(all_results)} combinations tested total)")

    match_path = os.path.join(output_root, f"matches_{scenario_key}.csv")
    with open(match_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"{eqA}.{pkA} %", f"{eqB}.{pkB} %", "INa % change", "IK % change", "score"])
        for r in sorted(matches, key=lambda r: r["score"]):
            w.writerow([r["a"], r["b"], f"{r['INa_pct']:.3f}", f"{r['IK_pct']:.3f}", f"{r['score']:.4g}"])
    print(f"  saved {match_path}  ({len(matches)} combinations within +/-{tolerance} pts)")

    if matches:
        print(f"\n  BEST MATCHES for {scenario_key}:")
        for r in sorted(matches, key=lambda r: r["score"])[:8]:
            print(f"    {eqA}.{pkA}={r['a']:+.2f}%   {eqB}.{pkB}={r['b']:+.2f}%"
                  f"   -> INa {r['INa_pct']:+.2f}%   IK {r['IK_pct']:+.2f}%   score={r['score']:.3f}")
    else:
        best = min(all_results, key=lambda r: r["score"])
        print(f"\n  No combination reached +/-{tolerance} pts after {round_num} rounds.")
        print(f"  Closest found: {eqA}.{pkA}={best['a']:+.2f}%  {eqB}.{pkB}={best['b']:+.2f}%"
              f"  -> INa {best['INa_pct']:+.2f}%  IK {best['IK_pct']:+.2f}%  (score {best['score']:.3f})")
        print(f"  Try: --lo/--hi to widen the range beyond [{lo}, {hi}], "
              f"or --max-rounds to allow more auto-zoom passes.")

    return all_results, matches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=["treatment_10min", "treatment_15min", "both"],
                     default="both")
    ap.add_argument("--lo", type=float, default=-100.0, help="Lower bound of the %% sweep range.")
    ap.add_argument("--hi", type=float, default=100.0, help="Upper bound of the %% sweep range.")
    ap.add_argument("--step", type=float, default=5.0, help="Coarse grid step in %%.")
    ap.add_argument("--tolerance", type=float, default=5.0,
                     help="Match window: +/- this many percentage points on BOTH INa and IK.")
    ap.add_argument("--max-rounds", type=int, default=4,
                     help="Max auto-zoom rounds if the coarse grid finds no match.")
    ap.add_argument("--output-root", default="scenario_plots_sweep")
    args = ap.parse_args()

    print("Running baseline (no overrides)...")
    base_INa, base_IK = timecourse_extremes([])
    print(f"  baseline INa_min = {base_INa:.6g} nA")
    print(f"  baseline IK_max  = {base_IK:.6g} nA")

    scenarios = ["treatment_10min", "treatment_15min"] if args.scenario == "both" else [args.scenario]
    for key in scenarios:
        sweep_scenario(key, base_INa, base_IK, args.lo, args.hi, args.step,
                        args.tolerance, args.max_rounds, args.output_root)

    print(f"\nAll done. Results in: {os.path.abspath(args.output_root)}")


if __name__ == "__main__":
    main()