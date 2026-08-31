"""
Hodgkin-Huxley Scenario Comparison — Treatment / Wash conditions
==================================================================
Instead of sweeping one parameter across a full percentage range, this
script runs a fixed set of NAMED SCENARIOS, each defined by one or more
SPECIFIC percentage changes applied simultaneously to specific rate-
equation parameters. Any parameter not explicitly listed for a scenario
stays at its baseline value.

Scenarios (all other parameters at baseline):
  treatment_5min  : -30% alpha_n.A   &  -40% alpha_m.A
  treatment_10min : -40% beta_h.A    &  -40% beta_m.C
  treatment_15min : -40% alpha_n.A   &  -50% beta_m.C
  wash_5min       : -40% alpha_m.A
  wash_10min      : -40% beta_n.A    &  +70% beta_h.B
  wash_15min      : +20% alpha_n.A   &  +60% beta_h.B

Produces the same 7 combined overlay figures as the old sweep script
(v, m, h, n, INa, IK, f-I curve) — but now overlaying baseline + the 6
scenarios instead of 16 % steps — plus an extremes_summary.txt with the
same kind of INa/IK extrema and spike-timing-shift information.

Saved to:  scenario_plots/{variable}.png  and  scenario_plots/extremes_summary.txt
"""

from brian2 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import numpy as np
import os
import csv
import argparse

prefs.codegen.target = 'numpy'

# ── Fixed biophysical parameters ──────────────────────────────────────────────
area = 20000 * umetre**2
Cm   = (1    * ufarad   * cm**-2) * area
gl   = (5e-5 * siemens  * cm**-2) * area
El   = -60 * mV;  EK = -90 * mV;  ENa = 50 * mV
g_na = (100  * msiemens * cm**-2) * area
g_kd = (30   * msiemens * cm**-2) * area
VT   = -63 * mV
taue = 5 * ms;  taui = 10 * ms
Ee   =   0 * mV;  Ei  = -80 * mV

PLOT_START_MS = 200   # time-domain plots only show t >= this value;
                       # the full simulation is still run/calculated either way

# ── Simulation time step ──────────────────────────────────────────────────────
TIME_STEP = 0.01 * ms
defaultclock.dt = TIME_STEP

BASE = {
    "alpha_m": {"A": 0.32,  "B": 4.0,  "C": 13.0},
    "beta_m":  {"A": 0.28,  "B": 5.0,  "C": 40.0},
    "alpha_h": {"A": 0.128, "B": 17.0, "C": 18.0},
    "beta_h":  {"A": 40.0,  "B": 5.0},
    "alpha_n": {"A": 0.032, "B": 5.0,  "C": 15.0},
    "beta_n":  {"A": 0.5,   "B": 10.0, "C": 40.0},
}

SOLVER = 'rk4'

# ── Scenario definitions ──────────────────────────────────────────────────────
# Each scenario: list of (eq_name, param_key, pct_change). Any parameter not
# listed here stays exactly at its BASE value.
SCENARIOS = {
    "treatment_5min": [
        ("alpha_n", "A", -30), #ok
        ("alpha_m", "A", -40), #ok
    ],
    "treatment_10min": [
        ("beta_h", "A", -29.44),  #ok
        ("beta_m", "C", 51.11),  #-50%=31.18. -40%=-20,64 -60%=84 -54%=32,57%=30 need=60
    ],
    "treatment_15min": [
        ("alpha_n", "A", 25), #-40%=-68.9 -30%=-65.12 -10%=-9 -20%=-61 -15%=-14 -19%=66 need=-40
        ("beta_m", "C", -45),  #-50%=73.2 -40%=69.9 -10%=-1 -20%=-18 -37%=66 -33%=-3 -35%=66 need=55
    ],
    "wash_5min": [
        ("alpha_m", "A", -40), #ok
    ],
    "wash_10min": [
        ("beta_n", "A", -47), # okokok -40%=-9.21 -30%=-10.31 -50%=-99 -43%=-8.9 -45%=8.6 need=5
        ("beta_h", "B", +70), #ok INa
    ],
    "wash_15min": [
        ("alpha_n", "A", +36), # okokok 20%=3.58, 10%=-3.17 30%=9.95 35%=13 need 15
        ("beta_h", "B", +60), #ok INa
    ],
}

SCENARIO_ORDER = ["baseline"] + list(SCENARIOS.keys())

# ── Colour / style scheme ─────────────────────────────────────────────────────
# Baseline = bold black. Treatment scenarios = solid, warming reds (deeper red
# = longer exposure). Wash scenarios = dashed, deepening blues (deeper blue =
# longer wash).
_treat_cmap = mpl_cm.Reds
_wash_cmap  = mpl_cm.Blues

SCENARIO_STYLE = {
    "baseline":        dict(color="black",              lw=2.5, ls='-',  zorder=10, alpha=1.00, label="Baseline"),
    "treatment_5min":  dict(color=_treat_cmap(0.45),     lw=1.7, ls='-',  zorder=6,  alpha=0.90, label="Treatment 5 min"),
    "treatment_10min": dict(color=_treat_cmap(0.70),     lw=1.7, ls='-',  zorder=6,  alpha=0.90, label="Treatment 10 min"),
    "treatment_15min": dict(color=_treat_cmap(0.95),     lw=1.7, ls='-',  zorder=6,  alpha=0.90, label="Treatment 15 min"),
    "wash_5min":       dict(color=_wash_cmap(0.45),      lw=1.7, ls='--', zorder=6,  alpha=0.90, label="Wash 5 min"),
    "wash_10min":      dict(color=_wash_cmap(0.70),      lw=1.7, ls='--', zorder=6,  alpha=0.90, label="Wash 10 min"),
    "wash_15min":      dict(color=_wash_cmap(0.95),      lw=1.7, ls='--', zorder=6,  alpha=0.90, label="Wash 15 min"),
}

# ── Equation builders ─────────────────────────────────────────────────────────

def build_eq_line(eq_name, params):
    p = params
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


def format_pct_change(pct):
    """Compact display for integer or decimal percentage changes."""
    pct = float(pct)
    if pct.is_integer():
        return f"{pct:+.0f}%"
    return f"{pct:+.3f}".rstrip("0").rstrip(".") + "%"


def build_scenario_params(overrides):
    """
    overrides: list of (eq_name, param_key, pct_change)
    Returns (params, applied):
      params  : full {eq_name: {A,B,C,...}} dict, BASE values except the
                overridden entries
      applied : list of (eq_name, param_key, pct, base_val, new_val) — kept
                for the text summary
    """
    params = {eq: dict(vals) for eq, vals in BASE.items()}
    applied = []
    for eq_name, param_key, pct in overrides:
        base_val = BASE[eq_name][param_key]
        new_val  = base_val * (1 + pct / 100)
        params[eq_name][param_key] = new_val
        applied.append((eq_name, param_key, pct, base_val, new_val))
    return params, applied


def get_eq_lines_from_params(params):
    lines = {n: build_eq_line(n, params[n]) for n in BASE}
    return (lines["alpha_m"], lines["beta_m"], lines["alpha_h"], lines["beta_h"],
            lines["alpha_n"], lines["beta_n"])


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

# ── Simulations ───────────────────────────────────────────────────────────────

def run_timecourse(eq_lines):
    am, bm, ah, bh, an, bn = eq_lines
    start_scope()
    eqs = Equations(eq_global.format(am=am, bm=bm, ah=ah, bh=bh, an=an, bn=bn))
    P = NeuronGroup(1, model=eqs, threshold='v>-20*mV',
                    refractory=3*ms, method=SOLVER)
    P.v  = 'El + (randn() * 5 - 5)*mV'
    P.I_ext = 0 * nA
    tr = StateMonitor(P, ['v', 'm', 'n', 'h'], record=[0])
    run(0.5 * second)
    t = tr.t / ms
    v_units = tr[0].v
    v = v_units / mV
    m = tr[0].m;  n = tr[0].n;  h = tr[0].h
    INa = (g_na * (m**3) * h * (v_units - ENa)) / nA
    IK  = (g_kd * (n**4)     * (v_units -  EK)) / nA
    return dict(t=t, v=v, m=m, n=n, h=h, INa=INa, IK=IK)


def run_fi_curve(eq_lines):
    am, bm, ah, bh, an, bn = eq_lines
    I_values = np.linspace(0, 5, 30) * nA

    start_scope()
    eqs_fi = Equations(eq_global.format(am=am, bm=bm, ah=ah, bh=bh, an=an, bn=bn))
    P = NeuronGroup(len(I_values), model=eqs_fi, threshold='v>-20*mV',
                    refractory=3*ms, method=SOLVER)
    P.v = El;  P.I_ext = I_values
    run(PLOT_START_MS * ms)   # drop leading transient (same window as time-course plots)
    sp = SpikeMonitor(P)
    duration_fi = 1200
    run(duration_fi * ms)

    firing_rates = (sp.count / (duration_fi * 1e-3))  # Hz
    return I_values / nA, np.array(firing_rates)


# ── Combined figure builder ───────────────────────────────────────────────────

PLOT_SPECS = [
    # (key,   x_key, y_key,  xlabel,                ylabel,           title)
    ("v",   "t",   "v",   "Time (ms)",            "v (mV)",         "Membrane Potential  v(t)"),
    ("m",   "t",   "m",   "Time (ms)",            "m",              "Na activation gate  m(t)"),
    ("h",   "t",   "h",   "Time (ms)",            "h",              "Na inactivation gate  h(t)"),
    ("n",   "t",   "n",   "Time (ms)",            "n",              "K activation gate  n(t)"),
    ("INa", "t",   "INa", "Time (ms)",            "I$_{Na}$ (nA)", "Sodium Current  I$_{Na}$(t)"),
    ("IK",  "t",   "IK",  "Time (ms)",            "I$_K$ (nA)",   "Potassium Current  I$_K$(t)"),
    ("fI",  "I",   "fr",  "Input current I (nA)", "Firing rate (Hz)", "f–I curve"),
]


def save_combined_figures_scenarios(folder, all_data):
    """
    all_data: list of (scenario_key, tc_dict, I_nA, fr) tuples
    Produces 7 PNG files in folder, overlaying baseline + all 6 scenarios.
    """
    os.makedirs(folder, exist_ok=True)
    data_by_key = {k: (tc, I_nA, fr) for k, tc, I_nA, fr in all_data}

    for key, xk, yk, xlabel, ylabel, base_title in PLOT_SPECS:
        fig, ax = plt.subplots(figsize=(10, 5))

        for scen_key in SCENARIO_ORDER:
            if scen_key not in data_by_key:
                continue
            tc, I_nA, fr = data_by_key[scen_key]
            style = SCENARIO_STYLE[scen_key]

            if xk == "I":
                x, y = I_nA, fr
            else:
                mask = tc["t"] >= PLOT_START_MS
                x = tc[xk][mask]
                y = tc[yk][mask]

            ax.plot(x, y, color=style["color"], lw=style["lw"], ls=style["ls"],
                    zorder=style["zorder"], alpha=style["alpha"], label=style["label"])

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{base_title}\nTreatment / Wash scenario comparison", fontsize=11)

        leg = ax.legend(loc='upper right', fontsize=8, framealpha=0.85, title="Scenario")
        for text in leg.get_texts():
            if text.get_text() == "Baseline":
                text.set_fontweight('bold')

        plt.tight_layout()
        fig.savefig(os.path.join(folder, f"{key}.png"), dpi=430)
        plt.close(fig)
        print(f"  saved {key}.png")


def compute_extremes_scenarios(all_data):
    INa_min, IK_max = {}, {}
    for key, tc, I_nA, fr in all_data:
        mask = tc["t"] >= PLOT_START_MS
        INa_min[key] = float(np.min(tc["INa"][mask]))
        IK_max[key]  = float(np.max(tc["IK"][mask]))
    return INa_min, IK_max


def compute_spike_shift_scenarios(all_data, threshold_mV=-20.0):
    spike_time = {}
    for key, tc, I_nA, fr in all_data:
        mask = tc["t"] >= PLOT_START_MS
        t = tc["t"][mask]
        v = tc["v"][mask]
        above = v > threshold_mV
        if above.any():
            idx = int(np.argmax(above))
            spike_time[key] = float(t[idx])
        else:
            spike_time[key] = None

    baseline_t = spike_time.get("baseline")
    shift = {}
    for key, st in spike_time.items():
        if st is not None and baseline_t is not None:
            shift[key] = st - baseline_t
        else:
            shift[key] = None
    return spike_time, shift


def save_extremes_txt_scenarios(folder, applied_log, INa_min, IK_max, spike_time, shift):
    baseline_INa = INa_min.get("baseline")
    baseline_IK  = IK_max.get("baseline")

    path = os.path.join(folder, "extremes_summary.txt")
    with open(path, "w") as f:
        f.write("Extreme-value summary — Treatment / Wash scenario comparison\n")
        f.write(f"  window   : t >= {PLOT_START_MS} ms\n")
        f.write(f"  dt       : {TIME_STEP}\n")
        f.write("=" * 92 + "\n\n")

        f.write("Scenario definitions (all other parameters at baseline)\n")
        f.write("-" * 92 + "\n")
        for key in SCENARIO_ORDER:
            if key not in applied_log:
                continue
            label = SCENARIO_STYLE[key]["label"]
            overrides = applied_log[key]
            if not overrides:
                f.write(f"  {label:<18}: (baseline — no changes)\n")
            else:
                parts = [
                    f"{format_pct_change(pct)} {eq_name}.{param_key} ({base_val:.4g} -> {new_val:.4g})"
                    for eq_name, param_key, pct, base_val, new_val in overrides
                ]
                f.write(f"  {label:<18}: " + "   &   ".join(parts) + "\n")
        f.write("\n")

        f.write("INa — most negative value reached (nA)\n")
        f.write("-" * 92 + "\n")
        f.write(f"{'scenario':<18} {'INa_min (nA)':>15} {'delta vs baseline':>20}\n")
        for key in SCENARIO_ORDER:
            if key not in INa_min:
                continue
            val = INa_min[key]
            label = SCENARIO_STYLE[key]["label"]
            if baseline_INa not in (None, 0):
                delta_pct = (val - baseline_INa) / abs(baseline_INa) * 100
                delta_str = f"{delta_pct:+.3f} %"
            else:
                delta_str = "n/a"
            f.write(f"{label:<18} {val:15.6g} {delta_str:>20}\n")
        f.write("\n")

        f.write("IK — most positive value reached (nA)\n")
        f.write("-" * 92 + "\n")
        f.write(f"{'scenario':<18} {'IK_max (nA)':>15} {'delta vs baseline':>20}\n")
        for key in SCENARIO_ORDER:
            if key not in IK_max:
                continue
            val = IK_max[key]
            label = SCENARIO_STYLE[key]["label"]
            if baseline_IK not in (None, 0):
                delta_pct = (val - baseline_IK) / abs(baseline_IK) * 100
                delta_str = f"{delta_pct:+.3f} %"
            else:
                delta_str = "n/a"
            f.write(f"{label:<18} {val:15.6g} {delta_str:>20}\n")
        f.write("\n")

        f.write("Spike/pulse timing shift, relative to baseline (ms)\n")
        f.write(f"  (spike time = first crossing of v > -20 mV, within t >= {PLOT_START_MS} ms)\n")
        f.write("  positive shift = spike occurs LATER than baseline\n")
        f.write("  negative shift = spike occurs EARLIER than baseline\n")
        f.write("-" * 92 + "\n")
        f.write(f"{'scenario':<18} {'spike time (ms)':>16} {'shift vs baseline (ms)':>24}\n")
        for key in SCENARIO_ORDER:
            if key not in spike_time:
                continue
            st = spike_time[key]
            sh = shift[key]
            label = SCENARIO_STYLE[key]["label"]
            st_str = f"{st:16.4f}" if st is not None else f"{'no spike':>16}"
            sh_str = f"{sh:+24.4f}" if sh is not None else f"{'n/a':>24}"
            f.write(f"{label:<18} {st_str} {sh_str}\n")

    print(f"  saved extremes_summary.txt")


# ── Extremes visualisation (ported from visualize_extremes.py) ───────────────
# The original visualize_extremes.py re-parsed extremes_summary.txt files
# produced by a single-parameter %-sweep script (rows keyed by -50% .. +100%
# of ONE parameter). That format doesn't apply here, since each scenario in
# this script changes several parameters at once and there's no single %-axis
# to plot against. Instead, the same idea (CSV table + heatmap + summary
# figure across all conditions, normalised to baseline) is rebuilt to work
# directly off the in-memory scenario results.

def build_scenario_delta_table(INa_min, IK_max, spike_time, shift):
    """
    Returns a dict: scenario_key -> dict(INa_delta_pct, IK_delta_pct,
    spike_time_ms, shift_ms), all relative to the 'baseline' scenario.
    """
    baseline_INa = INa_min.get("baseline")
    baseline_IK  = IK_max.get("baseline")

    table = {}
    for key in INa_min:
        ina_val = INa_min[key]
        ik_val  = IK_max[key]
        ina_delta = ((ina_val - baseline_INa) / abs(baseline_INa) * 100
                     if baseline_INa not in (None, 0) else None)
        ik_delta  = ((ik_val - baseline_IK) / abs(baseline_IK) * 100
                     if baseline_IK not in (None, 0) else None)
        table[key] = dict(
            INa_min=ina_val, IK_max=ik_val,
            INa_delta_pct=ina_delta, IK_delta_pct=ik_delta,
            spike_time_ms=spike_time.get(key), shift_ms=shift.get(key),
        )
    return table


def save_scenario_summary_csv(folder, table):
    """Writes the scenario x metric table as a CSV — opens directly in Excel/Sheets."""
    path = os.path.join(folder, "extremes_summary_table.csv")
    cols = ["INa_min (nA)", "INa % change vs baseline", "IK_max (nA)",
            "IK % change vs baseline", "spike time (ms)", "shift vs baseline (ms)"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario"] + cols)
        for key in SCENARIO_ORDER:
            if key not in table:
                continue
            row = table[key]
            label = SCENARIO_STYLE[key]["label"]

            def fmt(v, suffix=""):
                return f"{v:.4g}{suffix}" if v is not None else ""

            writer.writerow([
                label,
                fmt(row["INa_min"]),
                fmt(row["INa_delta_pct"], " %"),
                fmt(row["IK_max"]),
                fmt(row["IK_delta_pct"], " %"),
                fmt(row["spike_time_ms"]),
                fmt(row["shift_ms"]),
            ])
    print(f"  saved {path}")
    return path


def plot_scenario_heatmap(folder, table):
    """
    Excel-like colored grid: rows = scenarios, cols = [INa %change, IK %change,
    spike/pulse timing shift (ms)]. Mirrors visualize_extremes.py's
    plot_table_heatmap, but keyed by scenario instead of parameter x %-sweep.
    """
    keys = [k for k in SCENARIO_ORDER if k in table]
    if not keys:
        return
    labels = [SCENARIO_STYLE[k]["label"] for k in keys]
    col_names = ["INa % change", "IK % change", "Spike shift (ms)"]

    data = np.full((len(keys), 3), np.nan)
    for i, k in enumerate(keys):
        row = table[k]
        for j, val in enumerate([row["INa_delta_pct"], row["IK_delta_pct"], row["shift_ms"]]):
            if val is not None:
                data[i, j] = val

    finite = data[~np.isnan(data)]
    vmax = np.nanmax(np.abs(finite)) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig_h = max(4, 0.5 * len(keys) + 1.5)
    fig, ax = plt.subplots(figsize=(7, fig_h))

    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xticks(np.arange(-0.5, len(col_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(labels)):
        for j in range(len(col_names)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black")

    ax.set_title("Scenario comparison — % change vs baseline (currents), ms shift (timing)",
                  fontsize=11)
    fig.colorbar(im, ax=ax, label="value (see column)", fraction=0.04, pad=0.03)
    plt.tight_layout()
    path = os.path.join(folder, "extremes_summary_heatmap.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  saved {path}")
    return path


def plot_scenario_bar_summary(folder, table):
    """
    Bar-chart version of visualize_extremes.py's plot_folder_summary: 3 panels
    (INa_min, IK_max, spike shift), one bar per scenario (categorical x-axis,
    since scenarios aren't points along a single continuous % sweep).
    """
    keys = [k for k in SCENARIO_ORDER if k in table]
    if not keys:
        return
    labels = [SCENARIO_STYLE[k]["label"] for k in keys]
    colors = [SCENARIO_STYLE[k]["color"] for k in keys]
    x = np.arange(len(keys))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ina_vals = [table[k]["INa_min"] for k in keys]
    axes[0].bar(x, ina_vals, color=colors)
    axes[0].set_ylabel("INa_min (nA)")
    axes[0].set_title("Peak inward Na$^+$ current")

    ik_vals = [table[k]["IK_max"] for k in keys]
    axes[1].bar(x, ik_vals, color=colors)
    axes[1].set_ylabel("IK_max (nA)")
    axes[1].set_title("Peak outward K$^+$ current")

    shift_vals = [table[k]["shift_ms"] if table[k]["shift_ms"] is not None else 0 for k in keys]
    axes[2].bar(x, shift_vals, color=colors)
    axes[2].axhline(0, color='gray', lw=0.8, ls='--')
    axes[2].set_ylabel("Spike shift vs baseline (ms)")
    axes[2].set_title("Pulse timing shift")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)

    fig.suptitle("Extremes summary across scenarios (raw values)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(folder, "extremes_summary_bar.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  saved {path}")
    return path


def visualize_scenario_extremes(folder, INa_min, IK_max, spike_time, shift):
    """Builds the CSV table + heatmap + bar-chart summary for scenario results."""
    table = build_scenario_delta_table(INa_min, IK_max, spike_time, shift)
    save_scenario_summary_csv(folder, table)
    plot_scenario_heatmap(folder, table)
    plot_scenario_bar_summary(folder, table)


# ── Scenario tuning/search helpers ───────────────────────────────────────────

TUNING_TARGETS = {
    "treatment_10min": {"INa_delta_pct": 60.0, "IK_delta_pct": -33.0},
    "treatment_15min": {"INa_delta_pct": 55.0, "IK_delta_pct": -40.0},
    "wash_10min":      {"INa_delta_pct": 15.0, "IK_delta_pct": 5.0},
    "wash_15min":      {"INa_delta_pct": 12.5, "IK_delta_pct": 15.0},
}

# These are the scenario-specific parameters that the tuner is allowed to move.
# treatment_5min and wash_5min are deliberately omitted so they stay unchanged.
TUNING_PARAMETERS = {
    "treatment_10min": [("beta_h", "A"),  ("beta_m", "C")],
    "treatment_15min": [("alpha_n", "A"), ("beta_m", "C")],
    "wash_10min":      [("beta_n", "A"),  ("beta_h", "B")],
    "wash_15min":      [("alpha_n", "A"), ("beta_h", "B")],
}

TUNING_PCT_BOUNDS = (-90.0, 150.0)


def timecourse_extremes(tc):
    """Return the same INa_min and IK_max values used by the final summaries."""
    mask = tc["t"] >= PLOT_START_MS
    return float(np.min(tc["INa"][mask])), float(np.max(tc["IK"][mask]))


def delta_pct(value, baseline_value):
    if baseline_value in (None, 0):
        return None
    return (value - baseline_value) / abs(baseline_value) * 100


def pct_vector_to_overrides(tune_params, values):
    overrides = []
    for (eq_name, param_key), pct in zip(tune_params, values):
        pct = float(pct)
        if abs(pct) > 1e-9:
            overrides.append((eq_name, param_key, pct))
    return overrides


def initial_pct_vector(scenario_key, tune_params):
    current = {
        (eq_name, param_key): pct
        for eq_name, param_key, pct in SCENARIOS[scenario_key]
    }
    return np.array([float(current.get(param, 0.0)) for param in tune_params])


def tuning_score(ina_delta, ik_delta, target):
    if ina_delta is None or ik_delta is None:
        return np.inf, np.inf, np.inf
    ina_error = ina_delta - target["INa_delta_pct"]
    ik_error = ik_delta - target["IK_delta_pct"]
    score = (ina_error ** 2 + ik_error ** 2) ** 0.5
    return score, ina_error, ik_error


def evaluate_tuning_candidate(
    scenario_key, tune_params, values, baseline_INa, baseline_IK, target, history
):
    values = np.clip(np.array(values, dtype=float), *TUNING_PCT_BOUNDS)
    values = np.round(values, 3)
    overrides = pct_vector_to_overrides(tune_params, values)
    result = {
        "scenario_key": scenario_key,
        "values": values,
        "overrides": overrides,
        "INa_min": None,
        "IK_max": None,
        "INa_delta_pct": None,
        "IK_delta_pct": None,
        "INa_error": None,
        "IK_error": None,
        "score": np.inf,
        "error": "",
    }

    try:
        params, _ = build_scenario_params(overrides)
        eq_lines = get_eq_lines_from_params(params)
        np.random.seed(42)
        tc = run_timecourse(eq_lines)
        INa_min, IK_max = timecourse_extremes(tc)
        ina_delta = delta_pct(INa_min, baseline_INa)
        ik_delta = delta_pct(IK_max, baseline_IK)
        score, ina_error, ik_error = tuning_score(ina_delta, ik_delta, target)

        result.update({
            "INa_min": INa_min,
            "IK_max": IK_max,
            "INa_delta_pct": ina_delta,
            "IK_delta_pct": ik_delta,
            "INa_error": ina_error,
            "IK_error": ik_error,
            "score": score,
        })
    except Exception as exc:
        result["error"] = str(exc)

    history.append(result)
    return result


def tune_one_scenario(scenario_key, baseline_INa, baseline_IK, max_evals, tolerance):
    target = TUNING_TARGETS[scenario_key]
    tune_params = TUNING_PARAMETERS[scenario_key]
    history = []
    seen = {}

    def eval_values(values):
        clipped = np.clip(np.array(values, dtype=float), *TUNING_PCT_BOUNDS)
        key = tuple(np.round(clipped, 3))
        if key not in seen:
            seen[key] = evaluate_tuning_candidate(
                scenario_key, tune_params, clipped, baseline_INa, baseline_IK, target, history
            )
        return seen[key]

    best = eval_values(initial_pct_vector(scenario_key, tune_params))

    # Coarse local grid around the current hand-chosen values.
    offsets = [0, -60, 60, -40, 40, -20, 20]
    initial = initial_pct_vector(scenario_key, tune_params)
    if len(tune_params) <= 2:
        for off_a in offsets:
            for off_b in offsets:
                if len(history) >= max_evals:
                    break
                candidate = initial + np.array([off_a, off_b])
                result = eval_values(candidate)
                if result["score"] < best["score"]:
                    best = result
            if len(history) >= max_evals:
                break

    # Coordinate-descent refinement. Smaller steps are tried only if budget remains.
    for step in [20, 10, 5, 2.5, 1, 0.5]:
        improved = True
        while improved and len(history) < max_evals and best["score"] > tolerance:
            improved = False
            for idx in range(len(tune_params)):
                for direction in (-1, 1):
                    if len(history) >= max_evals:
                        break
                    candidate = best["values"].copy()
                    candidate[idx] += direction * step
                    result = eval_values(candidate)
                    if result["score"] < best["score"]:
                        best = result
                        improved = True
                if len(history) >= max_evals:
                    break

    return best, history


def overrides_to_string(overrides):
    if not overrides:
        return "(baseline)"
    return " & ".join(
        f"{format_pct_change(pct)} {eq_name}.{param_key}"
        for eq_name, param_key, pct in overrides
    )


def save_tuning_results_csv(folder, tuning_results):
    path = os.path.join(folder, "tuning_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "target INa %",
            "achieved INa %",
            "INa error",
            "target IK %",
            "achieved IK %",
            "IK error",
            "score",
            "chosen overrides",
        ])
        for key in TUNING_TARGETS:
            result = tuning_results[key]
            target = TUNING_TARGETS[key]

            def fmt(v):
                return f"{v:.6g}" if v is not None and np.isfinite(v) else ""

            writer.writerow([
                SCENARIO_STYLE[key]["label"],
                fmt(target["INa_delta_pct"]),
                fmt(result["INa_delta_pct"]),
                fmt(result["INa_error"]),
                fmt(target["IK_delta_pct"]),
                fmt(result["IK_delta_pct"]),
                fmt(result["IK_error"]),
                fmt(result["score"]),
                overrides_to_string(result["overrides"]),
            ])
    print(f"  saved {path}")
    return path


def save_tuning_history_csv(folder, histories):
    path = os.path.join(folder, "tuning_search_history.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "eval",
            "INa % change",
            "IK % change",
            "score",
            "overrides",
            "error",
        ])
        for key in TUNING_TARGETS:
            for i, result in enumerate(histories[key], start=1):

                def fmt(v):
                    return f"{v:.6g}" if v is not None and np.isfinite(v) else ""

                writer.writerow([
                    SCENARIO_STYLE[key]["label"],
                    i,
                    fmt(result["INa_delta_pct"]),
                    fmt(result["IK_delta_pct"]),
                    fmt(result["score"]),
                    overrides_to_string(result["overrides"]),
                    result["error"],
                ])
    print(f"  saved {path}")
    return path


def save_tuned_scenarios_py(folder, tuned_scenarios):
    path = os.path.join(folder, "tuned_scenarios.py")
    with open(path, "w") as f:
        f.write("# Paste this block over SCENARIOS in hh_scenarios&visualise.py\n")
        f.write("# if you want to make the tuned values the default.\n\n")
        f.write("SCENARIOS = {\n")
        for key in SCENARIOS:
            f.write(f'    "{key}": [\n')
            for eq_name, param_key, pct in tuned_scenarios[key]:
                f.write(f'        ("{eq_name}", "{param_key}", {pct:.6g}),\n')
            f.write("    ],\n")
        f.write("}\n")
    print(f"  saved {path}")
    return path


def run_tuning_workflow(output_root, max_evals=120, tolerance=1.0):
    os.makedirs(output_root, exist_ok=True)
    print("\nTuning targets")
    print("-" * 62)
    for key, target in TUNING_TARGETS.items():
        label = SCENARIO_STYLE[key]["label"]
        print(
            f"  {label:<18}: INa {target['INa_delta_pct']:+.2f}%"
            f"   IK {target['IK_delta_pct']:+.2f}%"
        )

    print("\nRunning baseline time-course for tuning...")
    base_params = {eq: dict(vals) for eq, vals in BASE.items()}
    eq_lines = get_eq_lines_from_params(base_params)
    np.random.seed(42)
    base_tc = run_timecourse(eq_lines)
    baseline_INa, baseline_IK = timecourse_extremes(base_tc)
    print(f"  baseline INa_min = {baseline_INa:.6g} nA")
    print(f"  baseline IK_max  = {baseline_IK:.6g} nA")

    tuning_results = {}
    histories = {}
    tuned_scenarios = {key: list(overrides) for key, overrides in SCENARIOS.items()}

    for key in TUNING_TARGETS:
        label = SCENARIO_STYLE[key]["label"]
        print(f"\nSearching {label} (max {max_evals} candidate sets)...")
        best, history = tune_one_scenario(key, baseline_INa, baseline_IK, max_evals, tolerance)
        tuning_results[key] = best
        histories[key] = history
        tuned_scenarios[key] = best["overrides"]

        print(f"  best: {overrides_to_string(best['overrides'])}")
        print(
            f"  got INa {best['INa_delta_pct']:+.3f}%"
            f" / target {TUNING_TARGETS[key]['INa_delta_pct']:+.3f}%"
        )
        print(
            f"  got IK  {best['IK_delta_pct']:+.3f}%"
            f" / target {TUNING_TARGETS[key]['IK_delta_pct']:+.3f}%"
        )
        print(f"  score: {best['score']:.4g} percentage points")

    save_tuning_results_csv(output_root, tuning_results)
    save_tuning_history_csv(output_root, histories)
    save_tuned_scenarios_py(output_root, tuned_scenarios)
    return tuned_scenarios


# ── Main ──────────────────────────────────────────────────────────────────────

def run_scenario_workflow(output_root, scenario_defs):
    os.makedirs(output_root, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  HH Scenario Comparison — Treatment / Wash conditions")
    print(f"  1 baseline + {len(scenario_defs)} scenarios  ×  7 plots")
    print(f"  Output: {os.path.abspath(output_root)}")
    print(f"{'='*62}\n")

    all_data = []       # (scenario_key, tc, I_nA, fr)
    applied_log = {}    # scenario_key -> list of applied overrides

    # Baseline (no changes)
    print("── baseline ──────────────────────────────────")
    np.random.seed(42)
    base_params = {eq: dict(vals) for eq, vals in BASE.items()}
    eq_lines = get_eq_lines_from_params(base_params)
    try:
        tc       = run_timecourse(eq_lines)
        I_nA, fr = run_fi_curve(eq_lines)
        all_data.append(("baseline", tc, I_nA, fr))
        applied_log["baseline"] = []
        print("  ✓")
    except Exception as exc:
        print(f"  FAILED: {exc}")

    # Named scenarios
    for name, overrides in scenario_defs.items():
        label = SCENARIO_STYLE[name]["label"]
        print(f"── {label} ──────────────────────────────────")
        params, applied = build_scenario_params(overrides)
        eq_lines = get_eq_lines_from_params(params)

        print(f"  {overrides_to_string(overrides)}", end="  ", flush=True)

        np.random.seed(42)
        try:
            tc       = run_timecourse(eq_lines)
            I_nA, fr = run_fi_curve(eq_lines)
            all_data.append((name, tc, I_nA, fr))
            applied_log[name] = applied
            print("✓")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if all_data:
        print(f"\n→ building combined figures in: {output_root}")
        save_combined_figures_scenarios(output_root, all_data)

        INa_min, IK_max     = compute_extremes_scenarios(all_data)
        spike_time, shift   = compute_spike_shift_scenarios(all_data)
        save_extremes_txt_scenarios(output_root, applied_log, INa_min, IK_max, spike_time, shift)

        print(f"\n→ building extremes visualisation in: {output_root}")
        visualize_scenario_extremes(output_root, INa_min, IK_max, spike_time, shift)

    print(f"\n{'='*62}")
    print(f"  Done.  All scenario plots saved to: {os.path.abspath(output_root)}")
    print(f"{'='*62}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HH treatment/wash scenarios, or tune selected scenarios to target INa/IK changes."
    )
    parser.add_argument(
        "--output-root",
        default="scenario_plots",
        help="Folder where plots, summaries, and tuning files are saved.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Search scenario percentage changes to match the requested INa/IK targets.",
    )
    parser.add_argument(
        "--tune-only",
        action="store_true",
        help="With --tune, write tuning CSV/snippet but skip final tuned plots.",
    )
    parser.add_argument(
        "--tune-max-evals",
        type=int,
        default=120,
        help="Maximum candidate parameter sets tested per target scenario.",
    )
    parser.add_argument(
        "--tune-tolerance",
        type=float,
        default=1.0,
        help="Stop refining a scenario when combined INa/IK error is below this many percentage points.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scenario_defs = SCENARIOS

    if args.tune:
        scenario_defs = run_tuning_workflow(
            args.output_root,
            max_evals=args.tune_max_evals,
            tolerance=args.tune_tolerance,
        )
        if args.tune_only:
            print(f"\nTuning done. Results saved to: {os.path.abspath(args.output_root)}\n")
            return

    run_scenario_workflow(args.output_root, scenario_defs)


if __name__ == "__main__":
    main()
