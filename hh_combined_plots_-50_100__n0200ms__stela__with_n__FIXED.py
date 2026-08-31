"""
Hodgkin-Huxley Parameter Sweep — Combined Overlay Plots
=========================================================
For every (equation, parameter) pair, runs all 16 percentage steps
(-50% to +100% in 10% increments) and produces 7 combined figures,
one per diagnostic variable (v, m, h, n, INa, IK, f-I curve).

Each figure overlays all 16 traces:
  • Colour gradient  blue → red  (-50% … +100%)
  • Baseline (0%)    bold black line
  • Dashed lines for negative %, solid for positive %

Time-domain traces (v, m, h, n, INa, IK) are simulated in full, but the
first PLOT_START_MS of each trace is cropped out before plotting — it
is still computed, just not shown. This holds for any total run length.

Saved to:  combined_plots/{eq_name}__{param_label}/{variable}.png
"""

from brian2 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import numpy as np
import os

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

STEPS = list(range(-50, 101, 10))   # -50 % … +100 %, 10 % increments (16 steps)

PLOT_START_MS = 200   # time-domain plots only show t >= this value;
                       # the full simulation is still run/calculated either way

# ── Simulation time step ──────────────────────────────────────────────────────
# Change this single value to alter the integration time step used by every
# simulation (time-course runs and f-I curve runs alike).
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

PARAM_LABELS = {
    "alpha_m": {"A": "0.32",  "B": "4mV",  "C": "13mV"},
    "beta_m":  {"A": "0.28",  "B": "5mV",  "C": "40mV"},
    "alpha_h": {"A": "0.128", "B": "17mV", "C": "18mV"},
    "beta_h":  {"A": "40mV",  "B": "5mV"},
    "alpha_n": {"A": "0.032", "B": "5mV",  "C": "15mV"},
    "beta_n":  {"A": "0.5",   "B": "10mV", "C": "40mV"},
}

SWEEP_PLAN = [
    ("alpha_m", ["A", "B", "C"]),
    ("beta_m",  ["A", "B", "C"]),
    ("alpha_h", ["A", "B", "C"]),
    ("beta_h",  ["A", "B"]),
    ("alpha_n", ["A", "B", "C"]),
    ("beta_n",  ["A", "B", "C"]),
]

SOLVER = 'rk4'

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

def get_eq_lines(eq_name, perturbed_params):
    lines = {n: build_eq_line(n, BASE[n]) for n in BASE}
    lines[eq_name] = build_eq_line(eq_name, perturbed_params)
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
    t = tr.t / ms # time (ms)
    v_units = tr[0].v          # keep as a proper Brian2 quantity (volts) for current calc
    v = v_units / mV           # numeric mV, used only for plotting v(t)
    m = tr[0].m;  n = tr[0].n;  h = tr[0].h
    # Compute currents the physically correct way: I = g * (gate terms) * (v - E_rev),
    # done entirely in Brian2 units (siemens * volt = amp), THEN convert to nA.
    # (Previously g_na/nA and g_kd/nA were divided by nA before multiplying by a
    #  bare mV-valued voltage difference, which is dimensionally wrong and produced
    #  numbers that are not actually nA.)
    INa = (g_na * (m**3) * h * (v_units - ENa)) / nA
    IK  = (g_kd * (n**4)     * (v_units -  EK)) / nA
    return dict(t=t, v=v, m=m, n=n, h=h, INa=INa, IK=IK)

def run_fi_curve(eq_lines):
    am, bm, ah, bh, an, bn = eq_lines
    I_values = np.linspace(0, 5, 30) * nA
    # firing_rates = []
    # for I_val in I_values:
    #     start_scope()
    #     eqs_fi = Equations(eq_global)
    #     P = NeuronGroup(1, model=eqs_fi, threshold='v>-20*mV',
    #                     refractory=3*ms, method=SOLVER)
    #     P.v = El;  P.I_ext = I_val
    #     sp = SpikeMonitor(P)
    #     run(500 * ms)
    #     firing_rates.append(sp.count[0] / 0.5)

    start_scope()
    eqs_fi = Equations(eq_global.format(am=am, bm=bm, ah=ah, bh=bh, an=an, bn=bn))

    P = NeuronGroup(len(I_values), model=eqs_fi, threshold='v>-20*mV',
                    refractory=3*ms, method=SOLVER)
    P.v = El;  P.I_ext = I_values
    run(PLOT_START_MS * ms) # drop leading transient (same window as time-course plots)
    sp = SpikeMonitor(P)
    duration_fi = 1200
    run(duration_fi * ms)


    firing_rates = (sp.count /(duration_fi*1e-3))  # Convert to Hz duration is 500 ms
    #print(firing_rates)
    return I_values / nA, np.array(firing_rates)

# ── Colour scheme ─────────────────────────────────────────────────────────────
# Blue (-50%) → red (+100%), baseline black

def step_style(pct):
    """Return (color, linewidth, linestyle, zorder, alpha) for a given pct."""
    if pct == 0:
        return 'black', 2.5, '-', 10, 1.0
    idx   = STEPS.index(pct)          # 0..len(STEPS)-1
    norm  = idx / (len(STEPS) - 1)    # 0..1
    color = mpl_cm.coolwarm(norm)
    lw    = 1.0
    ls    = '--' if pct < 0 else '-'
    alpha = 0.75
    return color, lw, ls, 5, alpha

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

def save_combined_figures(folder, all_data, eq_name, param_key):
    """
    all_data: list of (pct, tc_dict, I_nA, fr) tuples
    Produces 7 PNG files in folder.
    """
    os.makedirs(folder, exist_ok=True)
    orig_label = PARAM_LABELS[eq_name][param_key]

    for key, xk, yk, xlabel, ylabel, base_title in PLOT_SPECS:
        fig, ax = plt.subplots(figsize=(10, 5))

        for pct, tc, I_nA, fr in all_data:
            color, lw, ls, zo, alpha = step_style(pct)

            if xk == "I":
                x, y = I_nA, fr
            else:
                mask = tc["t"] >= PLOT_START_MS   # drop the leading transient
                x = tc[xk][mask]
                y = tc[yk][mask]

            label = f"{'+' if pct > 0 else ''}{pct}%"
            ax.plot(x, y, color=color, lw=lw, ls=ls, zorder=zo,
                    alpha=alpha, label=label)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f"{base_title}\n"
            f"eq: {eq_name}  |  param: {orig_label}  |  "
            f"sweep {STEPS[0]:+d} % to {STEPS[-1]:+d} %",
            fontsize=11
        )

        # ── Colour-bar style legend ───────────────────────────────────────
        # Sort legend entries from most negative to most positive
        handles, labels_l = ax.get_legend_handles_labels()
        order = sorted(range(len(labels_l)),
                       key=lambda i: int(labels_l[i].replace('%', '').replace('+', '')))
        leg = ax.legend(
            [handles[i] for i in order],
            [labels_l[i] for i in order],
            loc='upper right',
            fontsize=7.5,
            ncol=2,
            framealpha=0.85,
            title=f"% change of {orig_label}",
            title_fontsize=8,
        )
        # Bold the 0% legend entry
        for text in leg.get_texts():
            if text.get_text() == "0%":
                text.set_fontweight('bold')

        # ── Colour gradient colour-bar (decorative, shows the mapping) ────
        sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                   norm=plt.Normalize(vmin=STEPS[0], vmax=STEPS[-1]))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.025)
        cbar.set_label("% change", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        plt.tight_layout()
        fig.savefig(os.path.join(folder, f"{key}.png"), dpi=430)
        plt.close(fig)
        print(f"    saved {key}.png")

def compute_extremes(all_data):
    """
    For every percentage step, compute:
      - INa_min : the most negative value of INa(t) over the plotted window (t >= PLOT_START_MS)
      - IK_max  : the most positive value of IK(t)  over the plotted window (t >= PLOT_START_MS)
    Returns two dicts keyed by pct: {pct: value}
    """
    INa_min = {}
    IK_max = {}
    for pct, tc, I_nA, fr in all_data:
        mask = tc["t"] >= PLOT_START_MS
        INa_min[pct] = float(np.min(tc["INa"][mask]))
        IK_max[pct]  = float(np.max(tc["IK"][mask]))
    return INa_min, IK_max

def compute_spike_shift(all_data, threshold_mV=-20.0):
    """
    For every percentage step, find the time (within the plotted window,
    t >= PLOT_START_MS) at which v first crosses the model's spike
    threshold (v > -20 mV) — i.e. the time of the spike/pulse in that trace.

    Then compute how far that time is shifted relative to the 0% baseline
    trace's spike time, in ms. A positive shift means the spike happens
    LATER than baseline; negative means EARLIER.

    Returns two dicts keyed by pct:
      spike_time : {pct: spike time in ms, or None if no spike crossed threshold}
      shift      : {pct: (spike_time - baseline_spike_time) in ms, or None}
    """
    spike_time = {}
    for pct, tc, I_nA, fr in all_data:
        mask = tc["t"] >= PLOT_START_MS
        t = tc["t"][mask]
        v = tc["v"][mask]
        above = v > threshold_mV
        if above.any():
            idx = int(np.argmax(above))   # index of first crossing
            spike_time[pct] = float(t[idx])
        else:
            spike_time[pct] = None   # never reached threshold in this window

    baseline_t = spike_time.get(0)
    shift = {}
    for pct, st in spike_time.items():
        if st is not None and baseline_t is not None:
            shift[pct] = st - baseline_t
        else:
            shift[pct] = None
    return spike_time, shift

def save_extremes_txt(folder, eq_name, param_key, orig_label, base_val, INa_min, IK_max,
                       spike_time, shift):
    """
    Writes a text summary (NOT shown in the plot legends) listing, for every
    percentage step:
      - the most negative INa value reached, and its % change from the 0% baseline
      - the most positive IK  value reached, and its % change from the 0% baseline
      - the time at which the spike/pulse crosses threshold, and how far in
        time (ms) that spike is shifted relative to the 0% baseline trace
    Saved as extremes_summary.txt in the same folder as the PNGs.
    """
    baseline_INa = INa_min.get(0)
    baseline_IK  = IK_max.get(0)

    path = os.path.join(folder, "extremes_summary.txt")
    with open(path, "w") as f:
        f.write(f"Extreme-value summary\n")
        f.write(f"  equation : {eq_name}\n")
        f.write(f"  param    : {param_key}  ({orig_label}, base value = {base_val:.6g})\n")
        f.write(f"  window   : t >= {PLOT_START_MS} ms\n")
        f.write(f"  dt       : {TIME_STEP}\n")
        f.write("=" * 78 + "\n\n")

        f.write("INa — most negative value reached (nA)\n")
        f.write("-" * 78 + "\n")
        f.write(f"{'% change':>10}   {'INa_min (nA)':>15}   {'Δ from 0% baseline':>20}\n")
        for pct in sorted(INa_min):
            val = INa_min[pct]
            if baseline_INa not in (None, 0):
                delta_pct = (val - baseline_INa) / abs(baseline_INa) * 100
                delta_str = f"{delta_pct:+.3f} %"
            else:
                delta_str = "n/a"
            f.write(f"{pct:+9d}%   {val:15.6g}   {delta_str:>20}\n")

        f.write("\n")
        f.write("IK — most positive value reached (nA)\n")
        f.write("-" * 78 + "\n")
        f.write(f"{'% change':>10}   {'IK_max (nA)':>15}   {'Δ from 0% baseline':>20}\n")
        for pct in sorted(IK_max):
            val = IK_max[pct]
            if baseline_IK not in (None, 0):
                delta_pct = (val - baseline_IK) / abs(baseline_IK) * 100
                delta_str = f"{delta_pct:+.3f} %"
            else:
                delta_str = "n/a"
            f.write(f"{pct:+9d}%   {val:15.6g}   {delta_str:>20}\n")

        f.write("\n")
        f.write("Spike/pulse timing shift, relative to 0% baseline (ms)\n")
        f.write(f"  (spike time = first crossing of v > -20 mV, within t >= {PLOT_START_MS} ms)\n")
        f.write(f"  positive shift = spike occurs LATER than baseline\n")
        f.write(f"  negative shift = spike occurs EARLIER than baseline\n")
        f.write("-" * 78 + "\n")
        f.write(f"{'% change':>10}   {'spike time (ms)':>16}   {'shift vs 0% (ms)':>18}\n")
        for pct in sorted(spike_time):
            st = spike_time[pct]
            sh = shift[pct]
            st_str = f"{st:16.4f}" if st is not None else f"{'no spike':>16}"
            sh_str = f"{sh:+18.4f}" if sh is not None else f"{'n/a':>18}"
            f.write(f"{pct:+9d}%   {st_str}   {sh_str}\n")

    print(f"    saved extremes_summary.txt")

# ── Main ──────────────────────────────────────────────────────────────────────

OUTPUT_ROOT = "combined_plots"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

pair_count = sum(len(pk) for _, pk in SWEEP_PLAN)
print(f"\n{'='*62}")
print(f"  HH Combined Overlay Sweep")
print(f"  {pair_count} (equation, parameter) pairs  ×  {len(STEPS)} steps  ×  7 plots")
print(f"  Output: {os.path.abspath(OUTPUT_ROOT)}")
print(f"{'='*62}\n")

for eq_name, param_keys in SWEEP_PLAN:
    for param_key in param_keys:
        orig_label = PARAM_LABELS[eq_name][param_key]
        base_val   = BASE[eq_name][param_key]

        folder = os.path.join(OUTPUT_ROOT, f"{eq_name}__{orig_label}")
        print(f"\n── {eq_name}  param {param_key} ({orig_label}) ──────────────────")

        all_data = []   # (pct, tc, I_nA, fr)

        for pct in STEPS:
            new_val = base_val * (1 + pct / 100)
            params  = dict(BASE[eq_name])
            params[param_key] = new_val
            eq_lines = get_eq_lines(eq_name, params)

            sign_str = f"{'+' if pct > 0 else ''}{pct:+d}%".replace("++", "+")
            new_val_str = f"{new_val:.5g}"
            print(f"  {sign_str:>5}  ({orig_label} → {new_val_str})", end="  ", flush=True)

            np.random.seed(42)
            try:
                tc       = run_timecourse(eq_lines)
                I_nA, fr = run_fi_curve(eq_lines)
                all_data.append((pct, tc, I_nA, fr))
                print("✓")
            except Exception as exc:
                print(f"FAILED: {exc}")

        if all_data:
            print(f"  → building combined figures in: {folder}")
            save_combined_figures(folder, all_data, eq_name, param_key)

            INa_min, IK_max = compute_extremes(all_data)
            spike_time, shift = compute_spike_shift(all_data)
            save_extremes_txt(folder, eq_name, param_key, orig_label, base_val,
                               INa_min, IK_max, spike_time, shift)

print(f"\n{'='*62}")
print(f"  Done.  All combined plots saved to: {os.path.abspath(OUTPUT_ROOT)}")
print(f"{'='*62}\n")