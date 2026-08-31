"""
Visualise extremes_summary.txt files
=====================================
Walks a directory tree (default: ./combined_plots), finds every
extremes_summary.txt produced by the HH sweep script, parses out:

  - INa_min   (per pct, + % change vs 0% baseline)
  - IK_max    (per pct, + % change vs 0% baseline)
  - spike timing shift vs 0% baseline (ms)

...and produces, per (equation, parameter) folder:

  1. {folder}/extremes_summary_plot.png
     3 panels, y-axis = RAW VALUE: INa_min, IK_max, spike shift (all vs %change in param)

  2. {folder}/extremes_summary_pctchange_plot.png
     2 panels, y-axis = % CHANGE OF CURRENT vs 0% baseline: INa %change, IK %change
     (spike shift is already expressed as a difference from baseline, so it isn't
     duplicated here)

...and, per equation, overview figures across all its parameters (A/B/C):

  3. {OUTPUT_ROOT}/overview__{eq_name}.png
     Same 3 raw-value panels as (1), one line per parameter.

  4. {OUTPUT_ROOT}/overview__{eq_name}_pctchange.png
     Same 2 %-change panels as (2), one line per parameter.

Usage:
    python visualize_extremes.py [root_folder]

If root_folder is omitted, defaults to "combined_plots" in the current
working directory (matching OUTPUT_ROOT in the sweep script).
"""

import os
import re
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm


# ── Parsing ──────────────────────────────────────────────────────────────────

HEADER_RE = re.compile(
    r"equation\s*:\s*(?P<eq_name>\S+)\s*\n"
    r"\s*param\s*:\s*(?P<param_key>\S+)\s*\(\s*(?P<orig_label>[^,]+),\s*base value\s*=\s*(?P<base_val>[-\d.eE+]+)\s*\)"
)

# Matches rows like:  "     -50%          -1.23456           +3.210 %"
# or "n/a" / "no spike" in place of numeric fields.
ROW_RE = re.compile(
    r"^\s*([+-]?\d+)%\s+(\S+(?:\s+%)?)\s+(\S+(?:\s+%)?)\s*$",
    re.MULTILINE
)


def _parse_block(block_text):
    """
    Parses a table block into {pct: (col1_str, col2_str)}.
    Values are returned as raw strings ('n/a', 'no spike', or numeric strings
    with optional trailing '%') for the caller to interpret per-column.
    """
    out = {}
    for m in ROW_RE.finditer(block_text):
        pct = int(m.group(1))
        out[pct] = (m.group(2), m.group(3))
    return out


def _to_float(s):
    if s is None:
        return None
    s = s.strip()
    if s.lower() in ("n/a", "no", "spike", "no_spike", "no spike"):
        return None
    s = s.replace('%', '').replace('+', '')
    try:
        return float(s)
    except ValueError:
        return None


def parse_extremes_txt(path):
    """
    Parses one extremes_summary.txt file.
    Returns a dict:
      eq_name, param_key, orig_label, base_val,
      INa_min   : {pct: value_nA}
      INa_delta : {pct: pct_change_from_baseline or None}
      IK_max    : {pct: value_nA}
      IK_delta  : {pct: pct_change_from_baseline or None}
      spike_time: {pct: ms or None}
      spike_shift: {pct: ms or None}
    """
    with open(path) as f:
        text = f.read()

    hm = HEADER_RE.search(text)
    if not hm:
        return None
    eq_name    = hm.group('eq_name')
    param_key  = hm.group('param_key')
    orig_label = hm.group('orig_label').strip()
    base_val   = float(hm.group('base_val'))

    # Split into the three sections by their headers
    ina_split   = text.split("INa — most negative value reached")
    ik_split    = text.split("IK — most positive value reached")
    shift_split = text.split("Spike/pulse timing shift")

    INa_min, INa_delta = {}, {}
    if len(ina_split) > 1:
        section = ina_split[1].split("IK — most positive value reached")[0]
        rows = _parse_block(section)
        for pct, (v1, v2) in rows.items():
            INa_min[pct]   = _to_float(v1)
            INa_delta[pct] = _to_float(v2)

    IK_max, IK_delta = {}, {}
    if len(ik_split) > 1:
        section = ik_split[1].split("Spike/pulse timing shift")[0]
        rows = _parse_block(section)
        for pct, (v1, v2) in rows.items():
            IK_max[pct]   = _to_float(v1)
            IK_delta[pct] = _to_float(v2)

    spike_time, spike_shift = {}, {}
    if len(shift_split) > 1:
        section = shift_split[1]
        rows = _parse_block(section)
        for pct, (v1, v2) in rows.items():
            spike_time[pct]  = _to_float(v1)
            spike_shift[pct] = _to_float(v2)

    return dict(
        eq_name=eq_name, param_key=param_key, orig_label=orig_label, base_val=base_val,
        INa_min=INa_min, INa_delta=INa_delta,
        IK_max=IK_max, IK_delta=IK_delta,
        spike_time=spike_time, spike_shift=spike_shift,
    )


def find_summary_files(root):
    found = []
    for dirpath, _, filenames in os.walk(root):
        if "extremes_summary.txt" in filenames:
            found.append(os.path.join(dirpath, "extremes_summary.txt"))
    return sorted(found)


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _sorted_xy(d):
    pcts = sorted(k for k, v in d.items() if v is not None)
    xs = pcts
    ys = [d[p] for p in pcts]
    return xs, ys


def plot_folder_summary(record, out_path):
    """One figure per (equation, parameter): 3 panels, RAW VALUE vs % change in param."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    x1, y1 = _sorted_xy(record["INa_min"])
    axes[0].plot(x1, y1, 'o-', color='tab:blue')
    axes[0].axvline(0, color='gray', lw=0.8, ls='--')
    axes[0].set_xlabel("% change")
    axes[0].set_ylabel("INa_min (nA)")
    axes[0].set_title("Peak inward Na$^+$ current")

    x2, y2 = _sorted_xy(record["IK_max"])
    axes[1].plot(x2, y2, 'o-', color='tab:red')
    axes[1].axvline(0, color='gray', lw=0.8, ls='--')
    axes[1].set_xlabel("% change")
    axes[1].set_ylabel("IK_max (nA)")
    axes[1].set_title("Peak outward K$^+$ current")

    x3, y3 = _sorted_xy(record["spike_shift"])
    axes[2].plot(x3, y3, 'o-', color='tab:green')
    axes[2].axhline(0, color='gray', lw=0.8, ls='--')
    axes[2].axvline(0, color='gray', lw=0.8, ls='--')
    axes[2].set_xlabel("% change")
    axes[2].set_ylabel("Spike shift vs 0% baseline (ms)")
    axes[2].set_title("Pulse timing shift")

    fig.suptitle(
        f"{record['eq_name']}  |  param {record['param_key']} ({record['orig_label']}, "
        f"base = {record['base_val']:.6g})  —  raw values",
        fontsize=12
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_folder_pctchange(record, out_path):
    """One figure per (equation, parameter): 2 panels, y-axis = % CHANGE OF CURRENT
    vs 0% baseline (INa_delta, IK_delta), x-axis = % change in the swept parameter."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    x1, y1 = _sorted_xy(record["INa_delta"])
    axes[0].plot(x1, y1, 'o-', color='tab:blue')
    axes[0].axhline(0, color='gray', lw=0.8, ls='--')
    axes[0].axvline(0, color='gray', lw=0.8, ls='--')
    axes[0].set_xlabel("% change in parameter")
    axes[0].set_ylabel("INa_min % change vs 0% baseline")
    axes[0].set_title("Peak inward Na$^+$ current — % change")

    x2, y2 = _sorted_xy(record["IK_delta"])
    axes[1].plot(x2, y2, 'o-', color='tab:red')
    axes[1].axhline(0, color='gray', lw=0.8, ls='--')
    axes[1].axvline(0, color='gray', lw=0.8, ls='--')
    axes[1].set_xlabel("% change in parameter")
    axes[1].set_ylabel("IK_max % change vs 0% baseline")
    axes[1].set_title("Peak outward K$^+$ current — % change")

    fig.suptitle(
        f"{record['eq_name']}  |  param {record['param_key']} ({record['orig_label']}, "
        f"base = {record['base_val']:.6g})  —  % change vs baseline",
        fontsize=12
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_equation_overview(eq_name, records, out_path):
    """One figure per equation: overlays all its parameters (A/B/C), RAW VALUES, 3 panels."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = mpl_cm.tab10.colors

    for i, rec in enumerate(records):
        color = colors[i % len(colors)]
        label = f"{rec['param_key']} ({rec['orig_label']})"

        x1, y1 = _sorted_xy(rec["INa_min"])
        axes[0].plot(x1, y1, 'o-', color=color, label=label, ms=3)

        x2, y2 = _sorted_xy(rec["IK_max"])
        axes[1].plot(x2, y2, 'o-', color=color, label=label, ms=3)

        x3, y3 = _sorted_xy(rec["spike_shift"])
        axes[2].plot(x3, y3, 'o-', color=color, label=label, ms=3)

    titles = ["Peak inward Na$^+$ current", "Peak outward K$^+$ current", "Pulse timing shift"]
    ylabels = ["INa_min (nA)", "IK_max (nA)", "Spike shift vs 0% baseline (ms)"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.axvline(0, color='gray', lw=0.8, ls='--')
        ax.set_xlabel("% change")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[2].axhline(0, color='gray', lw=0.8, ls='--')
    axes[0].legend(fontsize=8, title="parameter")

    fig.suptitle(f"Equation: {eq_name}  —  all swept parameters (raw values)", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_equation_overview_pctchange(eq_name, records, out_path):
    """One figure per equation: overlays all its parameters (A/B/C), % CHANGE OF CURRENT,
    2 panels (INa %change, IK %change)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = mpl_cm.tab10.colors

    for i, rec in enumerate(records):
        color = colors[i % len(colors)]
        label = f"{rec['param_key']} ({rec['orig_label']})"

        x1, y1 = _sorted_xy(rec["INa_delta"])
        axes[0].plot(x1, y1, 'o-', color=color, label=label, ms=3)

        x2, y2 = _sorted_xy(rec["IK_delta"])
        axes[1].plot(x2, y2, 'o-', color=color, label=label, ms=3)

    titles = ["Peak inward Na$^+$ current — % change", "Peak outward K$^+$ current — % change"]
    ylabels = ["INa_min % change vs 0% baseline", "IK_max % change vs 0% baseline"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.axvline(0, color='gray', lw=0.8, ls='--')
        ax.set_xlabel("% change in parameter")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].legend(fontsize=8, title="parameter")

    fig.suptitle(f"Equation: {eq_name}  —  all swept parameters (% change vs baseline)", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ── Parameter x %-change summary tables (rows=parameters, cols=% sweep) ──────

def _row_label(rec):
    """e.g. 'alpha_m.A (0.32)' — equation.param_key with its original base value."""
    return f"{rec['eq_name']}.{rec['param_key']} ({rec['base_val']:.4g})"


def build_delta_tables(records):
    """
    Builds three tables (dicts of dicts) across ALL parsed records:
      ina_table[row_label][pct]   = INa % change vs 0% baseline
      ik_table[row_label][pct]    = IK  % change vs 0% baseline
      shift_table[row_label][pct] = spike/pulse timing shift vs 0% baseline (ms)
    """
    ina_table, ik_table, shift_table = {}, {}, {}
    for rec in records:
        label = _row_label(rec)
        ina_table[label]   = rec["INa_delta"]
        ik_table[label]    = rec["IK_delta"]
        shift_table[label] = rec["spike_shift"]
    return ina_table, ik_table, shift_table


def _all_pcts(table):
    return sorted({pct for row in table.values() for pct in row.keys()})


def write_table_csv(table, out_path, value_fmt="{:.2f}"):
    """Writes the table as a CSV — opens directly as a spreadsheet in Excel/Sheets."""
    pcts = _all_pcts(table)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter"] + [f"{p:+d}%" for p in pcts])
        for label in sorted(table.keys()):
            row = table[label]
            writer.writerow(
                [label] + [
                    (value_fmt.format(row[p]) if row.get(p) is not None else "")
                    for p in pcts
                ]
            )


def plot_table_heatmap(table, out_path, title, cbar_label="% change vs 0% baseline",
                        cell_fmt="{:.1f}"):
    """
    Renders the table as an Excel-like colored grid:
      rows = parameters (eq_name.param_key (base value))
      cols = % change in the swept parameter (-50% ... +100%)
      cell = value in the table (either % change of current, or ms timing shift)
    """
    pcts = _all_pcts(table)
    labels = sorted(table.keys())
    if not pcts or not labels:
        return

    data = np.full((len(labels), len(pcts)), np.nan)
    for i, label in enumerate(labels):
        row = table[label]
        for j, pct in enumerate(pcts):
            v = row.get(pct)
            if v is not None:
                data[i, j] = v

    finite = data[~np.isnan(data)]
    vmax = np.nanmax(np.abs(finite)) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig_w = max(8, 0.65 * len(pcts) + 2)
    fig_h = max(4, 0.4 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels([f"{p:+d}%" for p in pcts], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    # gridlines to make it look like a spreadsheet
    ax.set_xticks(np.arange(-0.5, len(pcts), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(labels)):
        for j in range(len(pcts)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, cell_fmt.format(v), ha="center", va="center",
                        fontsize=6.5, color="black")

    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.03, pad=0.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def build_summary_tables(records, root):
    """Builds and saves the INa, IK, and spike-timing-shift parameter x %-change
    summary tables (CSV + heatmap PNG) into the root output folder."""
    ina_table, ik_table, shift_table = build_delta_tables(records)

    ina_csv   = os.path.join(root, "INa_pctchange_table.csv")
    ik_csv    = os.path.join(root, "IK_pctchange_table.csv")
    shift_csv = os.path.join(root, "spike_shift_ms_table.csv")
    write_table_csv(ina_table, ina_csv)
    write_table_csv(ik_table, ik_csv)
    write_table_csv(shift_table, shift_csv, value_fmt="{:.3f}")
    print(f"  saved {ina_csv}")
    print(f"  saved {ik_csv}")
    print(f"  saved {shift_csv}")

    ina_png   = os.path.join(root, "INa_pctchange_table.png")
    ik_png    = os.path.join(root, "IK_pctchange_table.png")
    shift_png = os.path.join(root, "spike_shift_ms_table.png")
    plot_table_heatmap(ina_table, ina_png,
                        "INa peak — % change vs 0% baseline (rows = parameters, cols = % sweep)")
    plot_table_heatmap(ik_table, ik_png,
                        "IK peak — % change vs 0% baseline (rows = parameters, cols = % sweep)")
    plot_table_heatmap(shift_table, shift_png,
                        "Spike/pulse timing shift vs 0% baseline (rows = parameters, cols = % sweep)",
                        cbar_label="timing shift (ms)", cell_fmt="{:.2f}")
    print(f"  saved {ina_png}")
    print(f"  saved {ik_png}")
    print(f"  saved {shift_png}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "scenario_plots"
    if not os.path.isdir(root):
        print(f"Folder not found: {root}")
        sys.exit(1)

    files = find_summary_files(root)
    if not files:
        print(f"No extremes_summary.txt files found under: {root}")
        sys.exit(1)

    print(f"Found {len(files)} extremes_summary.txt file(s) under {root}\n")

    records = []
    for path in files:
        rec = parse_extremes_txt(path)
        if rec is None:
            print(f"  ! could not parse: {path}")
            continue
        rec["_folder"] = os.path.dirname(path)
        records.append(rec)

        raw_out = os.path.join(rec["_folder"], "extremes_summary_plot.png")
        plot_folder_summary(rec, raw_out)
        print(f"  saved {raw_out}")

        pct_out = os.path.join(rec["_folder"], "extremes_summary_pctchange_plot.png")
        plot_folder_pctchange(rec, pct_out)
        print(f"  saved {pct_out}")

    # Group by equation for overview figures
    by_eq = defaultdict(list)
    for rec in records:
        by_eq[rec["eq_name"]].append(rec)

    for eq_name, recs in by_eq.items():
        recs_sorted = sorted(recs, key=lambda r: r["param_key"])

        raw_out = os.path.join(root, f"overview__{eq_name}.png")
        plot_equation_overview(eq_name, recs_sorted, raw_out)
        print(f"  saved {raw_out}")

        pct_out = os.path.join(root, f"overview__{eq_name}_pctchange.png")
        plot_equation_overview_pctchange(eq_name, recs_sorted, pct_out)
        print(f"  saved {pct_out}")

    # Master parameter x %-change tables (rows=parameters, cols=% sweep) across
    # ALL equations/parameters at once — INa, IK, and spike timing shift.
    build_summary_tables(records, root)

    print(f"\nDone. {len(records)} folder(s) visualised, {len(by_eq)} equation overview(s) built "
          f"(raw + % change versions of each), plus INa/IK/spike-shift parameter x %-change summary tables.")


if __name__ == "__main__":
    main()