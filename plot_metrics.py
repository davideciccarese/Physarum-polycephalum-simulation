#!/usr/bin/env python3
"""
plot_metrics.py

Reads the metrics CSV files written by physarum_sim.py (one per scenario) and
produces, for each of the five scenarios, a PNG with the raw (non-normalised) time series of
the main network metrics, plus one Excel workbook that gathers the CSV
files as separate sheets.

Usage:
    python plot_metrics.py --outdir physarum_output
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [("occupied_area", "occupied cells"),
          ("network_length", "network length (D above prune threshold)"),
          ("consumed", "cumulative nutrient consumed"),
          ("mean_D", "mean conductivity of active tubes"),
          ("n_tubes", "number of tubes above prune threshold"),
          ("n_components", "connected components"),
          ("biomass", "total biomass"),
          ("cost_Dl", "network cost sum(D l)"),
          ("dissipation", "dissipation sum(Q^2 l / D)")]


def read_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    cols = {k: [float(r[k]) for r in rows] for k in rows[0].keys()}
    return cols


def plot_scenario(cols, scenario, out_png):
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    t = cols["time"]
    for ax, (key, label) in zip(axes.ravel(), PANELS):
        ax.plot(t, cols[key], lw=1.3)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("time")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Scenario {scenario}: network evolution metrics (raw values)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="physarum_output")
    args = ap.parse_args()
    try:
        from openpyxl import Workbook
    except ImportError:
        Workbook = None
    wb = Workbook() if Workbook else None
    first = True
    for sc in ("A", "B", "C", "D", "E"):
        path = os.path.join(args.outdir, f"metrics_scenario_{sc}.csv")
        if not os.path.exists(path):
            print(f"missing {path}")
            continue
        cols = read_csv(path)
        png = os.path.join(args.outdir, f"metrics_scenario_{sc}.png")
        plot_scenario(cols, sc, png)
        print(f"wrote {png}")
        if wb is not None:
            ws = wb.active if first else wb.create_sheet()
            ws.title = f"scenario_{sc}"
            first = False
            keys = list(cols.keys())
            ws.append(keys)
            for i in range(len(cols["step"])):
                ws.append([cols[k][i] for k in keys])
    if wb is not None:
        xlsx = os.path.join(args.outdir, "physarum_metrics_all_scenarios.xlsx")
        wb.save(xlsx)
        print(f"wrote {xlsx}")


if __name__ == "__main__":
    main()
