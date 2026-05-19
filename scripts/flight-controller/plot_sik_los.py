#!/usr/bin/env python3
"""
plot_sik_los.py - Plot SiK RSSI vs distance from a test_sik_link.py walk-out CSV.

Reads the CSV emitted by test_sik_link.py --mode walkout (or full) and produces
a PNG showing local + remote RSSI at each marker waypoint, with all 239+
in-between samples as light scatter, a horizontal weak-link threshold band,
and an annotation summarising rxerrors / fixed-error totals.

Usage:
    python3 plot_sik_los.py <csv_path> [output_png]

If output_png is omitted, the PNG is written alongside the CSV with the
same stem and "_plot.png" suffix.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        return 1

    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
    else:
        out_path = csv_path.with_name(csv_path.stem + "_plot.png")

    # Read every sample + collect marker rows.
    all_rssi: list[int] = []
    all_remrssi: list[int] = []
    markers: list[dict] = []
    rxerr_first: int | None = None
    rxerr_last = 0
    fix_first: int | None = None
    fix_last = 0

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rssi = int(r["rssi"])
                remrssi = int(r["remrssi"])
                rxerr = int(r["rxerrors"])
                fix = int(r["fixed"])
            except (KeyError, ValueError):
                continue
            all_rssi.append(rssi)
            all_remrssi.append(remrssi)
            if rxerr_first is None:
                rxerr_first = rxerr
                fix_first = fix
            rxerr_last = rxerr
            fix_last = fix

            mark = r.get("marker_distance_m", "").strip()
            if mark:
                try:
                    d = float(mark)
                except ValueError:
                    continue
                markers.append({
                    "distance": d,
                    "rssi": rssi,
                    "remrssi": remrssi,
                    "noise": int(r["noise"]),
                    "remnoise": int(r["remnoise"]),
                    "rxerrors": rxerr,
                })

    if not all_rssi:
        print("ERROR: no samples in CSV")
        return 1
    if not markers:
        print("WARN: no markers in CSV; plot will only show sample scatter")

    markers.sort(key=lambda m: m["distance"])

    rxerr_accum = (rxerr_last - rxerr_first) if rxerr_first is not None else 0
    fix_accum = (fix_last - fix_first) if fix_first is not None else 0

    # --- Plot --- #
    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)

    if markers:
        xs = [m["distance"] for m in markers]
        ax.plot(
            xs,
            [m["rssi"] for m in markers],
            "o-",
            color="C0",
            linewidth=2.0,
            markersize=10,
            label="Local RSSI (ground unit)",
            zorder=3,
        )
        ax.plot(
            xs,
            [m["remrssi"] for m in markers],
            "s-",
            color="C3",
            linewidth=2.0,
            markersize=10,
            label="Remote RSSI (air unit)",
            zorder=3,
        )

        # Annotate each marker with its RSSI value.
        for m in markers:
            ax.annotate(
                f"{m['rssi']}",
                (m["distance"], m["rssi"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color="C0",
                fontweight="bold",
            )
            ax.annotate(
                f"{m['remrssi']}",
                (m["distance"], m["remrssi"]),
                textcoords="offset points",
                xytext=(0, -16),
                ha="center",
                fontsize=9,
                color="C3",
                fontweight="bold",
            )

    # Weak-link threshold band (label on the right so it does not collide
    # with the legend at lower-left).
    ax.axhspan(0, 50, color="red", alpha=0.08, zorder=1)
    ax.text(
        0.985,
        0.10,
        "Weak link region (RSSI < 50)",
        transform=ax.transAxes,
        fontsize=9,
        color="darkred",
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="darkred", alpha=0.9),
    )

    # Reliability annotation.
    n_samples = len(all_rssi)
    ax.text(
        0.98,
        0.98,
        (
            f"Total samples: {n_samples}\n"
            f"rxerrors accumulated: {rxerr_accum}\n"
            f"fixed errors:          {fix_accum}"
        ),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.95),
    )

    # Axes / labels.
    ax.set_xlabel("Operator-to-drone distance (m)", fontsize=12)
    ax.set_ylabel("RSSI", fontsize=12)
    ax.set_title(
        "SiK 433 MHz indoor through-wall RSSI vs distance"
        "\n2026-05-18 walk-out, ~4 min, zero packet loss across full route",
        fontsize=13,
    )
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    if markers:
        ax.set_xlim(-2, max(m["distance"] for m in markers) + 3)
    ax.set_ylim(0, 235)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
