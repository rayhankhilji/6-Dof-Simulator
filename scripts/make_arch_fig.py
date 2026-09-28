#!/usr/bin/env python
"""Draw the GNC closed-loop block diagram -> docs/figures/architecture.png.

Single-row feedback loop (top row left->right, loop back along the bottom):

    Vehicle -> Sensors -> Navigation (EKF) -> Guidance -> Control
        ^                                                   |
        |__________________  Actuators  <___________________|

with ``Environment`` feeding the vehicle dynamics and the sensor
measurements from below (dashed).  Publication style, ~2200 x 900 px,
no title.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures", "architecture.png")

# ---------------------------------------------------------------------------
# Canvas: 112 x 50 units -> figsize (11.2, 5.0) at dpi 200 ~ 2240 x 1000 px.
# Top-row boxes y = 28..43 (h = 15, w = 15, gaps 8); bottom row y = 2..15.
# ---------------------------------------------------------------------------

BW, BH = 15.0, 15.0          # top-row box width / height
TY = 28.0                    # top-row box bottom

BOXES = [
    # key, x, title, sub-labels, edge color
    ("vehicle",    4.0, "Vehicle",
     ["6-DOF dynamics", "m, cg, J", "aero + thrust"], "#8c564b"),
    ("sensors",   27.0, "Sensors",
     ["IMU", "GPS · baro", "radar altimeter"], "#bcbd22"),
    ("nav",       50.0, "Navigation",
     ["EKF", "state estimate", "bias + covariance"], "#2ca02c"),
    ("guidance",  73.0, "Guidance",
     ["ZEM/ZEV", "polynomial", "optimal"], "#1f77b4"),
    ("control",   96.0, "Control",
     ["PID · LQR · MPC", "geometric"], "#9467bd"),
]
ACT = (94.0, 2.0, 17.0, 13.0)   # actuators box (x, y, w, h)
ENV = (44.0, 2.0, 18.0, 13.0)   # environment box


def edge_pt(box, edge, frac=0.5):
    """Point on a box edge; ``frac`` in [0,1] along the edge."""
    x, y, w, h = box
    return {
        "left": (x, y + frac * h), "right": (x + w, y + frac * h),
        "top": (x + frac * w, y + h), "bottom": (x + frac * w, y),
        "center": (x + w / 2.0, y + h / 2.0),
    }[edge]


def arrow(ax, p0, p1, style="-", color="#333333", lw=1.6):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=15, lw=lw, color=color,
        linestyle=style, shrinkA=1.5, shrinkB=1.5, zorder=2))


def rail_arrow(ax, pts, color="#333333", lw=1.6):
    """Polyline through ``pts`` with an arrowhead on the last segment."""
    for p0, p1 in zip(pts[:-2], pts[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw,
                solid_capstyle="round", zorder=2)
    ax.add_patch(FancyArrowPatch(
        pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=15, lw=lw,
        color=color, shrinkA=0, shrinkB=1.5, zorder=2))


def main(argv=None):
    fig, ax = plt.subplots(figsize=(11.2, 5.0), dpi=200)
    ax.set_xlim(0, 112)
    ax.set_ylim(-3.5, 46.5)
    ax.axis("off")

    geom = {k: (x, TY, BW, BH) for k, x, *_ in BOXES}
    geom["actuators"] = ACT
    geom["env"] = ENV

    # ---------------- forward chain (top row) ----------------
    fwd = [("vehicle", "sensors", "state $x$"),
           ("sensors", "nav", "meas. $z_k$"),
           ("nav", "guidance", "$\\hat{x}$"),
           ("guidance", "control", "$a_{cmd}$, on/off")]
    for src, dst, lab in fwd:
        p0 = edge_pt(geom[src], "right")
        p1 = edge_pt(geom[dst], "left")
        arrow(ax, p0, p1)
        ax.text((p0[0] + p1[0]) / 2, p0[1] + 1.6, lab, ha="center",
                va="bottom", fontsize=8.0, style="italic", color="#333333")

    # ---------------- control -> actuators ----------------
    p0 = edge_pt(geom["control"], "bottom", 0.5)
    p1 = edge_pt(geom["actuators"], "top", 0.5)
    arrow(ax, p0, p1)
    ax.text(p0[0] - 1.5, (p0[1] + p1[1]) / 2,
            "$\\delta$, $u$, $\\tau_{rcs}$", ha="right", va="center",
            fontsize=8.0, style="italic", color="#333333")

    # ---------------- actuators -> vehicle (bottom rail) ----------------
    ry = -1.2
    vx = geom["vehicle"][0] + 0.5 * BW
    axv = geom["actuators"][0] + 0.5 * ACT[2]
    rail_arrow(ax, [
        (axv, ACT[1]), (axv, ry), (vx, ry), (vx, TY),
    ])
    ax.text(56.0, ry - 1.2, "$T$, $\\tau$  (thrust & torque)", ha="center",
            va="top", fontsize=8.0, style="italic", color="#333333")

    # ---------------- environment (dashed feeds) ----------------
    arrow(ax, edge_pt(geom["env"], "top", 0.15),
          edge_pt(geom["vehicle"], "bottom", 0.85), style="--")
    arrow(ax, edge_pt(geom["env"], "top", 0.55),
          edge_pt(geom["sensors"], "bottom", 0.3), style="--")
    ax.text(30.0, 19.5, "$g$, $\\rho$, wind", fontsize=8.0, style="italic",
            color="#333333", ha="center")

    # ---------------- boxes ----------------
    def draw(box, title, subs, edge):
        x, y, w, h = box
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
            fc="white", ec=edge, lw=2.0, zorder=3))
        ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=edge, zorder=4)
        ax.text(x + w / 2, y + (h - 3.4) / 2 - 0.6, "\n".join(subs),
                ha="center", va="center", fontsize=8.2, color="#444444",
                zorder=4)

    for key, x, title, subs, edge in BOXES:
        draw(geom[key], title, subs, edge)
    draw(ACT, "Actuators", ["gimbal (TVC)", "throttle", "RCS"], "#d62728")
    draw(ENV, "Environment", ["gravity", "atmosphere",
                              "wind · gusts"], "#17becf")

    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
