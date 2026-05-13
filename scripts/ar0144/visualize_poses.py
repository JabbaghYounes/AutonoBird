#!/usr/bin/env python3
"""
=============================================================================
Generate pose-reference diagrams for guided_calibration.py.
=============================================================================

For each of the 40 calibration poses, produce a one-page diagram showing:
  - 3D view: camera viewing frustum + checkerboard plane in 3D space
  - Top-down view: where the board sits in front of the camera
  - Frame overlay: where the board should appear in the camera view
  - Camera-POV action text (translated from the script's board-POV wording)

Output: one multi-page PDF, one pose per page. Useful as both an operator
reference during calibration and as Appendix C of the dissertation.

Reads POSES from guided_calibration.py via AST so this script can run on
any machine with matplotlib + numpy — no cv2 / Hailo / camera required.

Usage:
    python3 visualize_poses.py [--out PATH]
=============================================================================
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =============================================================================
# Camera and board parameters
# =============================================================================

HFOV_DEG = 90.0                # AR0144 stereo wide-lens approx HFOV
ASPECT = 1280 / 720            # left-half rectified frame aspect
BOARD_W_M = 0.260              # 10 squares × 26 mm
BOARD_H_M = 0.182              # 7 squares × 26 mm

ZONE_DISTANCES_M = {
    "NEAR": 0.45,              # midpoint of 0.3-0.6 m
    "MID-NEAR": 0.90,          # midpoint of 0.6-1.2 m
    "MID": 1.85,               # midpoint of 1.2-2.5 m
    "FAR": 3.25,               # midpoint of 2.5-4.0 m
}


# =============================================================================
# Pose loading + interpretation
# =============================================================================

def load_poses_from_script(script_path: Path) -> list[dict]:
    """Extract the POSES list from guided_calibration.py via AST.

    Doesn't import the script (which would pull in cv2), just parses the
    source and literal-evals the POSES assignment.
    """
    src = script_path.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "POSES":
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"POSES list not found in {script_path}")


def zone_for(pose_name: str) -> str:
    return pose_name.split("]")[0].lstrip("[")


def zone_distance(pose_name: str) -> float:
    return ZONE_DISTANCES_M.get(zone_for(pose_name), 1.0)


def board_orientation(pose: dict) -> tuple[float, float, float]:
    """Return approximate (yaw, pitch, roll) in degrees from the pose info.

    Scans name + instruction + detail since some directional cues only appear
    in the instruction (e.g. "PITCH board ~30° upward" lives in the
    instruction; the name just says "Large Pitch ~30°").

    Conventions (right-handed, camera looks along +Z, world Y is up):
      - yaw +ve : board's RIGHT edge rotates AWAY from camera
      - pitch +ve: board's TOP edge AWAY (board tips away at top)
      - roll +ve : board rotates CCW when viewed from the camera
    """
    n = " ".join([pose.get("name", ""), pose.get("instruction", ""),
                   pose.get("detail", "")]).lower()
    yaw = pitch = roll = 0.0

    # Yaw
    if "large yaw" in n:
        yaw = -30 if "left" in n else +30
    elif "yaw left" in n:
        yaw = -25
    elif "yaw right" in n:
        yaw = +25

    # Pitch (large)
    if "large pitch" in n:
        pitch = +30 if "up" in n else -30
    elif "tilted upward" in n or "pitch up" in n:
        pitch = +20
    elif "tilted downward" in n or "pitch down" in n:
        pitch = -20

    # Slight tilts (overrides only if not already large)
    if "slight up" in n and pitch == 0:
        pitch = +10
    elif "slight down" in n and pitch == 0:
        pitch = -10

    # Roll
    if "counter-clockwise" in n:
        roll = -20
    elif "clockwise" in n:
        roll = +20

    # Diagonal corner-skews — combine
    if "tl to br" in n:
        pitch += +15
        yaw += -15
    elif "tr to bl" in n:
        pitch += +15
        yaw += +15

    # Perspective skew — light roll + small yaw, illustrative only
    if "perspective skew" in n:
        roll = roll or 15
        yaw = yaw or -10

    return yaw, pitch, roll


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Compose yaw (Y), pitch (X), roll (Z) into a 3x3 rotation matrix."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def board_position_from_region(region, distance_m, hfov_deg=HFOV_DEG, aspect=ASPECT):
    """Project the region's centre (frac coords) to a 3D point at Z=distance_m.

    Image origin is top-left (region y increases downward). World Y is flipped
    so positive Y is up — matches the matplotlib 3D plot convention used below.
    """
    rx, ry, rw, rh = region
    cx = rx + rw / 2
    cy = ry + rh / 2
    width_m = 2 * distance_m * np.tan(np.radians(hfov_deg / 2))
    height_m = width_m / aspect
    X = (cx - 0.5) * width_m
    Y = -(cy - 0.5) * height_m  # flip: image-down -> world-up
    Z = distance_m
    return np.array([X, Y, Z])


# =============================================================================
# 3D drawing
# =============================================================================

def draw_frustum(ax, far_m, hfov_deg=HFOV_DEG, aspect=ASPECT, color="gray"):
    """Draw the camera viewing frustum from origin out to far_m."""
    tan_h = np.tan(np.radians(hfov_deg / 2))
    tan_v = tan_h / aspect
    # Far rectangle (in camera-Z forward, X right, Y up after flip)
    w = far_m * tan_h
    h = far_m * tan_v
    corners = np.array([
        [-w, -h, far_m],
        [+w, -h, far_m],
        [+w, +h, far_m],
        [-w, +h, far_m],
    ])
    # Apex (origin) -> corners
    for c in corners:
        ax.plot([0, c[0]], [0, c[2]], [0, c[1]], color=color, lw=0.7, alpha=0.5)
    # Far rectangle edges
    loop = np.vstack([corners, corners[:1]])
    ax.plot(loop[:, 0], loop[:, 2], loop[:, 1], color=color, lw=0.7, alpha=0.5)


def draw_board_3d(ax, center, R, color="tab:blue"):
    """Draw the checkerboard plane in 3D given centre + rotation matrix."""
    hw = BOARD_W_M / 2
    hh = BOARD_H_M / 2
    corners_local = np.array([
        [-hw, -hh, 0],
        [+hw, -hh, 0],
        [+hw, +hh, 0],
        [-hw, +hh, 0],
    ])
    corners_world = (R @ corners_local.T).T + center
    verts = [list(zip(corners_world[:, 0], corners_world[:, 2], corners_world[:, 1]))]
    poly = Poly3DCollection(verts, facecolors=color, alpha=0.55, edgecolors="black")
    ax.add_collection3d(poly)
    # Mark TOP edge so the operator can orient the board correctly
    top_edge = corners_world[2:4]
    ax.plot(top_edge[:, 0], top_edge[:, 2], top_edge[:, 1],
            color="orange", lw=2.5)


def draw_3d_view(ax, pose):
    name = pose["name"]
    distance = zone_distance(name)
    pos = board_position_from_region(pose["region"], distance)
    yaw, pitch, roll = board_orientation(pose)
    R = rotation_matrix(yaw, pitch, roll)

    far_m = max(1.5, distance * 1.25)
    draw_frustum(ax, far_m=far_m)
    draw_board_3d(ax, pos, R)

    # Camera marker
    ax.scatter([0], [0], [0], color="red", s=40, zorder=10)
    ax.text(0, 0, 0.05, "camera", color="red", fontsize=8, zorder=10)

    extent = max(1.2, distance * 1.4)
    ax.set_xlim(-extent / 2, extent / 2)
    ax.set_ylim(0, extent)
    ax.set_zlim(-extent / 3, extent / 3)
    ax.set_xlabel("X — right (m)", fontsize=8)
    ax.set_ylabel("Z — depth (m)", fontsize=8)
    ax.set_zlabel("Y — up (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=20, azim=-65)
    ax.set_title(f"3D geometry — yaw {yaw:+.0f}°, pitch {pitch:+.0f}°, roll {roll:+.0f}°",
                 fontsize=9)


# =============================================================================
# 2D drawings
# =============================================================================

def draw_frame_overlay(ax, pose, frame_w=1280, frame_h=720):
    region = pose["region"]
    rx, ry, rw, rh = region

    ax.add_patch(Rectangle((0, 0), frame_w, frame_h,
                                edgecolor="black", facecolor="#f4f4f4", lw=1))
    x, y, w, h = rx * frame_w, ry * frame_h, rw * frame_w, rh * frame_h
    ax.add_patch(Rectangle((x, y), w, h,
                                edgecolor="green", facecolor="green",
                                alpha=0.18, lw=2))
    ax.plot([x + w / 2], [y + h / 2], "r+", markersize=14, mew=2)

    ax.set_xlim(0, frame_w)
    ax.set_ylim(frame_h, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("image x (px)", fontsize=8)
    ax.set_ylabel("image y (px)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("On-screen target zone (left camera)", fontsize=9)


def draw_topdown(ax, pose):
    name = pose["name"]
    distance = zone_distance(name)
    pos = board_position_from_region(pose["region"], distance)
    yaw, pitch, roll = board_orientation(pose)
    R = rotation_matrix(yaw, pitch, roll)

    # Camera
    ax.scatter([0], [0], color="red", s=60, zorder=5)
    ax.annotate("camera", (0, 0), xytext=(5, -15), textcoords="offset points",
                fontsize=8, color="red")

    # FOV cone (top-down: X vs Z)
    tan_h = np.tan(np.radians(HFOV_DEG / 2))
    far_z = max(1.5, distance * 1.4)
    ax.plot([0, far_z * tan_h], [0, far_z], color="gray", lw=0.7)
    ax.plot([0, -far_z * tan_h], [0, far_z], color="gray", lw=0.7)
    ax.plot([-far_z * tan_h, far_z * tan_h], [far_z, far_z],
            color="gray", lw=0.5, ls=":")

    # Board projection onto top-down plane — show as a line segment along
    # the board's local X axis (rotated by yaw)
    hw = BOARD_W_M / 2
    edge_local = np.array([[-hw, 0, 0], [+hw, 0, 0]])
    edge_world = (R @ edge_local.T).T + pos
    ax.plot(edge_world[:, 0], edge_world[:, 2], color="tab:blue", lw=4)
    ax.scatter([pos[0]], [pos[2]], color="tab:blue", s=30, zorder=4)

    ax.set_xlim(-far_z * tan_h * 1.15, far_z * tan_h * 1.15)
    ax.set_ylim(-0.1, far_z * 1.05)
    ax.set_xlabel("X (m, right)", fontsize=8)
    ax.set_ylabel("Z (m, depth)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.set_title("Top-down (camera looks +Z)", fontsize=9)
    ax.grid(True, alpha=0.3)


# =============================================================================
# Camera-POV instruction
# =============================================================================

def camera_pov_instruction(pose: dict) -> str:
    """Map the board-POV instruction to a camera-POV action list.

    Scans name + instruction + detail so cues that only appear in the
    instruction (e.g. "PITCH board ~30° upward") still trigger the right
    bullet.
    """
    name = " ".join([pose.get("name", ""), pose.get("instruction", ""),
                      pose.get("detail", "")]).lower()
    zone = zone_for(pose["name"])

    dist_str = {
        "NEAR": "0.3-0.6 m from board",
        "MID-NEAR": "0.6-1.2 m from board",
        "MID": "1.2-2.5 m from board",
        "FAR": "2.5-4 m from board",
    }.get(zone, "?")

    lines = [f"• Distance: {dist_str}"]

    # --- Pitch (board top edge away => camera tilts up at board) ---
    has_pitch_up = (
        "tilted upward" in name
        or "pitch up" in name
        or "upward pitch" in name
        or "tilt top of board away" in name
        or "tilt top" in name
        or "top of board away" in name
        or ("pitch board" in name and "upward" in name)
        or ("large pitch" in name and "upward" in name)
    )
    has_pitch_down = (
        "tilted downward" in name
        or "pitch down" in name
        or "downward pitch" in name
        or "tilt bottom of board away" in name
        or "tilt bottom" in name
        or "bottom of board away" in name
        or ("pitch board" in name and "downward" in name)
        or ("large pitch" in name and "downward" in name)
    )
    if has_pitch_up:
        lines.append("• Hold camera LOW, tilt camera UP toward the board.")
    if has_pitch_down:
        lines.append("• Hold camera HIGH, tilt camera DOWN toward the board.")

    # --- Centered, flat ---
    if ("centered, flat" in name or "center, flat" in name
            or "flat to camera" in name) and not (has_pitch_up or has_pitch_down):
        lines.append("• Aim camera straight at board (board centred in view).")

    # --- Yaw ---
    if "yaw left" in name or ("yaw" in name and "left" in name and "right" not in name):
        lines.append("• Step to YOUR LEFT, aim camera back at the board.")
    if "yaw right" in name or ("yaw" in name and "right" in name and "left" not in name):
        lines.append("• Step to YOUR RIGHT, aim camera back at the board.")

    # --- Roll ---
    if "counter-clockwise" in name:
        lines.append("• Roll camera counter-clockwise (around lens axis).")
    elif "clockwise" in name:
        lines.append("• Roll camera clockwise (around lens axis).")

    # --- Frame positioning ---
    if "top-left of frame" in name or "upper-left corner" in name:
        lines.append("• Aim DOWN-RIGHT of the board (board lands top-left).")
    if "top-right" in name or "upper-right corner" in name:
        lines.append("• Aim DOWN-LEFT of the board (board lands top-right).")
    if "bottom-right of frame" in name or "lower-right corner" in name:
        lines.append("• Aim UP-LEFT of the board (board lands bottom-right).")
    if "bottom-left" in name or "lower-left corner" in name:
        lines.append("• Aim UP-RIGHT of the board (board lands bottom-left).")
    if "far left edge" in name or "extreme left" in name:
        lines.append("• Aim HARD RIGHT — board barely fits on left of view.")
    if "far right edge" in name or "extreme right" in name:
        lines.append("• Aim HARD LEFT — board barely fits on right of view.")
    if "high in frame" in name or "upper third" in name or "upper half" in name:
        lines.append("• Aim DOWN so board sits in upper part of view.")
    if "low in frame" in name or "lower third" in name or "lower half" in name:
        lines.append("• Aim UP so board sits in lower part of view.")

    # --- Distance specials ---
    if "very close" in name:
        lines.append("• Walk to ~20-30 cm from wall, aim straight.")
    if "smallest detectable" in name:
        lines.append("• Step as far back as detection still works (~3.5-4 m).")
    if "small board" in name and "smallest" not in name:
        lines.append("• Step further back; aim straight.")

    # --- Combined / illustrative ---
    if "diagonal" in name:
        lines.append("• Combine: tilt camera so one CORNER of the board is")
        lines.append("  closest, the opposite corner furthest.")
    if "perspective skew" in name or "one corner closer" in name:
        lines.append("• Stand off-axis from the board and tilt camera so one")
        lines.append("  corner is obviously closer than the others.")
    if "yaw + roll" in name or "roll + yaw" in name or "yaw it slightly" in name:
        lines.append("• Combine a step sideways with a small camera roll.")

    return "\n".join(lines)


# =============================================================================
# Page rendering
# =============================================================================

def render_pose_page(pdf, pose, pose_idx):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"Pose {pose_idx + 1}/40 — {pose['name']}",
                 fontsize=14, fontweight="bold")

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    draw_3d_view(ax3d, pose)

    ax_frame = fig.add_subplot(2, 2, 2)
    draw_frame_overlay(ax_frame, pose)

    ax_top = fig.add_subplot(2, 2, 3)
    draw_topdown(ax_top, pose)

    ax_text = fig.add_subplot(2, 2, 4)
    ax_text.axis("off")
    ax_text.text(0, 0.97, "Camera-POV action", fontsize=10,
                 fontweight="bold", va="top")
    ax_text.text(0, 0.92, camera_pov_instruction(pose),
                 fontsize=9, family="DejaVu Sans", va="top", ha="left")
    ax_text.text(0, 0.32, "Script on-screen text (board-POV)",
                 fontsize=8, fontweight="bold", va="top")
    ax_text.text(0, 0.28, pose["instruction"], fontsize=8,
                 style="italic", va="top")
    ax_text.text(0, 0.18, pose["detail"], fontsize=7, color="gray", va="top")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_overview_page(pdf, poses):
    """Single page showing all 40 poses as top-down thumbnails."""
    fig, axes = plt.subplots(4, 10, figsize=(16, 7), constrained_layout=True)
    fig.suptitle("All 40 poses — top-down overview", fontsize=14, fontweight="bold")
    for i, pose in enumerate(poses):
        ax = axes.flat[i]
        # Mini top-down without labels
        name = pose["name"]
        distance = zone_distance(name)
        pos = board_position_from_region(pose["region"], distance)
        yaw, _, _ = board_orientation(pose)
        R = rotation_matrix(yaw, 0, 0)
        hw = BOARD_W_M / 2
        edge_local = np.array([[-hw, 0, 0], [+hw, 0, 0]])
        edge_world = (R @ edge_local.T).T + pos
        tan_h = np.tan(np.radians(HFOV_DEG / 2))
        far_z = 3.8
        ax.plot([0, far_z * tan_h], [0, far_z], color="lightgray", lw=0.5)
        ax.plot([0, -far_z * tan_h], [0, far_z], color="lightgray", lw=0.5)
        ax.scatter([0], [0], color="red", s=15)
        ax.plot(edge_world[:, 0], edge_world[:, 2], color="tab:blue", lw=2)
        ax.set_xlim(-3.8 * tan_h, 3.8 * tan_h)
        ax.set_ylim(-0.1, far_z)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        short = name.split("] ")[1] if "] " in name else name
        ax.set_title(f"{i+1}. {short}", fontsize=6)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    default_out = script_dir.parent.parent / "resources" / "pose_diagrams" / "poses.pdf"
    parser.add_argument("--out", default=str(default_out),
                        help=f"Output PDF path (default: {default_out})")
    parser.add_argument("--script", default=str(script_dir / "guided_calibration.py"),
                        help="Path to guided_calibration.py to read POSES from")
    parser.add_argument("--no-overview", action="store_true",
                        help="Skip the all-poses overview page")
    args = parser.parse_args()

    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"[ERROR] guided_calibration.py not found at: {script_path}")
        sys.exit(1)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    poses = load_poses_from_script(script_path)
    print(f"Loaded {len(poses)} poses from {script_path}")
    print(f"Generating diagrams -> {out_path}")

    with PdfPages(out_path) as pdf:
        if not args.no_overview:
            render_overview_page(pdf, poses)
        for i, pose in enumerate(poses):
            print(f"  Pose {i + 1}/{len(poses)}: {pose['name']}")
            render_pose_page(pdf, pose, i)

        d = pdf.infodict()
        d["Title"] = "AutonoBird Stereo Calibration Pose Reference"
        d["Subject"] = "Camera-POV geometry for the 40-pose calibration sequence"
        d["Author"] = "AutonoBird"

    print(f"\nDone. {len(poses)} per-pose pages + "
          f"{'1 overview' if not args.no_overview else '0 overview'} pages "
          f"written to {out_path}.")


if __name__ == "__main__":
    main()
