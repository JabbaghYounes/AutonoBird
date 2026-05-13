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

def camera_world_pose(pose):
    """Compute the camera's position + orientation in a board-fixed world frame.

    World frame conventions:
      X: right (when facing the wall)
      Y: up
      Z: away from wall, toward the operator (positive)
    Board is at world origin with its face normal pointing in +Z.

    Returns (cam_pos_world, R_cam_to_world) where R_cam_to_world's columns
    are the camera's local axes expressed in world coords. The camera's
    optical axis (its local +Z) points toward the board for centred poses.
    """
    distance = zone_distance(pose["name"])
    t_b = board_position_from_region(pose["region"], distance)  # board in cam frame
    R_b = rotation_matrix(*board_orientation(pose))             # board in cam frame

    # Invert: camera position in BOARD frame
    cam_pos_board = -R_b.T @ t_b
    R_cam_to_board = R_b.T

    # Flip Z so the camera ends up at positive world-Z (operator side).
    M = np.diag([1.0, 1.0, -1.0])
    cam_pos_world = M @ cam_pos_board
    R_cam_to_world = M @ R_cam_to_board
    return cam_pos_world, R_cam_to_world


def draw_camera_3d(ax, cam_pos, R_cw, frustum_len=0.4, color="red"):
    """Draw the camera as a small pyramid frustum at the given world pose."""
    tan_h = np.tan(np.radians(HFOV_DEG / 2))
    tan_v = tan_h / ASPECT
    w = frustum_len * tan_h
    h = frustum_len * tan_v
    # Far rectangle in camera-local coords (apex at origin, +Z is forward)
    corners_local = np.array([
        [-w, -h, frustum_len],
        [+w, -h, frustum_len],
        [+w, +h, frustum_len],
        [-w, +h, frustum_len],
    ])
    corners_world = (R_cw @ corners_local.T).T + cam_pos

    # Apex marker
    ax.scatter([cam_pos[0]], [cam_pos[2]], [cam_pos[1]],
               color=color, s=45, zorder=10)
    # Apex-to-corner lines
    for c in corners_world:
        ax.plot([cam_pos[0], c[0]],
                [cam_pos[2], c[2]],
                [cam_pos[1], c[1]],
                color=color, lw=1.0, alpha=0.8)
    # Far rectangle loop
    loop = np.vstack([corners_world, corners_world[:1]])
    ax.plot(loop[:, 0], loop[:, 2], loop[:, 1],
            color=color, lw=1.0, alpha=0.8)

    # "Up" tick (camera's local +Y axis) to show roll orientation
    up_local = np.array([0, frustum_len * 0.6, 0])
    up_world = R_cw @ up_local + cam_pos
    ax.plot([cam_pos[0], up_world[0]],
            [cam_pos[2], up_world[2]],
            [cam_pos[1], up_world[1]],
            color="orange", lw=2.0)


def draw_wall_and_board(ax, wall_extent=2.2):
    """Draw the wall plane at world Z=0 and the checkerboard mounted on it."""
    we = wall_extent
    # Wall — large translucent rectangle at Z=0 (X-Y plane)
    wall_corners = np.array([
        [-we, -we * 0.6, 0],
        [+we, -we * 0.6, 0],
        [+we, +we * 0.6, 0],
        [-we, +we * 0.6, 0],
    ])
    wall_verts = [list(zip(wall_corners[:, 0],
                            wall_corners[:, 2],
                            wall_corners[:, 1]))]
    wall_poly = Poly3DCollection(wall_verts,
                                  facecolors="#ececec",
                                  alpha=0.35,
                                  edgecolors="#888")
    ax.add_collection3d(wall_poly)

    # Board — sits at origin, face in +Z. Render at z=+0.005 to avoid
    # z-fighting with the wall.
    hw, hh = BOARD_W_M / 2, BOARD_H_M / 2
    board_corners = np.array([
        [-hw, -hh, 0.005],
        [+hw, -hh, 0.005],
        [+hw, +hh, 0.005],
        [-hw, +hh, 0.005],
    ])
    board_verts = [list(zip(board_corners[:, 0],
                             board_corners[:, 2],
                             board_corners[:, 1]))]
    board_poly = Poly3DCollection(board_verts,
                                   facecolors="white",
                                   alpha=0.95,
                                   edgecolors="black")
    ax.add_collection3d(board_poly)
    # Mark the TOP edge in orange so the operator can read orientation
    ax.plot([-hw, +hw], [0.005, 0.005], [+hh, +hh],
            color="orange", lw=2.5)


def draw_3d_view(ax, pose):
    yaw, pitch, roll = board_orientation(pose)
    cam_pos, R_cw = camera_world_pose(pose)
    distance = zone_distance(pose["name"])

    draw_wall_and_board(ax)
    draw_camera_3d(ax, cam_pos, R_cw)

    # Light line from camera apex to board centre to show aim
    ax.plot([cam_pos[0], 0],
            [cam_pos[2], 0],
            [cam_pos[1], 0],
            color="red", lw=0.5, ls=":", alpha=0.5)

    # Axis limits — span from wall (Z=0) to a bit past the camera
    extent_z = max(1.5, abs(cam_pos[2]) * 1.25, distance * 1.25)
    extent_xy = max(1.2, abs(cam_pos[0]) * 1.8, abs(cam_pos[1]) * 1.8, 1.0)
    ax.set_xlim(-extent_xy, extent_xy)
    ax.set_ylim(0, extent_z)
    ax.set_zlim(-extent_xy / 1.5, extent_xy / 1.5)
    ax.set_xlabel("X — right (m)", fontsize=8)
    ax.set_ylabel("Z — depth from wall (m)", fontsize=8)
    ax.set_zlabel("Y — up (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=15, azim=-70)
    ax.set_title(
        f"Board on wall (origin) — Camera at "
        f"({cam_pos[0]:+.2f}, {cam_pos[1]:+.2f}, {cam_pos[2]:+.2f}) m\n"
        f"yaw {yaw:+.0f}°, pitch {pitch:+.0f}°, roll {roll:+.0f}° "
        f"(board-in-camera frame)",
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
    """Floor-plan view (board-fixed): wall at top, operator/camera below.

    - Wall is the horizontal line at Y=0 (top of plot, since we invert Y).
    - Board sits on the wall at X=0.
    - Camera position is at (X, Z) in world; we plot it at (X, -Z) so positive
      depth from wall reads as "down" in the plot — feels like a floor plan
      with the operator standing at the bottom.
    - Camera's facing direction is drawn as an arrow from its position.
    """
    distance = zone_distance(pose["name"])
    cam_pos, R_cw = camera_world_pose(pose)

    far_depth = max(1.5, abs(cam_pos[2]) * 1.4, distance * 1.4)
    plot_w_half = max(1.5, abs(cam_pos[0]) * 2.0, 1.5)

    # Wall — horizontal line at the top of plot (depth=0)
    ax.axhline(y=0, color="dimgray", lw=2.0)
    ax.fill_between([-plot_w_half, plot_w_half], 0, -0.05,
                     facecolor="lightgray", alpha=0.5, edgecolor=None)
    # Board on wall — short bold segment at X=0
    bw = BOARD_W_M / 2
    ax.plot([-bw, +bw], [0, 0], color="tab:blue", lw=5)
    ax.text(0, 0.03, "board", fontsize=7, ha="center", color="tab:blue")

    # Camera position (plotted at depth = -cam_pos[2] so "down in plot = far from wall")
    cam_x = cam_pos[0]
    cam_depth = cam_pos[2]
    ax.scatter([cam_x], [cam_depth], color="red", s=70, zorder=5)
    ax.annotate("camera", (cam_x, cam_depth), xytext=(7, -2),
                textcoords="offset points", fontsize=8, color="red")

    # Camera's optical-axis direction in world (R_cw[:, 2]) — show as an arrow
    fx = R_cw[0, 2]  # X component of camera's forward axis in world
    fz = R_cw[2, 2]  # Z component
    arrow_len = 0.35
    ax.annotate("", xy=(cam_x + fx * arrow_len, cam_depth + fz * arrow_len),
                xytext=(cam_x, cam_depth),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    # Camera FOV cone — project the world frustum to the top-down plane
    tan_h = np.tan(np.radians(HFOV_DEG / 2))
    fov_len = max(distance * 1.1, 1.2)
    # Two corner rays in camera-local: (+/- tan_h, 0, 1) at unit depth
    for sign in (-1, +1):
        ray_local = np.array([sign * tan_h, 0, 1]) * fov_len
        ray_world = R_cw @ ray_local + cam_pos
        ax.plot([cam_x, ray_world[0]], [cam_depth, ray_world[2]],
                color="red", lw=0.5, alpha=0.4)

    # Axis setup — invert Y so wall (depth=0) is at the top
    ax.set_xlim(-plot_w_half, plot_w_half)
    ax.set_ylim(far_depth * 1.05, -0.15)  # inverted: large depth at bottom
    ax.set_xlabel("X — right (m)", fontsize=8)
    ax.set_ylabel("depth from wall (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.set_title("Floor plan — wall at top, board fixed, camera moves",
                  fontsize=9)
    ax.grid(True, alpha=0.3)
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

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_overview_page(pdf, poses):
    """Single page showing all 40 camera positions as top-down thumbnails.

    Wall is the line at top of each thumbnail; board sits at X=0 on it.
    The camera position for each pose is shown as a red dot with a short
    arrow indicating its aim direction.
    """
    fig, axes = plt.subplots(4, 10, figsize=(16, 7), constrained_layout=True)
    fig.suptitle("All 40 poses — board fixed on wall, camera positions",
                  fontsize=14, fontweight="bold")
    for i, pose in enumerate(poses):
        ax = axes.flat[i]
        cam_pos, R_cw = camera_world_pose(pose)
        # Wall at top
        ax.axhline(y=0, color="dimgray", lw=1.2)
        bw = BOARD_W_M / 2
        ax.plot([-bw, +bw], [0, 0], color="tab:blue", lw=2.5)
        # Camera
        ax.scatter([cam_pos[0]], [cam_pos[2]], color="red", s=20, zorder=5)
        # Arrow showing camera aim direction
        fx, fz = R_cw[0, 2], R_cw[2, 2]
        arrow_len = 0.25
        ax.annotate("", xy=(cam_pos[0] + fx * arrow_len,
                             cam_pos[2] + fz * arrow_len),
                    xytext=(cam_pos[0], cam_pos[2]),
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
        # Bounds — accommodate FAR poses (~3.5 m camera depth)
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(4.0, -0.2)  # inverted Y so wall is at top
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        short = pose["name"].split("] ")[1] if "] " in pose["name"] else pose["name"]
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
