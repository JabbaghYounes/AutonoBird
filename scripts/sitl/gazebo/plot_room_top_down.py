#!/usr/bin/env python3
"""Render a top-down PNG blueprint of the cs_classroom Gazebo world."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

import build_room_world as r


def main():
    _, ax = plt.subplots(figsize=(8, 11))

    ax.add_patch(patches.Rectangle(
        (r.ROOM_X_MIN, r.ROOM_Y_MIN),
        r.ROOM_EW, r.ROOM_NS,
        facecolor="#f5f0e0", edgecolor="black", linewidth=2, zorder=1,
    ))

    ax.add_patch(patches.Rectangle(
        (r.ROOM_X_MAX - 0.08, r.ROOM_Y_MIN + r.ROOM_NS * 0.15),
        0.08, r.ROOM_NS * 0.7,
        facecolor="#a8c9f5", edgecolor="#5080c0", alpha=0.6, zorder=2,
    ))
    ax.add_patch(patches.Rectangle(
        (r.ROOM_X_MIN + r.ROOM_EW * 0.15, r.ROOM_Y_MIN),
        r.ROOM_EW * 0.7, 0.08,
        facecolor="#a8c9f5", edgecolor="#5080c0", alpha=0.6, zorder=2,
    ))

    wb_y = r.ROOM_Y_MAX - 0.05
    for label, x_centre in [("WB1", r.WB1_X), ("WB2", r.WB2_X)]:
        ax.add_patch(patches.Rectangle(
            (x_centre - r.WB_W / 2, wb_y - 0.2),
            r.WB_W, 0.2,
            facecolor="white", edgecolor="black", zorder=3,
        ))
        ax.text(x_centre, wb_y - 0.1, label, ha="center", va="center", fontsize=8)

    door_y0 = r.DOOR_Y - r.DOOR_W / 2
    ax.add_patch(patches.Rectangle(
        (r.ROOM_X_MIN - 0.1, door_y0),
        0.2, r.DOOR_W,
        facecolor="#cce6ff", edgecolor="black", zorder=3,
    ))
    ax.text(r.ROOM_X_MIN - 0.3, r.DOOR_Y, "DOOR", ha="right", va="center", fontsize=9)

    ax.add_patch(patches.Rectangle(
        (r.TEACHER_DESK_X - r.TEACHER_DESK_SX / 2,
         r.TEACHER_DESK_Y - r.TEACHER_DESK_SY / 2),
        r.TEACHER_DESK_SX, r.TEACHER_DESK_SY,
        facecolor="#a07050", edgecolor="black", zorder=3,
    ))
    ax.text(r.TEACHER_DESK_X, r.TEACHER_DESK_Y, "Teacher\ndesk",
            ha="center", va="center", fontsize=7, color="white")
    ax.add_patch(patches.Rectangle(
        (r.TEACHER_DESK_X - r.CHAIR_SX / 2,
         r.TEACHER_DESK_Y - r.TEACHER_DESK_SY / 2 - r.CHAIR_OFFSET - r.CHAIR_SY / 2),
        r.CHAIR_SX, r.CHAIR_SY,
        facecolor="#1a1a1a", edgecolor="none", zorder=4,
    ))

    ax.add_patch(patches.Rectangle(
        (r.SERVER_X - r.SERVER_SX / 2, r.SERVER_Y - r.SERVER_SY / 2),
        r.SERVER_SX, r.SERVER_SY,
        facecolor="#202020", edgecolor="black", zorder=3,
    ))
    ax.text(r.SERVER_X + 0.7, r.SERVER_Y, "Servers", ha="left", va="center", fontsize=8)

    ax.add_patch(patches.Rectangle(
        (r.STORAGE_X - r.STORAGE_SX / 2, r.STORAGE_Y - r.STORAGE_SY / 2),
        r.STORAGE_SX, r.STORAGE_SY,
        facecolor="#808080", edgecolor="black", zorder=3,
    ))
    ax.text(r.STORAGE_X, r.STORAGE_Y, "Storage", ha="center", va="center", fontsize=8)

    ax.add_patch(patches.Rectangle(
        (r.T12_X - r.T12_W / 2, r.T12_Y - r.T12_LEN_NS / 2),
        r.T12_W, r.T12_LEN_NS,
        facecolor="#8b6234", edgecolor="black", zorder=3,
    ))
    ax.add_patch(patches.Rectangle(
        (r.T12_X - r.CABLE_TRAY_W / 2, r.T12_Y - r.T12_LEN_NS / 2),
        r.CABLE_TRAY_W, r.T12_LEN_NS,
        facecolor="#3a3a3a", edgecolor="none", zorder=4,
    ))
    ax.text(r.T12_X, r.T12_Y + r.T12_LEN_NS / 2 + 0.25, "T1 / T2",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.add_patch(patches.Rectangle(
        (r.T3_X - r.T3_W / 2, r.T3_Y - r.T3_LEN_NS / 2),
        r.T3_W, r.T3_LEN_NS,
        facecolor="#8b6234", edgecolor="black", zorder=3,
    ))
    ax.text(r.T3_X, r.T3_Y + r.T3_LEN_NS / 2 + 0.25, "T3",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

    t12_west = r.T12_X - r.T12_W / 2
    t12_east = r.T12_X + r.T12_W / 2
    t3_west = r.T3_X - r.T3_W / 2
    t1_ws_x = t12_west + 0.25
    t2_ws_x = t12_east - 0.25
    t1_chair_x = t12_west - r.CHAIR_OFFSET
    t2_chair_x = t12_east + r.CHAIR_OFFSET
    t3_chair_x = t3_west - r.CHAIR_OFFSET

    ws_y_start = r.T12_Y - r.T12_LEN_NS / 2 + 0.5
    ws_y_step = (r.T12_LEN_NS - 1.0) / (r.WS_COUNT - 1)
    for ws_idx in range(r.WS_COUNT):
        ws_y = ws_y_start + ws_idx * ws_y_step
        for ws_x in (t1_ws_x, t2_ws_x, r.T3_X):
            ax.add_patch(patches.Rectangle(
                (ws_x - r.WS_SX / 2, ws_y - r.WS_SY / 2),
                r.WS_SX, r.WS_SY,
                facecolor="#1a1a1a", edgecolor="none", zorder=4,
            ))
        for chair_x in (t1_chair_x, t2_chair_x, t3_chair_x):
            ax.add_patch(patches.Rectangle(
                (chair_x - r.CHAIR_SX / 2, ws_y - r.CHAIR_SY / 2),
                r.CHAIR_SX, r.CHAIR_SY,
                facecolor="#404040", edgecolor="black", linewidth=0.5, zorder=4,
            ))

    ax.add_patch(patches.Circle(
        (r.OBSTACLE_X, r.OBSTACLE_Y), r.OBSTACLE_R,
        facecolor="#d83030", edgecolor="black", zorder=5,
    ))
    ax.text(r.OBSTACLE_X - 0.4, r.OBSTACLE_Y, "obstacle",
            ha="right", va="center", fontsize=8)

    ax.plot(r.DRONE_SPAWN_X, r.DRONE_SPAWN_Y, marker="^",
            markersize=18, markerfacecolor="#4080ff", markeredgecolor="black",
            zorder=6)
    ax.text(r.DRONE_SPAWN_X - 0.4, r.DRONE_SPAWN_Y, "drone\nstart (facing N)",
            ha="right", va="center", fontsize=8)

    ax.annotate("flight path",
                xy=(r.OBSTACLE_X, r.OBSTACLE_Y - r.OBSTACLE_R),
                xytext=(r.DRONE_SPAWN_X, r.DRONE_SPAWN_Y + 0.5),
                arrowprops=dict(arrowstyle="->", color="#4080ff", lw=2,
                                connectionstyle="arc3,rad=0"),
                fontsize=8, color="#4080ff",
                ha="center", va="bottom")

    ax.text(0, r.ROOM_Y_MAX + 0.5, "NORTH", ha="center", fontsize=12,
            fontweight="bold", color="#444")
    ax.text(0, r.ROOM_Y_MIN - 0.5, "SOUTH", ha="center", fontsize=12,
            fontweight="bold", color="#444")
    ax.text(r.ROOM_X_MIN - 0.7, 0, "WEST", ha="right", va="center",
            fontsize=12, fontweight="bold", color="#444", rotation=90)
    ax.text(r.ROOM_X_MAX + 0.5, 0, "EAST", ha="left", va="center",
            fontsize=12, fontweight="bold", color="#444", rotation=270)

    ax.set_xlim(r.ROOM_X_MIN - 1.5, r.ROOM_X_MAX + 1.5)
    ax.set_ylim(r.ROOM_Y_MIN - 1.5, r.ROOM_Y_MAX + 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (east)  [m]")
    ax.set_ylabel("Y (north)  [m]")
    ax.set_title(f"cs_classroom Gazebo world (top-down) — {r.ROOM_EW} x {r.ROOM_NS} x {r.CEILING} m")
    ax.grid(True, alpha=0.3, zorder=0)

    out_path = Path(__file__).parent / "worlds" / "cs_classroom_topdown.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
