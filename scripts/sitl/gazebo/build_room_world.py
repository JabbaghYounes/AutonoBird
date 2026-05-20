#!/usr/bin/env python3
"""Generate cs_classroom.sdf — Gazebo world that mirrors the AutonoBird demo room.

Run:
    python3 scripts/sitl/gazebo/build_room_world.py

Edits the file at scripts/sitl/gazebo/worlds/cs_classroom.sdf.

Room model (ENU world frame: +X=east, +Y=north, +Z=up; origin = room centre):
  - 10 m N-S x 7 m E-W x 3 m ceiling
  - Door on west wall near NW corner
  - Two whiteboards on north wall
  - Teacher's desk in NE corner with a chair beside it
  - Server rack along upper-west wall (the SERVERS column in the layout sketch)
  - Storage cabinet in SW corner
  - T1+T2 = one back-to-back double-sided bench in the room centre, with a
    cable management strip running along its north-south spine
  - T3 = a single-sided bench further east, separated from T2 by a walkway
  - 6 workstations per table side (18 monitor stand-ins total)
  - 18 chairs — T1 west / T2 east / T3 west, mirroring the layout sketch
  - Windows on the east and south walls (cosmetic light-blue panels)
  - One demo obstacle ("person" cylinder) in the western walkway

The iris_with_lidar drone spawns in the western walkway, facing north (+Y).
"""

from pathlib import Path

ROOM_NS = 10.0
ROOM_EW = 7.0
CEILING = 3.0
WALL_T = 0.10

ROOM_X_MIN = -ROOM_EW / 2
ROOM_X_MAX = +ROOM_EW / 2
ROOM_Y_MIN = -ROOM_NS / 2
ROOM_Y_MAX = +ROOM_NS / 2

DOOR_Y = ROOM_Y_MAX - 1.2
DOOR_W = 0.9

WB1_X = -2.0
WB2_X = +0.5
WB_W = 2.0
WB_H_LO = 1.0
WB_H_HI = 2.2

TEACHER_DESK_X = ROOM_X_MAX - 0.7
TEACHER_DESK_Y = ROOM_Y_MAX - 0.5
TEACHER_DESK_SX = 1.0
TEACHER_DESK_SY = 0.7
TEACHER_DESK_SZ = 0.75

SERVER_X = ROOM_X_MIN + 0.35
SERVER_Y = +1.5
SERVER_SX = 0.6
SERVER_SY = 3.5
SERVER_SZ = 1.9

STORAGE_X = ROOM_X_MIN + 0.8
STORAGE_Y = ROOM_Y_MIN + 1.0
STORAGE_SX = 1.5
STORAGE_SY = 1.8
STORAGE_SZ = 1.5

T12_X = +0.5
T12_W = 1.4
T12_LEN_NS = 5.0
T12_Y = -0.5

T3_X = +3.0
T3_W = 0.7
T3_LEN_NS = 5.0
T3_Y = -0.5

TABLE_H = 0.75
CABLE_TRAY_W = 0.06
CABLE_TRAY_H = 0.4

WS_COUNT = 6
WS_SX = 0.4
WS_SY = 0.3
WS_SZ = 0.4

CHAIR_SX = 0.45
CHAIR_SY = 0.45
CHAIR_SZ = 0.45
CHAIR_OFFSET = 0.45

WINDOW_LO = 1.0
WINDOW_HI = 2.0
WINDOW_T = 0.04

OBSTACLE_X = -1.5
OBSTACLE_Y = -1.0
OBSTACLE_R = 0.25
OBSTACLE_H = 1.7

DRONE_SPAWN_X = -1.5
DRONE_SPAWN_Y = -3.0
DRONE_SPAWN_Z = 0.195
DRONE_SPAWN_YAW_DEG = 90


def color_material(r, g, b, a=1.0):
    return f"""
          <material>
            <ambient>{r} {g} {b} {a}</ambient>
            <diffuse>{r} {g} {b} {a}</diffuse>
            <specular>0.3 0.3 0.3 {a}</specular>
          </material>"""


def box_model(name, x, y, z, sx, sy, sz, color, static=True, with_collision=True):
    r, g, b = color[:3]
    alpha = color[3] if len(color) == 4 else 1.0
    collision_block = ""
    if with_collision:
        collision_block = f"""
        <collision name="collision">
          <geometry>
            <box><size>{sx} {sy} {sz}</size></box>
          </geometry>
        </collision>"""
    return f"""    <model name="{name}">
      <static>{'true' if static else 'false'}</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">{collision_block}
        <visual name="visual">
          <geometry>
            <box><size>{sx} {sy} {sz}</size></box>
          </geometry>{color_material(r, g, b, alpha)}
        </visual>
      </link>
    </model>"""


def cylinder_model(name, x, y, z, radius, length, color):
    r, g, b = color
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder><radius>{radius}</radius><length>{length}</length></cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder><radius>{radius}</radius><length>{length}</length></cylinder>
          </geometry>{color_material(r, g, b)}
        </visual>
      </link>
    </model>"""


def build_walls():
    floor_color = (0.55, 0.55, 0.55)
    ceiling_color = (0.92, 0.92, 0.92)
    wall_color = (0.92, 0.90, 0.85)

    floor = box_model(
        "floor",
        0, 0, -0.05,
        ROOM_EW + 2 * WALL_T, ROOM_NS + 2 * WALL_T, 0.1,
        floor_color,
    )
    ceiling = box_model(
        "ceiling",
        0, 0, CEILING + 0.05,
        ROOM_EW + 2 * WALL_T, ROOM_NS + 2 * WALL_T, 0.1,
        ceiling_color,
    )

    north_wall = box_model(
        "wall_north",
        0, ROOM_Y_MAX + WALL_T / 2, CEILING / 2,
        ROOM_EW + 2 * WALL_T, WALL_T, CEILING,
        wall_color,
    )
    south_wall = box_model(
        "wall_south",
        0, ROOM_Y_MIN - WALL_T / 2, CEILING / 2,
        ROOM_EW + 2 * WALL_T, WALL_T, CEILING,
        wall_color,
    )
    east_wall = box_model(
        "wall_east",
        ROOM_X_MAX + WALL_T / 2, 0, CEILING / 2,
        WALL_T, ROOM_NS, CEILING,
        wall_color,
    )

    west_lower_len = (DOOR_Y - DOOR_W / 2) - ROOM_Y_MIN
    west_lower_y = ROOM_Y_MIN + west_lower_len / 2
    west_upper_len = ROOM_Y_MAX - (DOOR_Y + DOOR_W / 2)
    west_upper_y = ROOM_Y_MAX - west_upper_len / 2

    west_wall_lower = box_model(
        "wall_west_lower",
        ROOM_X_MIN - WALL_T / 2, west_lower_y, CEILING / 2,
        WALL_T, west_lower_len, CEILING,
        wall_color,
    )
    west_wall_upper = box_model(
        "wall_west_upper",
        ROOM_X_MIN - WALL_T / 2, west_upper_y, CEILING / 2,
        WALL_T, west_upper_len, CEILING,
        wall_color,
    )

    return [floor, ceiling, north_wall, south_wall, east_wall,
            west_wall_lower, west_wall_upper]


def build_whiteboards():
    wb_color = (1.0, 1.0, 1.0)
    z_mid = (WB_H_LO + WB_H_HI) / 2
    wb_h = WB_H_HI - WB_H_LO
    wb_y = ROOM_Y_MAX - WALL_T / 2 - 0.02
    wb1 = box_model(
        "whiteboard_1",
        WB1_X, wb_y, z_mid,
        WB_W, 0.02, wb_h,
        wb_color,
    )
    wb2 = box_model(
        "whiteboard_2",
        WB2_X, wb_y, z_mid,
        WB_W, 0.02, wb_h,
        wb_color,
    )
    return [wb1, wb2]


def build_windows():
    window_color = (0.45, 0.65, 0.95, 0.45)
    pane_z = (WINDOW_LO + WINDOW_HI) / 2
    pane_h = WINDOW_HI - WINDOW_LO

    east_window = box_model(
        "window_east",
        ROOM_X_MAX - WINDOW_T - 0.01, 0, pane_z,
        WINDOW_T, ROOM_NS * 0.7, pane_h,
        window_color, with_collision=False,
    )
    south_window = box_model(
        "window_south",
        0, ROOM_Y_MIN + WINDOW_T + 0.01, pane_z,
        ROOM_EW * 0.7, WINDOW_T, pane_h,
        window_color, with_collision=False,
    )
    return [east_window, south_window]


def build_furniture():
    desk_color = (0.40, 0.28, 0.18)
    server_color = (0.10, 0.10, 0.10)
    storage_color = (0.60, 0.60, 0.62)
    chair_color = (0.05, 0.05, 0.05)

    teacher_desk = box_model(
        "teacher_desk",
        TEACHER_DESK_X, TEACHER_DESK_Y, TEACHER_DESK_SZ / 2,
        TEACHER_DESK_SX, TEACHER_DESK_SY, TEACHER_DESK_SZ,
        desk_color,
    )
    teacher_chair = box_model(
        "teacher_chair",
        TEACHER_DESK_X, TEACHER_DESK_Y - TEACHER_DESK_SY / 2 - CHAIR_OFFSET,
        CHAIR_SZ / 2,
        CHAIR_SX, CHAIR_SY, CHAIR_SZ,
        chair_color,
    )
    server_rack = box_model(
        "server_rack",
        SERVER_X, SERVER_Y, SERVER_SZ / 2,
        SERVER_SX, SERVER_SY, SERVER_SZ,
        server_color,
    )
    storage = box_model(
        "storage_cabinet",
        STORAGE_X, STORAGE_Y, STORAGE_SZ / 2,
        STORAGE_SX, STORAGE_SY, STORAGE_SZ,
        storage_color,
    )
    return [teacher_desk, teacher_chair, server_rack, storage]


def workstation_y_positions(table_y, table_len):
    ws_y_start = table_y - table_len / 2 + 0.5
    ws_y_step = (table_len - 1.0) / (WS_COUNT - 1)
    return [ws_y_start + i * ws_y_step for i in range(WS_COUNT)]


def build_tables_workstations_chairs():
    table_color = (0.50, 0.35, 0.22)
    cable_color = (0.20, 0.20, 0.22)
    ws_color = (0.08, 0.08, 0.08)
    chair_color = (0.05, 0.05, 0.05)
    models = []

    t12_bench = box_model(
        "table_T1T2",
        T12_X, T12_Y, TABLE_H / 2,
        T12_W, T12_LEN_NS, TABLE_H,
        table_color,
    )
    models.append(t12_bench)

    cable_tray = box_model(
        "cable_tray_T1T2",
        T12_X, T12_Y, TABLE_H + CABLE_TRAY_H / 2,
        CABLE_TRAY_W, T12_LEN_NS, CABLE_TRAY_H,
        cable_color,
    )
    models.append(cable_tray)

    t12_west_edge = T12_X - T12_W / 2
    t12_east_edge = T12_X + T12_W / 2
    t1_ws_x = t12_west_edge + 0.25
    t2_ws_x = t12_east_edge - 0.25
    t1_chair_x = t12_west_edge - CHAIR_OFFSET
    t2_chair_x = t12_east_edge + CHAIR_OFFSET

    for ws_idx, ws_y in enumerate(workstation_y_positions(T12_Y, T12_LEN_NS), start=1):
        models.append(box_model(
            f"ws_T1_{ws_idx}",
            t1_ws_x, ws_y, TABLE_H + WS_SZ / 2,
            WS_SX, WS_SY, WS_SZ,
            ws_color,
        ))
        models.append(box_model(
            f"ws_T2_{ws_idx}",
            t2_ws_x, ws_y, TABLE_H + WS_SZ / 2,
            WS_SX, WS_SY, WS_SZ,
            ws_color,
        ))
        models.append(box_model(
            f"chair_T1_{ws_idx}",
            t1_chair_x, ws_y, CHAIR_SZ / 2,
            CHAIR_SX, CHAIR_SY, CHAIR_SZ,
            chair_color,
        ))
        models.append(box_model(
            f"chair_T2_{ws_idx}",
            t2_chair_x, ws_y, CHAIR_SZ / 2,
            CHAIR_SX, CHAIR_SY, CHAIR_SZ,
            chair_color,
        ))

    t3_bench = box_model(
        "table_T3",
        T3_X, T3_Y, TABLE_H / 2,
        T3_W, T3_LEN_NS, TABLE_H,
        table_color,
    )
    models.append(t3_bench)

    t3_west_edge = T3_X - T3_W / 2
    t3_ws_x = T3_X
    t3_chair_x = t3_west_edge - CHAIR_OFFSET

    for ws_idx, ws_y in enumerate(workstation_y_positions(T3_Y, T3_LEN_NS), start=1):
        models.append(box_model(
            f"ws_T3_{ws_idx}",
            t3_ws_x, ws_y, TABLE_H + WS_SZ / 2,
            WS_SX, WS_SY, WS_SZ,
            ws_color,
        ))
        models.append(box_model(
            f"chair_T3_{ws_idx}",
            t3_chair_x, ws_y, CHAIR_SZ / 2,
            CHAIR_SX, CHAIR_SY, CHAIR_SZ,
            chair_color,
        ))

    return models


def build_demo_obstacle():
    obs_color = (0.85, 0.20, 0.20)
    return cylinder_model(
        "demo_person",
        OBSTACLE_X, OBSTACLE_Y, OBSTACLE_H / 2,
        OBSTACLE_R, OBSTACLE_H,
        obs_color,
    )


def world_header_and_footer():
    header = """<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="cs_classroom">
    <physics name="1ms" type="ignore">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system"
      name="gz::sim::systems::Physics">
    </plugin>
    <plugin
      filename="gz-sim-sensors-system"
      name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-user-commands-system"
      name="gz::sim::systems::UserCommands">
    </plugin>
    <plugin filename="gz-sim-scene-broadcaster-system"
      name="gz::sim::systems::SceneBroadcaster">
    </plugin>
    <plugin filename="gz-sim-imu-system"
      name="gz::sim::systems::Imu">
    </plugin>
    <plugin filename="gz-sim-navsat-system"
      name="gz::sim::systems::NavSat">
    </plugin>

    <scene>
      <ambient>0.7 0.7 0.7</ambient>
      <background>0.6 0.6 0.65</background>
    </scene>

    <spherical_coordinates>
      <latitude_deg>-35.363262</latitude_deg>
      <longitude_deg>149.165237</longitude_deg>
      <elevation>584</elevation>
      <heading_deg>0</heading_deg>
      <surface_model>EARTH_WGS84</surface_model>
    </spherical_coordinates>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 8 0 0 0</pose>
      <diffuse>0.85 0.85 0.85 1</diffuse>
      <specular>0.4 0.4 0.4 1</specular>
      <attenuation>
        <range>30</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.3 -0.2 -0.9</direction>
    </light>

    <light type="point" name="room_fill">
      <pose>0 0 2.7 0 0 0</pose>
      <diffuse>0.5 0.5 0.5 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation>
        <range>15</range>
        <constant>0.5</constant>
        <linear>0.05</linear>
        <quadratic>0.01</quadratic>
      </attenuation>
    </light>
"""

    footer = f"""
    <include>
      <uri>model://iris_with_lidar</uri>
      <pose degrees="true">{DRONE_SPAWN_X} {DRONE_SPAWN_Y} {DRONE_SPAWN_Z} 0 0 {DRONE_SPAWN_YAW_DEG}</pose>
    </include>

  </world>
</sdf>
"""
    return header, footer


def main():
    header, footer = world_header_and_footer()
    parts = []
    parts.append(header)
    parts.append("\n    <!-- Room shell: floor, ceiling, walls (west wall split at door) -->")
    parts.extend(build_walls())
    parts.append("\n    <!-- Whiteboards on north wall -->")
    parts.extend(build_whiteboards())
    parts.append("\n    <!-- Cosmetic window panels (east + south walls), visual-only -->")
    parts.extend(build_windows())
    parts.append("\n    <!-- Furniture: teacher's desk + chair, server rack, storage cabinet -->")
    parts.extend(build_furniture())
    parts.append("\n    <!-- T1+T2 back-to-back bench (cable tray spine) + T3 separate bench -->")
    parts.append("\n    <!-- Plus 18 workstation boxes and 18 chairs -->")
    parts.extend(build_tables_workstations_chairs())
    parts.append("\n    <!-- Demo obstacle: person-stand-in cylinder for the planner sidestep -->")
    parts.append(build_demo_obstacle())
    parts.append(footer)

    out_path = Path(__file__).parent / "worlds" / "cs_classroom.sdf"
    out_path.write_text("\n".join(parts))
    print(f"Wrote {out_path}")
    print(f"  Room: {ROOM_EW} m E-W x {ROOM_NS} m N-S x {CEILING} m ceiling")
    print(f"  Drone spawn: ({DRONE_SPAWN_X}, {DRONE_SPAWN_Y}, {DRONE_SPAWN_Z}) yaw {DRONE_SPAWN_YAW_DEG} deg")
    print(f"  Obstacle: cylinder r={OBSTACLE_R}m, h={OBSTACLE_H}m at ({OBSTACLE_X}, {OBSTACLE_Y})")


if __name__ == "__main__":
    main()
