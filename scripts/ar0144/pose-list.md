Below is a structured 40-pose calibration plan designed specifically for:

52 mm baseline stereo

Drone obstacle avoidance

0.3–4 m operational range

Strong 3D parameter coverage

You can either move the board or the camera — just create these relative poses.

## Distance Zones (Critical for Drone Use)

We’ll divide into 4 distance bands:

Near: 0.3–0.6 m (precision tuning)

Mid-Near: 0.6–1.2 m

Mid: 1.2–2.5 m

Far: 2.5–4 m (important for early obstacle detection)

10 poses per zone = 40 total

## Zone 1 — NEAR RANGE (0.3-0.6 m) — 10 Poses

These tune close-range depth (landing, tight spaces).

Centered, flat to camera

Tilted upward ~20° (top closer)

Tilted downward ~20°

Rotated left (yaw ~25°)

Rotated right (yaw ~25°)

Roll clockwise ~20°

Roll counterclockwise ~20°

Top-left of image frame

Bottom-right of image frame

Very close, filling ~80% of frame

## Zone 2 — MID-NEAR (0.6-1.2 m) — 10 Poses

Important for obstacle avoidance at moderate speed.

Centered flat

Yaw left + slight upward tilt

Yaw right + slight downward tilt

Roll + yaw combination

Board high in frame

Board low in frame

Board far left edge

Board far right edge

Diagonal (top-left to bottom-right tilt)

Opposite diagonal

## Zone 3 — MID RANGE (1.2-2.5 m) — 10 Poses

Critical for navigation.

Center flat

Large yaw (~30°)

Large pitch (~30°)

Yaw + roll combo

Board at upper third of frame

Lower third

Extreme left edge

Extreme right edge

Slight perspective skew (one corner closer)

Board small in frame (~30% coverage)

## Zone 4 — FAR RANGE (2.5-4 m) — 10 Poses

Important for long-range disparity calibration.

Center flat

Yaw left

Yaw right

Pitch up

Pitch down

Upper-left corner of image

Upper-right

Lower-left

Lower-right

Smallest visible but still detectable board
