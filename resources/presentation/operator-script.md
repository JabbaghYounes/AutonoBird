# Live demo operator script

For the 21 May 2026 AutonoBird presentation. The live segment runs the airframe-mounted Pi in monitor mode, with motors physically locked out (props removed). Audience sees two cv2 windows + orchestrator terminal on the projector via VNC.

Target duration: **4–5 minutes**.

---

## Pre-event (day before)

1. **Charge the 4S LiPo** to 4.0 V/cell (or storage charge 3.85 V if standing by). Confirm balance.
2. **Bench-boot the Pi** with the UBEC: SSH from your laptop, confirm boot, confirm AI HAT+ enumerates (`lsusb | grep Hailo` should match).
3. **Verify perception venv health** — `source scripts/perception/venv/bin/activate && python -c "import hailo_platform, cv2, numpy; print('OK')"`.
4. **Verify VNC handshake** — open VNC viewer on your laptop, confirm the wayvnc session is reachable, confirm `$DISPLAY=:0` works for opening a cv2 window.
5. **Smoke-run** `./physical-drone-test.sh --show-depth` for ~30 s — confirm both cv2 windows render (YOLO + depth heatmap), `dmesg` is clean of under-voltage, orchestrator status line streams. Then `./physical-drone-test.sh stop`.
6. **Confirm props are removed and stored separately.** Verify with eyes-on before packing.

---

## Day of (T-30 min, in the demo room)

1. **Place the airframe** on the teacher's desk (or a stable, raised surface visible to audience). Centre it so the camera faces the area you'll walk in.
2. **Power up via the 4S LiPo** (connect the XT60 — no transmitter needed; ARMING_CHECK is the safety net).
3. **Connect your laptop** to the same WiFi network as the Pi (or use the SiK ground unit for telemetry visibility on screen).
4. **Open the VNC viewer** in fullscreen so the audience sees the same screen you do.
5. **Run a 30-second test of the demo command**:
   ```
   cd ~/Documents/AutonoBird/scripts/autonomy
   ./physical-drone-test.sh --show-depth
   ```
   Watch for: both cv2 windows render, orchestrator prints STATUS within 5 s, no `BridgeError` or `Hailo` exceptions. Then tear down: `./physical-drone-test.sh stop`.
6. **Leave the airframe on, prompt at the home directory** ready to re-run.

---

## During the talk — live segment

### Step 1 — Intro slide hand-off (~30 s)

After the "Live demonstration" cue slide, say:

> "What you'll see now is the actual airframe-mounted Pi running the same perception pipeline you've seen in the slides. The drone is powered on but the propellers are removed — this is a static demonstration of the perception loop, not a flight. The motors and ESCs are wired but cannot spin."

Switch the projector to the VNC viewer.

### Step 2 — Boot the orchestrator (~30 s)

Type into the SSH terminal (visible on screen):

```
./physical-drone-test.sh --show-depth
```

Two cv2 windows open within ~5 seconds. The orchestrator terminal starts streaming.

Say:

> "The orchestrator brings up the MAVLink bridge first, then the flight state machine, then the reactive planner in monitor mode — meaning the planner *would* command sidesteps but the autonomous-command path is gated off. On the left, the perception cv2 window shows YOLO detections with bounding boxes and depth labels. On the right, the colourised depth heatmap — red is close, blue is far, black is beyond the accurate calibration band."

### Step 3 — Walk in front, near distance (~1 min)

Stand approximately **1 m in front of the airframe**, facing it. Hold still for 3 seconds.

Point at the screen:

> "YOLO identifies me as 'person' with high confidence — you can see the label and the depth annotation, around 1 metre. On the heatmap, my silhouette is solid red — well inside the accurate band of 0.3 to 1.5 metres."

Move laterally (left and right) slowly. The bounding box tracks. Say:

> "The bounding box tracks the centroid — that's what the depth fusion samples. The median-of-five-by-five-patch lookup gives us a stable depth even when the SGBM is noisy at the silhouette edges."

### Step 4 — Step back, mid distance (~45 s)

Step back to **~2.5 m**.

> "At 2.5 metres I'm just outside the accurate calibration band — the depth value is still useful but starts to lose precision. Notice the heatmap silhouette getting warmer-to-cooler — that's the depth gradient. Detection confidence is still high; YOLO is robust to distance."

### Step 5 — Different object (~45 s)

Pick up a different object — backpack, chair, laptop, whatever's available.

> "YOLO is trained on 80 COCO classes. We measured 14 distinct classes appearing during our handheld walkaround test on the 18th of May — so anything in the COCO vocabulary you put in front of the camera, the system identifies it and gives you its depth in real time."

If multiple objects can be in frame at once (a person and a chair), do that — show simultaneous detections with different depths.

### Step 6 — Planner state, FSM transitions (~45 s)

Bring attention to the orchestrator terminal.

> "Down here in the orchestrator log, you can see the flight state machine — it's reporting DISCONNECTED right now because the FC is in its pre-flight arming-check state without an outdoor compass calibration. If the planner were in active mode and an obstacle entered the 1.5-metre threshold, you'd see CRUISING-to-AVOIDING transition logs scrolling here, and a velocity setpoint would be emitted. In monitor mode it's all observation — no commands actually flow to the autopilot."

### Step 7 — Shutdown (~30 s)

Stop the demo:

```
./physical-drone-test.sh stop
```

> "That closes the live demonstration. The full pipeline ran end-to-end on the airframe-mounted Pi, with the real Pixhawk in the loop, the AR0144 stereo camera feeding live frames, and the Hailo-8 NPU doing YOLO inference at the 138 ms loop time we measured in the dissertation."

Switch back to the slide deck for Q&A.

---

## Recovery procedures

If something fails on stage, fall back gracefully — don't debug live.

### Pi doesn't boot / no SSH

- Visible symptom: no green LED on the Pi, or LED is on but VNC session won't connect.
- **Fallback**: skip the live segment. Say "the airframe is having a boot issue, so I'll show the recorded walkaround instead." Play the recorded walkaround perception video (figure-6 source frames or the cv2 capture if you have one).

### cv2 windows don't render

- Visible symptom: terminal says "Qt platform plugin failed" or windows just don't appear.
- **Try once**: in the SSH session, `export DISPLAY=:0` then re-run.
- **If still failing**: skip live, fall back to recorded.

### No detections firing

- Visible symptom: cv2 window opens, video is live, but no bounding boxes ever appear.
- **Likely cause**: HEF missing or hailo_platform not loaded.
- **Try once**: `ls scripts/perception/models/model.hef` — confirm file is there. `source scripts/perception/venv/bin/activate && python -c "import hailo_platform"` — confirm import.
- **If still failing**: skip live, fall back to recorded.

### AR0144 not enumerated

- Visible symptom: `dmesg | tail` shows no USB camera, cv2 fails to open `/dev/video0`.
- **Try once**: re-plug the USB cable on the AR0144. Re-run.
- **If still failing**: skip live, fall back to recorded.

### Under-voltage warning

- Visible symptom: orchestrator says "FAIL: bridge error" or `dmesg` shows undervolt.
- **Likely cause**: LiPo voltage dropped under load.
- **Action**: stop immediately. Don't try to debug live. Fall back to recorded.

### Orchestrator says "BridgeError: heartbeat timeout"

- Visible symptom: orchestrator can't find the Pixhawk on `/dev/ttyACM0`.
- **Try once**: `ls /dev/ttyACM*` — confirm the device exists. If not, re-plug the FC USB cable.
- **If still failing**: skip live, fall back to recorded.

---

## After the demo

1. Power off the airframe (unplug the XT60).
2. Disconnect VNC, close SSH.
3. If anything went wrong during the demo, jot a note for the post-mortem.
