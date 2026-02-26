import { useState, useEffect, useCallback } from "react";

// ==========================================================================
// ZONE 1: NEAR (0.3–0.6 m) — precision close-range depth
// ==========================================================================
const NEAR_COLOR = ["#FF6B35", "#FF5252", "#FF7043", "#F4511E", "#E64A19", "#FF8A65", "#FF7043", "#D84315", "#BF360C", "#FF3D00"];
// ==========================================================================
// ZONE 2: MID-NEAR (0.6–1.2 m) — moderate-speed obstacle avoidance
// ==========================================================================
const MID_NEAR_COLOR = ["#F7C948", "#FFB300", "#FDD835", "#FFD54F", "#FFCA28", "#FFC107", "#FFB74D", "#FFA726", "#FF9800", "#FFD600"];
// ==========================================================================
// ZONE 3: MID (1.2–2.5 m) — critical navigation range
// ==========================================================================
const MID_COLOR = ["#4ECDC4", "#26C6DA", "#00BCD4", "#00ACC1", "#0097A7", "#00838F", "#4DD0E1", "#80DEEA", "#009688", "#26A69A"];
// ==========================================================================
// ZONE 4: FAR (2.5–4 m) — early obstacle detection
// ==========================================================================
const FAR_COLOR = ["#C44DFF", "#AB47BC", "#9C27B0", "#8E24AA", "#7B1FA2", "#CE93D8", "#BA68C8", "#AA00FF", "#E040FB", "#D500F9"];

const POSES = [
  // ZONE 1: NEAR (0.3–0.6 m)
  { name: "[NEAR] Centered, Flat", instruction: "Hold board in CENTER, flat to camera", detail: "About 30-60 cm away", icon: "⊕", region: { x: 10, y: 10, w: 80, h: 80 }, color: NEAR_COLOR[0] },
  { name: "[NEAR] Tilted Up ~20°", instruction: "Hold CLOSE, tilt TOP away from camera", detail: "About 30-60 cm — top edge further", icon: "⤒", region: { x: 15, y: 5, w: 70, h: 80 }, color: NEAR_COLOR[1] },
  { name: "[NEAR] Tilted Down ~20°", instruction: "Hold CLOSE, tilt BOTTOM away from camera", detail: "About 30-60 cm — bottom edge further", icon: "⤓", region: { x: 15, y: 15, w: 70, h: 80 }, color: NEAR_COLOR[2] },
  { name: "[NEAR] Yaw Left ~25°", instruction: "Hold CLOSE, rotate LEFT edge toward camera", detail: "About 30-60 cm — left side closer", icon: "⟲", region: { x: 10, y: 10, w: 75, h: 80 }, color: NEAR_COLOR[3] },
  { name: "[NEAR] Yaw Right ~25°", instruction: "Hold CLOSE, rotate RIGHT edge toward camera", detail: "About 30-60 cm — right side closer", icon: "⟳", region: { x: 15, y: 10, w: 75, h: 80 }, color: NEAR_COLOR[4] },
  { name: "[NEAR] Roll CW ~20°", instruction: "Hold CLOSE, ROTATE board clockwise ~20°", detail: "About 30-60 cm — tilt like a clock hand", icon: "⊕", region: { x: 10, y: 10, w: 80, h: 80 }, color: NEAR_COLOR[5] },
  { name: "[NEAR] Roll CCW ~20°", instruction: "Hold CLOSE, ROTATE board counter-clockwise ~20°", detail: "About 30-60 cm — opposite rotation", icon: "⊕", region: { x: 10, y: 10, w: 80, h: 80 }, color: NEAR_COLOR[6] },
  { name: "[NEAR] Top-Left", instruction: "Hold CLOSE, position board in TOP-LEFT", detail: "About 30-60 cm — upper-left area", icon: "↖", region: { x: 0, y: 0, w: 55, h: 55 }, color: NEAR_COLOR[7] },
  { name: "[NEAR] Bottom-Right", instruction: "Hold CLOSE, position board in BOTTOM-RIGHT", detail: "About 30-60 cm — lower-right area", icon: "↘", region: { x: 45, y: 45, w: 55, h: 55 }, color: NEAR_COLOR[8] },
  { name: "[NEAR] Very Close", instruction: "Move board VERY CLOSE to fill ~80% of frame", detail: "About 20-30 cm — as close as possible", icon: "↑", region: { x: 5, y: 5, w: 90, h: 90 }, color: NEAR_COLOR[9] },

  // ZONE 2: MID-NEAR (0.6–1.2 m)
  { name: "[MID-NEAR] Centered, Flat", instruction: "Hold board in CENTER, flat to camera", detail: "About 60-120 cm — roughly arm's length", icon: "⊕", region: { x: 20, y: 15, w: 60, h: 70 }, color: MID_NEAR_COLOR[0] },
  { name: "[MID-NEAR] Yaw Left + Up", instruction: "Yaw board LEFT and tilt TOP slightly away", detail: "About 60-120 cm — combined rotation", icon: "↖", region: { x: 10, y: 5, w: 65, h: 70 }, color: MID_NEAR_COLOR[1] },
  { name: "[MID-NEAR] Yaw Right + Down", instruction: "Yaw board RIGHT and tilt BOTTOM slightly away", detail: "About 60-120 cm — combined rotation", icon: "↘", region: { x: 25, y: 25, w: 65, h: 70 }, color: MID_NEAR_COLOR[2] },
  { name: "[MID-NEAR] Roll + Yaw", instruction: "ROTATE board ~15° AND yaw it slightly right", detail: "About 60-120 cm — combine roll with yaw", icon: "⟳", region: { x: 15, y: 15, w: 70, h: 70 }, color: MID_NEAR_COLOR[3] },
  { name: "[MID-NEAR] High", instruction: "Position board in the UPPER half of frame", detail: "About 60-120 cm — board near top", icon: "⤒", region: { x: 20, y: 0, w: 60, h: 50 }, color: MID_NEAR_COLOR[4] },
  { name: "[MID-NEAR] Low", instruction: "Position board in the LOWER half of frame", detail: "About 60-120 cm — board near bottom", icon: "⤓", region: { x: 20, y: 50, w: 60, h: 50 }, color: MID_NEAR_COLOR[5] },
  { name: "[MID-NEAR] Far Left", instruction: "Position board at the FAR LEFT of frame", detail: "About 60-120 cm — board at left edge", icon: "←", region: { x: 0, y: 15, w: 40, h: 70 }, color: MID_NEAR_COLOR[6] },
  { name: "[MID-NEAR] Far Right", instruction: "Position board at the FAR RIGHT of frame", detail: "About 60-120 cm — board at right edge", icon: "→", region: { x: 60, y: 15, w: 40, h: 70 }, color: MID_NEAR_COLOR[7] },
  { name: "[MID-NEAR] Diag TL→BR", instruction: "Tilt board diagonally — TOP-LEFT corner closer", detail: "About 60-120 cm — diagonal perspective", icon: "↘", region: { x: 15, y: 15, w: 70, h: 70 }, color: MID_NEAR_COLOR[8] },
  { name: "[MID-NEAR] Diag TR→BL", instruction: "Tilt board diagonally — TOP-RIGHT corner closer", detail: "About 60-120 cm — opposite diagonal", icon: "↙", region: { x: 15, y: 15, w: 70, h: 70 }, color: MID_NEAR_COLOR[9] },

  // ZONE 3: MID (1.2–2.5 m)
  { name: "[MID] Center, Flat", instruction: "Hold board in CENTER, flat to camera", detail: "About 1.2-2.5 m away", icon: "⊕", region: { x: 25, y: 20, w: 50, h: 60 }, color: MID_COLOR[0] },
  { name: "[MID] Large Yaw ~30°", instruction: "Hold at MID distance, YAW board ~30° left", detail: "About 1.2-2.5 m — strong left rotation", icon: "⟲", region: { x: 20, y: 15, w: 60, h: 70 }, color: MID_COLOR[1] },
  { name: "[MID] Large Pitch ~30°", instruction: "Hold at MID distance, PITCH board ~30° upward", detail: "About 1.2-2.5 m — strong upward tilt", icon: "⤒", region: { x: 20, y: 10, w: 60, h: 70 }, color: MID_COLOR[2] },
  { name: "[MID] Yaw + Roll", instruction: "Hold at MID distance, YAW right + ROLL slightly", detail: "About 1.2-2.5 m — combined rotation", icon: "⟳", region: { x: 20, y: 15, w: 60, h: 70 }, color: MID_COLOR[3] },
  { name: "[MID] Upper Third", instruction: "Position board in the UPPER THIRD of frame", detail: "About 1.2-2.5 m — board near top", icon: "⤒", region: { x: 25, y: 0, w: 50, h: 40 }, color: MID_COLOR[4] },
  { name: "[MID] Lower Third", instruction: "Position board in the LOWER THIRD of frame", detail: "About 1.2-2.5 m — board near bottom", icon: "⤓", region: { x: 25, y: 60, w: 50, h: 40 }, color: MID_COLOR[5] },
  { name: "[MID] Extreme Left", instruction: "Position board at EXTREME LEFT edge", detail: "About 1.2-2.5 m — board at far left", icon: "←", region: { x: 0, y: 20, w: 35, h: 60 }, color: MID_COLOR[6] },
  { name: "[MID] Extreme Right", instruction: "Position board at EXTREME RIGHT edge", detail: "About 1.2-2.5 m — board at far right", icon: "→", region: { x: 65, y: 20, w: 35, h: 60 }, color: MID_COLOR[7] },
  { name: "[MID] Perspective Skew", instruction: "Hold at MID distance, one CORNER closer", detail: "About 1.2-2.5 m — perspective distortion", icon: "⊕", region: { x: 20, y: 15, w: 60, h: 70 }, color: MID_COLOR[8] },
  { name: "[MID] Small Board", instruction: "Move FURTHER back so board is SMALL in frame", detail: "Near 2-2.5 m — board covers ~30% of zone", icon: "↓", region: { x: 30, y: 25, w: 40, h: 50 }, color: MID_COLOR[9] },

  // ZONE 4: FAR (2.5–4 m)
  { name: "[FAR] Center, Flat", instruction: "Hold board in CENTER, flat to camera", detail: "About 2.5-4 m — board will look small", icon: "⊕", region: { x: 30, y: 25, w: 40, h: 50 }, color: FAR_COLOR[0] },
  { name: "[FAR] Yaw Left", instruction: "Hold FAR, YAW board to the LEFT", detail: "About 2.5-4 m — rotate left edge toward camera", icon: "⟲", region: { x: 20, y: 20, w: 50, h: 60 }, color: FAR_COLOR[1] },
  { name: "[FAR] Yaw Right", instruction: "Hold FAR, YAW board to the RIGHT", detail: "About 2.5-4 m — rotate right edge toward camera", icon: "⟳", region: { x: 30, y: 20, w: 50, h: 60 }, color: FAR_COLOR[2] },
  { name: "[FAR] Pitch Up", instruction: "Hold FAR, TILT top away from camera", detail: "About 2.5-4 m — upward pitch", icon: "⤒", region: { x: 25, y: 10, w: 50, h: 50 }, color: FAR_COLOR[3] },
  { name: "[FAR] Pitch Down", instruction: "Hold FAR, TILT bottom away from camera", detail: "About 2.5-4 m — downward pitch", icon: "⤓", region: { x: 25, y: 40, w: 50, h: 50 }, color: FAR_COLOR[4] },
  { name: "[FAR] Upper-Left", instruction: "Hold FAR, position board in UPPER-LEFT", detail: "About 2.5-4 m — small board in top-left", icon: "↖", region: { x: 2, y: 2, w: 40, h: 40 }, color: FAR_COLOR[5] },
  { name: "[FAR] Upper-Right", instruction: "Hold FAR, position board in UPPER-RIGHT", detail: "About 2.5-4 m — small board in top-right", icon: "↗", region: { x: 58, y: 2, w: 40, h: 40 }, color: FAR_COLOR[6] },
  { name: "[FAR] Lower-Left", instruction: "Hold FAR, position board in LOWER-LEFT", detail: "About 2.5-4 m — small board in bottom-left", icon: "↙", region: { x: 2, y: 58, w: 40, h: 40 }, color: FAR_COLOR[7] },
  { name: "[FAR] Lower-Right", instruction: "Hold FAR, position board in LOWER-RIGHT", detail: "About 2.5-4 m — small board in bottom-right", icon: "↘", region: { x: 58, y: 58, w: 40, h: 40 }, color: FAR_COLOR[8] },
  { name: "[FAR] Smallest Board", instruction: "Move as FAR as possible while board is detected", detail: "About 3.5-4 m — smallest visible board", icon: "↓", region: { x: 30, y: 25, w: 40, h: 50 }, color: FAR_COLOR[9] },
];

function ProgressRing({ progress, size = 80, strokeWidth = 5 }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - progress * circumference;
  const color = progress >= 1 ? "#00E676" : "#FFB300";

  return (
    <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
      <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="#1a1a2e" strokeWidth={strokeWidth} />
      <circle
        cx={size / 2} cy={size / 2} r={radius} fill="none"
        stroke={color} strokeWidth={strokeWidth + 1}
        strokeDasharray={circumference} strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: "stroke-dashoffset 0.3s ease, stroke 0.3s ease" }}
      />
      <text
        x={size / 2} y={size / 2}
        textAnchor="middle" dominantBaseline="central"
        fill="#fff" fontSize="14" fontWeight="700" fontFamily="'JetBrains Mono', monospace"
        style={{ transform: "rotate(90deg)", transformOrigin: "center" }}
      >
        {Math.round(progress * 100)}%
      </text>
    </svg>
  );
}

function CameraViewport({ pose, holdProgress, status }) {
  const region = pose.region;

  return (
    <div style={{
      position: "relative",
      width: "100%",
      aspectRatio: "4/3",
      background: "linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0d0d2b 100%)",
      borderRadius: "12px",
      overflow: "hidden",
      border: "2px solid #2a2a4a",
      boxShadow: "0 0 40px rgba(0,0,0,0.5), inset 0 0 60px rgba(0,0,0,0.3)",
    }}>
      {/* Scanline effect */}
      <div style={{
        position: "absolute", inset: 0, zIndex: 1, pointerEvents: "none", opacity: 0.03,
        background: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.1) 2px, rgba(255,255,255,0.1) 4px)",
      }} />

      {/* Simulated camera noise texture */}
      <div style={{
        position: "absolute", inset: 0, zIndex: 0, opacity: 0.15,
        background: `radial-gradient(ellipse at 40% 50%, rgba(80,80,120,0.4) 0%, transparent 60%),
                     radial-gradient(ellipse at 70% 30%, rgba(60,60,100,0.3) 0%, transparent 50%)`,
      }} />

      {/* Target region */}
      <div style={{
        position: "absolute",
        left: `${region.x}%`, top: `${region.y}%`,
        width: `${region.w}%`, height: `${region.h}%`,
        border: `2px ${status === "in-zone" ? "solid" : "dashed"} ${status === "in-zone" ? "#00E676" : pose.color}`,
        borderRadius: "8px",
        background: status === "in-zone"
          ? "rgba(0, 230, 118, 0.06)"
          : `${pose.color}08`,
        transition: "all 0.4s ease",
        zIndex: 2,
        boxShadow: status === "in-zone"
          ? "0 0 20px rgba(0,230,118,0.2), inset 0 0 20px rgba(0,230,118,0.05)"
          : `0 0 15px ${pose.color}15`,
      }}>
        {/* Corner brackets */}
        {[
          { top: -1, left: -1, borderTop: "3px solid", borderLeft: "3px solid" },
          { top: -1, right: -1, borderTop: "3px solid", borderRight: "3px solid" },
          { bottom: -1, left: -1, borderBottom: "3px solid", borderLeft: "3px solid" },
          { bottom: -1, right: -1, borderBottom: "3px solid", borderRight: "3px solid" },
        ].map((style, i) => (
          <div key={i} style={{
            position: "absolute", width: "16px", height: "16px",
            borderColor: status === "in-zone" ? "#00E676" : pose.color,
            ...style,
          }} />
        ))}

        {/* Direction icon */}
        {status !== "in-zone" && (
          <div style={{
            position: "absolute", top: "50%", left: "50%",
            transform: "translate(-50%, -50%)",
            fontSize: "36px", color: pose.color, opacity: 0.7,
            textShadow: `0 0 20px ${pose.color}40`,
            animation: "pulse 2s ease-in-out infinite",
          }}>
            {pose.icon}
          </div>
        )}
      </div>

      {/* Simulated checkerboard (when in zone) */}
      {status === "in-zone" && (
        <div style={{
          position: "absolute",
          left: `${region.x + region.w * 0.15}%`,
          top: `${region.y + region.h * 0.15}%`,
          width: `${region.w * 0.7}%`,
          height: `${region.h * 0.7}%`,
          zIndex: 3, opacity: 0.5,
          display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gridTemplateRows: "repeat(5, 1fr)",
          borderRadius: "2px", overflow: "hidden",
          boxShadow: "0 0 10px rgba(0,230,118,0.3)",
        }}>
          {Array.from({ length: 35 }).map((_, i) => (
            <div key={i} style={{
              background: (Math.floor(i / 7) + i % 7) % 2 === 0 ? "#ddd" : "#333",
            }} />
          ))}
        </div>
      )}

      {/* Hold steady bar */}
      {status === "in-zone" && (
        <div style={{
          position: "absolute", bottom: "12%", left: "50%", transform: "translateX(-50%)",
          width: "60%", zIndex: 10,
        }}>
          <div style={{
            fontSize: "11px", color: holdProgress >= 1 ? "#00E676" : "#FFB300",
            fontFamily: "'JetBrains Mono', monospace", fontWeight: 600,
            marginBottom: "4px", textAlign: "center", letterSpacing: "1px",
            textTransform: "uppercase",
          }}>
            {holdProgress >= 1 ? "✓ CAPTURED" : "Hold steady..."}
          </div>
          <div style={{
            height: "6px", background: "#1a1a3e", borderRadius: "3px",
            border: "1px solid #2a2a4a", overflow: "hidden",
          }}>
            <div style={{
              height: "100%", borderRadius: "3px",
              width: `${Math.min(holdProgress * 100, 100)}%`,
              background: holdProgress >= 1
                ? "linear-gradient(90deg, #00E676, #69F0AE)"
                : "linear-gradient(90deg, #FFB300, #FDD835)",
              transition: "width 0.15s linear",
              boxShadow: holdProgress >= 1
                ? "0 0 8px rgba(0,230,118,0.5)"
                : "0 0 8px rgba(255,179,0,0.4)",
            }} />
          </div>
        </div>
      )}

      {/* Status text */}
      <div style={{
        position: "absolute", bottom: "4%", left: "50%", transform: "translateX(-50%)",
        zIndex: 10, fontSize: "11px",
        color: status === "in-zone" ? "#00E676" : status === "detected" ? "#FFB300" : "#ff5252",
        fontFamily: "'JetBrains Mono', monospace", fontWeight: 500,
        background: "rgba(0,0,0,0.6)", padding: "4px 12px", borderRadius: "4px",
        letterSpacing: "0.5px", whiteSpace: "nowrap",
      }}>
        {status === "in-zone" ? "● BOARD IN POSITION" :
         status === "detected" ? "● DETECTED — Move to target zone" :
         "○ No board detected — adjust position"}
      </div>
    </div>
  );
}

function PoseTimeline({ poses, currentIndex, completedPoses }) {
  return (
    <div style={{
      display: "flex", gap: "3px", padding: "8px 0", flexWrap: "wrap", justifyContent: "center",
    }}>
      {poses.map((pose, i) => (
        <div key={i} style={{
          width: "28px", height: "28px", borderRadius: "6px",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "10px", fontWeight: 700,
          fontFamily: "'JetBrains Mono', monospace",
          background: completedPoses.has(i) ? "#00E67622" :
                      i === currentIndex ? `${pose.color}30` : "#1a1a2e",
          border: completedPoses.has(i) ? "1.5px solid #00E676" :
                  i === currentIndex ? `1.5px solid ${pose.color}` : "1.5px solid #2a2a4a",
          color: completedPoses.has(i) ? "#00E676" :
                 i === currentIndex ? pose.color : "#555",
          transition: "all 0.3s ease",
          cursor: "default",
        }}
          title={pose.name}
        >
          {completedPoses.has(i) ? "✓" : i + 1}
        </div>
      ))}
    </div>
  );
}

export default function GuidedCalibrationDemo() {
  const [currentPose, setCurrentPose] = useState(0);
  const [status, setStatus] = useState("none");
  const [holdProgress, setHoldProgress] = useState(0);
  const [completedPoses, setCompletedPoses] = useState(new Set());
  const [isRunning, setIsRunning] = useState(false);
  const [showComplete, setShowComplete] = useState(false);

  const pose = POSES[currentPose];

  const simulateCapture = useCallback(() => {
    if (!isRunning || currentPose >= POSES.length) return;

    // Simulate: none → detected → in-zone → hold → captured → next
    setStatus("none");
    setHoldProgress(0);

    const t1 = setTimeout(() => setStatus("detected"), 800);
    const t2 = setTimeout(() => setStatus("in-zone"), 2000);

    let holdStep = 0;
    const holdInterval = setInterval(() => {
      holdStep += 0.08;
      setHoldProgress(Math.min(holdStep, 1));
      if (holdStep >= 1) {
        clearInterval(holdInterval);
        setCompletedPoses(prev => {
          const next = new Set(prev);
          next.add(currentPose);
          return next;
        });
        setTimeout(() => {
          if (currentPose < POSES.length - 1) {
            setCurrentPose(prev => prev + 1);
          } else {
            setIsRunning(false);
            setShowComplete(true);
          }
        }, 600);
      }
    }, 100);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearInterval(holdInterval);
    };
  }, [currentPose, isRunning]);

  useEffect(() => {
    if (isRunning) {
      const cleanup = simulateCapture();
      return cleanup;
    }
  }, [currentPose, isRunning, simulateCapture]);

  const handleStart = () => {
    setCurrentPose(0);
    setCompletedPoses(new Set());
    setIsRunning(true);
    setShowComplete(false);
    setStatus("none");
    setHoldProgress(0);
  };

  const handleReset = () => {
    setIsRunning(false);
    setCurrentPose(0);
    setCompletedPoses(new Set());
    setShowComplete(false);
    setStatus("none");
    setHoldProgress(0);
  };

  const overallProgress = completedPoses.size / POSES.length;

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(160deg, #05050f 0%, #0a0a20 40%, #0f0f2a 100%)",
      color: "#e0e0e0",
      fontFamily: "'Segoe UI', -apple-system, sans-serif",
      padding: "20px",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
        @keyframes pulse { 0%, 100% { opacity: 0.5; transform: translate(-50%, -50%) scale(1); } 50% { opacity: 1; transform: translate(-50%, -50%) scale(1.15); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes glow { 0%, 100% { box-shadow: 0 0 20px rgba(0,230,118,0.2); } 50% { box-shadow: 0 0 40px rgba(0,230,118,0.4); } }
      `}</style>

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: "16px", maxWidth: "500px" }}>
        <div style={{
          fontSize: "10px", letterSpacing: "4px", textTransform: "uppercase",
          color: "#666", fontFamily: "'JetBrains Mono', monospace", marginBottom: "6px",
          fontWeight: 600,
        }}>
          Waveshare AR0144 Stereo
        </div>
        <h1 style={{
          fontSize: "28px", fontWeight: 800, margin: "0 0 4px 0",
          fontFamily: "'Outfit', sans-serif",
          background: "linear-gradient(135deg, #fff 0%, #aab 100%)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          letterSpacing: "-0.5px",
        }}>
          Guided Calibration
        </h1>
        <p style={{
          fontSize: "13px", color: "#667", margin: 0, fontWeight: 400,
          fontFamily: "'Outfit', sans-serif",
        }}>
          Face-scan style — follow the prompts, auto-captures when ready
        </p>
      </div>

      {/* Main content */}
      <div style={{ width: "100%", maxWidth: "480px" }}>

        {/* Progress ring + pose info */}
        <div style={{
          display: "flex", alignItems: "center", gap: "16px",
          marginBottom: "12px", padding: "12px 16px",
          background: "#0d0d22", borderRadius: "10px", border: "1px solid #1a1a3e",
        }}>
          <ProgressRing progress={overallProgress} size={64} strokeWidth={4} />
          <div style={{ flex: 1 }}>
            <div style={{
              fontSize: "15px", fontWeight: 700, color: pose.color,
              fontFamily: "'Outfit', sans-serif", marginBottom: "2px",
            }}>
              {isRunning ? `${currentPose + 1}/${POSES.length}: ${pose.name}` :
               showComplete ? "Calibration Complete!" : "Ready to Begin"}
            </div>
            <div style={{
              fontSize: "12px", color: "#888",
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              {isRunning ? pose.instruction : showComplete ?
               `${completedPoses.size} poses captured successfully` :
               "40 guided poses across 4 distance zones"}
            </div>
            {isRunning && (
              <div style={{
                fontSize: "11px", color: "#556", marginTop: "2px",
                fontFamily: "'JetBrains Mono', monospace",
              }}>
                {pose.detail}
              </div>
            )}
          </div>
        </div>

        {/* Camera viewport */}
        <CameraViewport pose={pose} holdProgress={holdProgress} status={status} />

        {/* Pose timeline */}
        <div style={{ margin: "12px 0" }}>
          <PoseTimeline poses={POSES} currentIndex={currentPose} completedPoses={completedPoses} />
        </div>

        {/* Controls */}
        <div style={{ display: "flex", gap: "8px", justifyContent: "center" }}>
          {!isRunning && !showComplete && (
            <button onClick={handleStart} style={{
              padding: "12px 32px", fontSize: "14px", fontWeight: 700,
              fontFamily: "'Outfit', sans-serif",
              background: "linear-gradient(135deg, #00E676, #00C853)",
              color: "#000", border: "none", borderRadius: "8px", cursor: "pointer",
              letterSpacing: "0.5px",
              boxShadow: "0 4px 20px rgba(0,230,118,0.3)",
              transition: "transform 0.15s ease, box-shadow 0.15s ease",
            }}
              onMouseEnter={e => { e.target.style.transform = "translateY(-1px)"; e.target.style.boxShadow = "0 6px 25px rgba(0,230,118,0.4)"; }}
              onMouseLeave={e => { e.target.style.transform = ""; e.target.style.boxShadow = "0 4px 20px rgba(0,230,118,0.3)"; }}
            >
              ▶ Start Guided Capture
            </button>
          )}
          {isRunning && (
            <button onClick={() => {
              if (currentPose < POSES.length - 1) {
                setCurrentPose(prev => prev + 1);
              }
            }} style={{
              padding: "10px 24px", fontSize: "13px", fontWeight: 600,
              fontFamily: "'JetBrains Mono', monospace",
              background: "transparent", color: "#888",
              border: "1px solid #2a2a4a", borderRadius: "8px", cursor: "pointer",
            }}>
              Skip Pose →
            </button>
          )}
          {(isRunning || showComplete) && (
            <button onClick={handleReset} style={{
              padding: "10px 24px", fontSize: "13px", fontWeight: 600,
              fontFamily: "'JetBrains Mono', monospace",
              background: "transparent", color: "#666",
              border: "1px solid #2a2a4a", borderRadius: "8px", cursor: "pointer",
            }}>
              Reset
            </button>
          )}
          {showComplete && (
            <button onClick={() => alert("In the real script, this runs: python guided_calibration.py calibrate")}
              style={{
                padding: "10px 24px", fontSize: "13px", fontWeight: 700,
                fontFamily: "'Outfit', sans-serif",
                background: "linear-gradient(135deg, #FFB300, #FF8F00)",
                color: "#000", border: "none", borderRadius: "8px", cursor: "pointer",
                boxShadow: "0 4px 15px rgba(255,179,0,0.3)",
              }}>
              Run Calibration →
            </button>
          )}
        </div>

        {/* Info box */}
        <div style={{
          marginTop: "20px", padding: "14px 16px",
          background: "#0a0a1e", borderRadius: "8px",
          border: "1px solid #1a1a3e",
          fontSize: "12px", color: "#556",
          fontFamily: "'JetBrains Mono', monospace",
          lineHeight: 1.6,
        }}>
          <div style={{ color: "#889", fontWeight: 600, marginBottom: "6px", fontSize: "11px", letterSpacing: "1px", textTransform: "uppercase" }}>
            How it works
          </div>
          <div>This is a <span style={{ color: "#aab" }}>simulation</span> of the guided calibration UI. The actual Python script (<span style={{ color: pose.color }}>guided_calibration.py</span>) does the same thing with your real camera:</div>
          <div style={{ marginTop: "8px" }}>
            <span style={{ color: "#00E676" }}>1.</span> Shows target zones on the live camera feed{"\n"}
            <span style={{ color: "#FFB300" }}>2.</span> Detects your checkerboard with OpenCV{"\n"}
            <span style={{ color: "#FF6B35" }}>3.</span> Auto-captures when board is in-zone for ~1s{"\n"}
            <span style={{ color: "#C44DFF" }}>4.</span> Walks through 40 poses across 4 distance zones
          </div>
        </div>
      </div>
    </div>
  );
}
