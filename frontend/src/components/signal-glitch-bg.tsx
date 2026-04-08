"use client";

import type { CSSProperties } from "react";
import "./signal-glitch-bg.css";

type SignalGlitchBgProps = {
  isDark: boolean;
};

const BARS = [
  { top: "8%", delay: "0.15s" },
  { top: "16%", delay: "0.65s" },
  { top: "25%", delay: "1.05s" },
  { top: "37%", delay: "0.4s" },
  { top: "49%", delay: "1.45s" },
  { top: "61%", delay: "0.95s" },
  { top: "73%", delay: "1.75s" },
  { top: "87%", delay: "0.3s" },
];

export function SignalGlitchBg({ isDark }: SignalGlitchBgProps) {
  return (
    <div className={`signal-glitch-bg ${isDark ? "dark" : "light"}`} aria-hidden>
      <div className="signal-glitch-bg__base" />
      <div className="signal-glitch-bg__grid" />
      <div className="signal-glitch-bg__sweep" />
      <div className="signal-glitch-bg__noise" />
      <div className="signal-glitch-bg__bars">
        {BARS.map((bar, index) => (
          <span key={index} style={{ "--top": bar.top, "--delay": bar.delay } as CSSProperties} />
        ))}
      </div>
    </div>
  );
}
