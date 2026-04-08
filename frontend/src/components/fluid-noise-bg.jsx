"use client";

import { useEffect, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import "./fluid-noise-bg.css";

const vertexShader = `
precision highp float;
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fragmentShader = `
precision highp float;
uniform vec2 uResolution;
uniform float uTime;
uniform float uSpeed;
uniform float uFreq;
uniform float uAmp;
uniform vec3 uBase;
uniform vec3 uMid;
uniform vec3 uHigh;
uniform float uDark;
varying vec2 vUv;

vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
vec2 fade(vec2 t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

float cnoise(vec2 P) {
  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod289(Pi);
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;
  vec4 i = permute(permute(ix) + iy);

  vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0;
  vec4 gy = abs(gx) - 0.5;
  vec4 tx = floor(gx + 0.5);
  gx = gx - tx;

  vec2 g00 = vec2(gx.x, gy.x);
  vec2 g10 = vec2(gx.y, gy.y);
  vec2 g01 = vec2(gx.z, gy.z);
  vec2 g11 = vec2(gx.w, gy.w);

  vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 *= norm.x;
  g01 *= norm.y;
  g10 *= norm.z;
  g11 *= norm.w;

  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));

  vec2 nxy = mix(vec2(n00, n01), vec2(n10, n11), fade(Pf.xy).x);
  return 2.3 * mix(nxy.x, nxy.y, fade(Pf.xy).y);
}

float fbm(vec2 p) {
  float value = 0.0;
  float amp = 0.55;
  float freq = uFreq;
  for (int i = 0; i < 5; i++) {
    value += amp * cnoise(p * freq);
    freq *= 1.85;
    amp *= uAmp;
  }
  return value;
}

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

void main() {
  vec2 uv = vUv;
  vec2 p = uv - 0.5;
  p.x *= uResolution.x / max(uResolution.y, 1.0);

  float t = uTime * uSpeed;
  float n1 = fbm(p + vec2(t * 0.8, -t * 0.55));
  float n2 = fbm((p * 1.4) - vec2(t * 0.35, t * 0.7));
  float ridge = smoothstep(0.1, 0.8, abs(n1 + n2 * 0.6));

  vec3 col = mix(uBase, uMid, ridge);
  col = mix(col, uHigh, smoothstep(0.4, 1.0, n2 + 0.5));

  float quant = 6.0;
  float grain = (hash(gl_FragCoord.xy + uTime * 50.0) - 0.5) * 0.04;
  col = floor((col + grain) * quant) / quant;

  float vignette = smoothstep(1.05, 0.1, length(uv - 0.5));
  col *= mix(0.86, 1.07, vignette);

  float scan = sin((uv.y + t * 0.12) * 220.0) * 0.008;
  col += scan;

  // Keep light mode cleaner and dark mode deeper.
  if (uDark < 0.5) {
    col = mix(col, vec3(0.96, 0.98, 1.0), 0.08);
  }

  gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`;

function FluidPlane({ isDark }) {
  const meshRef = useRef(null);
  const { viewport, size, gl } = useThree();

  const uniformsRef = useRef({
    uResolution: new THREE.Uniform(new THREE.Vector2(size.width, size.height)),
    uTime: new THREE.Uniform(0),
    uSpeed: new THREE.Uniform(0.22),
    uFreq: new THREE.Uniform(1.7),
    uAmp: new THREE.Uniform(0.56),
    uBase: new THREE.Uniform(new THREE.Color(0.02, 0.08, 0.12)),
    uMid: new THREE.Uniform(new THREE.Color(0.05, 0.2, 0.26)),
    uHigh: new THREE.Uniform(new THREE.Color(0.72, 0.5, 0.2)),
    uDark: new THREE.Uniform(isDark ? 1 : 0),
  });

  useEffect(() => {
    const dpr = gl.getPixelRatio();
    uniformsRef.current.uResolution.value.set(size.width * dpr, size.height * dpr);
  }, [size, gl]);

  useEffect(() => {
    const u = uniformsRef.current;
    u.uDark.value = isDark ? 1 : 0;
    if (isDark) {
      u.uBase.value.setRGB(0.01, 0.07, 0.1);
      u.uMid.value.setRGB(0.04, 0.2, 0.26);
      u.uHigh.value.setRGB(0.76, 0.45, 0.16);
    } else {
      u.uBase.value.setRGB(0.88, 0.94, 0.98);
      u.uMid.value.setRGB(0.62, 0.8, 0.88);
      u.uHigh.value.setRGB(0.91, 0.67, 0.43);
    }
  }, [isDark]);

  useFrame((_, delta) => {
    uniformsRef.current.uTime.value += delta;
  });

  return (
    <mesh ref={meshRef} scale={[viewport.width, viewport.height, 1]}>
      <planeGeometry args={[1, 1]} />
      <shaderMaterial
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniformsRef.current}
      />
    </mesh>
  );
}

export default function FluidNoiseBg({ isDark }) {
  return (
    <Canvas className="fluid-noise-bg" camera={{ position: [0, 0, 4.5] }} dpr={1} gl={{ antialias: true }}>
      <FluidPlane isDark={isDark} />
    </Canvas>
  );
}
