import * as THREE from "three";
import { HandLandmarker, FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

// ---------- Config ----------
const FOV_DEG = 60;                       // virtual camera vertical FOV (placement only; sizing is FOV-independent, see below)
const HAND_WIDTH_M = 0.075;               // avg index_mcp<->pinky_mcp physical width, meters (used only for depth/placement)
const WRIST_DIAMETER_TO_HAND_WIDTH = 0.68; // avg wrist diameter as a fraction of index<->pinky MCP width
const BAND_RADIUS_FACTOR = 1.18;          // band sits slightly proud of the wrist
const FOREARM_OFFSET_M = 0.02;            // how far up the forearm (from wrist) the band centers
const FACE_EYE_WIDTH_M = 0.09;            // avg outer-eye-corner to outer-eye-corner distance, meters

const LOCK_ANIM_DURATION_MS = 800;        // "tying on" animation length when a wrist locks on
const TILAK_UP_OFFSET_FACTOR = 0.5;       // how far above the between-eyebrows point the tilak sits, in eye-widths

const JAW_OPEN_THRESHOLD = 0.5;           // jawOpen blendshape score that triggers the sweet-feed
const JAW_CLOSE_THRESHOLD = 0.3;          // must drop back below this before it can re-trigger
const SWEET_FLY_DURATION_MS = 650;
const SWEET_COOLDOWN_MS = 500;

const MAX_RECORD_MS = 15000;              // safety cap so recordings can't grow unbounded on-device

// Path to a pre-recorded welcome/instruction voice message, played once the
// first time a rakhi appears. Drop the file at this path (or change the path
// below) to wire it up.
const WELCOME_AUDIO_SRC = 'audio/welcome-message.mp3';

// ---------- DOM ----------
const video = document.getElementById('video');
const glCanvas = document.getElementById('gl');
const flash = document.getElementById('flash');
const celebrateEl = document.getElementById('celebrate');

const welcomeScreen = document.getElementById('welcomeScreen');
const welcomeText = document.getElementById('welcomeText');
const startBackBtn = document.getElementById('startBackBtn');
const startSelfieBtn = document.getElementById('startSelfieBtn');

const instructionCard = document.getElementById('instructionCard');
const instructionText = document.getElementById('instructionText');

const controls = document.getElementById('controls');
const captureBtn = document.getElementById('captureBtn');
const recordBtn = document.getElementById('recordBtn');
const switchCameraBtn = document.getElementById('switchCameraBtn');
const replayBtn = document.getElementById('replayBtn');

let handLandmarker = null;
let faceLandmarker = null;
let faceModelLoading = false;
let running = false;
let lastVideoTime = -1;
let lastFrameTime = performance.now();
let rakhiScale = 1.3; // default = old baseline size + 2 taps of "Bigger" (2 * 0.15)
let selfieMode = false;
let currentStream = null;
let welcomeMessagePlayedOnce = false;

// Shows plain-language guidance in the top card instead of raw tracking status.
function setInstruction(text) {
  instructionText.textContent = text;
  instructionCard.classList.add('show');
}
// Routes a "getting ready" message to whichever screen the user can currently see.
function setLoadingMessage(text) {
  if (running) setInstruction(text);
  else welcomeText.textContent = text;
}

// ---------- Audio (synthesized, no external assets) ----------
let audioCtx = null;
function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') audioCtx.resume();
}
function tone(freq, startOffset, duration, gainPeak, type = 'sine') {
  const t0 = audioCtx.currentTime + startOffset;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, t0);
  gain.gain.setValueAtTime(0, t0);
  gain.gain.linearRampToValueAtTime(gainPeak, t0 + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, t0 + duration);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start(t0);
  osc.stop(t0 + duration + 0.05);
}
function playChime() {
  if (!audioCtx) return;
  // Small bell: a fundamental plus a couple of harmonics, staggered slightly.
  tone(1046.5, 0, 0.9, 0.12, 'sine');
  tone(1568.0, 0.02, 0.7, 0.07, 'sine');
  tone(2093.0, 0.04, 0.5, 0.04, 'sine');
}
function playPop() {
  if (!audioCtx) return;
  tone(523.25, 0, 0.18, 0.15, 'triangle');
  tone(783.99, 0.05, 0.15, 0.1, 'triangle');
}
function playCheer() {
  if (!audioCtx) return;
  // Quick ascending arpeggio plus a couple of high "sparkle" tones.
  tone(523.25, 0, 0.35, 0.13, 'triangle');
  tone(659.25, 0.08, 0.35, 0.13, 'triangle');
  tone(783.99, 0.16, 0.4, 0.13, 'triangle');
  tone(1046.5, 0.26, 0.55, 0.14, 'triangle');
  tone(1568.0, 0.3, 0.4, 0.06, 'sine');
  tone(2093.0, 0.42, 0.35, 0.05, 'sine');
}

// ---------- Welcome voice message ----------
let welcomeAudioEl = null;
function ensureWelcomeAudioEl() {
  if (!welcomeAudioEl) {
    welcomeAudioEl = new Audio(WELCOME_AUDIO_SRC);
    welcomeAudioEl.preload = 'auto';
  }
  return welcomeAudioEl;
}
// Unlocks the <audio> element for later programmatic playback. Must be called
// synchronously from within a real user-gesture handler (e.g. a button click).
function unlockWelcomeAudio() {
  const el = ensureWelcomeAudioEl();
  el.muted = true;
  el.play().then(() => { el.pause(); el.currentTime = 0; el.muted = false; }).catch(() => { el.muted = false; });
}
function playWelcomeMessage() {
  const el = ensureWelcomeAudioEl();
  el.currentTime = 0;
  el.play().catch((err) => console.warn('Welcome audio could not play (missing file or blocked):', err));
  replayBtn.style.display = 'flex';
}

// ---------- Three.js scene ----------
const renderer = new THREE.WebGLRenderer({ canvas: glCanvas, alpha: true, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(FOV_DEG, 1, 0.01, 10);
camera.position.set(0, 0, 0);
camera.lookAt(0, 0, -1);

scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.2));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(0.5, 1, 0.5);
scene.add(dirLight);

// One anchor per tracked hand, created lazily, reused across frames.
const anchors = [];

function makeOccluder() {
  const geo = new THREE.CylinderGeometry(1, 1, 0.14, 20, 1, true);
  const mat = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: true, transparent: true });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.renderOrder = -1; // irrelevant for depth test correctness, but keeps draw order tidy
  return mesh;
}

// ---------- Procedural rakhi ----------
// Palette inspired by a classic mauli-thread rakhi: red medallion face, gold
// petals, red gem rings, silver trim, maroon thread, ivory knot beads.
// Two gem palettes let a second tracked wrist ("sibling mode") read as a
// distinct-but-matching design rather than an identical clone.
const GEM_VARIANTS = [
  { gem: 0xd81e2c, gemDeep: 0x8f0f1a, disc: 0xd81e2c }, // classic ruby red
  { gem: 0x1f9e6e, gemDeep: 0x0e5a3d, disc: 0x1f9e6e }, // emerald green
];
const COLOR_GOLD = 0xf2c230;
const COLOR_GOLD_DEEP = 0xb8860b;
const COLOR_SILVER = 0xd8dce2;
const COLOR_THREAD = 0x7a1620;
const COLOR_IVORY = 0xf5efe0;

function ringLayout(count, radius, z, build) {
  const group = new THREE.Group();
  for (let i = 0; i < count; i++) {
    const angle = (i / count) * Math.PI * 2;
    const mesh = build(angle, i);
    mesh.position.set(Math.cos(angle) * radius, Math.sin(angle) * radius, z);
    group.add(mesh);
  }
  return group;
}

function makeGemRing(count, radius, gemRadius, z, color) {
  const geo = new THREE.SphereGeometry(gemRadius, 12, 10);
  const mat = new THREE.MeshStandardMaterial({
    color, roughness: 0.2, metalness: 0.15, emissive: color, emissiveIntensity: 0.12, transparent: true
  });
  return ringLayout(count, radius, z, () => new THREE.Mesh(geo, mat));
}

function makePetalRing(count, radius, length, width, z, color, colorDeep) {
  const geo = new THREE.SphereGeometry(1, 12, 10);
  geo.scale(width, length, width * 0.35);
  const mat = new THREE.MeshStandardMaterial({
    color, roughness: 0.25, metalness: 0.55, emissive: colorDeep, emissiveIntensity: 0.1, transparent: true
  });
  return ringLayout(count, radius, z, (angle) => {
    const petal = new THREE.Mesh(geo, mat);
    petal.rotation.z = angle - Math.PI / 2; // point long axis radially outward
    return petal;
  });
}

function makeCenterFlower(z) {
  const group = new THREE.Group();
  const petals = makePetalRing(8, 0.16, 0.15, 0.065, z + 0.01, COLOR_GOLD, COLOR_GOLD_DEEP);
  group.add(petals);
  const core = new THREE.Mesh(
    new THREE.SphereGeometry(0.06, 14, 12),
    new THREE.MeshStandardMaterial({ color: COLOR_GOLD, roughness: 0.15, metalness: 0.6, emissive: COLOR_GOLD_DEEP, emissiveIntensity: 0.25, transparent: true })
  );
  core.position.set(0, 0, z + 0.03);
  group.add(core);
  return group;
}

function makeMedallion(variant) {
  const group = new THREE.Group();

  // Silver trim ring (sits just behind the disc face)
  const trim = new THREE.Mesh(
    new THREE.TorusGeometry(0.56, 0.045, 10, 40),
    new THREE.MeshStandardMaterial({ color: COLOR_SILVER, roughness: 0.3, metalness: 0.85, transparent: true })
  );
  group.add(trim);

  // Disc face
  const disc = new THREE.Mesh(
    new THREE.CylinderGeometry(0.55, 0.55, 0.05, 40),
    new THREE.MeshStandardMaterial({ color: variant.disc, roughness: 0.4, metalness: 0.1, transparent: true })
  );
  disc.rotation.x = Math.PI / 2;
  group.add(disc);

  group.add(makeCenterFlower(0.03));
  group.add(makeGemRing(14, 0.34, 0.055, 0.05, variant.gemDeep));
  group.add(makePetalRing(16, 0.78, 0.28, 0.12, 0.03, COLOR_GOLD, COLOR_GOLD_DEEP));

  // Thin gold ring linking the outer petals, echoing the looped-thread look
  const petalLink = new THREE.Mesh(
    new THREE.TorusGeometry(0.78, 0.015, 8, 48),
    new THREE.MeshStandardMaterial({ color: COLOR_GOLD_DEEP, roughness: 0.35, metalness: 0.7, transparent: true })
  );
  petalLink.position.z = 0.01;
  group.add(petalLink);

  group.add(makeGemRing(20, 0.98, 0.05, 0.03, variant.gemDeep));

  return group;
}

function makeKnotBead(angleFromFront) {
  const geo = new THREE.CapsuleGeometry(0.09, 0.16, 4, 8);
  const mat = new THREE.MeshStandardMaterial({ color: COLOR_IVORY, roughness: 0.5, metalness: 0.05, transparent: true });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.z = Math.PI / 2;
  mesh.position.set(Math.sin(angleFromFront), 0, Math.cos(angleFromFront));
  return mesh;
}

function makeRakhi(variantIndex) {
  const variant = GEM_VARIANTS[variantIndex % GEM_VARIANTS.length];
  const group = new THREE.Group();

  // Thread band (torus, authored around local Y axis, encircling the wrist)
  const band = new THREE.Mesh(
    new THREE.TorusGeometry(1, 0.075, 12, 48),
    new THREE.MeshStandardMaterial({ color: COLOR_THREAD, roughness: 0.7, metalness: 0.05, transparent: true })
  );
  band.rotation.x = Math.PI / 2;
  group.add(band);

  // Ivory knot-bead accents to either side of the medallion, along the thread
  group.add(makeKnotBead(THREE.MathUtils.degToRad(65)));
  group.add(makeKnotBead(-THREE.MathUtils.degToRad(65)));

  // Medallion sits on the front face (+Z in anchor space)
  const medallion = makeMedallion(variant);
  medallion.position.set(0, 0, 1.0);
  group.add(medallion);

  return group;
}

function makeAnchor(variantIndex) {
  const anchor = new THREE.Group();
  const occluder = makeOccluder();
  const rakhi = makeRakhi(variantIndex);
  anchor.add(occluder);
  anchor.add(rakhi);
  const materials = [];
  rakhi.traverse((obj) => { if (obj.material) materials.push(obj.material); });
  anchor.userData = { occluder, rakhiRoot: rakhi, materials, lockTime: null, wasVisible: false };
  scene.add(anchor);
  return anchor;
}

function setAnchorGeometry(anchor, wristRadiusM, bandRadiusM, scaleFactor, animT) {
  const { occluder, rakhiRoot } = anchor.userData;
  // Occluder tracks the same user scale factor as the band, so the ratio between
  // "wrist" and "band" — and therefore how much of the band is hidden — stays
  // consistent as the rakhi is made bigger or smaller.
  const occluderRadiusM = wristRadiusM * scaleFactor;
  occluder.scale.set(occluderRadiusM, 1, occluderRadiusM);
  rakhiRoot.scale.setScalar(bandRadiusM * animT);
}

// Eased "tying on" animation: overshoot-and-settle scale/opacity, run once per lock-on.
function easeOutBack(t) {
  const c1 = 1.70158, c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

// ---------- Sparkle particle burst (celebration) ----------
const sparkleParticles = [];
function spawnSparkles(origin) {
  // Small and quick to disperse outward, so it reads as a glint of sparkle
  // rather than a cluster of bubbles lingering over the subject's face.
  const count = 16;
  for (let i = 0; i < count; i++) {
    const geo = new THREE.SphereGeometry(0.006 + Math.random() * 0.006, 6, 6);
    const mat = new THREE.MeshBasicMaterial({ color: Math.random() > 0.5 ? 0xffd54a : 0xffffff, transparent: true });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(origin);
    const dir = new THREE.Vector3((Math.random() - 0.5) * 2, Math.random() * 0.8 + 0.3, (Math.random() - 0.5)).normalize();
    const speed = 0.55 + Math.random() * 0.5;
    scene.add(mesh);
    sparkleParticles.push({ mesh, vel: dir.multiplyScalar(speed), born: performance.now(), life: 450 + Math.random() * 250 });
  }
}
function updateSparkles(dtSeconds) {
  const now = performance.now();
  for (let i = sparkleParticles.length - 1; i >= 0; i--) {
    const p = sparkleParticles[i];
    const age = now - p.born;
    if (age > p.life) {
      scene.remove(p.mesh);
      p.mesh.geometry.dispose();
      p.mesh.material.dispose();
      sparkleParticles.splice(i, 1);
      continue;
    }
    const t = age / p.life;
    p.mesh.position.addScaledVector(p.vel, dtSeconds);
    p.vel.y -= 0.8 * dtSeconds;
    p.mesh.material.opacity = 1 - t;
    p.mesh.scale.setScalar(1 - t * 0.5);
  }
}

function showCelebration() {
  celebrateEl.classList.add('show');
  clearTimeout(showCelebration._t);
  showCelebration._t = setTimeout(() => celebrateEl.classList.remove('show'), 2600);
}

// ---------- Face (selfie mode: tilak + sweet-feeding) ----------
const faceGroup = new THREE.Group();
scene.add(faceGroup);

function makeTilak() {
  const group = new THREE.Group();
  const geo = new THREE.SphereGeometry(1, 14, 12);
  geo.scale(0.055, 0.09, 0.03);
  const mark = new THREE.Mesh(
    geo,
    new THREE.MeshStandardMaterial({ color: 0xb8102a, roughness: 0.45, metalness: 0.1, emissive: 0x4a0510, emissiveIntensity: 0.2 })
  );
  mark.position.set(0, 0.01, 0.01);
  group.add(mark);

  // Traditional rice-grain accents beneath the mark
  for (const dx of [-0.045, 0, 0.045]) {
    const grainGeo = new THREE.SphereGeometry(1, 8, 6);
    grainGeo.scale(0.014, 0.03, 0.014);
    const grain = new THREE.Mesh(grainGeo, new THREE.MeshStandardMaterial({ color: 0xf7f1df, roughness: 0.6 }));
    grain.position.set(dx, -0.075, 0.01);
    group.add(grain);
  }
  return group;
}

function makeSweet() {
  const group = new THREE.Group();
  const core = new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 14),
    new THREE.MeshStandardMaterial({ color: 0xdb9a3c, roughness: 0.6, metalness: 0.05, emissive: 0x552f0c, emissiveIntensity: 0.1 })
  );
  group.add(core);
  // small surface nubs for a laddu-like texture
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    const nub = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, 8, 6),
      new THREE.MeshStandardMaterial({ color: 0xcf8a30, roughness: 0.7 })
    );
    nub.position.set(Math.cos(a) * 0.7, Math.sin(a) * 0.7, 0.55);
    group.add(nub);
  }
  return group;
}

const tilak = makeTilak();
faceGroup.add(tilak);

const sweet = makeSweet();
sweet.visible = false;
scene.add(sweet);

const faceState = { sweetPhase: 'idle', flightStart: 0, cooldownUntil: 0, sweetStartPos: new THREE.Vector3() };

function nearestHandWorldPos() {
  for (const a of anchors) {
    if (a.visible) return a.position;
  }
  return null;
}

function updateFace(landmarks, jawOpenScore) {
  const forehead = landmarks[9];
  const noseTip = landmarks[1];
  const eyeL = landmarks[33];
  const eyeR = landmarks[263];
  const lipTop = landmarks[13];
  const lipBottom = landmarks[14];

  const foreheadPx = landmarkToScreenPx(forehead);
  const noseTipPx = landmarkToScreenPx(noseTip);
  const eyeLPx = landmarkToScreenPx(eyeL);
  const eyeRPx = landmarkToScreenPx(eyeR);
  const lipTopPx = landmarkToScreenPx(lipTop);
  const lipBottomPx = landmarkToScreenPx(lipBottom);

  const eyeWidthPx = Math.hypot(eyeRPx[0] - eyeLPx[0], eyeRPx[1] - eyeLPx[1]);
  const depthM = estimateDepthM(eyeWidthPx, FACE_EYE_WIDTH_M);

  const foreheadW = screenToWorld(foreheadPx[0], foreheadPx[1], depthM);
  const noseTipW = screenToWorld(noseTipPx[0], noseTipPx[1], depthM);
  const eyeLW = screenToWorld(eyeLPx[0], eyeLPx[1], depthM);
  const eyeRW = screenToWorld(eyeRPx[0], eyeRPx[1], depthM);
  const lipTopW = screenToWorld(lipTopPx[0], lipTopPx[1], depthM);
  const lipBottomW = screenToWorld(lipBottomPx[0], lipBottomPx[1], depthM);
  const mouthW = lipTopW.clone().add(lipBottomW).multiplyScalar(0.5);

  // Orient the tilak flat against the forehead, upright with head tilt.
  const axisRight = eyeRW.clone().sub(eyeLW).normalize();
  let axisUp = foreheadW.clone().sub(noseTipW).normalize();
  const axisNormal = new THREE.Vector3().crossVectors(axisRight, axisUp).normalize();
  axisUp = new THREE.Vector3().crossVectors(axisNormal, axisRight).normalize();
  const basis = new THREE.Matrix4().makeBasis(axisRight, axisUp, axisNormal);
  faceGroup.quaternion.setFromRotationMatrix(basis);

  const eyeWidthWorld = eyeRW.distanceTo(eyeLW);
  faceGroup.position.copy(foreheadW)
    .add(axisNormal.clone().multiplyScalar(0.005))
    .add(axisUp.clone().multiplyScalar(eyeWidthWorld * TILAK_UP_OFFSET_FACTOR));
  tilak.scale.setScalar(eyeWidthWorld * 1.15);

  // ---------- Sweet-feeding gag ----------
  const now = performance.now();
  const st = faceState;

  if (st.sweetPhase === 'idle' && jawOpenScore > JAW_OPEN_THRESHOLD && now > st.cooldownUntil) {
    const handPos = nearestHandWorldPos();
    st.sweetStartPos.copy(handPos ? handPos : mouthW.clone().add(new THREE.Vector3(0.18, -0.28, 0.05)));
    st.flightStart = now;
    st.sweetPhase = 'flying';
    sweet.visible = true;
  }

  if (st.sweetPhase === 'flying') {
    const t = Math.min((now - st.flightStart) / SWEET_FLY_DURATION_MS, 1);
    const eased = t * t * (3 - 2 * t); // smoothstep
    const pos = st.sweetStartPos.clone().lerp(mouthW, eased);
    pos.y += Math.sin(t * Math.PI) * 0.06; // little arc
    sweet.position.copy(pos);
    const s = eyeWidthWorld * (0.55 - 0.3 * eased); // shrinks as it "enters" the mouth
    sweet.scale.setScalar(Math.max(s, 0.001));
    if (t >= 1) {
      sweet.visible = false;
      st.sweetPhase = 'cooldown';
      st.cooldownUntil = now + SWEET_COOLDOWN_MS;
      playPop();
      playCheer();
      spawnSparkles(mouthW);
      showCelebration();
    }
  }

  if (st.sweetPhase === 'cooldown' && jawOpenScore < JAW_CLOSE_THRESHOLD && now > st.cooldownUntil) {
    st.sweetPhase = 'idle';
  }
}

// ---------- Camera / MediaPipe ----------
function resizeRenderer() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resizeRenderer);

async function initModel() {
  setLoadingMessage('Getting things ready — this can take a few seconds the first time...');
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2
  });
  return vision;
}

async function initFaceModel(vision) {
  if (faceLandmarker || faceModelLoading) return;
  faceModelLoading = true;
  setLoadingMessage('Loading face tracking...');
  const filesetVision = vision || await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetVision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true
  });
  faceModelLoading = false;
}

async function startCamera(facingMode) {
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  });
  currentStream = stream;
  video.srcObject = stream;
  await new Promise((r) => { video.onloadedmetadata = r; });
  video.play();
  resizeRenderer();
}

// Map a normalized MediaPipe landmark to pixel coords on the (object-fit: cover) video element.
function landmarkToScreenPx(lm) {
  const vAspect = video.videoWidth / video.videoHeight;
  const cAspect = window.innerWidth / window.innerHeight;
  let sx, sy, scale;
  if (cAspect > vAspect) {
    scale = window.innerWidth / video.videoWidth;
    sx = 0;
    sy = (window.innerHeight - video.videoHeight * scale) / 2;
  } else {
    scale = window.innerHeight / video.videoHeight;
    sy = 0;
    sx = (window.innerWidth - video.videoWidth * scale) / 2;
  }
  return [lm.x * video.videoWidth * scale + sx, lm.y * video.videoHeight * scale + sy];
}

// Back-project a screen pixel to a 3D world point at a given depth (meters), assuming
// the camera sits at the origin looking down -Z with vertical FOV = FOV_DEG (matches `camera` above).
const fovRad = THREE.MathUtils.degToRad(FOV_DEG);
function screenToWorld(px, py, depthM) {
  const ndcX = (px / window.innerWidth) * 2 - 1;
  const ndcY = -((py / window.innerHeight) * 2 - 1);
  const halfH = depthM * Math.tan(fovRad / 2);
  const halfW = halfH * (window.innerWidth / window.innerHeight);
  return new THREE.Vector3(ndcX * halfW, ndcY * halfH, -depthM);
}

function estimateDepthM(widthPx, referenceWidthM) {
  const focalPx = (window.innerHeight / 2) / Math.tan(fovRad / 2);
  return (referenceWidthM * focalPx) / Math.max(widthPx, 1);
}

// Converts a length measured in screen pixels to world-space meters at a given depth,
// using the SAME projection as screenToWorld(). Because it shares that projection, the
// rendered object's on-screen size tracks the pixel measurement directly (self-consistent)
// even if our FOV/focal-length assumption doesn't exactly match the real camera lens —
// any calibration error cancels out between placement and sizing instead of shrinking/
// growing the rakhi relative to the real wrist.
function pxLengthToWorld(lengthPx, depthM) {
  const halfH = depthM * Math.tan(fovRad / 2);
  return (lengthPx / window.innerHeight) * (2 * halfH);
}

function updateHand(anchor, lm) {
  const wristPx = landmarkToScreenPx(lm[0]);
  const middleMcpPx = landmarkToScreenPx(lm[9]);
  const indexMcpPx = landmarkToScreenPx(lm[5]);
  const pinkyMcpPx = landmarkToScreenPx(lm[17]);
  const handWidthPx = Math.hypot(indexMcpPx[0] - pinkyMcpPx[0], indexMcpPx[1] - pinkyMcpPx[1]);

  const depthM = estimateDepthM(handWidthPx, HAND_WIDTH_M);
  const wristW = screenToWorld(wristPx[0], wristPx[1], depthM);
  const middleW = screenToWorld(middleMcpPx[0], middleMcpPx[1], depthM);
  const indexW = screenToWorld(indexMcpPx[0], indexMcpPx[1], depthM);
  const pinkyW = screenToWorld(pinkyMcpPx[0], pinkyMcpPx[1], depthM);

  // Forearm axis: points from fingers, through wrist, into the forearm.
  const forearmDir = wristW.clone().sub(middleW).normalize();

  // Palm normal (roll reference), Gram-Schmidt'd orthogonal to forearmDir.
  let palmNormal = new THREE.Vector3().crossVectors(
    indexW.clone().sub(wristW), pinkyW.clone().sub(wristW)
  ).normalize();
  palmNormal.sub(forearmDir.clone().multiplyScalar(palmNormal.dot(forearmDir))).normalize();
  if (palmNormal.lengthSq() < 1e-6) palmNormal.set(1, 0, 0); // degenerate fallback

  const axisX = new THREE.Vector3().crossVectors(forearmDir, palmNormal).normalize();
  const basis = new THREE.Matrix4().makeBasis(axisX, forearmDir, palmNormal);
  anchor.quaternion.setFromRotationMatrix(basis);

  anchor.position.copy(wristW).add(forearmDir.clone().multiplyScalar(FOREARM_OFFSET_M));

  // "Tying on" animation: trigger once whenever this anchor transitions from hidden to visible.
  const ud = anchor.userData;
  if (!ud.wasVisible) {
    ud.lockTime = performance.now();
    playChime();
    if (!welcomeMessagePlayedOnce) {
      welcomeMessagePlayedOnce = true;
      playWelcomeMessage();
    }
  }
  ud.wasVisible = true;

  const animRaw = ud.lockTime === null ? 1 : Math.min((performance.now() - ud.lockTime) / LOCK_ANIM_DURATION_MS, 1);
  const animT = ud.lockTime === null ? 1 : easeOutBack(animRaw);
  const opacity = ud.lockTime === null ? 1 : Math.min(animRaw * 1.6, 1);
  for (const m of ud.materials) m.opacity = opacity;

  const wristRadiusPx = (handWidthPx * WRIST_DIAMETER_TO_HAND_WIDTH) / 2;
  const wristRadiusM = pxLengthToWorld(wristRadiusPx, depthM);
  setAnchorGeometry(anchor, wristRadiusM, wristRadiusM * BAND_RADIUS_FACTOR * rakhiScale, rakhiScale, Math.max(animT, 0));
  anchor.visible = true;
}

function computeInstruction(handsCount, faceTracked) {
  if (selfieMode) {
    if (handsCount > 0 && faceTracked) return 'Perfect! Open your mouth wide for a sweet 🍬, or tap 📸 to save this moment.';
    if (faceTracked) return 'Looking good! Hold up your wrist to tie a Rakhi too, or open your mouth for a sweet 🍬.';
    if (handsCount > 0) return 'Rakhi tied! Now bring your face into view for your tilak 🙏.';
    return 'Bring your face into view, nice and centered 🙂';
  }
  if (handsCount > 0) return '✨ Your Rakhi is on! Tap 📸 to save a photo, or 🎥 to record a video.';
  return 'Hold your wrist up to the camera, like checking a watch ⌚';
}

function renderLoop() {
  if (!running) return;
  requestAnimationFrame(renderLoop);

  const now = performance.now();
  const dt = Math.min((now - lastFrameTime) / 1000, 0.05);
  lastFrameTime = now;
  updateSparkles(dt);

  if (video.currentTime !== lastVideoTime && handLandmarker) {
    lastVideoTime = video.currentTime;
    const result = handLandmarker.detectForVideo(video, now);
    const hands = result.landmarks || [];

    while (anchors.length < hands.length) {
      anchors.push(makeAnchor(anchors.length));
    }
    anchors.forEach((a, i) => {
      const visible = i < hands.length;
      if (!visible) a.userData.wasVisible = false;
      a.visible = visible;
    });

    let faceTracked = false;
    if (selfieMode && faceLandmarker) {
      const faceResult = faceLandmarker.detectForVideo(video, now);
      const faceLm = faceResult.faceLandmarks && faceResult.faceLandmarks[0];
      if (faceLm) {
        faceTracked = true;
        const blendshapes = faceResult.faceBlendshapes && faceResult.faceBlendshapes[0];
        let jawOpen = 0;
        if (blendshapes) {
          const cat = blendshapes.categories.find((c) => c.categoryName === 'jawOpen');
          if (cat) jawOpen = cat.score;
        }
        updateFace(faceLm, jawOpen);
      }
      faceGroup.visible = faceTracked;
      sweet.visible = sweet.visible && faceTracked;
    } else {
      faceGroup.visible = false;
      sweet.visible = false;
    }

    if (hands.length > 0) {
      hands.forEach((lm, i) => updateHand(anchors[i], lm));
    }

    setInstruction(computeInstruction(hands.length, faceTracked));
  }

  renderer.render(scene, camera);
  if (isRecording) compositeInto(recordCtx, recordCanvas.width, recordCanvas.height);
}

async function boot(facingMode) {
  startBackBtn.disabled = true;
  startSelfieBtn.disabled = true;
  try {
    const [vision] = await Promise.all([initModel(), startCamera(facingMode)]);
    if (selfieMode) await initFaceModel(vision);
    welcomeScreen.style.display = 'none';
    controls.style.display = 'flex';
    unlockWelcomeAudio();
    ensureAudio();
    running = true;
    lastFrameTime = performance.now();
    setInstruction(selfieMode ? 'Bring your face into view, nice and centered 🙂' : 'Hold your wrist up to the camera, like checking a watch ⌚');
    renderLoop();
  } catch (e) {
    welcomeText.textContent = 'Something went wrong: ' + e.message + ' — please try again.';
    startBackBtn.disabled = false;
    startSelfieBtn.disabled = false;
    console.error(e);
  }
}

startBackBtn.addEventListener('click', () => { selfieMode = false; boot('environment'); });
startSelfieBtn.addEventListener('click', () => {
  selfieMode = true;
  document.body.classList.add('mirrored');
  boot('user');
});

controls.addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (!btn) return;
  const action = btn.dataset.action;
  if (action === 'scale-') rakhiScale = Math.max(0.4, rakhiScale - 0.15);
  if (action === 'scale+') rakhiScale = Math.min(2.5, rakhiScale + 0.15);
});

switchCameraBtn.addEventListener('click', async () => {
  switchCameraBtn.disabled = true;
  selfieMode = !selfieMode;
  document.body.classList.toggle('mirrored', selfieMode);
  setInstruction('Switching camera...');
  try {
    await startCamera(selfieMode ? 'user' : 'environment');
    if (selfieMode) await initFaceModel();
    setInstruction(selfieMode ? 'Bring your face into view, nice and centered 🙂' : 'Hold your wrist up to the camera, like checking a watch ⌚');
  } catch (e) {
    setInstruction('Camera switch failed: ' + e.message);
  }
  switchCameraBtn.disabled = false;
});

replayBtn.addEventListener('click', () => playWelcomeMessage());

// ---------- Shared compositing (photo capture + video recording) ----------
// Draws the camera feed (replicating its on-screen object-fit:cover crop,
// since drawImage ignores CSS) plus the WebGL rakhi layer into a 2D context.
function compositeInto(ctx, width, height) {
  const vAspect = video.videoWidth / video.videoHeight;
  const cAspect = width / height;
  let sx, sy, scale;
  if (cAspect > vAspect) {
    scale = width / video.videoWidth;
    sx = 0;
    sy = (height - video.videoHeight * scale) / 2;
  } else {
    scale = height / video.videoHeight;
    sy = 0;
    sx = (width - video.videoWidth * scale) / 2;
  }

  if (selfieMode) {
    // Mirror the composite to match the on-screen (CSS-mirrored) selfie preview.
    ctx.save();
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, sx, sy, video.videoWidth * scale, video.videoHeight * scale);
  ctx.drawImage(glCanvas, 0, 0, width, height);
  if (selfieMode) ctx.restore();
}

function compositeFrame() {
  const out = document.createElement('canvas');
  out.width = window.innerWidth;
  out.height = window.innerHeight;
  compositeInto(out.getContext('2d'), out.width, out.height);
  return out;
}

// Shares or downloads a captured file, preferring Web Share (the reliable
// "Save to gallery"/"Save Image" path on iOS Safari, and works on Android
// too) and falling back to a direct download where file sharing isn't
// available (mainly desktop browsers).
async function saveFile(blob, filename, mimeType, savedMessage) {
  const file = new File([blob], filename, { type: mimeType });

  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: 'AR Rakhi' });
      setInstruction('Choose "Save" to add it to your gallery.');
    } catch (err) {
      if (err.name !== 'AbortError') setInstruction('Share cancelled or failed.');
    }
    return;
  }

  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  setInstruction(savedMessage);
}

async function captureAndSave() {
  captureBtn.disabled = true;
  const out = compositeFrame();

  flash.style.opacity = '1';
  setTimeout(() => { flash.style.opacity = '0'; }, 150);

  out.toBlob(async (blob) => {
    if (!blob) {
      setInstruction('Photo capture failed — please try again.');
      captureBtn.disabled = false;
      return;
    }
    await saveFile(blob, `rakhi-${Date.now()}.png`, 'image/png', 'Photo saved to Downloads.');
    captureBtn.disabled = false;
  }, 'image/png');
}

captureBtn.addEventListener('click', captureAndSave);

// ---------- Video recording ----------
const recordCanvas = document.createElement('canvas');
const recordCtx = recordCanvas.getContext('2d');
let isRecording = false;
let mediaRecorder = null;
let recordedChunks = [];
let recordStartTime = 0;
let recordTimerHandle = null;
let recordAutoStopHandle = null;

const RECORD_MIME_CANDIDATES = [
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
  'video/mp4'
];
function pickRecordMimeType() {
  if (typeof MediaRecorder === 'undefined') return null;
  return RECORD_MIME_CANDIDATES.find((t) => MediaRecorder.isTypeSupported(t)) || null;
}

function updateRecordButtonLabel() {
  const elapsed = Math.floor((performance.now() - recordStartTime) / 1000);
  const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const ss = String(elapsed % 60).padStart(2, '0');
  recordBtn.textContent = `⏹ Stop (${mm}:${ss})`;
}

function startRecording() {
  const mimeType = pickRecordMimeType();
  if (!mimeType) {
    setInstruction("Video recording isn't supported on this browser — you can still take photos.");
    return;
  }

  recordCanvas.width = window.innerWidth;
  recordCanvas.height = window.innerHeight;
  recordedChunks = [];

  const stream = recordCanvas.captureStream(30);
  mediaRecorder = new MediaRecorder(stream, { mimeType });
  mediaRecorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = async () => {
    clearInterval(recordTimerHandle);
    clearTimeout(recordAutoStopHandle);
    recordBtn.classList.remove('recording');
    recordBtn.textContent = '🎥 Record';
    recordBtn.disabled = false;
    isRecording = false;

    const blob = new Blob(recordedChunks, { type: mimeType });
    const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
    setInstruction('Saving your video...');
    await saveFile(blob, `rakhi-video-${Date.now()}.${ext}`, mimeType.split(';')[0], 'Video saved to Downloads.');
  };

  mediaRecorder.start();
  isRecording = true;
  recordStartTime = performance.now();
  recordBtn.classList.add('recording');
  updateRecordButtonLabel();
  recordTimerHandle = setInterval(updateRecordButtonLabel, 500);
  recordAutoStopHandle = setTimeout(stopRecording, MAX_RECORD_MS);
  setInstruction('Recording... tap Stop when you\'re done (max 15 seconds).');
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    recordBtn.disabled = true;
    mediaRecorder.stop();
  }
}

recordBtn.addEventListener('click', () => {
  if (isRecording) stopRecording();
  else startRecording();
});
