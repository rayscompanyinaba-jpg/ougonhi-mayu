// 黄金比眉（ブラウザ版）: カメラ映像に眉毛の黄金バランスのガイドを重ねる
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";
import { buildGuide, LANDMARKS } from "./guide.js";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const COLORS = { pink: "#e91e63", red: "#ff2626", white: "#ffffff", yellow: "#ffe000" };

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const ctx = canvas.getContext("2d");

const state = {
  selection: "right",       // ガイドを出す眉（使う人から見た左右）: left / both / right
  color: "pink",
  showLines: true,
  showRecommended: true,
  showPoints: false,
  scoreOpen: false,
};

let landmarker = null;
let source = null;            // <video>（カメラ）か <img>（確認用モード）
let face = null;              // 最新のコマの顔（なめらかにした点・顔の向き）
let rawPoints = null;         // 採点用: なめらかにする前の点
let lastTimestamp = -1;

// ---- 起動 ----

$("startButton").addEventListener("click", start);
const testImage = new URLSearchParams(location.search).get("img");
if (testImage) start(); // 確認用モード: カメラの代わりに写真を使う

async function start() {
  $("startError").hidden = true;
  $("start").hidden = true;
  $("loading").hidden = false;
  try {
    const [sourceElement, faceLandmarker] = await Promise.all([openSource(), createLandmarker()]);
    source = sourceElement;
    landmarker = faceLandmarker;
    $("loading").hidden = true;
    $("controls").hidden = false;
    requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);
    $("loading").hidden = true;
    $("start").hidden = false;
    $("startError").textContent = error.name === "NotAllowedError"
      ? "カメラの使用が許可されていません。Safariの設定（ぁあ → Webサイトの設定 → カメラ）で許可してから、もう一度押してください。"
      : "起動できませんでした。通信状況を確認して、もう一度お試しください。";
    $("startError").hidden = false;
  }
}

async function openSource() {
  if (testImage) {
    const img = new Image();
    img.src = testImage;
    await img.decode();
    return img;
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 1280 } },
    audio: false,
  });
  const video = document.createElement("video");
  video.playsInline = true;
  video.muted = true;
  video.srcObject = stream;
  await video.play();
  return video;
}

async function createLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
  });
  try {
    return await FaceLandmarker.createFromOptions(vision, options("GPU"));
  } catch {
    return await FaceLandmarker.createFromOptions(vision, options("CPU"));
  }
}

function sourceSize() {
  return source instanceof HTMLVideoElement
    ? { width: source.videoWidth, height: source.videoHeight }
    : { width: source.naturalWidth, height: source.naturalHeight };
}

// ---- 毎コマの処理 ----

function loop() {
  const { width, height } = sourceSize();
  if (width && height) {
    let timestamp = performance.now();
    if (timestamp <= lastTimestamp) timestamp = lastTimestamp + 1;
    lastTimestamp = timestamp;
    const result = landmarker.detectForVideo(source, timestamp);
    updateFace(result, width, height);
    draw(width, height);
  }
  requestAnimationFrame(loop);
}

function updateFace(result, width, height) {
  const landmarks = result.faceLandmarks?.[0];
  if (!landmarks) {
    face = null;
    rawPoints = null;
    return;
  }
  // 映像は鏡と同じ映り方で表示するので、点も左右を反転させてピクセルの座標にする
  const points = landmarks.map((p) => ({ x: (1 - p.x) * width, y: p.y * height }));
  rawPoints = points;
  // 前のコマの位置と混ぜて、点の細かい揺れを抑える
  const previous = face?.points;
  const smoothed = previous && previous.length === points.length
    ? points.map((p, i) => ({ x: previous[i].x * 0.5 + p.x * 0.5, y: previous[i].y * 0.5 + p.y * 0.5 }))
    : points;
  face = { points: smoothed, ...faceAngles(result.facialTransformationMatrixes?.[0]?.data) };
}

/** 顔の向き（度）。yaw: 左右を向く、pitch: 上下を向く */
function faceAngles(m) {
  if (!m) return { yaw: 0, pitch: 0 };
  // 4×4 の行列（列ごとに並んでいる）の回転部分から角度を取り出す
  const r20 = m[2], r21 = m[6], r22 = m[10];
  const toDegrees = (radians) => (radians * 180) / Math.PI;
  return {
    yaw: toDegrees(Math.asin(Math.max(-1, Math.min(1, -r20)))),
    pitch: toDegrees(Math.atan2(r21, r22)),
  };
}

/** 正面からどれだけ外れているか（度）。左右・上下のうち大きいほう */
const angleFromFront = (f) => Math.max(Math.abs(f.yaw), Math.abs(f.pitch));
const isFacingFront = (f) => angleFromFront(f) < 12;
/** ガイドの濃さ。正面から外れるほど薄くする（8度まではそのまま、18度でほぼ見えない） */
const guideOpacity = (f) => 1 - Math.min(Math.max((angleFromFront(f) - 8) / 10, 0), 1) * 0.9;

// ---- 描画 ----

let viewTransform = null; // 映像の座標 → 画面の座標

function draw(width, height) {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  // 映像を画面いっぱいに拡大し、はみ出した部分を切り取って、鏡のように左右反転して描く
  const scale = Math.max(cw / width, ch / height);
  const offsetX = (cw - width * scale) / 2, offsetY = (ch - height * scale) / 2;
  viewTransform = (p) => ({ x: p.x * scale + offsetX, y: p.y * scale + offsetY });
  ctx.save();
  ctx.translate(offsetX + width * scale, offsetY);
  ctx.scale(-scale, scale);
  ctx.drawImage(source, 0, 0, width, height);
  ctx.restore();

  updateMessage();
  $("angle").hidden = !(state.showPoints && face);
  if (!face) return;
  if (state.showPoints) {
    $("angle").textContent = `左右 ${Math.round(face.yaw)}°  上下 ${Math.round(face.pitch)}°`;
  }

  const guide = buildGuide(face.points);
  if (guide) {
    ctx.save();
    ctx.globalAlpha = guideOpacity(face);
    for (const side of guide.sides) {
      if (state.selection === "both" || state.selection === side.name) drawSide(side);
    }
    ctx.restore();
  }
  if (state.showPoints) drawLandmarks(face.points);
}

function drawSide(side) {
  const color = COLORS[state.color];
  const T = viewTransform;
  if (state.showLines) {
    // 眉山ゾーン（b〜c の間）を薄く塗る
    ctx.fillStyle = withAlpha(color, 0.08);
    fillPolygon(side.archZone.map(T));
    // a〜e の線（眉を描く邪魔にならないよう、細く薄く）
    ctx.strokeStyle = withAlpha(color, 0.45);
    ctx.lineWidth = 1;
    ctx.fillStyle = withAlpha(color, 0.8);
    ctx.font = "bold 14px -apple-system, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const [label, segment] of side.lines) {
      const from = T(segment.from), to = T(segment.to);
      ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke();
      ctx.fillText(label, to.x, to.y - 10);
    }
  }
  // おすすめの眉の輪郭を点線で描く
  if (state.showRecommended && side.recommendedBrow.length) {
    const outline = side.recommendedBrow.map(T);
    ctx.fillStyle = withAlpha(color, 0.06);
    fillPolygon(outline);
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.lineJoin = "round";
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = withAlpha(color, 0.75);
    tracePolygon(outline);
    ctx.stroke();
    ctx.restore();
  }
  // 基準にした点（目頭・黒目の外側・目尻・小鼻）は、ほぼ見えない程度に
  ctx.fillStyle = "rgba(255,255,255,0.15)";
  for (const p of side.anchors.map(T)) {
    ctx.beginPath(); ctx.arc(p.x, p.y, 2, 0, Math.PI * 2); ctx.fill();
  }
}

/** 検出した顔のパーツの点（動作確認用）。使っている点だけを薄く表示する */
function drawLandmarks(points) {
  const groups = [
    ["#ff4f9a", (s) => [...s.browUpper, ...s.browLower]],
    ["#38d6ff", (s) => [s.eyeInner, s.eyeOuter, s.eyeTop]],
    ["#ffd400", (s) => s.iris],
    ["#3ddc84", (s) => [s.noseWing]],
  ];
  for (const sideIds of [LANDMARKS.right, LANDMARKS.left]) {
    for (const [color, pick] of groups) {
      ctx.fillStyle = withAlpha(color, 0.5);
      for (const i of pick(sideIds)) {
        const p = viewTransform(points[i]);
        ctx.beginPath(); ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2); ctx.fill();
      }
    }
  }
}

function tracePolygon(points) {
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.closePath();
}
function fillPolygon(points) { tracePolygon(points); ctx.fill(); }

function withAlpha(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

// ---- 案内 ----

let noticeText = null, noticeTimer = null;

function showNotice(text) {
  noticeText = text;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { noticeText = null; }, 2500);
}

function updateMessage() {
  const text = noticeText
    ?? (!face ? "顔を画面に映してください" : !isFacingFront(face) ? "顔を正面に向けてください" : null);
  const message = $("message");
  message.hidden = !text;
  if (text && message.textContent !== text) message.textContent = text;
}

// ---- 採点 ----

$("scoreButton").addEventListener("click", () => {
  if (!rawPoints || !face) return showNotice("顔を画面に映してから押してください");
  if (!isFacingFront(face)) return showNotice("正面を向いてから押してください");
  const guide = buildGuide(rawPoints, captureLuma());
  const entries = (guide?.sides ?? [])
    .filter((side) => state.selection === "both" || state.selection === side.name)
    // 画面での位置に合わせて、左眉を先に並べる
    .sort((a, b) => (a.name === "left" ? -1 : 1) - (b.name === "left" ? -1 : 1))
    .filter((side) => side.score);
  if (!entries.length) return showNotice("眉毛がうまく読み取れませんでした。明るい場所でもう一度押してください");
  showScore(entries);
});

/** 今のコマの明るさ（鏡像にそろえた映像の座標） */
function captureLuma() {
  const { width, height } = sourceSize();
  const work = document.createElement("canvas");
  work.width = width;
  work.height = height;
  const w = work.getContext("2d", { willReadFrequently: true });
  w.translate(width, 0);
  w.scale(-1, 1);
  w.drawImage(source, 0, 0, width, height);
  const rgba = w.getImageData(0, 0, width, height).data;
  const data = new Uint8Array(width * height);
  for (let i = 0, j = 0; j < data.length; i += 4, j++) {
    data[j] = (rgba[i] * 299 + rgba[i + 1] * 587 + rgba[i + 2] * 114) / 1000;
  }
  return { width, height, data };
}

function showScore(sides) {
  const container = $("scoreEntries");
  container.replaceChildren(...sides.map((side) => {
    const entry = element("div", "entry");
    const head = element("div", "entry-head");
    head.append(
      element("span", "title", side.name === "left" ? "左眉" : "右眉"),
      element("span", "total", String(side.score.total)),
      element("span", "", `点 / ${side.score.max}点`),
    );
    entry.append(head);
    for (const item of side.score.items) {
      const row = element("div", "item");
      const line = element("div", "item-row");
      line.append(element("span", "", item.name), element("span", "", `${item.points} / ${item.max}`));
      const bar = element("div", "bar");
      const fill = element("span", item.points === item.max ? "full" : "");
      fill.style.width = `${(item.points / item.max) * 100}%`;
      bar.append(fill);
      row.append(line, bar);
      if (item.advice) row.append(element("span", "advice", item.advice));
      entry.append(row);
    }
    return entry;
  }));
  $("scorecard").hidden = false;
  $("scoreButton").hidden = true;
}

$("closeScore").addEventListener("click", () => {
  $("scorecard").hidden = true;
  $("scoreButton").hidden = false;
});

function element(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text != null) e.textContent = text;
  return e;
}

// ---- 操作ボタン ----

function toggle(id, key) {
  const button = $(id);
  button.addEventListener("click", () => {
    state[key] = !state[key];
    button.setAttribute("aria-pressed", String(state[key]));
  });
}
toggle("linesTool", "showLines");
toggle("browTool", "showRecommended");
toggle("pointsTool", "showPoints");

$("colorSelect").addEventListener("change", (event) => {
  state.color = event.target.value;
  document.querySelector("#colorTool .swatch").style.background = COLORS[state.color];
});

const sideButtons = document.querySelectorAll(".segmented button");
function updateSideButtons() {
  for (const button of sideButtons) button.setAttribute("aria-checked", String(button.dataset.side === state.selection));
}
for (const button of sideButtons) {
  button.addEventListener("click", () => { state.selection = button.dataset.side; updateSideButtons(); });
}
updateSideButtons();
