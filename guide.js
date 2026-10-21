// 眉毛の黄金バランスのガイド線とおすすめ眉の計算（iPhoneアプリの BrowGuide.swift / BrowTemplate.swift と同じ考え方）
//
// - a: 目頭の真上 → 眉頭
// - b: 黒目の外側の真上 → 眉山（内側の目安）
// - c: 目尻の真上 → 眉山（外側の目安）
// - d: 小鼻と目尻を結んだ線の延長 → 眉尻
// - e: 眉頭の下端を通る横線 → 眉尻はこれより下がりすぎない

// MediaPipe Face Landmarker の点の番号（使う人から見た右・左）
export const LANDMARKS = {
  right: {
    eyeInner: 133, eyeOuter: 33, eyeTop: 159,
    iris: [468, 469, 470, 471, 472], // 先頭が黒目の中心、残りが黒目のふち
    browUpper: [70, 63, 105, 66, 107], // 眉尻 → 眉頭
    browLower: [46, 53, 52, 65, 55],   // 眉尻 → 眉頭
    noseWing: 64,
  },
  left: {
    eyeInner: 362, eyeOuter: 263, eyeTop: 386,
    iris: [473, 474, 475, 476, 477],
    browUpper: [300, 293, 334, 296, 336],
    browLower: [276, 283, 282, 295, 285],
    noseWing: 294,
  },
};

// UTOWA講師の資料「ゴールデンバランス」に描かれた眉の形。
// 資料の画像（525×729ピクセル）から、眉の上端と下端を縦の列ごとに読み取った値
const TEMPLATE = {
  aX: 223.5, bX: 329.5, cX: 354.5, tipX: 416, eY: 276, tipY: 257,
  raw: [
    [219, 266, 273], [222, 258, 273], [225, 253, 273], [228, 251, 273],
    [231, 248, 273], [234, 247, 273], [237, 245, 273], [240, 244, 273],
    [243, 243, 273], [246, 243, 273], [249, 242, 273], [252, 241, 273],
    [255, 240, 273], [258, 240, 273], [261, 240, 273], [264, 239, 273],
    [267, 238, 273], [270, 237, 273], [273, 235, 272], [276, 235, 271],
    [279, 235, 269], [282, 233, 268], [285, 233, 266], [288, 232, 265],
    [291, 231, 264], [294, 231, 262], [297, 231, 261], [300, 231, 261],
    [303, 231, 260], [306, 230, 259], [309, 230, 259], [312, 229, 258],
    [315, 229, 257], [318, 229, 256], [321, 229, 256], [324, 229, 255],
    [327, 229, 255], [330, 229, 255], [333, 229, 255], [336, 228, 254],
    [339, 228, 253], [342, 229, 252], [345, 229, 252], [348, 229, 251],
    [351, 229, 251], [354, 230, 250], [357, 230, 249], [360, 231, 248],
    [363, 231, 248], [366, 232, 248], [369, 232, 249], [372, 233, 249],
    [375, 235, 249], [378, 236, 249], [381, 237, 249], [384, 239, 250],
    [387, 240, 250], [390, 241, 250], [393, 243, 250], [396, 244, 251],
    [399, 246, 252], [402, 247, 253], [405, 249, 254], [408, 250, 255],
    [411, 252, 257], [414, 255, 258],
  ],
};
TEMPLATE.eyeWidth = TEMPLATE.cX - TEMPLATE.aX;
// 前後2列ずつと平均して、ピクセルの段差をなくす（眉頭の丸みの部分の上端は角ばるのでそのまま）
TEMPLATE.columns = TEMPLATE.raw.map(([x, top, bottom], i, raw) => {
  const from = Math.max(i - 2, 0), to = Math.min(i + 2, raw.length - 1);
  const range = raw.slice(from, to + 1);
  const average = (k) => range.reduce((sum, c) => sum + c[k], 0) / range.length;
  return { x, top: i < 5 ? top : average(1), bottom: average(2) };
});

/** 黒目の半径を推定するときの割合（黒目のふちの点が取れないときだけ使う） */
const IRIS_RADIUS_RATIO = 0.2;

/**
 * 顔の傾きに合わせた座標系。両方の黒目を結ぶ線を横軸、それに直角な向きを縦軸（上がプラス）にする
 */
class FaceFrame {
  constructor(p, q) {
    const [l, r] = p.x < q.x ? [p, q] : [q, p];
    this.center = { x: (l.x + r.x) / 2, y: (l.y + r.y) / 2 };
    const dx = r.x - l.x, dy = r.y - l.y;
    const length = Math.max(Math.hypot(dx, dy), 1);
    this.right = { x: dx / length, y: dy / length };
    this.up = { x: this.right.y, y: -this.right.x };
  }
  /** 映像の座標 → 顔の座標 */
  local(p) {
    const vx = p.x - this.center.x, vy = p.y - this.center.y;
    return { x: vx * this.right.x + vy * this.right.y, y: vx * this.up.x + vy * this.up.y };
  }
  /** 顔の座標 → 映像の座標 */
  world(p) {
    return {
      x: this.center.x + this.right.x * p.x + this.up.x * p.y,
      y: this.center.y + this.right.y * p.x + this.up.y * p.y,
    };
  }
}

/**
 * ガイドを作る
 * @param points 映像の座標（左上原点・ピクセル、鏡像にそろえた後）の顔の点の並び
 * @param luma 採点するときだけ渡す、そのコマの明るさ（{width, height, data}）
 */
export function buildGuide(points, luma = null) {
  const frame = new FaceFrame(points[LANDMARKS.right.iris[0]], points[LANDMARKS.left.iris[0]]);
  const sides = ["right", "left"].map((side) => makeSide(side, points, frame, luma)).filter(Boolean);
  return sides.length ? { sides } : null;
}

function makeSide(sideName, points, frame, luma) {
  const ids = LANDMARKS[sideName];
  const L = (i) => frame.local(points[i]);

  const innerCorner = L(ids.eyeInner);   // 目頭
  const outerCorner = L(ids.eyeOuter);   // 目尻
  // 顔の中心から見た外向きの向き（画面の左側なら -1、右側なら +1）
  const o = Math.sign(outerCorner.x) || 1;
  const outwardness = (p) => o * p.x;

  const eyeWidth = Math.abs(outerCorner.x - innerCorner.x);
  if (!(eyeWidth > 0)) return null;

  // 黒目の外側: 黒目のふちの点のうち一番外側（取れなければ目の幅から推定）
  const pupil = L(ids.iris[0]);
  const ring = ids.iris.slice(1).map(L);
  const outerRing = ring.reduce((a, b) => (outwardness(b) > outwardness(a) ? b : a), ring[0]);
  const irisOuterX = outerRing && outwardness(outerRing) > outwardness(pupil)
    ? outerRing.x
    : pupil.x + o * eyeWidth * IRIS_RADIUS_RATIO;
  const irisOuter = { x: irisOuterX, y: pupil.y };

  const browUpper = ids.browUpper.map(L);
  const browLower = ids.browLower.map(L);
  const browHeadBottom = browLower[browLower.length - 1]; // 眉頭の下端
  const browTop = Math.max(...browUpper.map((p) => p.y));
  const browBottom = Math.min(...browLower.map((p) => p.y));
  const noseWing = L(ids.noseWing); // 小鼻

  // 縦線の上端（眉の少し上）と下端（目の少し下）
  const top = browTop + eyeWidth * 0.35;
  const bottom = Math.min(innerCorner.y, outerCorner.y) - eyeWidth * 0.6;
  const W = (p) => frame.world(p);
  const vertical = (x) => ({ from: W({ x, y: bottom }), to: W({ x, y: top }) });

  // d: 小鼻 → 目尻 の向きに、縦線の上端の高さまで伸ばす
  const direction = { x: outerCorner.x - noseWing.x, y: outerCorner.y - noseWing.y };
  if (!(direction.y > 0)) return null;
  const dXAt = (y) => noseWing.x + direction.x * (y - noseWing.y) / direction.y;
  const dEnd = { x: dXAt(top), y: top };

  // e: 眉頭の下端の高さの横線
  const eY = browHeadBottom.y;
  const eStart = { x: browHeadBottom.x - o * eyeWidth * 0.25, y: eY };
  const eEnd = { x: outerCorner.x + o * eyeWidth * 0.8, y: eY };

  const zoneBottom = browBottom - eyeWidth * 0.1;
  const archZone = [
    { x: irisOuter.x, y: zoneBottom }, { x: outerCorner.x, y: zoneBottom },
    { x: outerCorner.x, y: top }, { x: irisOuter.x, y: top },
  ].map(W);

  const recommendedLocal = recommendedBrowOutline({
    aX: innerCorner.x, bX: irisOuter.x, cX: outerCorner.x, eY, dXAt, eyeWidth, o,
  });

  const side = {
    name: sideName,
    isScreenLeft: o < 0,
    lines: [
      ["a", vertical(innerCorner.x)],
      ["b", vertical(irisOuter.x)],
      ["c", vertical(outerCorner.x)],
      ["d", { from: W(noseWing), to: W(dEnd) }],
      ["e", { from: W(eStart), to: W(eEnd) }],
    ],
    archZone,
    anchors: [innerCorner, irisOuter, outerCorner, noseWing].map(W),
    recommendedBrow: recommendedLocal.map(W),
    score: null,
    measuredBrowPoints: [],
  };

  // 採点のときだけ: 映像の明るさから実際の眉毛を読み取って判定する
  if (luma && recommendedLocal.length) {
    const tipX = recommendedLocal.reduce((a, p) => (o * p.x > o * a ? p.x : a), recommendedLocal[0].x);
    const eyeTopY = L(ids.eyeTop).y;
    const measurement = measureBrow({
      brightness: (p) => brightnessAt(luma, W(p)),
      outline: recommendedLocal, aX: innerCorner.x, tipX, eY, eyeTopY, eyeWidth, o,
    });
    if (measurement) {
      side.measuredBrowPoints = measurement.browPoints.map(W);
      side.score = evaluate(measurement, {
        aX: innerCorner.x, bX: irisOuter.x, cX: outerCorner.x,
        dX: dXAt(measurement.tailY), eY, eyeWidth, o,
      });
    }
  }
  return side;
}

/**
 * おすすめの眉の輪郭を、顔の座標で返す。
 * 講師の資料の眉の形を、この顔の a・b・c・d・e の線に合わせて伸び縮みさせる
 */
function recommendedBrowOutline({ aX, bX, cX, eY, dXAt, eyeWidth, o }) {
  const t = TEMPLATE;
  const heightOnFace = (templateY) => eY + ((t.eY - templateY) / t.eyeWidth) * eyeWidth;
  // 眉尻の先: d の線の上で、資料と同じ高さになる位置
  const tipY = heightOnFace(t.tipY);
  const tipX = dXAt(tipY);
  if (!(o * (tipX - cX) > 0 && o * (cX - bX) > 0 && o * (bX - aX) > 0)) return [];

  // 資料の x → この顔の x（a・b・c・眉尻の先を基準に、区間ごとにまっすぐ伸ばす）
  const anchors = [[t.aX, aX], [t.bX, bX], [t.cX, cX], [t.tipX, tipX]];
  const xOnFace = (templateX) => {
    let i = 0;
    for (let k = 0; k < anchors.length - 1; k++) if (anchors[k][0] <= templateX) i = k;
    const [t0, f0] = anchors[i], [t1, f1] = anchors[i + 1];
    return f0 + ((f1 - f0) * (templateX - t0)) / (t1 - t0);
  };
  const upper = t.columns.map((c) => ({ x: xOnFace(c.x), y: heightOnFace(c.top) }));
  const lower = [...t.columns].reverse().map((c) => ({ x: xOnFace(c.x), y: heightOnFace(c.bottom) }));
  return [...upper, { x: tipX, y: tipY }, ...lower];
}

function brightnessAt(luma, p) {
  const x = Math.round(p.x), y = Math.round(p.y);
  if (x < 0 || y < 0 || x >= luma.width || y >= luma.height) return null;
  return luma.data[y * luma.width + x];
}

function pointInPolygon(p, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i], b = polygon[j];
    if ((a.y > p.y) !== (b.y > p.y) && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

/** 明るさの分布を「暗いグループ」と「明るいグループ」に一番よく分ける境目（大津の方法） */
function otsuThreshold(values) {
  const histogram = new Array(256).fill(0);
  for (const v of values) histogram[Math.min(Math.max(Math.floor(v), 0), 255)]++;
  const total = values.length;
  const sumAll = histogram.reduce((s, n, level) => s + n * level, 0);
  let sumDark = 0, countDark = 0, best = { level: 128, score: -1 };
  for (let level = 0; level < 256; level++) {
    countDark += histogram[level];
    sumDark += level * histogram[level];
    const countBright = total - countDark;
    if (countDark === 0 || countBright === 0) continue;
    const meanDark = sumDark / countDark, meanBright = (sumAll - sumDark) / countBright;
    const score = countDark * countBright * (meanDark - meanBright) ** 2;
    if (score > best.score) best = { level, score };
  }
  return best.level;
}

/** 眉のまわりを細かいマス目に分けて明るさを調べ、周りの肌より暗いマスを眉毛とみなす */
function measureBrow({ brightness, outline, aX, tipX, eY, eyeTopY, eyeWidth, o }) {
  const step = eyeWidth / 50;
  const uStart = o * aX - eyeWidth * 0.25; // u = o * x（外側ほど大きい）
  const uEnd = o * tipX + eyeWidth * 0.25;
  // 下端は e の線のすぐ下まで（それより下は、目頭の上のくぼみの影が眉毛と同じくらい暗いため）
  const yStart = Math.max(eY - eyeWidth * 0.1, eyeTopY + eyeWidth * 0.25);
  const yEnd = eY + eyeWidth * 0.75;
  const columns = Math.floor((uEnd - uStart) / step), rows = Math.floor((yEnd - yStart) / step);
  if (columns <= 0 || rows <= 0) return null;
  const point = (c, r) => ({ x: o * (uStart + c * step), y: yStart + r * step });

  const values = new Array(columns * rows);
  for (let r = 0; r < rows; r++) for (let c = 0; c < columns; c++) values[r * columns + c] = brightness(point(c, r));
  const known = values.filter((v) => v != null).sort((a, b) => a - b);
  if (known.length <= values.length / 2) return null;

  // 眉間などの影まで眉毛と間違えないよう、肌の明るさの6割強より暗いことも条件にする
  const skin = known[Math.floor(known.length * 0.75)];
  const threshold = Math.min(otsuThreshold(known), skin * 0.62);
  const dark = values.map((v) => (v ?? 255) < threshold);

  // 小さな点（ほくろ・ノイズ）は眉毛として数えない
  const visited = new Array(dark.length).fill(false);
  for (let start = 0; start < dark.length; start++) {
    if (!dark[start] || visited[start]) continue;
    const group = [start], stack = [start];
    visited[start] = true;
    while (stack.length) {
      const index = stack.pop(), r = Math.floor(index / columns), c = index % columns;
      for (const [dc, dr] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nc = c + dc, nr = r + dr;
        if (nc < 0 || nr < 0 || nc >= columns || nr >= rows) continue;
        const n = nr * columns + nc;
        if (dark[n] && !visited[n]) { visited[n] = true; group.push(n); stack.push(n); }
      }
    }
    if (group.length < 6) for (const i of group) dark[i] = false;
  }

  // おすすめ眉の輪郭の中と外で、眉毛のマスを数える
  let templateCells = 0, darkInside = 0, darkOutside = 0;
  const columnCounts = new Array(columns).fill(0), columnTops = new Array(columns).fill(-1);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns; c++) {
      const inside = pointInPolygon(point(c, r), outline);
      if (inside) templateCells++;
      if (!dark[r * columns + c]) continue;
      if (inside) darkInside++; else darkOutside++;
      columnCounts[c]++;
      columnTops[c] = Math.max(columnTops[c], r);
    }
  }
  if (templateCells === 0 || darkInside + darkOutside < templateCells / 6) return null;

  // 2マス以上眉毛がある列だけを使って、眉頭・眉山・眉尻を決める
  const browColumns = [...Array(columns).keys()].filter((c) => columnCounts[c] >= 2);
  if (!browColumns.length) return null;
  const first = browColumns[0], last = browColumns[browColumns.length - 1];
  const highest = Math.max(...browColumns.map((c) => columnTops[c]));
  const archColumns = browColumns.filter((c) => columnTops[c] >= highest - 1);
  const archColumn = archColumns.reduce((s, c) => s + c, 0) / archColumns.length;

  // 眉尻の高さ: 外側の 1 割の列にある眉毛のマスの平均の高さ
  const tailStart = last - Math.max(Math.floor((last - first) / 10), 1);
  const tailRows = [];
  for (let c = tailStart; c <= last; c++) for (let r = 0; r < rows; r++) if (dark[r * columns + c]) tailRows.push(r);
  const tailRow = tailRows.length ? tailRows.reduce((s, r) => s + r, 0) / tailRows.length : 0;

  const xOf = (c) => o * (uStart + c * step);
  return {
    innerX: xOf(first), archX: xOf(archColumn), outerX: xOf(last),
    tailY: yStart + tailRow * step,
    coverage: darkInside / templateCells,
    overflow: darkOutside / templateCells,
    browPoints: dark.flatMap((d, i) => (d ? [point(i % columns, Math.floor(i / columns))] : [])),
  };
}

/** 読み取った眉毛を、ガイドの線とおすすめ眉と比べて点数を付ける（100点満点） */
function evaluate(brow, { aX, bX, cX, dX, eY, eyeWidth, o }) {
  // allowance 以内は満点、zeroAt で 0 点（その間はずれに比例して減らす）
  const points = (max, error, allowance, zeroAt) => {
    const ratio = Math.min(Math.max((error - allowance) / (zeroAt - allowance), 0), 1);
    return Math.round(max * (1 - ratio));
  };
  // 外側（こめかみ側）へ行くほどプラス。目の幅に対する割合
  const outwardOffset = (x, reference) => (o * (x - reference)) / eyeWidth;
  const item = (name, got, max, advice) => ({ name, points: got, max, advice: got === max ? null : advice });

  const overflowPoints = points(25, brow.overflow, 0.1, 0.8);
  const coveragePoints = points(20, 1 - brow.coverage, 0.15, 0.65);

  const headOffset = outwardOffset(brow.innerX, aX);
  const headPoints = points(15, Math.abs(headOffset), 0.05, 0.35);

  const archFromB = outwardOffset(brow.archX, bX), archFromC = outwardOffset(brow.archX, cX);
  const archError = archFromB < 0 ? -archFromB : Math.max(archFromC, 0);
  const archPoints = points(15, archError, 0.03, 0.35);

  const tailOffset = outwardOffset(brow.outerX, dX);
  const tailPoints = points(15, Math.abs(tailOffset), 0.05, 0.35);

  const drop = Math.max(eY - brow.tailY, 0) / eyeWidth;
  const heightPoints = points(10, drop, 0.02, 0.25);

  const items = [
    item("余分な毛（はみ出し）", overflowPoints, 25, "おすすめ眉の線からはみ出した毛を整えましょう"),
    item("形の埋まり具合", coveragePoints, 20, "おすすめ眉の線の中で、足りない部分を描き足しましょう"),
    item("眉頭の位置", headPoints, 15, headOffset > 0
      ? "眉頭を鼻側へ描き足して、a の線にそろえましょう"
      : "眉頭が鼻側に寄りすぎています。a の線までにしましょう"),
    item("眉山の位置", archPoints, 15, archFromB < 0
      ? "眉山をもう少しこめかみ側（b〜c の間）にしましょう"
      : "眉山をもう少し鼻側（b〜c の間）にしましょう"),
    item("眉尻の長さ", tailPoints, 15, tailOffset < 0
      ? "眉尻を d の線まで伸ばしましょう"
      : "眉尻が長すぎます。d の線までにしましょう"),
    item("眉尻の高さ", heightPoints, 10, "眉尻が下がりすぎています。e の線より上で止めましょう"),
  ];
  return { items, total: items.reduce((s, i) => s + i.points, 0), max: 100 };
}
