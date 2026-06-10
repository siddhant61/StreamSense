// Azure Kinect skeleton geometry for the preview canvas.
// Bone connectivity mirrors streamer/kinect_support.py BONES (32-joint order).

export const BONES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 26], [26, 27],
  [27, 28], [28, 29], [27, 30], [30, 31],
  [2, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [7, 10],
  [2, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [14, 17],
  [0, 18], [18, 19], [19, 20], [20, 21],
  [0, 22], [22, 23], [23, 24], [24, 25],
];

export type Point = [number, number, number]; // [x, y, confidence]

export interface ScreenPoint {
  x: number;
  y: number;
  conf: number;
}

/**
 * Fit the joint cloud into a w×h canvas (preserve aspect, pad, flip Y so up is up).
 * Joints with confidence 0 are treated as missing and reported as conf 0.
 */
export function fitToCanvas(points: Point[], w: number, h: number, pad = 20): ScreenPoint[] {
  const valid = points.filter((p) => p[2] > 0);
  if (valid.length === 0) {
    return points.map(() => ({ x: 0, y: 0, conf: 0 }));
  }
  const xs = valid.map((p) => p[0]);
  const ys = valid.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const scale = Math.min((w - 2 * pad) / spanX, (h - 2 * pad) / spanY);
  const offX = (w - spanX * scale) / 2;
  const offY = (h - spanY * scale) / 2;
  return points.map((p) => ({
    x: offX + (p[0] - minX) * scale,
    y: h - (offY + (p[1] - minY) * scale), // flip Y
    conf: p[2],
  }));
}
