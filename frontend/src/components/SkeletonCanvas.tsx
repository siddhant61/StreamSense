import { useEffect, useRef } from "react";
import { BONES, fitToCanvas, type Point } from "../skeleton";

const SIZE = 320;

export function SkeletonCanvas({ points }: { points: Point[] | null }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = "#0b0f14";
    ctx.fillRect(0, 0, SIZE, SIZE);

    if (!points) {
      ctx.fillStyle = "#8b98a5";
      ctx.font = "13px system-ui";
      ctx.textAlign = "center";
      ctx.fillText("awaiting Kinect joints…", SIZE / 2, SIZE / 2);
      return;
    }

    const sp = fitToCanvas(points, SIZE, SIZE);
    ctx.strokeStyle = "#4a9eff";
    ctx.lineWidth = 2;
    for (const [a, b] of BONES) {
      const pa = sp[a], pb = sp[b];
      if (pa.conf > 0 && pb.conf > 0) {
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      }
    }
    ctx.fillStyle = "#e6edf3";
    for (const p of sp) {
      if (p.conf > 0) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [points]);

  return <canvas ref={ref} width={SIZE} height={SIZE} className="skeleton" />;
}
