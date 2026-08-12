import React, { useEffect, useRef, useState } from 'react';

interface StrokePoint {
  x: number;
  y: number;
}

interface Stroke {
  points: StrokePoint[];
}

const CANVAS_HEIGHT = 300;
const STROKE_COLOR = 'rgba(255, 255, 255, 0.95)';
const STROKE_WIDTH = 2.5;

export const DrawingCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const strokesRef = useRef<Stroke[]>([]);
  const drawingRef = useRef(false);
  const currentRef = useRef<StrokePoint[]>([]);

  const commitStrokes = (next: Stroke[]) => {
    strokesRef.current = next;
    setStrokes(next);
  };

  const redraw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = STROKE_COLOR;
    ctx.lineWidth = STROKE_WIDTH;

    const drawPoints = (points: StrokePoint[]) => {
      if (points.length === 0) return;
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i += 1) {
        ctx.lineTo(points[i].x, points[i].y);
      }
      ctx.stroke();
    };

    for (const stroke of strokesRef.current) {
      drawPoints(stroke.points);
    }
    drawPoints(currentRef.current);
  };

  const resizeCanvas = () => {
    const canvas = canvasRef.current;
    const container = canvas?.parentElement;
    if (!canvas || !container) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(CANVAS_HEIGHT * dpr));
  };

  useEffect(() => {
    const container = canvasRef.current?.parentElement;
    if (!container) return;
    const observer = new ResizeObserver(() => {
      resizeCanvas();
      redraw();
    });
    observer.observe(container);
    resizeCanvas();
    redraw();
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    redraw();
  }, [strokes]);

  const getPoint = (event: React.PointerEvent): StrokePoint => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  const finishStroke = () => {
    drawingRef.current = false;
    if (currentRef.current.length > 0) {
      commitStrokes([...strokesRef.current, { points: currentRef.current }]);
    }
    currentRef.current = [];
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      canvas.setPointerCapture(event.pointerId);
    } catch {
      // pointer may already be inactive
    }
    drawingRef.current = true;
    currentRef.current = [getPoint(event)];
    redraw();
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    currentRef.current.push(getPoint(event));
    redraw();
  };

  const handlePointerUp = () => {
    if (drawingRef.current) finishStroke();
  };

  return (
    <div className="space-y-3">
      <div
        className="relative rounded-xl border border-white/10 bg-black/40 overflow-hidden"
        style={{ height: CANVAS_HEIGHT }}
      >
        <canvas
          ref={canvasRef}
          className="w-full h-full touch-none cursor-crosshair block"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        />
        {strokes.length === 0 && (
          <p className="pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2 text-center text-xs text-gray-500 select-none">
            Patient draws here (mouse or touch)
          </p>
        )}
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => commitStrokes(strokesRef.current.slice(0, -1))}
          disabled={strokes.length === 0}
          className="px-4 py-1.5 rounded-lg text-sm font-semibold border border-white/10 bg-white/5 text-gray-300 transition-all duration-200 hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Undo
        </button>
        <button
          type="button"
          onClick={() => commitStrokes([])}
          disabled={strokes.length === 0}
          className="px-4 py-1.5 rounded-lg text-sm font-semibold border border-white/10 bg-white/5 text-gray-300 transition-all duration-200 hover:bg-white/10 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Clear
        </button>
      </div>
    </div>
  );
};
