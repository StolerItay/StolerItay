import { useRef, useState, useEffect, useCallback } from 'react';
import { Eraser, Paintbrush, Wand2, RotateCcw } from 'lucide-react';
import { submitInpaint, pollJob } from '../api';
import ModelSelector from './ModelSelector';

const INPAINT_MODELS = [
  {
    id: 'gemini-edit',
    name: 'Gemini Edit',
    tag: 'recommended' as const,
    note: 'Gemini sees the full render and the painted mask together. Understands architectural context holistically — materials, shadows, perspective — for natural-looking edits. No Replicate token needed.',
  },
  {
    id: 'flux-fill-pro',
    name: 'Flux Fill Pro',
    tag: 'precise' as const,
    note: 'Pixel-precise mask adherence. Best when the masked boundary must be followed exactly.',
  },
  {
    id: 'flux-fill-dev',
    name: 'Flux Fill Dev',
    tag: 'fast' as const,
    note: 'Faster and cheaper than Pro. Slightly less refined blending.',
  },
  {
    id: 'sd-inpainting',
    name: 'SD Inpainting',
    tag: 'fallback' as const,
    note: 'Classic Stable Diffusion inpainting. Quick & low cost.',
  },
];

interface Props {
  imageUrl: string;
  onDone: () => void;
  onNewResult: (url: string) => void;
}

export default function InpaintCanvas({ imageUrl, onDone, onNewResult }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageCanvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isErasing, setIsErasing] = useState(false);
  const [brushSize, setBrushSize] = useState(30);
  const [prompt, setPrompt] = useState('');
  const [inpaintModel, setInpaintModel] = useState('gemini-edit');
  const [generating, setGenerating] = useState(false);
  const [dims, setDims] = useState({ w: 800, h: 500 });
  const lastPos = useRef<{ x: number; y: number } | null>(null);

  // Load image onto canvas
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const maxW = containerRef.current?.clientWidth ?? 800;
      const scale = Math.min(1, maxW / img.naturalWidth);
      const w = Math.round(img.naturalWidth * scale);
      const h = Math.round(img.naturalHeight * scale);
      setDims({ w, h });

      setTimeout(() => {
        const ic = imageCanvasRef.current;
        const mc = maskCanvasRef.current;
        if (!ic || !mc) return;
        ic.width = w;
        ic.height = h;
        mc.width = w;
        mc.height = h;
        const ctx = ic.getContext('2d')!;
        ctx.drawImage(img, 0, 0, w, h);
      }, 0);
    };
    img.src = imageUrl;
  }, [imageUrl]);

  const getPos = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = maskCanvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    if ('touches' in e) {
      const t = e.touches[0];
      return {
        x: (t.clientX - rect.left) * scaleX,
        y: (t.clientY - rect.top) * scaleY,
      };
    }
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const draw = useCallback(
    (pos: { x: number; y: number }) => {
      const mc = maskCanvasRef.current;
      if (!mc) return;
      const ctx = mc.getContext('2d')!;
      ctx.globalCompositeOperation = isErasing ? 'destination-out' : 'source-over';
      ctx.fillStyle = 'rgba(255, 200, 0, 0.7)';
      ctx.lineWidth = brushSize;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      if (lastPos.current) {
        ctx.beginPath();
        ctx.moveTo(lastPos.current.x, lastPos.current.y);
        ctx.lineTo(pos.x, pos.y);
        ctx.strokeStyle = isErasing ? 'rgba(0,0,0,1)' : 'rgba(255, 200, 0, 0.7)';
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, brushSize / 2, 0, Math.PI * 2);
      ctx.fill();
      lastPos.current = pos;
    },
    [isErasing, brushSize]
  );

  const onMouseDown = (e: React.MouseEvent) => {
    setIsDrawing(true);
    lastPos.current = null;
    draw(getPos(e));
  };
  const onMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing) return;
    draw(getPos(e));
  };
  const onMouseUp = () => {
    setIsDrawing(false);
    lastPos.current = null;
  };

  const clearMask = () => {
    const mc = maskCanvasRef.current;
    if (!mc) return;
    const ctx = mc.getContext('2d')!;
    ctx.clearRect(0, 0, mc.width, mc.height);
  };

  const getMaskDataUrl = (): string => {
    const mc = maskCanvasRef.current!;
    const w = mc.width, h = mc.height;
    // White canvas masked by the painted strokes' alpha → white where painted, transparent elsewhere
    const whiteCanvas = document.createElement('canvas');
    whiteCanvas.width = w; whiteCanvas.height = h;
    const wCtx = whiteCanvas.getContext('2d')!;
    wCtx.fillStyle = 'white';
    wCtx.fillRect(0, 0, w, h);
    wCtx.globalCompositeOperation = 'destination-in';
    wCtx.drawImage(mc, 0, 0);
    // Composite white strokes onto black background → standard B&W inpaint mask
    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = w; finalCanvas.height = h;
    const fCtx = finalCanvas.getContext('2d')!;
    fCtx.fillStyle = 'black';
    fCtx.fillRect(0, 0, w, h);
    fCtx.drawImage(whiteCanvas, 0, 0);
    return finalCanvas.toDataURL('image/png');
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      alert('Please describe what you want to change in the marked area.');
      return;
    }
    setGenerating(true);
    try {
      const maskDataUrl = getMaskDataUrl();
      const { jobId } = await submitInpaint(imageUrl, maskDataUrl, prompt, inpaintModel);
      // Poll
      let done = false;
      while (!done) {
        await new Promise((r) => setTimeout(r, 2000));
        const result = await pollJob(jobId);
        if (result.status === 'done' && result.output_url) {
          onNewResult(result.output_url);
          done = true;
        } else if (result.status === 'error') {
          alert('Inpainting failed: ' + result.error);
          done = true;
        }
      }
    } catch (err) {
      alert('Error: ' + err);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={() => setIsErasing(false)}
          className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-all
            ${!isErasing ? 'bg-yellow-500 text-black' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'}`}
        >
          <Paintbrush size={13} /> Paint
        </button>
        <button
          onClick={() => setIsErasing(true)}
          className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-all
            ${isErasing ? 'bg-yellow-500 text-black' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'}`}
        >
          <Eraser size={13} /> Erase
        </button>
        <button
          onClick={clearMask}
          className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 transition-all"
        >
          <RotateCcw size={13} /> Clear
        </button>
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-xs text-gray-500">Brush</span>
          <input
            type="range"
            min={5}
            max={80}
            value={brushSize}
            onChange={(e) => setBrushSize(Number(e.target.value))}
            className="w-24 accent-yellow-400"
          />
          <span className="text-xs text-gray-400 w-6">{brushSize}</span>
        </div>
      </div>

      {/* Canvas */}
      <div
        ref={containerRef}
        className="relative rounded-xl overflow-hidden cursor-crosshair select-none"
        style={{ width: '100%', height: dims.h }}
      >
        <canvas
          ref={imageCanvasRef}
          style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
        />
        <canvas
          ref={maskCanvasRef}
          style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', opacity: 0.6 }}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
        />
      </div>

      <p className="text-xs text-gray-500 -mt-2">
        Paint over the area you want to change (shown in yellow), then describe what to put there.
      </p>

      {/* Prompt + Generate */}
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={2}
        placeholder="Describe the change: e.g. 'replace with dark granite cladding and floor-to-ceiling glass'"
        className="w-full rounded-xl bg-gray-900 border border-gray-700 text-gray-200 placeholder-gray-600
          px-4 py-3 text-sm resize-none focus:outline-none focus:border-yellow-500/60 transition-colors"
      />
      <ModelSelector models={INPAINT_MODELS} value={inpaintModel} onChange={setInpaintModel} />
      <div className="flex gap-3">
        <button
          onClick={handleGenerate}
          disabled={generating}
          className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl bg-yellow-500 hover:bg-yellow-400
            disabled:opacity-50 disabled:cursor-not-allowed text-black font-semibold text-sm transition-all"
        >
          {generating ? (
            <>
              <div className="w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin" />
              Generating…
            </>
          ) : (
            <>
              <Wand2 size={15} /> Apply Inpaint
            </>
          )}
        </button>
        <button
          onClick={onDone}
          className="px-5 py-3 rounded-xl bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm transition-all"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
