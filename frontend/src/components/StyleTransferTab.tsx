import { useState } from 'react';
import { Wand2 } from 'lucide-react';
import DropZone from './DropZone';
import PromptInput from './PromptInput';
import ModelSelector from './ModelSelector';
import ResultPanel from './ResultPanel';
import { submitStyleTransfer } from '../api';

const MODELS = [
  {
    id: 'gemini-25-pro',
    name: 'Gemini 2.5 Pro → Image Gen',
    tag: 'recommended' as const,
    note: 'Two-step: Gemini 2.5 Pro deeply analyzes the images and writes a rich architectural prompt, then the image-gen model renders it. With original geometry uploaded, the analyst identifies the design delta so intentional changes are never auto-corrected.',
  },
  {
    id: 'gemini-25-flash',
    name: 'Gemini 2.5 Flash → Image Gen',
    tag: 'fast' as const,
    note: 'Two-step: Gemini 2.5 Flash analyzes the images and writes a detailed prompt, then the image-gen model renders. Faster than Pro. Supports 3-image delta mode.',
  },
  {
    id: 'gemini-direct',
    name: 'Gemini 2.0 Flash (Direct)',
    tag: 'fast' as const,
    note: 'Gemini 2.0 Flash sees all images and generates the render in one shot. With original geometry, it sees the delta directly and preserves your design changes.',
  },
  {
    id: 'flux-redux-controlnet',
    name: 'Flux Canny Pro + Gemini Prompt',
    tag: 'precise' as const,
    note: 'Gemini describes the style → Flux Canny Pro applies it constrained to mass edges. With original geometry, uses the delta-aware analyst for a richer structural prompt.',
  },
  {
    id: 'flux-redux-only',
    name: 'Flux Redux Only',
    tag: 'fast' as const,
    note: 'Single step. Looser mass adherence but faster and often more creative.',
  },
  {
    id: 'sdxl-img2img',
    name: 'SDXL img2img',
    tag: 'fallback' as const,
    note: 'Classic img2img. Cheapest option, lower fidelity. Supports 3-image delta prompt.',
  },
];

export default function StyleTransferTab() {
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [massFile, setMassFile] = useState<File | null>(null);
  const [originalMassFile, setOriginalMassFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState(MODELS[0].id);
  const [jobId, setJobId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const canSubmit = referenceFile && massFile;
  const threeImageMode = !!(referenceFile && massFile && originalMassFile);

  const handleSubmit = async () => {
    if (!referenceFile || !massFile) return;
    setLoading(true);
    try {
      const { jobId } = await submitStyleTransfer(
        referenceFile, massFile, prompt, model, undefined, originalMassFile
      );
      setJobId(jobId);
    } catch (err) {
      alert('Submission failed: ' + err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <p className="text-sm text-gray-400">
        Transfer the visual style, materials, and atmosphere of a reference image onto a new
        building mass. The AI applies the render aesthetic to your new design.
        Optionally upload the original geometry so the AI can identify design deltas and
        preserve your intentional changes.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <DropZone
          label="Style Reference Image"
          file={referenceFile}
          onFile={setReferenceFile}
          hint="Any render or photo with the desired style"
        />
        <DropZone
          label="New Building Mass"
          file={massFile}
          onFile={setMassFile}
          hint="Wireframe, schematic or mass model screenshot"
        />
      </div>

      <div className="flex flex-col gap-2">
        <DropZone
          label="Original Building Mass (optional — enables 3-image delta mode)"
          file={originalMassFile}
          onFile={setOriginalMassFile}
          hint="The baseline geometry before your design edits — helps the AI preserve intentional changes"
        />
        {threeImageMode && (
          <p className="text-xs text-yellow-400 font-medium">
            ✦ 3-image delta mode active — the analyst will identify geometry changes and lock them in
          </p>
        )}
      </div>

      <PromptInput
        label="Additional Style Prompt"
        value={prompt}
        onChange={setPrompt}
        placeholder="e.g. biophilic design, warm wood lattice, floor-to-ceiling glass, golden hour lighting"
      />

      <ModelSelector models={MODELS} value={model} onChange={setModel} />

      <button
        onClick={handleSubmit}
        disabled={!canSubmit || loading}
        className="flex items-center justify-center gap-2 py-3.5 rounded-xl bg-yellow-500 hover:bg-yellow-400
          disabled:opacity-40 disabled:cursor-not-allowed text-black font-bold text-sm transition-all"
      >
        {loading ? (
          <>
            <div className="w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin" />
            Submitting…
          </>
        ) : (
          <>
            <Wand2 size={15} /> Transfer Style
          </>
        )}
      </button>

      <ResultPanel jobId={jobId} />
    </div>
  );
}
