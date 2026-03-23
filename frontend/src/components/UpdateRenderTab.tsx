import { useState } from 'react';
import { Wand2 } from 'lucide-react';
import DropZone from './DropZone';
import PromptInput from './PromptInput';
import ModelSelector from './ModelSelector';
import ResultPanel from './ResultPanel';
import { submitUpdateRender } from '../api';

const MODELS = [
  {
    id: 'gemini-25-pro',
    name: 'Gemini 2.5 Pro → Image Gen',
    tag: 'recommended' as const,
    note: 'Two-step: Gemini 2.5 Pro reads both the existing render (style) and the new mass (geometry), writes a rich architectural prompt, then generates the render. Sees full visual context — best quality.',
  },
  {
    id: 'gemini-25-flash',
    name: 'Gemini 2.5 Flash → Image Gen',
    tag: 'fast' as const,
    note: 'Same two-step pipeline with Flash. Faster than Pro. Gemini reads both images directly — no information lost to text conversion.',
  },
  {
    id: 'gemini-direct',
    name: 'Gemini Direct (one-shot)',
    tag: 'fast' as const,
    note: 'Gemini sees both images in one shot and generates the render immediately. No separate analysis step.',
  },
  {
    id: 'flux-controlnet-canny',
    name: 'Flux Canny Pro (BFL)',
    tag: 'precise' as const,
    note: 'Official Black Forest Labs edge-guided model. Very precise on silhouette and line work.',
  },
  {
    id: 'flux-controlnet-depth',
    name: 'Flux Depth Pro (BFL)',
    tag: 'precise' as const,
    note: 'Best for masses with strong 3D depth variation. Official BFL depth-guided model.',
  },
  {
    id: 'sdxl-controlnet',
    name: 'SDXL ControlNet — Canny',
    tag: 'fallback' as const,
    note: 'Older model, faster & cheaper. Good for quick tests.',
  },
];

export default function UpdateRenderTab() {
  const [renderFile, setRenderFile] = useState<File | null>(null);
  const [massFile, setMassFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState('');
  const [model, setModel] = useState(MODELS[0].id);
  const [jobId, setJobId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const canSubmit = renderFile && massFile;

  const handleSubmit = async () => {
    if (!renderFile || !massFile) return;
    setLoading(true);
    try {
      const { jobId } = await submitUpdateRender(renderFile, massFile, prompt, model);
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
        Upload your existing render and a new building mass (image/wireframe). The AI will replace
        the old building with a photorealistic render of the new mass, preserving the scene.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <DropZone
          label="Existing Render"
          file={renderFile}
          onFile={setRenderFile}
          hint="Photo or render of the current building"
        />
        <DropZone
          label="New Building Mass"
          file={massFile}
          onFile={setMassFile}
          hint="Wireframe, schematic or mass model screenshot"
        />
      </div>

      <PromptInput
        label="Style & Materiality Prompt"
        value={prompt}
        onChange={setPrompt}
        placeholder="e.g. glass curtain wall with gold accents, photorealistic dusk render, lush greenery, dramatic lighting"
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
            <Wand2 size={15} /> Update Render
          </>
        )}
      </button>

      <ResultPanel jobId={jobId} />
    </div>
  );
}
