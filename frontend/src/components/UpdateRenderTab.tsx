import { useState } from 'react';
import { Wand2 } from 'lucide-react';
import DropZone from './DropZone';
import PromptInput from './PromptInput';
import ResultPanel from './ResultPanel';
import { submitUpdateRender } from '../api';

export default function UpdateRenderTab() {
  const [renderFile, setRenderFile] = useState<File | null>(null);
  const [massFile, setMassFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState('');
  const [jobId, setJobId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const canSubmit = renderFile && massFile;

  const handleSubmit = async () => {
    if (!renderFile || !massFile) return;
    setLoading(true);
    try {
      const { jobId } = await submitUpdateRender(renderFile, massFile, prompt);
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
