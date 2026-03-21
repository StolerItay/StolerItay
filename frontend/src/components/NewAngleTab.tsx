import { useState } from 'react';
import { Wand2 } from 'lucide-react';
import DropZone from './DropZone';
import PromptInput from './PromptInput';
import ResultPanel from './ResultPanel';
import { submitNewAngle } from '../api';

const ANGLE_PRESETS = [
  { label: 'Worm\'s Eye View', value: "extreme low angle looking up, worm's eye view, dramatic perspective" },
  { label: 'Bird\'s Eye', value: "aerial bird's eye view from above, looking down at the building" },
  { label: 'Street Level', value: "street level view, pedestrian perspective, eye-level shot" },
  { label: 'Corner View', value: "dynamic corner view, 45-degree angle, showing two facades" },
  { label: 'Distant Skyline', value: "distant view from far away, full building in skyline context" },
];

export default function NewAngleTab() {
  const [renderFile, setRenderFile] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [anglePrompt, setAnglePrompt] = useState('');
  const [stylePrompt, setStylePrompt] = useState('');
  const [jobId, setJobId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const canSubmit = renderFile && (anglePrompt.trim() || referenceFile);

  const handleSubmit = async () => {
    if (!renderFile) return;
    setLoading(true);
    try {
      const { jobId } = await submitNewAngle(renderFile, referenceFile, anglePrompt, stylePrompt);
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
        Generate a new camera angle of a building from an existing render. Provide an angle
        reference image or describe the desired viewpoint via prompt.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <DropZone
          label="Source Render"
          file={renderFile}
          onFile={setRenderFile}
          hint="Existing render or photo of the building"
        />
        <DropZone
          label="Angle Reference (optional)"
          file={referenceFile}
          onFile={setReferenceFile}
          hint="Reference image showing desired camera angle"
        />
      </div>

      {/* Angle presets */}
      <div className="flex flex-col gap-2">
        <label className="text-xs font-semibold uppercase tracking-widest text-gray-400">
          Quick Angle Presets
        </label>
        <div className="flex flex-wrap gap-2">
          {ANGLE_PRESETS.map((p) => (
            <button
              key={p.label}
              onClick={() => setAnglePrompt(p.value)}
              className={`text-xs px-3 py-1.5 rounded-lg border transition-all
                ${anglePrompt === p.value
                  ? 'border-yellow-500 bg-yellow-500/10 text-yellow-400'
                  : 'border-gray-700 bg-gray-900/50 text-gray-400 hover:border-gray-500'
                }`}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <PromptInput
        label="Angle & Composition Prompt"
        value={anglePrompt}
        onChange={setAnglePrompt}
        placeholder="e.g. dramatic low-angle view from plaza level, looking up at the towers"
      />

      <PromptInput
        label="Style & Materiality Prompt"
        value={stylePrompt}
        onChange={setStylePrompt}
        placeholder="e.g. photorealistic, golden hour, glass and gold facade, lush greenery"
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
            <Wand2 size={15} /> Generate New Angle
          </>
        )}
      </button>

      <ResultPanel jobId={jobId} />
    </div>
  );
}
