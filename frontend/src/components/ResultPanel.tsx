import { useEffect, useRef, useState } from 'react';
import { Download, Paintbrush } from 'lucide-react';
import { pollJob } from '../api';
import InpaintCanvas from './InpaintCanvas';

interface Props {
  jobId: string | null;
}

export default function ResultPanel({ jobId }: Props) {
  const [status, setStatus] = useState<string>('pending');
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [inpaintMode, setInpaintMode] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!jobId) return;
    setStatus('pending');
    setOutputUrl(null);
    setError(null);
    setInpaintMode(false);

    intervalRef.current = setInterval(async () => {
      try {
        const result = await pollJob(jobId);
        setStatus(result.status);
        if (result.status === 'done' && result.output_url) {
          setOutputUrl(result.output_url);
          clearInterval(intervalRef.current!);
        } else if (result.status === 'error') {
          setError(result.error ?? 'Unknown error');
          clearInterval(intervalRef.current!);
        }
      } catch {
        // will retry
      }
    }, 2000);

    return () => clearInterval(intervalRef.current!);
  }, [jobId]);

  if (!jobId) return null;

  return (
    <div className="mt-8 rounded-2xl border border-gray-800 bg-gray-900/60 overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3 border-b border-gray-800">
        <span className="text-sm font-semibold text-gray-300">Result</span>
        {outputUrl && (
          <div className="flex gap-2">
            <button
              onClick={() => setInpaintMode(!inpaintMode)}
              className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-all
                ${inpaintMode ? 'bg-yellow-500 text-black' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'}`}
            >
              <Paintbrush size={13} />
              {inpaintMode ? 'Done Painting' : 'Inpaint'}
            </button>
            <a
              href={outputUrl}
              download="render.png"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 transition-all"
            >
              <Download size={13} />
              Download
            </a>
          </div>
        )}
      </div>

      <div className="p-5">
        {(status === 'pending' || status === 'processing') && (
          <div className="flex flex-col items-center justify-center py-16 gap-4">
            <div className="w-10 h-10 border-2 border-yellow-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-gray-500 capitalize">{status}…</span>
          </div>
        )}

        {status === 'error' && (
          <div className="py-8 text-center text-red-400 text-sm">{error}</div>
        )}

        {status === 'done' && outputUrl && (
          inpaintMode ? (
            <InpaintCanvas
              imageUrl={outputUrl}
              onDone={() => setInpaintMode(false)}
              onNewResult={(url) => {
                setOutputUrl(url);
                setInpaintMode(false);
              }}
            />
          ) : (
            <img
              src={outputUrl}
              alt="Generated render"
              className="w-full rounded-xl object-contain max-h-[600px]"
            />
          )
        )}
      </div>
    </div>
  );
}
