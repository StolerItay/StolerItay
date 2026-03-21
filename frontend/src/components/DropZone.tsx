import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, Image } from 'lucide-react';

interface Props {
  label: string;
  accept?: string[];
  file: File | null;
  onFile: (f: File | null) => void;
  hint?: string;
}

export default function DropZone({ label, accept, file, onFile, hint }: Props) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted[0]) onFile(accepted[0]);
    },
    [onFile]
  );

  const acceptObj = accept
    ? Object.fromEntries(accept.map((t) => [t, []]))
    : { 'image/*': [] };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptObj,
    maxFiles: 1,
  });

  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-semibold uppercase tracking-widest text-gray-400">
        {label}
      </label>
      {file ? (
        <div className="relative rounded-xl overflow-hidden border border-yellow-500/30 bg-gray-900">
          {file.type.startsWith('image/') ? (
            <img
              src={URL.createObjectURL(file)}
              alt="preview"
              className="w-full h-40 object-cover"
            />
          ) : (
            <div className="h-40 flex flex-col items-center justify-center gap-2 text-gray-400">
              <Image size={32} />
              <span className="text-sm">{file.name}</span>
            </div>
          )}
          <button
            onClick={() => onFile(null)}
            className="absolute top-2 right-2 bg-black/70 hover:bg-black rounded-full p-1 text-white"
          >
            <X size={14} />
          </button>
        </div>
      ) : (
        <div
          {...getRootProps()}
          className={`h-40 rounded-xl border-2 border-dashed flex flex-col items-center justify-center gap-2 cursor-pointer transition-all
            ${isDragActive ? 'border-yellow-400 bg-yellow-400/5' : 'border-gray-700 hover:border-gray-500 bg-gray-900/50'}`}
        >
          <input {...getInputProps()} />
          <Upload size={24} className="text-gray-500" />
          <span className="text-sm text-gray-500">
            {isDragActive ? 'Drop here' : 'Drop or click to upload'}
          </span>
          {hint && <span className="text-xs text-gray-600">{hint}</span>}
        </div>
      )}
    </div>
  );
}
