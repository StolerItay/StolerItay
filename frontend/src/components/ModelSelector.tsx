interface Model {
  id: string;
  name: string;
  tag?: 'recommended' | 'fast' | 'fallback' | 'precise';
  note: string;
}

interface Props {
  models: Model[];
  value: string;
  onChange: (id: string) => void;
}

const TAG_STYLES: Record<string, string> = {
  recommended: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/40',
  fast: 'bg-blue-500/20 text-blue-400 border-blue-500/40',
  fallback: 'bg-gray-700/40 text-gray-500 border-gray-600/40',
  precise: 'bg-purple-500/20 text-purple-400 border-purple-500/40',
};

export default function ModelSelector({ models, value, onChange }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <label className="text-xs font-semibold uppercase tracking-widest text-gray-400">
        Model
      </label>
      <div className="flex flex-col gap-2">
        {models.map((m) => (
          <button
            key={m.id}
            onClick={() => onChange(m.id)}
            className={`flex items-start justify-between gap-3 px-4 py-3 rounded-xl border text-left transition-all
              ${value === m.id
                ? 'border-yellow-500/60 bg-yellow-500/5'
                : 'border-gray-800 bg-gray-900/40 hover:border-gray-700'
              }`}
          >
            <div className="flex flex-col gap-0.5">
              <span className={`text-sm font-medium ${value === m.id ? 'text-white' : 'text-gray-300'}`}>
                {m.name}
              </span>
              <span className="text-xs text-gray-500">{m.note}</span>
            </div>
            {m.tag && (
              <span className={`shrink-0 text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full border ${TAG_STYLES[m.tag]}`}>
                {m.tag}
              </span>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}
