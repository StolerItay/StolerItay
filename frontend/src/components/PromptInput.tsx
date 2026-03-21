interface Props {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  label?: string;
}

export default function PromptInput({ value, onChange, placeholder, label }: Props) {
  return (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label className="text-xs font-semibold uppercase tracking-widest text-gray-400">
          {label}
        </label>
      )}
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={3}
        placeholder={placeholder ?? 'Describe materiality, texture, render style…'}
        className="w-full rounded-xl bg-gray-900 border border-gray-700 text-gray-200 placeholder-gray-600
          px-4 py-3 text-sm resize-none focus:outline-none focus:border-yellow-500/60 transition-colors"
      />
    </div>
  );
}
