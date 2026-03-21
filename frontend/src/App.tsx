import { useState } from 'react';
import { RefreshCw, Palette, Camera } from 'lucide-react';
import UpdateRenderTab from './components/UpdateRenderTab';
import StyleTransferTab from './components/StyleTransferTab';
import NewAngleTab from './components/NewAngleTab';
import type { Mode } from './types';

const TABS: { id: Mode; label: string; icon: React.ReactNode; description: string }[] = [
  {
    id: 'update-render',
    label: 'Update Render',
    icon: <RefreshCw size={15} />,
    description: 'Replace building with new mass',
  },
  {
    id: 'style-transfer',
    label: 'Style Transfer',
    icon: <Palette size={15} />,
    description: 'Apply style to new mass',
  },
  {
    id: 'new-angle',
    label: 'New Angle',
    icon: <Camera size={15} />,
    description: 'Generate new viewpoint',
  },
];

export default function App() {
  const [mode, setMode] = useState<Mode>('update-render');

  return (
    <div className="min-h-screen bg-[#0f0f13] text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-[#0f0f13]/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-yellow-500 flex items-center justify-center">
              <span className="text-black font-bold text-sm">A</span>
            </div>
            <div>
              <h1 className="text-base font-bold text-white leading-tight">ArchRender AI</h1>
              <p className="text-xs text-gray-500">Architectural Render Editor</p>
            </div>
          </div>
          <span className="text-xs text-gray-600 hidden sm:block">
            Powered by Stable Diffusion + ControlNet
          </span>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-8">
        {/* Tab selector */}
        <div className="grid grid-cols-3 gap-3 mb-8">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setMode(tab.id)}
              className={`flex flex-col items-start gap-1 p-4 rounded-xl border text-left transition-all
                ${mode === tab.id
                  ? 'border-yellow-500/60 bg-yellow-500/5 text-white'
                  : 'border-gray-800 bg-gray-900/40 text-gray-400 hover:border-gray-700 hover:text-gray-300'
                }`}
            >
              <div className={`flex items-center gap-2 font-semibold text-sm
                ${mode === tab.id ? 'text-yellow-400' : ''}`}>
                {tab.icon}
                {tab.label}
              </div>
              <span className="text-xs text-gray-500 hidden sm:block">{tab.description}</span>
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="rounded-2xl border border-gray-800 bg-gray-900/30 p-6">
          {mode === 'update-render' && <UpdateRenderTab />}
          {mode === 'style-transfer' && <StyleTransferTab />}
          {mode === 'new-angle' && <NewAngleTab />}
        </div>

        {/* Footer note */}
        <p className="mt-6 text-center text-xs text-gray-600">
          Set your{' '}
          <code className="bg-gray-800 px-1.5 py-0.5 rounded text-gray-400">REPLICATE_API_TOKEN</code>{' '}
          in{' '}
          <code className="bg-gray-800 px-1.5 py-0.5 rounded text-gray-400">backend/.env</code>{' '}
          to enable AI generation.
        </p>
      </main>
    </div>
  );
}
