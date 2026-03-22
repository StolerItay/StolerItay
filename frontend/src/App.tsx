import { useState } from 'react';
import { RefreshCw, Palette, Camera, Key, CheckCircle, XCircle, Loader } from 'lucide-react';
import axios from 'axios';

const BASE = 'http://localhost:8000/api';
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
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [tokenStatus, setTokenStatus] = useState<'idle' | 'checking' | 'valid' | 'invalid'>('idle');
  const [tokenUser, setTokenUser] = useState('');

  async function validateToken() {
    if (!apiKey) return;
    setTokenStatus('checking');
    try {
      const form = new FormData();
      form.append('token', apiKey);
      const { data } = await axios.post(`${BASE}/validate-token`, form);
      if (data.valid) {
        setTokenStatus('valid');
        setTokenUser(data.username || '');
      } else {
        setTokenStatus('invalid');
      }
    } catch {
      setTokenStatus('invalid');
    }
  }

  return (
    <div className="min-h-screen bg-[#0f0f13] text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-[#0f0f13]/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 shrink-0">
            <div className="w-8 h-8 rounded-lg bg-yellow-500 flex items-center justify-center">
              <span className="text-black font-bold text-sm">A</span>
            </div>
            <div>
              <h1 className="text-base font-bold text-white leading-tight">ArchRender AI</h1>
              <p className="text-xs text-gray-500">Architectural Render Editor</p>
            </div>
          </div>

          {/* API key input */}
          <div className="flex items-center gap-2 flex-1 max-w-sm">
            <Key size={13} className="text-gray-500 shrink-0" />
            <input
              type={showKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => { setApiKey(e.target.value); setTokenStatus('idle'); }}
              placeholder="Replicate API token"
              className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 text-xs text-gray-200
                placeholder-gray-600 focus:outline-none focus:border-yellow-500/60 transition-colors"
            />
            <button
              onClick={() => setShowKey((v) => !v)}
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors shrink-0"
            >
              {showKey ? 'Hide' : 'Show'}
            </button>
            <button
              onClick={validateToken}
              disabled={!apiKey || tokenStatus === 'checking'}
              className="text-xs px-2 py-1 rounded bg-gray-800 border border-gray-700 text-gray-300
                hover:border-yellow-500/60 hover:text-yellow-400 transition-colors shrink-0 disabled:opacity-40"
            >
              {tokenStatus === 'checking' ? <Loader size={11} className="animate-spin" /> : 'Test'}
            </button>
            {tokenStatus === 'valid' && (
              <span className="flex items-center gap-1 text-xs text-green-400 shrink-0">
                <CheckCircle size={13} /> {tokenUser}
              </span>
            )}
            {tokenStatus === 'invalid' && (
              <span className="flex items-center gap-1 text-xs text-red-400 shrink-0">
                <XCircle size={13} /> Invalid
              </span>
            )}
          </div>
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
          {mode === 'update-render' && <UpdateRenderTab apiKey={apiKey} />}
          {mode === 'style-transfer' && <StyleTransferTab apiKey={apiKey} />}
          {mode === 'new-angle' && <NewAngleTab apiKey={apiKey} />}
        </div>

        {!apiKey && (
          <p className="mt-4 text-center text-xs text-yellow-600">
            Enter your Replicate API token above to enable AI generation.
          </p>
        )}
      </main>
    </div>
  );
}
