import React, { useState, useEffect } from 'react';
import { 
  Sparkles, 
  Cpu, 
  GitFork, 
  Layers, 
  Paintbrush, 
  Upload, 
  CheckCircle, 
  AlertCircle, 
  RefreshCw,
  Sliders,
  Image as ImageIcon
} from 'lucide-react';

const TABS = [
  { id: 'universal', name: 'Universal Restoration', icon: Sparkles, desc: 'Single End-to-End Autoencoder (Task 1)' },
  { id: 'routing', name: 'Hard-Routed Restoration', icon: GitFork, desc: 'Classifier + 3 Dedicated Expert Autoencoders (Task 2)' },
  { id: 'moe', name: 'Soft Mixture-of-Experts Restoration', icon: Layers, desc: 'Soft Gating Network with Weighted Blending (Task 3)' },
  { id: 'sketch', name: 'Face-to-Sketch Generator', icon: Paintbrush, desc: 'FS2K Paired pix2pix GAN Synthesis (Task 4)' }
];

export default function App() {
  const [activeTab, setActiveTab] = useState('universal');
  const [health, setHealth] = useState(null);
  const [loadingHealth, setLoadingHealth] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [resultMeta, setResultMeta] = useState(null);
  const [selectedStyle, setSelectedStyle] = useState(0);

  useEffect(() => {
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    setLoadingHealth(true);
    try {
      const res = await fetch('http://localhost:8000/health');
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      } else {
        setHealth({ status: 'offline' });
      }
    } catch (e) {
      setHealth({ status: 'offline' });
    } finally {
      setLoadingHealth(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setOutputImage(null);
      setResultMeta(null);
    }
  };

  const handleRunInference = async () => {
    if (!selectedFile) return;
    setProcessing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    let endpoint = 'http://localhost:8000/universal-restoration';
    if (activeTab === 'routing') endpoint = 'http://localhost:8000/hard-routing';
    if (activeTab === 'moe') endpoint = 'http://localhost:8000/soft-mixture';
    if (activeTab === 'sketch') {
      endpoint = 'http://localhost:8000/face-to-sketch';
      formData.append('style', selectedStyle.toString());
    }

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const data = await res.json();
        const base64Data = data.output_image_base64 || data.sketch_image_base64;
        if (base64Data) {
          setOutputImage(`data:image/png;base64,${base64Data}`);
        }
        setResultMeta(data);
      } else {
        alert('Inference request failed.');
      }
    } catch (err) {
      console.error(err);
      alert('Could not reach backend API at http://localhost:8000.');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                GenAI Restoration & Synthesis Studio
              </h1>
              <p className="text-xs text-slate-400">Deep Learning Vision Workspace</p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-800/60 border border-slate-700/50 text-xs">
              <div className={`w-2 h-2 rounded-full ${health?.status === 'ok' ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
              <span className="text-slate-300">
                Backend: <strong className={health?.status === 'ok' ? 'text-emerald-400' : 'text-rose-400'}>
                  {health?.status === 'ok' ? 'Online' : 'Offline'}
                </strong>
              </span>
              <button onClick={fetchHealth} disabled={loadingHealth} className="hover:text-cyan-400 transition-colors ml-1">
                <RefreshCw className={`w-3.5 h-3.5 ${loadingHealth ? 'animate-spin' : ''}`} />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main layout */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full flex flex-col">
        {/* Navigation Tabs */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id);
                  setOutputImage(null);
                  setResultMeta(null);
                }}
                className={`flex items-start p-4 rounded-xl border text-left transition-all duration-200 ${
                  isActive
                    ? 'bg-gradient-to-b from-cyan-500/10 to-blue-600/10 border-cyan-500/50 shadow-lg shadow-cyan-500/10'
                    : 'bg-slate-900/40 border-slate-800/60 hover:bg-slate-850 hover:border-slate-700'
                }`}
              >
                <div className={`p-2.5 rounded-lg mr-3.5 ${
                  isActive ? 'bg-cyan-500 text-slate-950 font-semibold' : 'bg-slate-800 text-slate-400'
                }`}>
                  <Icon className="w-5 h-5" />
                </div>
                <div>
                  <h3 className={`text-sm font-semibold ${isActive ? 'text-cyan-300' : 'text-slate-200'}`}>
                    {tab.name}
                  </h3>
                  <p className="text-xs text-slate-400 mt-1 line-clamp-2">{tab.desc}</p>
                </div>
              </button>
            );
          })}
        </div>

        {/* Active Workspace */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Controls & Upload Column */}
          <div className="lg:col-span-4 flex flex-col space-y-6">
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-xl">
              <h2 className="text-base font-semibold text-slate-200 mb-4 flex items-center">
                <Sliders className="w-4 h-4 mr-2 text-cyan-400" />
                Input & Parameters
              </h2>

              {/* Upload Dropzone */}
              <div className="mb-5">
                <label className="block text-xs font-medium text-slate-400 mb-2">Source Image</label>
                <label className="flex flex-col items-center justify-center w-full h-40 border-2 border-dashed border-slate-700 rounded-xl cursor-pointer hover:border-cyan-500/50 hover:bg-slate-800/30 transition-all group">
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <Upload className="w-8 h-8 text-slate-500 group-hover:text-cyan-400 mb-2 transition-colors" />
                    <p className="text-xs text-slate-300 font-medium">Click or drag image to upload</p>
                    <p className="text-[10px] text-slate-500 mt-1">PNG, JPG, JPEG (RGB)</p>
                  </div>
                  <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
                </label>
              </div>

              {/* Task 4 Specific Style Selector */}
              {activeTab === 'sketch' && (
                <div className="mb-5">
                  <label className="block text-xs font-medium text-slate-400 mb-2">Sketch Style Tier</label>
                  <div className="grid grid-cols-3 gap-2">
                    {[0, 1, 2].map((s) => (
                      <button
                        key={s}
                        type="button"
                        onClick={() => setSelectedStyle(s)}
                        className={`py-2 px-3 text-xs font-medium rounded-lg border transition-all ${
                          selectedStyle === s
                            ? 'bg-cyan-500/20 border-cyan-500 text-cyan-300'
                            : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Style {s}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Run Button */}
              <button
                disabled={!selectedFile || processing}
                onClick={handleRunInference}
                className={`w-full py-3 px-4 rounded-xl font-medium text-sm flex items-center justify-center transition-all ${
                  !selectedFile || processing
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                    : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-semibold shadow-lg shadow-cyan-500/20'
                }`}
              >
                {processing ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    Run Restoration / Synthesis
                  </>
                )}
              </button>
            </div>

            {/* Inference Metadata Card */}
            {resultMeta && (
              <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-xl">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
                  Inference Diagnostics
                </h3>
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between py-1 border-b border-slate-800">
                    <span className="text-slate-400">Endpoint Status</span>
                    <span className="text-emerald-400 font-medium">{resultMeta.status}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-800">
                    <span className="text-slate-400">Latency</span>
                    <span className="text-slate-200 font-mono">{resultMeta.latency_ms} ms</span>
                  </div>
                  {resultMeta.detected_corruption && (
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400">Detected Corruption</span>
                      <span className="text-cyan-300 font-semibold">{resultMeta.detected_corruption}</span>
                    </div>
                  )}
                  {resultMeta.confidence && (
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400">Classifier Confidence</span>
                      <span className="text-slate-200 font-mono">{(resultMeta.confidence * 100).toFixed(1)}%</span>
                    </div>
                  )}
                  {resultMeta.expert_weights && (
                    <div className="pt-2">
                      <span className="text-slate-400 block mb-2">Expert Gating Weights:</span>
                      <div className="space-y-1.5">
                        {Object.entries(resultMeta.expert_weights).map(([k, v]) => (
                          <div key={k} className="flex items-center justify-between text-[11px]">
                            <span className="text-slate-400">{k}</span>
                            <span className="text-cyan-300 font-mono">{(v * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Canvas & Comparison Column */}
          <div className="lg:col-span-8 flex flex-col">
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-xl flex-1 flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-base font-semibold text-slate-200">
                  {TABS.find(t => t.id === activeTab)?.name} Workspace
                </h2>
                <span className="text-xs px-2.5 py-1 rounded-md bg-slate-800 text-cyan-400 font-mono">
                  {activeTab === 'sketch' ? '256x256 RGB' : '128x128 RGB'}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 flex-1 items-stretch">
                {/* Input Preview */}
                <div className="flex flex-col bg-slate-950/80 border border-slate-800 rounded-xl p-4 items-center justify-center min-h-[300px]">
                  <span className="text-xs font-medium text-slate-400 mb-3">Input Image</span>
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Input Preview"
                      className="max-h-64 object-contain rounded-lg border border-slate-800 shadow-md"
                    />
                  ) : (
                    <div className="text-center text-slate-600">
                      <ImageIcon className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p className="text-xs">No image loaded</p>
                    </div>
                  )}
                </div>

                {/* Output Restored/Generated */}
                <div className="flex flex-col bg-slate-950/80 border border-slate-800 rounded-xl p-4 items-center justify-center min-h-[300px]">
                  <span className="text-xs font-medium text-slate-400 mb-3">
                    {activeTab === 'sketch' ? 'Generated Sketch' : 'Restored Output'}
                  </span>
                  {outputImage ? (
                    <img
                      src={outputImage}
                      alt="Output Result"
                      className="max-h-64 object-contain rounded-lg border border-cyan-500/30 shadow-lg shadow-cyan-500/10"
                    />
                  ) : (
                    <div className="text-center text-slate-600">
                      <Sparkles className="w-12 h-12 mx-auto mb-2 opacity-30" />
                      <p className="text-xs">Model output will appear here</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
