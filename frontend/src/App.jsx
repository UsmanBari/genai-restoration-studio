import React, { useState, useEffect, useRef } from 'react';
import { 
  Sparkles, 
  GitFork, 
  Layers, 
  Paintbrush, 
  Upload, 
  Camera,
  Download,
  RefreshCw,
  Sliders,
  Image as ImageIcon,
  CheckCircle2,
  AlertTriangle,
  Info,
  ArrowRight,
  ShieldCheck,
  Zap,
  Activity
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const WORKSPACES = [
  { 
    id: 'universal', 
    name: 'Universal Restoration', 
    icon: Sparkles, 
    badge: 'Task 1',
    desc: 'Single shared autoencoder reconstructing clean targets across all corruption conditions.' 
  },
  { 
    id: 'hard_routing', 
    name: 'Hard-Routed Restoration', 
    icon: GitFork, 
    badge: 'Task 2',
    desc: '4-class corruption classifier dispatching to dedicated specialist autoencoders with clean bypass.' 
  },
  { 
    id: 'soft_moe', 
    name: 'Soft Mixture-of-Experts Restoration', 
    icon: Layers, 
    badge: 'Task 3',
    desc: 'Differentiable temperature-gated soft mixture blending 4 expert branches continuously.' 
  },
  { 
    id: 'face_to_sketch', 
    name: 'Face-to-Sketch Generator', 
    icon: Paintbrush, 
    badge: 'Task 4',
    desc: 'FS2K paired conditional GAN generator conditioned on Style 1, Style 2, or Style 3.' 
  }
];

const STYLES = [
  { id: 0, label: 'Style 1', subtitle: 'Pencil / Classic', badge: 'Style ID: 0' },
  { id: 1, label: 'Style 2', subtitle: 'Sketch / Artistic', badge: 'Style ID: 1' },
  { id: 2, label: 'Style 3', subtitle: 'Caricature / Graphic', badge: 'Style ID: 2' },
];

const CORRUPTION_OPTIONS = [
  { id: 'clean', label: 'Clean (None)' },
  { id: 'salt_and_pepper', label: 'Salt & Pepper Noise' },
  { id: 'gaussian_blur', label: 'Gaussian Blur' },
  { id: 'rectangular_occlusion', label: 'Rectangular Occlusion' }
];

const SEVERITY_TIERS = [
  { id: 'low', label: 'Low Severity' },
  { id: 'medium', label: 'Medium Severity' },
  { id: 'high', label: 'High Severity' }
];

export default function App() {
  const [activeTab, setActiveTab] = useState('universal');
  const [health, setHealth] = useState(null);
  const [loadingHealth, setLoadingHealth] = useState(false);
  
  // Image & Processing state
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [resultMeta, setResultMeta] = useState(null);
  
  // Gallery
  const [samples, setSamples] = useState([]);
  
  // Workspace controls
  const [selectedStyle, setSelectedStyle] = useState(0);
  const [selectedCorruption, setSelectedCorruption] = useState('gaussian_blur');
  const [selectedTier, setSelectedTier] = useState('medium');
  const [corrupting, setCorrupting] = useState(false);
  
  // Webcam state
  const [webcamActive, setWebcamActive] = useState(false);
  const videoRef = useRef(null);

  useEffect(() => {
    fetchHealth();
    fetchSamples();
  }, []);

  const fetchHealth = async () => {
    setLoadingHealth(true);
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      } else {
        setHealth({ status: 'offline' });
      }
    } catch {
      setHealth({ status: 'offline' });
    } finally {
      setLoadingHealth(false);
    }
  };

  const fetchSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/samples`);
      if (res.ok) {
        const data = await res.json();
        setSamples(data.samples || []);
      }
    } catch (e) {
      console.warn('Could not fetch samples:', e);
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

  const handleSelectSample = (sample) => {
    const sampleUrl = sample.url.startsWith('http') ? sample.url : `${API_BASE}${sample.url}`;
    setPreviewUrl(sampleUrl);
    setOutputImage(null);
    setResultMeta(null);
    
    // Convert static image URL to File object for backend endpoints
    fetch(sampleUrl)
      .then(res => res.blob())
      .then(blob => {
        const file = new File([blob], sample.id, { type: 'image/png' });
        setSelectedFile(file);
      })
      .catch(err => console.error('Failed to load sample blob:', err));
  };


  const handleApplyCorruption = async () => {
    if (!selectedFile) return;
    setCorrupting(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('corruption_type', selectedCorruption);
      formData.append('severity_tier', selectedTier);

      const res = await fetch(`${API_BASE}/api/corrupt`, {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const data = await res.json();
        const b64Url = `data:image/png;base64,${data.corrupted_image_base64}`;
        setPreviewUrl(b64Url);
        setOutputImage(null);
        setResultMeta(null);
        
        // Update selectedFile with corrupted blob
        const blobRes = await fetch(b64Url);
        const blob = await blobRes.blob();
        setSelectedFile(new File([blob], 'corrupted_input.png', { type: 'image/png' }));
      }
    } catch (err) {
      alert('Corruption simulation failed: ' + err.message);
    } finally {
      setCorrupting(false);
    }
  };

  const handleRunInference = async () => {
    if (!selectedFile) return;
    setProcessing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    let endpoint = `${API_BASE}/api/universal-restoration`;
    if (activeTab === 'hard_routing') endpoint = `${API_BASE}/api/hard-routing`;
    if (activeTab === 'soft_moe') endpoint = `${API_BASE}/api/soft-mixture`;
    if (activeTab === 'face_to_sketch') {
      endpoint = `${API_BASE}/api/face-to-sketch`;
      formData.append('style_id', selectedStyle.toString());
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
        const errData = await res.json().catch(() => ({}));
        alert(`Inference failed: ${errData.detail || res.statusText}`);
      }
    } catch (err) {
      alert(`Could not reach backend API at ${API_BASE}. Make sure the server is running.`);
    } finally {
      setProcessing(false);
    }
  };

  const startWebcam = async () => {
    setWebcamActive(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 256, height: 256 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      alert('Webcam access error: ' + err.message);
      setWebcamActive(false);
    }
  };

  const captureWebcam = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, 128, 128);
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], 'webcam_portrait.png', { type: 'image/png' });
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setOutputImage(null);
        setResultMeta(null);
        stopWebcam();
      }
    }, 'image/png');
  };

  const stopWebcam = () => {
    if (videoRef.current?.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
    }
    setWebcamActive(false);
  };

  const handleDownloadOutput = () => {
    if (!outputImage) return;
    const link = document.createElement('a');
    link.href = outputImage;
    link.download = `${activeTab}_output_${Date.now()}.png`;
    link.click();
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800/80 bg-slate-900/70 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                GenAI Restoration & Synthesis Studio
              </h1>
              <p className="text-[11px] text-slate-400">Assignment 1 Multi-Model Workspace (Tasks 1–4 ONNX Runtime)</p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-800/80 border border-slate-700/60 text-xs shadow-inner">
              <div className={`w-2 h-2 rounded-full ${health?.status === 'ok' ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
              <span className="text-slate-300">
                Backend: <strong className={health?.status === 'ok' ? 'text-emerald-400' : 'text-rose-400'}>
                  {health?.status === 'ok' ? 'Online (CPU)' : 'Offline'}
                </strong>
              </span>
              <button onClick={fetchHealth} disabled={loadingHealth} className="hover:text-cyan-400 transition-colors ml-1 p-0.5" title="Refresh health status">
                <RefreshCw className={`w-3.5 h-3.5 ${loadingHealth ? 'animate-spin' : ''}`} />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 flex-1 w-full flex flex-col space-y-6">
        {/* Navigation Tabs (Exact Spec Names) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {WORKSPACES.map((tab) => {
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
                className={`relative flex items-start p-4 rounded-xl border text-left transition-all duration-200 ${
                  isActive
                    ? 'bg-gradient-to-b from-cyan-500/15 to-blue-600/10 border-cyan-500/60 shadow-lg shadow-cyan-500/10 ring-1 ring-cyan-500/40'
                    : 'bg-slate-900/50 border-slate-800/80 hover:bg-slate-850 hover:border-slate-700'
                }`}
              >
                <div className={`p-2.5 rounded-lg mr-3 flex-shrink-0 ${
                  isActive ? 'bg-cyan-500 text-slate-950 font-bold' : 'bg-slate-800 text-slate-400'
                }`}>
                  <Icon className="w-5 h-5" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <h3 className={`text-xs font-bold truncate ${isActive ? 'text-cyan-300' : 'text-slate-200'}`}>
                      {tab.name}
                    </h3>
                    <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                      {tab.badge}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">{tab.desc}</p>
                </div>
              </button>
            );
          })}
        </div>

        {/* Workspace Body */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
          {/* Controls Column */}
          <div className="lg:col-span-4 flex flex-col space-y-4">
            {/* Input & Parameters Card */}
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-xl shadow-xl flex flex-col space-y-4">
              <h2 className="text-sm font-semibold text-slate-200 flex items-center">
                <Sliders className="w-4 h-4 mr-2 text-cyan-400" />
                Input & Controls
              </h2>

              {/* Sample Presets Gallery */}
              {samples.length > 0 && (
                <div>
                  <label className="block text-[11px] font-medium text-slate-400 mb-1.5">Preset Sample Gallery</label>
                  <div className="grid grid-cols-3 gap-2">
                    {samples.map((s) => (
                      <button
                        key={s.id}
                        type="button"
                        onClick={() => handleSelectSample(s)}
                        className="group flex flex-col items-center p-1.5 rounded-lg border border-slate-800 bg-slate-950/60 hover:border-cyan-500/50 hover:bg-slate-850 transition-all text-center"
                      >
                        <img 
                          src={s.url.startsWith('http') ? s.url : `${API_BASE}${s.url}`} 
                          alt={s.name}
                          className="w-12 h-12 object-cover rounded mb-1 border border-slate-800 group-hover:border-cyan-500/40"
                        />

                        <span className="text-[10px] text-slate-300 font-medium truncate w-full">{s.name}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* File Upload Dropzone */}
              <div>
                <label className="block text-[11px] font-medium text-slate-400 mb-1.5">Upload Custom Image</label>
                <div className="flex space-x-2">
                  <label className="flex-1 flex flex-col items-center justify-center h-28 border-2 border-dashed border-slate-700/80 rounded-xl cursor-pointer hover:border-cyan-500/60 hover:bg-slate-850/50 transition-all group">
                    <Upload className="w-5 h-5 text-slate-500 group-hover:text-cyan-400 mb-1 transition-colors" />
                    <span className="text-xs text-slate-300 font-medium">Choose Image</span>
                    <span className="text-[10px] text-slate-500">128×128 RGB recommended</span>
                    <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
                  </label>

                  {activeTab === 'face_to_sketch' && (
                    <button
                      type="button"
                      onClick={startWebcam}
                      className="w-24 flex flex-col items-center justify-center border border-slate-700 rounded-xl bg-slate-800/40 hover:bg-cyan-500/10 hover:border-cyan-500/50 text-slate-300 hover:text-cyan-300 transition-all"
                    >
                      <Camera className="w-5 h-5 mb-1" />
                      <span className="text-[10px] font-medium">Webcam</span>
                    </button>
                  )}
                </div>
              </div>

              {/* Corruption Simulator (For Tasks 1, 2, 3) */}
              {activeTab !== 'face_to_sketch' && (
                <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-2.5">
                  <span className="text-[11px] font-semibold text-slate-300 flex items-center">
                    <Zap className="w-3.5 h-3.5 text-amber-400 mr-1.5" />
                    Simulate Programmatic Corruption
                  </span>
                  
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <select
                      value={selectedCorruption}
                      onChange={(e) => setSelectedCorruption(e.target.value)}
                      className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 text-xs focus:ring-1 focus:ring-cyan-500 outline-none"
                    >
                      {CORRUPTION_OPTIONS.map(c => (
                        <option key={c.id} value={c.id}>{c.label}</option>
                      ))}
                    </select>

                    <select
                      value={selectedTier}
                      onChange={(e) => setSelectedTier(e.target.value)}
                      className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 text-xs focus:ring-1 focus:ring-cyan-500 outline-none"
                    >
                      {SEVERITY_TIERS.map(t => (
                        <option key={t.id} value={t.id}>{t.label}</option>
                      ))}
                    </select>
                  </div>

                  <button
                    type="button"
                    disabled={!selectedFile || corrupting}
                    onClick={handleApplyCorruption}
                    className="w-full py-1.5 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition-colors flex items-center justify-center"
                  >
                    {corrupting ? <RefreshCw className="w-3 h-3 animate-spin mr-1.5" /> : null}
                    Apply Runtime Corruption
                  </button>
                </div>
              )}

              {/* Task 4: Style Condition Selector */}
              {activeTab === 'face_to_sketch' && (
                <div className="space-y-2">
                  <label className="block text-[11px] font-medium text-slate-400">
                    FS2K Categorical Style Condition
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    {STYLES.map((st) => (
                      <button
                        key={st.id}
                        type="button"
                        onClick={() => setSelectedStyle(st.id)}
                        className={`p-2.5 rounded-xl border text-left transition-all ${
                          selectedStyle === st.id
                            ? 'bg-cyan-500/20 border-cyan-500 text-cyan-200 ring-1 ring-cyan-500/50'
                            : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-850'
                        }`}
                      >
                        <span className="block text-xs font-bold">{st.label}</span>
                        <span className="block text-[10px] text-slate-400 truncate">{st.subtitle}</span>
                        <span className="block text-[9px] font-mono text-cyan-400/80 mt-1">{st.badge}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Run Button */}
              <button
                disabled={!selectedFile || processing}
                onClick={handleRunInference}
                className={`w-full py-3 px-4 rounded-xl text-xs font-bold flex items-center justify-center transition-all ${
                  !selectedFile || processing
                    ? 'bg-slate-800/80 text-slate-500 cursor-not-allowed border border-slate-700/50'
                    : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 shadow-lg shadow-cyan-500/25 active:scale-[0.99]'
                }`}
              >
                {processing ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Executing ONNX Inference...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    {activeTab === 'face_to_sketch' ? 'Synthesize Facial Sketch' : 'Run Deep Restoration'}
                  </>
                )}
              </button>
            </div>

            {/* Inference Diagnostics Card */}
            {resultMeta && (
              <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-xl shadow-xl space-y-3">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center justify-between">
                  <span>Execution Diagnostics</span>
                  <span className="text-[10px] font-mono text-emerald-400 px-1.5 py-0.5 rounded bg-emerald-950/80 border border-emerald-800/50">
                    {resultMeta.latency_ms} ms
                  </span>
                </h3>

                {/* Hard Routing Diagnostics */}
                {activeTab === 'hard_routing' && resultMeta.probabilities && (
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400">Predicted Corruption</span>
                      <span className="text-cyan-300 font-bold uppercase">{resultMeta.predicted_corruption}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400">Selected Expert</span>
                      <span className="text-emerald-400 font-mono text-[11px]">{resultMeta.selected_expert}</span>
                    </div>
                    <div className="pt-1">
                      <span className="text-[11px] text-slate-400 block mb-1.5 font-medium">Classifier Probabilities:</span>
                      <div className="space-y-1.5">
                        {Object.entries(resultMeta.probabilities).map(([name, prob]) => (
                          <div key={name} className="space-y-0.5">
                            <div className="flex justify-between text-[10px]">
                              <span className="text-slate-400">{name}</span>
                              <span className="text-slate-200 font-mono">{(prob * 100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                              <div 
                                className={`h-full rounded-full ${name === resultMeta.predicted_corruption ? 'bg-cyan-400' : 'bg-slate-600'}`}
                                style={{ width: `${prob * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Soft MoE Diagnostics */}
                {activeTab === 'soft_moe' && resultMeta.routing_weights && (
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400">Dominant Expert</span>
                      <span className="text-cyan-300 font-bold uppercase">{resultMeta.dominant_expert}</span>
                    </div>
                    <div className="pt-1">
                      <span className="text-[11px] text-slate-400 block mb-1.5 font-medium">Continuous Gating Weights:</span>
                      <div className="space-y-1.5">
                        {Object.entries(resultMeta.routing_weights).map(([name, weight]) => (
                          <div key={name} className="space-y-0.5">
                            <div className="flex justify-between text-[10px]">
                              <span className="text-slate-400">{name.replace('_', ' ')}</span>
                              <span className="text-slate-200 font-mono">{(weight * 100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                              <div 
                                className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 rounded-full"
                                style={{ width: `${weight * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Face-to-Sketch Diagnostics */}
                {activeTab === 'face_to_sketch' && (
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400">Applied Style</span>
                      <span className="text-cyan-300 font-semibold">{resultMeta.style_name}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400">Conditioning Tensor</span>
                      <span className="text-slate-200 font-mono">[Style ID: {resultMeta.style_id}]</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Visualization Canvas Column */}
          <div className="lg:col-span-8 flex flex-col space-y-4">
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-xl shadow-xl flex-1 flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-bold text-slate-200 flex items-center">
                  <Activity className="w-4 h-4 mr-2 text-cyan-400" />
                  {WORKSPACES.find(w => w.id === activeTab)?.name} Comparative Display
                </h2>
                
                {outputImage && (
                  <button
                    onClick={handleDownloadOutput}
                    className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-cyan-300 border border-slate-700 transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download Result</span>
                  </button>
                )}
              </div>

              {/* Side-by-Side Dual View */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 flex-1 items-stretch">
                {/* Input Panel */}
                <div className="flex flex-col bg-slate-950/80 border border-slate-800/90 rounded-2xl p-5 items-center justify-center min-h-[340px] shadow-inner">
                  <span className="text-xs font-semibold text-slate-400 mb-3 tracking-wider uppercase">
                    Input Image
                  </span>
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Input Preview"
                      className="max-h-72 object-contain rounded-xl border border-slate-800 shadow-2xl"
                    />
                  ) : (
                    <div className="text-center text-slate-600">
                      <ImageIcon className="w-14 h-14 mx-auto mb-2 opacity-40" />
                      <p className="text-xs">Select or upload an image to begin</p>
                    </div>
                  )}
                </div>

                {/* Output Panel */}
                <div className="flex flex-col bg-slate-950/80 border border-slate-800/90 rounded-2xl p-5 items-center justify-center min-h-[340px] shadow-inner">
                  <span className="text-xs font-semibold text-slate-400 mb-3 tracking-wider uppercase">
                    {activeTab === 'face_to_sketch' ? 'Synthesized Sketch' : 'Restored Image'}
                  </span>
                  {outputImage ? (
                    <img
                      src={outputImage}
                      alt="Output Result"
                      className="max-h-72 object-contain rounded-xl border border-cyan-500/40 shadow-2xl shadow-cyan-500/10"
                    />
                  ) : (
                    <div className="text-center text-slate-600">
                      <Sparkles className="w-14 h-14 mx-auto mb-2 opacity-30" />
                      <p className="text-xs">Generated output will appear here</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Webcam Modal */}
      {webcamActive && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 max-w-sm w-full space-y-4 shadow-2xl">
            <h3 className="text-sm font-bold text-slate-200 flex items-center">
              <Camera className="w-4 h-4 mr-2 text-cyan-400" />
              Capture Facial Portrait
            </h3>
            <div className="relative rounded-xl overflow-hidden border border-slate-800 bg-black aspect-square flex items-center justify-center">
              <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover" />
            </div>
            <div className="flex space-x-2">
              <button
                type="button"
                onClick={captureWebcam}
                className="flex-1 py-2 px-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs transition-colors"
              >
                Capture Photo
              </button>
              <button
                type="button"
                onClick={stopWebcam}
                className="py-2 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

