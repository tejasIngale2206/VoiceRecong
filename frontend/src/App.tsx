import React from 'react';
import { Navbar } from './components/Navbar';
import { SecurityHeader } from './components/SecurityHeader';
import { Oscilloscope } from './components/Oscilloscope';
import { MetricsGrid } from './components/MetricsGrid';
import { ThreatLog } from './components/ThreatLog';
import ForensicInspector from './components/ForensicInspector';
import { useVoiceShieldStream } from './hooks/useVoiceShieldStream';
import { AlertCircle, ShieldCheck, Database } from 'lucide-react';


export const App: React.FC = () => {
  const {
    isConnected,
    isStreaming,
    threatStatus,
    latestResponse,
    logs,
    analyserNode,
    error,
    startProtection,
    stopProtection,
    simulateSyntheticAttack,
    clearLogs,
  } = useVoiceShieldStream('ws://127.0.0.1:8000/ws/stream');

  const handleToggleProtection = () => {
    if (isStreaming) {
      stopProtection();
    } else {
      startProtection();
    }
  };

  return (
    <div className="min-h-screen bg-[#07090e] text-slate-100 flex flex-col font-sans selection:bg-emerald-500/30 selection:text-emerald-300">
      {/* Top Navbar */}
      <Navbar
        threatStatus={threatStatus}
        isStreaming={isStreaming}
        isConnected={isConnected}
        latencyMs={latestResponse?.latency_ms}
        onToggleProtection={handleToggleProtection}
        onSimulateAttack={simulateSyntheticAttack}
      />

      {/* Main Security Dashboard Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Error Boundary / Connection Warning Banner */}
        {error && (
          <div className="p-4 rounded-xl bg-red-950/60 border border-red-500/80 text-red-200 flex items-start space-x-3 text-sm">
            <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <strong className="font-bold">Audio / WebSocket Error:</strong> {error}
              <div className="text-xs text-red-300/80 mt-1">
                Make sure the backend is active on <code className="bg-red-900/40 px-1 py-0.5 rounded font-mono">http://127.0.0.1:8000</code> and microphone permissions are granted.
              </div>
            </div>
          </div>
        )}

        {/* Dual-State Threat Header (Emerald Green <-> Crimson Red) */}
        <SecurityHeader
          threatStatus={threatStatus}
          isStreaming={isStreaming}
          metrics={latestResponse?.metrics}
        />

        {/* Real-time Oscilloscope with live WaveSurfer & Canvas rendering */}
        <Oscilloscope
          analyserNode={analyserNode}
          threatStatus={threatStatus}
          isStreaming={isStreaming}
          energyDb={latestResponse?.metrics?.energy_db}
        />

        {/* Real-time Telemetry Metrics Grid (4 core security cards) */}
        <MetricsGrid
          latestResponse={latestResponse}
          threatStatus={threatStatus}
          isStreaming={isStreaming}
        />

        {/* Forensic Audio File Inspector Component */}
        <ForensicInspector />

        {/* Event Stream Terminal & Architecture Specs */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Real-time Threat Event Log (Takes 2 columns on wide screens) */}
          <div className="lg:col-span-2">
            <ThreatLog logs={logs} onClear={clearLogs} />
          </div>

          {/* System Protocol & Compliance Specs Panel */}
          <div className="rounded-2xl border border-slate-800 bg-[#0d121f] p-5 shadow-xl flex flex-col justify-between">
            <div>
              <h2 className="text-sm font-bold tracking-wider text-slate-200 uppercase font-mono mb-3 flex items-center gap-2">
                <Database className="w-4 h-4 text-cyan-400" />
                Pipeline Architecture
              </h2>

              <ul className="space-y-2.5 text-xs text-slate-400 font-mono">
                <li className="flex items-start space-x-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0" />
                  <span>
                    <strong className="text-slate-200">Audio Ingestion:</strong> 16,000 Hz, 16-bit mono PCM chunks (8,000 samples / 500ms).
                  </span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 mt-1.5 shrink-0" />
                  <span>
                    <strong className="text-slate-200">Zero-Disk Policy:</strong> In-memory volatile RAM buffering (<code className="text-cyan-300">io.BytesIO</code>) exclusively.
                  </span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-purple-400 mt-1.5 shrink-0" />
                  <span>
                    <strong className="text-slate-200">Score Fusion:</strong> 60% RawNet2 1D + 40% ResNet18 Log-Mel Spectrogram.
                  </span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5 shrink-0" />
                  <span>
                    <strong className="text-slate-200">Hysteresis:</strong> 3 consecutive frames &ge; 0.40 to trip alert.
                  </span>
                </li>
                <li className="flex items-start space-x-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5 shrink-0" />
                  <span>
                    <strong className="text-slate-200">Latency Target:</strong> &lt; 50 ms feature extraction + inference runtime.
                  </span>
                </li>
              </ul>
            </div>

            <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] font-mono text-slate-500 flex justify-between items-center">
              <span>VoiceShield AI SIH26104</span>
              <span className="flex items-center gap-1 text-emerald-400 font-bold">
                <ShieldCheck className="w-3.5 h-3.5" />
                Zero-Trust Active
              </span>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-[#070a12] py-4 text-center text-xs text-slate-500 font-mono">
        VoiceShield AI &copy; 2026 — Smart India Hackathon (SIH26104) Zero-Trust Audio Security Engine
      </footer>
    </div>
  );
};

export default App;