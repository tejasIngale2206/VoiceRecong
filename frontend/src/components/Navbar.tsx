import React from 'react';
import { Shield, ShieldAlert, Mic, MicOff, Radio, Zap, AlertTriangle } from 'lucide-react';
import type { ThreatStatus } from '../types';

interface NavbarProps {
  threatStatus: ThreatStatus;
  isStreaming: boolean;
  isConnected: boolean;
  latencyMs?: number;
  onToggleProtection: () => void;
  onSimulateAttack: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  threatStatus,
  isStreaming,
  isConnected,
  latencyMs,
  onToggleProtection,
  onSimulateAttack,
}) => {
  const isAlert = threatStatus === 'SYNTHETIC_ALERT';

  return (
    <header className="border-b border-slate-800 bg-[#0b0f19]/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Logo and branding */}
        <div className="flex items-center space-x-3">
          <div
            className={`p-2 rounded-xl transition-all duration-300 ${
              isAlert
                ? 'bg-red-500/20 text-red-400 ring-2 ring-red-500/50 animate-pulse'
                : 'bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/30'
            }`}
          >
            {isAlert ? <ShieldAlert className="w-6 h-6" /> : <Shield className="w-6 h-6" />}
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-lg font-bold tracking-wider text-white">VOICESHIELD</span>
              <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                AI CORE
              </span>
              <span className="text-xs px-1.5 py-0.5 rounded text-slate-400 border border-slate-800">
                SIH26104
              </span>
            </div>
            <p className="text-[11px] text-slate-400 hidden sm:block">
              Zero-Trust Real-Time Voice Anti-Spoofing & Deepfake Detection Engine
            </p>
          </div>
        </div>

        {/* Action Controls and Live Badges */}
        <div className="flex items-center space-x-3 sm:space-x-4">
          {/* Latency Pill */}
          {latencyMs !== undefined && (
            <div className="hidden md:flex items-center space-x-1.5 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs font-mono">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-slate-400">Latency:</span>
              <span className={latencyMs < 50 ? 'text-emerald-400 font-bold' : 'text-amber-400'}>
                {latencyMs.toFixed(1)}ms
              </span>
            </div>
          )}

          {/* WebSocket Status Indicator */}
          <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs font-mono">
            <Radio
              className={`w-3.5 h-3.5 ${
                isConnected ? 'text-emerald-400 animate-pulse' : 'text-slate-500'
              }`}
            />
            <span className={isConnected ? 'text-emerald-400' : 'text-slate-500'}>
              {isConnected ? 'LIVE WS' : 'OFFLINE'}
            </span>
          </div>

          {/* Simulation Attack Button for Live Demonstrations */}
          {isStreaming && (
            <button
              onClick={onSimulateAttack}
              title="Simulate synthetic attack frames to test hysteresis trigger"
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-red-400 bg-red-950/40 border border-red-800/60 hover:bg-red-900/50 hover:border-red-600 transition"
            >
              <AlertTriangle className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Simulate Attack</span>
            </button>
          )}

          {/* Start/Stop Main Button */}
          <button
            onClick={onToggleProtection}
            className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all shadow-lg ${
              isStreaming
                ? 'bg-red-600/80 hover:bg-red-600 text-white shadow-red-900/30'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/30 hover:scale-[1.02]'
            }`}
          >
            {isStreaming ? (
              <>
                <MicOff className="w-4 h-4" />
                <span>Stop Protection</span>
              </>
            ) : (
              <>
                <Mic className="w-4 h-4" />
                <span>Start Protection</span>
              </>
            )}
          </button>
        </div>
      </div>
    </header>
  );
};
