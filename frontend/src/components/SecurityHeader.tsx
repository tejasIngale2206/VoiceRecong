import React from 'react';
import { ShieldCheck, AlertOctagon, Volume2, ShieldAlert, Cpu } from 'lucide-react';
import type { ThreatStatus, StreamMetrics } from '../types';

interface SecurityHeaderProps {
  threatStatus: ThreatStatus;
  isStreaming: boolean;
  metrics?: StreamMetrics;
}

export const SecurityHeader: React.FC<SecurityHeaderProps> = ({
  threatStatus,
  isStreaming,
  metrics,
}) => {
  const isAlert = threatStatus === 'SYNTHETIC_ALERT';
  const isSilence = threatStatus === 'SILENCE';
  const isSafe = threatStatus === 'SAFE';

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border p-6 transition-all duration-500 shadow-2xl ${
        isAlert
          ? 'bg-gradient-to-r from-red-950/90 via-red-900/60 to-black border-red-500 ring-2 ring-red-500/30'
          : isSafe
          ? 'bg-gradient-to-r from-emerald-950/80 via-slate-900/90 to-black border-emerald-500/50'
          : 'bg-gradient-to-r from-slate-900 via-slate-900/80 to-black border-slate-800'
      }`}
    >
      {/* Background glow effects */}
      <div
        className={`absolute -right-16 -top-16 w-64 h-64 rounded-full blur-3xl opacity-25 pointer-events-none transition-all duration-700 ${
          isAlert ? 'bg-red-500' : isSafe ? 'bg-emerald-500' : 'bg-slate-700'
        }`}
      />

      <div className="relative z-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        {/* State Icon & Main Title */}
        <div className="flex items-center space-x-4">
          <div
            className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-all duration-500 ${
              isAlert
                ? 'bg-red-500 text-white shadow-lg shadow-red-500/40 animate-bounce'
                : isSafe
                ? 'bg-emerald-500/20 border border-emerald-500/40 text-emerald-400'
                : 'bg-slate-800 text-slate-400'
            }`}
          >
            {isAlert ? (
              <AlertOctagon className="w-8 h-8" />
            ) : isSafe ? (
              <ShieldCheck className="w-8 h-8" />
            ) : (
              <ShieldAlert className="w-8 h-8" />
            )}
          </div>

          <div>
            <div className="flex items-center space-x-2.5">
              <span
                className={`text-xs font-mono font-bold tracking-widest px-2.5 py-0.5 rounded-full border ${
                  isAlert
                    ? 'bg-red-500/20 text-red-400 border-red-500/40'
                    : isSafe
                    ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40'
                    : 'bg-slate-800 text-slate-400 border-slate-700'
                }`}
              >
                {isAlert
                  ? 'CRITICAL DEFENSE BREACH'
                  : isSafe
                  ? 'THREAT LEVEL: LOW'
                  : 'STREAM INACTIVE'}
              </span>

              {metrics && (
                <span className="text-xs font-mono text-slate-400 hidden sm:inline">
                  Hysteresis State: {metrics.consecutive_threats}/3 threats
                </span>
              )}
            </div>

            <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-white mt-1">
              {isAlert ? (
                <span className="text-red-400 flex items-center gap-2">
                  🚨 AI GENERATED / SPOOFED VOICE DETECTED
                </span>
              ) : isSafe ? (
                <span className="text-emerald-400 flex items-center gap-2">
                  ✅ BONA FIDE / REAL HUMAN VOICE DETECTED
                </span>
              ) : isSilence ? (
                <span className="text-slate-200">
                  MONITORING AUDIO — BACKGROUND SILENCE
                </span>
              ) : isStreaming ? (
                <span className="text-slate-300">ESTABLISHING ZERO-TRUST PIPELINE...</span>
              ) : (
                <span className="text-slate-400">READY TO PROTECT AUDIO STREAM</span>
              )}
            </h1>

            <p className="text-xs sm:text-sm text-slate-300 mt-0.5">
              {isAlert
                ? 'Dual-core neural engine confirmed synthetic spoof signature across 3 consecutive frames.'
                : isSafe
                ? 'In-memory biometric feature extraction validating legitimate human speech in real time.'
                : 'Zero-Disk volatile RAM streaming over 16kHz binary PCM WebSocket pipe.'}
            </p>
          </div>
        </div>

        {/* State Telemetry Pills */}
        <div className="flex flex-wrap items-center gap-2.5 sm:self-center">
          {/* VAD Status */}
          <div className="px-3 py-1.5 rounded-xl bg-black/40 border border-slate-800 flex items-center space-x-2">
            <Volume2
              className={`w-4 h-4 ${
                isSilence
                  ? 'text-slate-500'
                  : isStreaming
                  ? 'text-cyan-400 animate-pulse'
                  : 'text-slate-600'
              }`}
            />
            <div className="text-left">
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">
                Voice Gate
              </div>
              <div
                className={`text-xs font-mono font-bold ${
                  isSilence ? 'text-slate-500' : 'text-cyan-300'
                }`}
              >
                {isSilence ? 'GATE LOCKED' : 'SPEECH ACTIVE'}
              </div>
            </div>
          </div>

          {/* Engine Mode */}
          <div className="px-3 py-1.5 rounded-xl bg-black/40 border border-slate-800 flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-purple-400" />
            <div className="text-left">
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">
                Inference Core
              </div>
              <div className="text-xs font-mono font-bold text-purple-300">
                RAWNET2 + RESNET18
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
