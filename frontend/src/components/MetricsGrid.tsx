import React from 'react';
import { Gauge, Mic, Layers, Clock, Cpu, CheckCircle2 } from 'lucide-react';
import type { StreamResponse, ThreatStatus } from '../types';

interface MetricsGridProps {
  latestResponse: StreamResponse | null;
  threatStatus: ThreatStatus;
  isStreaming: boolean;
}

export const MetricsGrid: React.FC<MetricsGridProps> = ({
  latestResponse,
  threatStatus,
  isStreaming,
}) => {
  const spoofScore = latestResponse?.spoof_score ?? 0.0;
  const vadActive = latestResponse?.vad_active ?? false;
  const latencyMs = latestResponse?.latency_ms ?? 0.0;
  const metrics = latestResponse?.metrics;

  const rawnet2Score = metrics?.rawnet2_score ?? 0.0;
  const resnetScore = metrics?.resnet_score ?? 0.0;
  const energyDb = metrics?.energy_db ?? -100.0;
  const consecutiveThreats = metrics?.consecutive_threats ?? 0;
  const device = metrics?.device ?? 'cpu';

  // Spoof score color scheme (Threshold: >= 0.40 triggers SYNTHETIC_ALERT)
  const getScoreColor = (score: number) => {
    if (score >= 0.40) return 'text-red-400';
    if (score >= 0.25) return 'text-amber-400';
    return 'text-emerald-400';
  };

  const getScoreBg = (score: number) => {
    if (score >= 0.40) return 'bg-red-500';
    if (score >= 0.25) return 'bg-amber-500';
    return 'bg-emerald-500';
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* 1. Spoof Probability Gauge */}
      <div className="rounded-2xl border border-slate-800 bg-[#0d121f] p-5 shadow-lg flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400">
          <span className="text-xs font-mono uppercase tracking-wider font-semibold">
            Threat Probability
          </span>
          <div className="flex items-center space-x-1.5">
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400">
              {threatStatus}
            </span>
            <Gauge className="w-4 h-4 text-cyan-400" />
          </div>
        </div>

        <div className="my-4 text-center">
          <div className={`text-4xl font-black font-mono tracking-tight ${getScoreColor(spoofScore)}`}>
            {(spoofScore * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-slate-400 font-mono mt-1">
            Score: {spoofScore.toFixed(4)} / 1.0000
          </div>
        </div>

        {/* Threat Progress Bar */}
        <div>
          <div className="w-full h-2 rounded-full bg-slate-900 overflow-hidden border border-slate-800">
            <div
              className={`h-full transition-all duration-300 ${getScoreBg(spoofScore)}`}
              style={{ width: `${Math.min(100, Math.max(0, spoofScore * 100))}%` }}
            />
          </div>
          <div className="flex justify-between items-center text-[10px] font-mono text-slate-500 mt-1.5">
            <span>0.00 SAFE</span>
            <span className="text-red-400 font-semibold">0.40 ALERT GATE</span>
            <span>1.00</span>
          </div>
        </div>
      </div>

      {/* 2. VAD Activity Indicator */}
      <div className="rounded-2xl border border-slate-800 bg-[#0d121f] p-5 shadow-lg flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400">
          <span className="text-xs font-mono uppercase tracking-wider font-semibold">
            Voice Activity Gate
          </span>
          <Mic className="w-4 h-4 text-purple-400" />
        </div>

        <div className="my-4 text-center">
          <div
            className={`inline-flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-mono font-bold border ${
              vadActive
                ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40'
                : 'bg-slate-800/80 text-slate-400 border-slate-700'
            }`}
          >
            {isStreaming && vadActive ? (
              <>
                <CheckCircle2 className="w-4 h-4" />
                <span>SPEECH DETECTED</span>
              </>
            ) : isStreaming ? (
              <>
                <span className="w-2 h-2 rounded-full bg-slate-500" />
                <span>SILENCE / NOISE</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-slate-600" />
                <span>GATE STANDBY</span>
              </>
            )}
          </div>
          <div className="text-xs text-slate-400 font-mono mt-2">
            Signal: {energyDb.toFixed(1)} dBFS
          </div>
        </div>

        <div className="text-[11px] font-mono text-slate-400 bg-slate-900/60 p-2 rounded-lg border border-slate-800">
          <div className="flex justify-between">
            <span>Noise Gate:</span>
            <span className="text-white font-semibold">-45.0 dBFS</span>
          </div>
          <div className="flex justify-between mt-0.5">
            <span>Storage:</span>
            <span className="text-emerald-400">Zero-Disk RAM (io.BytesIO)</span>
          </div>
        </div>
      </div>

      {/* 3. Dual-Model Breakdown */}
      <div className="rounded-2xl border border-slate-800 bg-[#0d121f] p-5 shadow-lg flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400">
          <span className="text-xs font-mono uppercase tracking-wider font-semibold">
            Dual AI Stream Score
          </span>
          <Layers className="w-4 h-4 text-amber-400" />
        </div>

        <div className="my-2 space-y-2 text-xs font-mono">
          <div>
            <div className="flex justify-between text-slate-300 mb-1">
              <span>RawNet2 (1D Wave, 60%):</span>
              <strong className={getScoreColor(rawnet2Score)}>{(rawnet2Score * 100).toFixed(1)}%</strong>
            </div>
            <div className="w-full h-1.5 rounded-full bg-slate-900 overflow-hidden">
              <div
                className={`h-full ${getScoreBg(rawnet2Score)}`}
                style={{ width: `${Math.min(100, rawnet2Score * 100)}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-slate-300 mb-1">
              <span>ResNet18 (2D Spec, 40%):</span>
              <strong className={getScoreColor(resnetScore)}>{(resnetScore * 100).toFixed(1)}%</strong>
            </div>
            <div className="w-full h-1.5 rounded-full bg-slate-900 overflow-hidden">
              <div
                className={`h-full ${getScoreBg(resnetScore)}`}
                style={{ width: `${Math.min(100, resnetScore * 100)}%` }}
              />
            </div>
          </div>
        </div>

        <div className="text-[10px] font-mono text-slate-500 bg-slate-900/60 p-2 rounded-lg border border-slate-800 flex justify-between items-center">
          <span>S = (0.6 * RawNet) + (0.4 * ResNet)</span>
          <span className="uppercase text-purple-400 font-bold flex items-center gap-1">
            <Cpu className="w-3 h-3" />
            {device}
          </span>
        </div>
      </div>

      {/* 4. Processing Latency (ms) */}
      <div className="rounded-2xl border border-slate-800 bg-[#0d121f] p-5 shadow-lg flex flex-col justify-between">
        <div className="flex items-center justify-between text-slate-400">
          <span className="text-xs font-mono uppercase tracking-wider font-semibold">
            Processing Latency
          </span>
          <Clock className="w-4 h-4 text-emerald-400" />
        </div>

        <div className="my-4 text-center">
          <div
            className={`text-4xl font-black font-mono tracking-tight ${
              latencyMs < 30 ? 'text-emerald-400' : latencyMs < 50 ? 'text-amber-400' : 'text-red-400'
            }`}
          >
            {latencyMs.toFixed(1)} <span className="text-lg font-normal text-slate-400">ms</span>
          </div>
          <div className="text-xs text-slate-400 font-mono mt-1">
            Latency Budget: &lt; 50 ms / 500ms chunk
          </div>
        </div>

        <div className="text-[11px] font-mono text-slate-400 bg-slate-900/60 p-2 rounded-lg border border-slate-800">
          <div className="flex justify-between">
            <span>Hysteresis Streak:</span>
            <span className={consecutiveThreats >= 3 ? 'text-red-400 font-bold' : 'text-slate-300'}>
              {consecutiveThreats} / 3 frames
            </span>
          </div>
          <div className="flex justify-between mt-0.5">
            <span>Budget Utilized:</span>
            <span className="text-emerald-400 font-semibold">
              {((latencyMs / 50) * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
