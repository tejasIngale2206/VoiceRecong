import React from 'react';
import { Terminal, Trash2, ShieldAlert, ShieldCheck, VolumeX } from 'lucide-react';
import type { ThreatLogItem } from '../types';

interface ThreatLogProps {
  logs: ThreatLogItem[];
  onClear: () => void;
}

export const ThreatLog: React.FC<ThreatLogProps> = ({ logs, onClear }) => {
  return (
    <div className="rounded-2xl border border-slate-800 bg-[#0d121f] p-5 shadow-xl flex flex-col h-80">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <Terminal className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-bold tracking-wider text-slate-200 uppercase font-mono">
            Zero-Trust Event Stream Log
          </h2>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs font-mono text-slate-500">
            {logs.length} events buffered
          </span>
          {logs.length > 0 && (
            <button
              onClick={onClear}
              className="p-1 rounded text-slate-500 hover:text-slate-300 hover:bg-slate-800 transition"
              title="Clear event log"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Log list */}
      <div className="flex-1 overflow-y-auto mt-3 space-y-1.5 font-mono text-xs pr-1">
        {logs.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-600 text-xs">
            Awaiting WebSocket audio frames from microphone stream...
          </div>
        ) : (
          logs.map((log) => {
            const isAlert = log.status === 'SYNTHETIC_ALERT';
            const isSilence = log.status === 'SILENCE';

            return (
              <div
                key={log.id}
                className={`p-2 rounded-lg border flex items-center justify-between transition-colors ${
                  isAlert
                    ? 'bg-red-950/30 border-red-500/40 text-red-300'
                    : isSilence
                    ? 'bg-slate-900/40 border-slate-800/80 text-slate-400'
                    : 'bg-emerald-950/20 border-emerald-500/30 text-emerald-300'
                }`}
              >
                <div className="flex items-center space-x-3">
                  {isAlert ? (
                    <ShieldAlert className="w-3.5 h-3.5 text-red-400 shrink-0" />
                  ) : isSilence ? (
                    <VolumeX className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                  ) : (
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                  )}

                  <span className="text-slate-500 text-[11px]">{log.timestamp}</span>
                  <span className="text-white font-bold">Chunk #{log.chunk_id}</span>
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                      isAlert
                        ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                        : isSilence
                        ? 'bg-slate-800 text-slate-400'
                        : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    }`}
                  >
                    {log.status}
                  </span>
                </div>

                <div className="flex items-center space-x-4 text-[11px]">
                  <span>
                    Score: <strong className="text-white">{log.spoof_score.toFixed(4)}</strong>
                  </span>
                  <span>
                    Latency: <strong className="text-slate-300">{log.latency_ms.toFixed(1)}ms</strong>
                  </span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
