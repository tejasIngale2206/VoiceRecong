import React, { useEffect, useRef } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { Activity, Radio } from 'lucide-react';
import type { ThreatStatus } from '../types';

interface OscilloscopeProps {
  analyserNode: AnalyserNode | null;
  threatStatus: ThreatStatus;
  isStreaming: boolean;
  energyDb?: number;
}

export const Oscilloscope: React.FC<OscilloscopeProps> = ({
  analyserNode,
  threatStatus,
  isStreaming,
  energyDb = -100,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const waveContainerRef = useRef<HTMLDivElement | null>(null);
  const waveSurferRef = useRef<WaveSurfer | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);

  const isAlert = threatStatus === 'SYNTHETIC_ALERT';
  const isSafe = threatStatus === 'SAFE';

  // Initialize WaveSurfer container
  useEffect(() => {
    if (!waveContainerRef.current) return;

    try {
      const ws = WaveSurfer.create({
        container: waveContainerRef.current,
        waveColor: isAlert ? '#ef4444' : isSafe ? '#10b981' : '#475569',
        progressColor: isAlert ? '#b91c1c' : '#059669',
        height: 60,
        normalize: true,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        cursorWidth: 0,
        interact: false,
      });

      waveSurferRef.current = ws;

      return () => {
        ws.destroy();
        waveSurferRef.current = null;
      };
    } catch (err) {
      console.warn('WaveSurfer initialization note:', err);
    }
  }, [isAlert, isSafe]);

  // Live 60fps high-precision Audio Oscilloscope rendering via AnalyserNode
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let isRunning = true;
    const bufferLength = analyserNode ? analyserNode.fftSize : 512;
    const dataArray = new Uint8Array(bufferLength);

    const renderOscilloscope = () => {
      if (!isRunning) return;

      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      // Cyber oscilloscope background grid
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 1;
      const gridSize = 24;
      for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Center baseline
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();

      if (analyserNode && isStreaming) {
        analyserNode.getByteTimeDomainData(dataArray);

        // Dynamic theme line styling
        if (isAlert) {
          ctx.strokeStyle = '#ef4444';
          ctx.shadowColor = 'rgba(239, 68, 68, 0.8)';
          ctx.shadowBlur = 12;
        } else if (isSafe) {
          ctx.strokeStyle = '#10b981';
          ctx.shadowColor = 'rgba(16, 185, 129, 0.8)';
          ctx.shadowBlur = 10;
        } else {
          ctx.strokeStyle = '#38bdf8';
          ctx.shadowColor = 'rgba(56, 189, 248, 0.5)';
          ctx.shadowBlur = 6;
        }

        ctx.lineWidth = 2.5;
        ctx.beginPath();

        const sliceWidth = width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0;
          const y = (v * height) / 2;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
          x += sliceWidth;
        }

        ctx.lineTo(width, height / 2);
        ctx.stroke();
        ctx.shadowBlur = 0; // reset
      } else {
        // Flatline / Idle state
        ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
      }

      animationFrameIdRef.current = requestAnimationFrame(renderOscilloscope);
    };

    renderOscilloscope();

    return () => {
      isRunning = false;
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
      }
    };
  }, [analyserNode, isStreaming, isAlert, isSafe]);

  // Map energy dB (-100 to 0) to percentage (0% to 100%)
  const dbPercentage = Math.min(100, Math.max(0, ((energyDb + 70) / 70) * 100));
  const noiseGatePercent = (( -45 + 70 ) / 70) * 100; // ~35.7%

  return (
    <div className="rounded-2xl border border-slate-800 bg-[#0d121f] p-5 shadow-xl">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <Activity className={`w-4 h-4 ${isAlert ? 'text-red-400' : 'text-emerald-400'}`} />
          <h2 className="text-sm font-bold tracking-wider text-slate-200 uppercase font-mono">
            Live Audio Oscilloscope & WaveSurfer Monitor
          </h2>
        </div>
        <div className="flex items-center space-x-3 text-xs font-mono text-slate-400">
          <span className="flex items-center space-x-1">
            <Radio className={`w-3 h-3 ${isStreaming ? 'text-emerald-400 animate-ping' : 'text-slate-600'}`} />
            <span>16kHz Mono PCM</span>
          </span>
          <span className="hidden sm:inline px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
            500ms Frame Window
          </span>
        </div>
      </div>

      {/* Main Canvas Oscilloscope */}
      <div className="relative w-full h-36 bg-[#070a12] rounded-xl overflow-hidden border border-slate-800/80">
        <canvas
          ref={canvasRef}
          width={800}
          height={144}
          className="w-full h-full block"
        />

        {/* WaveSurfer Container Placeholder for Timeline Sync */}
        <div ref={waveContainerRef} className="hidden" />

        {/* Watermark badge */}
        <div className="absolute top-2 left-3 pointer-events-none">
          <span className="text-[10px] font-mono tracking-widest text-slate-600 uppercase">
            CH-1 TIME DOMAIN | 16000 S/SEC
          </span>
        </div>

        {/* Threat State Overlay Pill */}
        <div className="absolute bottom-2 right-3 pointer-events-none">
          <span
            className={`text-[11px] font-mono px-2 py-0.5 rounded border font-semibold ${
              isAlert
                ? 'bg-red-500/20 text-red-400 border-red-500/40 animate-pulse'
                : isSafe
                ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
                : 'bg-slate-800/80 text-slate-400 border-slate-700'
            }`}
          >
            {threatStatus}
          </span>
        </div>
      </div>

      {/* dBFS Audio Energy Level Bar with -45dB Noise Gate Threshold */}
      <div className="mt-4">
        <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 mb-1">
          <span>RMS Energy: <strong className="text-white">{energyDb.toFixed(1)} dBFS</strong></span>
          <span className="text-cyan-400">Gate Threshold: -45.0 dBFS</span>
        </div>
        <div className="relative w-full h-2 rounded-full bg-slate-900 border border-slate-800 overflow-hidden">
          {/* Active Level */}
          <div
            className={`h-full transition-all duration-100 ${
              energyDb >= -45.0 ? 'bg-gradient-to-r from-emerald-500 to-cyan-400' : 'bg-slate-700'
            }`}
            style={{ width: `${dbPercentage}%` }}
          />
          {/* -45dB threshold marker line */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-yellow-400 shadow-sm shadow-yellow-300"
            style={{ left: `${noiseGatePercent}%` }}
            title="VAD Noise Gate: -45 dBFS"
          />
        </div>
      </div>
    </div>
  );
};
