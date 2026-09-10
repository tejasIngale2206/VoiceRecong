import { useState, useEffect, useRef, useCallback } from 'react';
import type {
  StreamResponse,
  ThreatStatus,
  ThreatLogItem,
} from '../types';
import {
  TARGET_SAMPLE_RATE,
  SAMPLES_PER_CHUNK,
  floatTo16BitPCM,
  generateSyntheticChunk,
} from '../utils/audio';

const DEFAULT_WS_URL = 'ws://127.0.0.1:8000/ws/stream';

export interface UseVoiceShieldStreamReturn {
  isConnected: boolean;
  isStreaming: boolean;
  threatStatus: ThreatStatus;
  latestResponse: StreamResponse | null;
  logs: ThreatLogItem[];
  analyserNode: AnalyserNode | null;
  error: string | null;
  startProtection: () => Promise<void>;
  stopProtection: () => void;
  simulateSyntheticAttack: () => void;
  clearLogs: () => void;
}

export function useVoiceShieldStream(
  wsUrl: string = DEFAULT_WS_URL
): UseVoiceShieldStreamReturn {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [threatStatus, setThreatStatus] = useState<ThreatStatus>('DISCONNECTED');
  const [latestResponse, setLatestResponse] = useState<StreamResponse | null>(null);
  const [logs, setLogs] = useState<ThreatLogItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null);
  const pcmAccumulatorRef = useRef<Float32Array>(new Float32Array(0));

  // Clean up audio hardware and nodes
  const cleanupAudio = useCallback(() => {
    if (processorNodeRef.current) {
      processorNodeRef.current.disconnect();
      processorNodeRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    setAnalyserNode(null);
    pcmAccumulatorRef.current = new Float32Array(0);
    setIsStreaming(false);
  }, []);

  // Gracefully close WebSocket
  const disconnectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
    setThreatStatus('DISCONNECTED');
  }, []);

  const stopProtection = useCallback(() => {
    cleanupAudio();
    disconnectWebSocket();
  }, [cleanupAudio, disconnectWebSocket]);

  // Handle incoming message from backend
  const handleWebSocketMessage = useCallback((event: MessageEvent) => {
    try {
      const response: StreamResponse = JSON.parse(event.data);
      setLatestResponse(response);
      setThreatStatus(response.status);

      const logItem: ThreatLogItem = {
        id: Date.now() + Math.random(),
        timestamp: new Date().toLocaleTimeString(),
        chunk_id: response.chunk_id,
        vad_active: response.vad_active,
        spoof_score: response.spoof_score,
        status: response.status,
        latency_ms: response.latency_ms,
        consecutive_threats: response.metrics?.consecutive_threats,
      };

      setLogs((prev) => [logItem, ...prev.slice(0, 99)]); // Keep last 100 logs
    } catch (err) {
      console.error('Failed to parse incoming WebSocket message:', err);
    }
  }, []);

  // Resample audio if device sample rate differs from 16kHz
  const resampleTo16k = (
    input: Float32Array,
    inputRate: number
  ): Float32Array => {
    if (inputRate === TARGET_SAMPLE_RATE) return input;
    const ratio = inputRate / TARGET_SAMPLE_RATE;
    const outputLength = Math.round(input.length / ratio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const sourceIndex = i * ratio;
      const indexFloor = Math.floor(sourceIndex);
      const indexCeil = Math.min(input.length - 1, indexFloor + 1);
      const fraction = sourceIndex - indexFloor;
      output[i] =
        input[indexFloor] * (1 - fraction) + input[indexCeil] * fraction;
    }
    return output;
  };

  // Start real-time audio pipeline
  const startProtection = useCallback(async () => {
    setError(null);
    setThreatStatus('CONNECTING');

    try {
      // 1. Establish WebSocket connection
      const socket = new WebSocket(wsUrl);
      socket.binaryType = 'arraybuffer';

      await new Promise<void>((resolve, reject) => {
        socket.onopen = () => {
          setIsConnected(true);
          setThreatStatus('SAFE');
          resolve();
        };
        socket.onerror = () => {
          reject(new Error('WebSocket connection error. Is the backend running on port 8000?'));
        };
        socket.onclose = () => {
          setIsConnected(false);
          setIsStreaming(false);
          setThreatStatus('DISCONNECTED');
        };
      });

      socket.onmessage = handleWebSocketMessage;
      wsRef.current = socket;

      // 2. Access User Microphone
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;

      // 3. Setup Web Audio API Pipeline
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioCtx({ sampleRate: TARGET_SAMPLE_RATE });
      audioContextRef.current = audioCtx;

      // Resume context if suspended by browser autoplay policies
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.8;
      source.connect(analyser);
      setAnalyserNode(analyser);

      // ScriptProcessorNode for accumulating exact 500ms chunks (8000 samples @ 16kHz)
      const bufferSize = 4096;
      const processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
      processorNodeRef.current = processor;

      processor.onaudioprocess = (audioProcessingEvent) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          return;
        }

        const inputChannelData = audioProcessingEvent.inputBuffer.getChannelData(0);
        // Resample if necessary
        const resampled = resampleTo16k(inputChannelData, audioCtx.sampleRate);

        // Accumulate into buffer
        const existing = pcmAccumulatorRef.current;
        const merged = new Float32Array(existing.length + resampled.length);
        merged.set(existing, 0);
        merged.set(resampled, existing.length);
        pcmAccumulatorRef.current = merged;

        // While we have at least 8,000 samples (500ms), dispatch binary PCM payload
        while (pcmAccumulatorRef.current.length >= SAMPLES_PER_CHUNK) {
          const chunk = pcmAccumulatorRef.current.slice(0, SAMPLES_PER_CHUNK);
          pcmAccumulatorRef.current = pcmAccumulatorRef.current.slice(SAMPLES_PER_CHUNK);

          const pcm16Buffer = floatTo16BitPCM(chunk);
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(pcm16Buffer);
          }
        }
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
      setIsStreaming(true);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown audio error';
      setError(message);
      setThreatStatus('DISCONNECTED');
      cleanupAudio();
      disconnectWebSocket();
    }
  }, [wsUrl, handleWebSocketMessage, cleanupAudio, disconnectWebSocket]);

  // Inject synthetic frames to test the Red Alert Hysteresis transition in real time
  const simulateSyntheticAttack = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Please start protection before running simulation attack.');
      return;
    }

    // Stream 3 consecutive synthetic chunks spaced 100ms apart
    for (let i = 0; i < 4; i++) {
      setTimeout(() => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          const syntheticChunk = generateSyntheticChunk(880 + i * 50, 0.75);
          wsRef.current.send(syntheticChunk);
        }
      }, i * 120);
    }
  }, []);

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopProtection();
    };
  }, [stopProtection]);

  return {
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
  };
}
