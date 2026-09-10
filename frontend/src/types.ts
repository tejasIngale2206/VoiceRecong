/**
 * VoiceShield AI TypeScript Interfaces (SIH26104)
 */

export interface StreamMetrics {
  energy_db: number;
  rawnet2_score: number;
  resnet_score: number;
  fused_score: number;
  consecutive_threats: number;
  consecutive_safe: number;
  device?: string;
}

export interface StreamResponse {
  chunk_id: number;
  vad_active: boolean;
  spoof_score: number;
  status: 'SAFE' | 'SYNTHETIC_ALERT' | 'SILENCE';
  latency_ms: number;
  metrics?: StreamMetrics;
}

export type ThreatStatus = 'SAFE' | 'SYNTHETIC_ALERT' | 'SILENCE' | 'DISCONNECTED' | 'CONNECTING';

export interface ThreatLogItem {
  id: number;
  timestamp: string;
  chunk_id: number;
  vad_active: boolean;
  spoof_score: number;
  status: 'SAFE' | 'SYNTHETIC_ALERT' | 'SILENCE';
  latency_ms: number;
  consecutive_threats?: number;
}
