/**
 * Audio DSP & PCM conversion utilities for VoiceShield AI.
 * Strict specification: 16,000 Hz, mono, 16-bit PCM little-endian.
 */

export const TARGET_SAMPLE_RATE = 16000;
export const CHUNK_DURATION_SEC = 0.5; // 500ms
export const SAMPLES_PER_CHUNK = TARGET_SAMPLE_RATE * CHUNK_DURATION_SEC; // 8000 samples
export const BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * 2; // 16,000 bytes

/**
 * Convert 32-bit float audio samples [-1.0, 1.0] to 16-bit linear PCM little-endian buffer.
 */
export function floatTo16BitPCM(input: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(input.length * 2);
  const view = new DataView(buffer);

  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return buffer;
}

/**
 * Calculate Root Mean Square (RMS) in dBFS for a float audio buffer.
 */
export function calculateDbFS(samples: Float32Array): number {
  if (samples.length === 0) return -100.0;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    sum += samples[i] * samples[i];
  }
  const rms = Math.sqrt(sum / samples.length);
  if (rms <= 1e-5) return -100.0;
  return Math.max(-100.0, 20 * Math.log10(rms));
}

/**
 * Synthesize a 500ms mock 16kHz PCM chunk for testing and demonstration.
 */
export function generateSyntheticChunk(
  frequencyHz: number = 440,
  amplitude: number = 0.7
): ArrayBuffer {
  const samples = new Float32Array(SAMPLES_PER_CHUNK);
  for (let i = 0; i < SAMPLES_PER_CHUNK; i++) {
    const t = i / TARGET_SAMPLE_RATE;
    // Synthesize synthetic chirp and non-linear phase modulation matching vocoder artifacts
    samples[i] =
      Math.sin(2 * Math.PI * (frequencyHz + 2500 * t) * t) * amplitude;
  }
  return floatTo16BitPCM(samples);
}
