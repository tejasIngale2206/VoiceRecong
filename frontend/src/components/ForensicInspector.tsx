import React, { useState, useEffect, useRef } from 'react';
// @ts-ignore
import jsPDF from 'jspdf';
import {
    Upload,
    FileAudio,
    AlertTriangle,
    CheckCircle,
    RefreshCw,
    XCircle,
    Play,
    Pause,
    Activity,
    Download
} from 'lucide-react';

interface TimelinePoint {
    timestamp_sec: number;
    threat_score: number;
    is_silent?: boolean;
}

interface ScanResult {
    status: string;
    fused_score: number;
    rawnet2_score: number;
    resnet_score: number;
    device?: string;
    timeline?: TimelinePoint[];
}

export const ForensicInspector: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [scanning, setScanning] = useState<boolean>(false);
    const [result, setResult] = useState<ScanResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [selectedTime, setSelectedTime] = useState<number | null>(null);
    const [isPlaying, setIsPlaying] = useState<boolean>(false);

    const audioRef = useRef<HTMLAudioElement | null>(null);

    useEffect(() => {
        if (file) {
            const url = URL.createObjectURL(file);
            setAudioUrl(url);
            return () => URL.revokeObjectURL(url);
        } else {
            setAudioUrl(null);
        }
    }, [file]);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setResult(null);
            setError(null);
            setSelectedTime(null);
        }
    };

    const handleClearFile = () => {
        setFile(null);
        setResult(null);
        setError(null);
        setSelectedTime(null);
        if (audioRef.current) {
            audioRef.current.pause();
        }
    };

    const handleScan = async () => {
        if (!file) return;

        setScanning(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/api/v1/scan-file', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Server returned error status ${response.status}`);
            }

            const data: ScanResult = await response.json();
            setResult(data);
        } catch (err: any) {
            setError(err.message || 'An error occurred during scanning.');
        } finally {
            setScanning(false);
        }
    };

    const handleBarClick = (timestamp: number) => {
        setSelectedTime(timestamp);
        if (audioRef.current) {
            audioRef.current.currentTime = timestamp;
            audioRef.current.play();
            setIsPlaying(true);
        }
    };

    const togglePlay = () => {
        if (!audioRef.current) return;
        if (isPlaying) {
            audioRef.current.pause();
            setIsPlaying(false);
        } else {
            audioRef.current.play();
            setIsPlaying(true);
        }
    };

    const formatScore = (score?: number) => {
        if (score === undefined || score === null) return '0.0%';
        const val = score <= 1.0 ? score * 100 : score;
        return `${val.toFixed(1)}%`;
    };

    const isThreat = (status?: string) => {
        if (!status) return false;
        const s = status.toUpperCase();
        return s === 'SPOOF' || s === 'SYNTHETIC_ALERT' || s === 'THREAT';
    };

    const generatePDFReport = () => {
        if (!result) return;

        const doc = new jsPDF();
        const dateStr = new Date().toLocaleString();

        doc.setFont("helvetica", "bold");
        doc.setFontSize(18);
        doc.setTextColor(13, 18, 31);
        doc.text("VOICESHIELD - FORENSIC AUDIT REPORT", 14, 20);

        doc.setFontSize(10);
        doc.setFont("helvetica", "normal");
        doc.setTextColor(100, 116, 139);
        doc.text(`Generated on: ${dateStr}`, 14, 28);
        doc.text(`Analyzed File: ${file?.name || "Unknown File"}`, 14, 34);

        doc.setLineWidth(0.5);
        doc.setDrawColor(203, 213, 225);
        doc.line(14, 38, 196, 38);

        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");
        doc.setTextColor(15, 23, 42);
        doc.text("1. Overall Forensic Verdict", 14, 48);

        const isSpoof = isThreat(result.status);
        doc.setFontSize(12);
        doc.setTextColor(isSpoof ? 220 : 16, isSpoof ? 38 : 185, isSpoof ? 38 : 129);
        doc.text(`Status: ${result.status} (${isSpoof ? 'DEEPFAKE DETECTED' : 'AUTHENTIC AUDIO'})`, 14, 56);

        doc.setTextColor(51, 65, 85);
        doc.setFont("helvetica", "normal");
        doc.text(`Fused Threat Score: ${formatScore(result.fused_score)}`, 14, 64);

        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");
        doc.setTextColor(15, 23, 42);
        doc.text("2. Multi-Model Ensemble Breakdown", 14, 78);

        doc.setFontSize(11);
        doc.setFont("helvetica", "normal");
        doc.text(`* RawNet2 (1D Waveform Analysis): ${formatScore(result.rawnet2_score)}`, 18, 88);
        doc.text(`* ResNet18 (2D Spectrogram Analysis): ${formatScore(result.resnet_score)}`, 18, 96);

        if (result.timeline && result.timeline.length > 0) {
            doc.setFontSize(14);
            doc.setFont("helvetica", "bold");
            doc.setTextColor(15, 23, 42);
            doc.text("3. Temporal Threat Distribution Summary", 14, 110);

            doc.setFontSize(10);
            doc.setFont("helvetica", "normal");

            let yPos = 120;
            const totalFrames = result.timeline.length;
            const threatFrames = result.timeline.filter(p => {
                const val = p.threat_score <= 1.0 ? p.threat_score * 100 : p.threat_score;
                return val >= 50.0;
            }).length;

            doc.text(`Total Evaluated Time Frames: ${totalFrames}`, 18, yPos);
            yPos += 6;
            doc.text(`Flagged High-Risk Frames (>=50% Threat): ${threatFrames}`, 18, yPos);
        }

        doc.save(`Forensic_Report_${file?.name || 'audio'}.pdf`);
    };

    return (
        <div className="rounded-2xl border border-slate-800 bg-[#0d121f] p-6 shadow-xl text-slate-100 font-sans">
            <div className="flex items-center justify-between mb-2">
                <h2 className="text-base font-bold tracking-wider text-slate-200 font-mono flex items-center gap-2">
                    <FileAudio className="w-5 h-5 text-cyan-400" />
                    FORENSIC AUDIO FILE INSPECTOR
                </h2>
                <span className="text-xs text-slate-500 font-mono">POST /api/v1/scan-file</span>
            </div>

            <p className="text-xs text-slate-400 font-mono mb-6">
                Upload pre-recorded audio samples (.mp3, .wav, .flac) for offline deepfake analysis and score fusion verification.
            </p>

            <div className="flex flex-col sm:flex-row items-center gap-4 mb-4">
                <div className="relative flex-1 w-full">
                    <label className="flex items-center justify-between px-4 py-3 border border-dashed border-slate-700 rounded-xl cursor-pointer hover:border-cyan-500/50 bg-slate-900/50 transition-all">
                        <div className="flex items-center space-x-3 overflow-hidden">
                            <Upload className="w-5 h-5 text-cyan-400 shrink-0" />
                            <span className="text-sm font-mono text-slate-300 truncate">
                                {file ? file.name : 'Choose or drop audio file (.wav, .mp3, .flac)'}
                            </span>
                        </div>
                        <input
                            type="file"
                            accept=".wav,.mp3,.flac,.ogg,audio/*"
                            className="hidden"
                            onChange={handleFileChange}
                        />
                    </label>

                    {file && (
                        <button
                            onClick={handleClearFile}
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-red-400 transition-colors"
                            title="Remove File"
                        >
                            <XCircle className="w-4 h-4" />
                        </button>
                    )}
                </div>

                <button
                    onClick={handleScan}
                    disabled={!file || scanning}
                    className={`w-full sm:w-auto px-6 py-3 rounded-xl font-mono font-bold text-sm transition-all flex items-center justify-center gap-2 shrink-0 ${scanning || !file
                        ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50'
                        : 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg shadow-cyan-950/50'
                        }`}
                >
                    {scanning ? (
                        <>
                            <RefreshCw className="w-4 h-4 animate-spin" />
                            SCANNING SPECTRUM...
                        </>
                    ) : (
                        'RUN FORENSIC SCAN'
                    )}
                </button>
            </div>

            {error && (
                <div className="p-3.5 rounded-xl bg-red-950/60 border border-red-500/80 text-red-200 text-xs font-mono flex items-center gap-2 mb-4">
                    <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                    <span>{error}</span>
                </div>
            )}

            {result && (
                <div className="mt-6 p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-6">
                    <div className="flex justify-between items-center pb-3 border-b border-slate-800/80 font-mono">
                        <span className="text-xs font-bold tracking-wider text-slate-400 uppercase">
                            Evaluation Result:
                        </span>
                        <div className="flex items-center gap-3">
                            <button
                                onClick={generatePDFReport}
                                className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-cyan-400 border border-cyan-800/50 rounded-lg text-xs font-mono font-bold flex items-center gap-1.5 transition-all shadow-md"
                            >
                                <Download className="w-3.5 h-3.5" />
                                EXPORT PDF REPORT
                            </button>
                            <span
                                className={`px-3 py-1 text-xs font-bold rounded-full flex items-center gap-1.5 ${isThreat(result.status)
                                    ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                                    : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                                    }`}
                            >
                                {isThreat(result.status) ? (
                                    <>
                                        <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
                                        {result.status} (DEEPFAKE)
                                    </>
                                ) : (
                                    <>
                                        <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                                        {result.status} (AUTHENTIC)
                                    </>
                                )}
                            </span>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono text-center">
                        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                            <p className="text-xs text-slate-400 mb-1 font-sans">Fused Threat Score</p>
                            <p className={`text-2xl font-extrabold ${isThreat(result.status) ? 'text-red-400' : 'text-emerald-400'}`}>
                                {formatScore(result.fused_score)}
                            </p>
                        </div>

                        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                            <p className="text-xs text-slate-400 mb-1 font-sans">RawNet2 (1D Waveform)</p>
                            <p className="text-xl font-bold text-cyan-300">
                                {formatScore(result.rawnet2_score)}
                            </p>
                        </div>

                        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                            <p className="text-xs text-slate-400 mb-1 font-sans">ResNet18 (2D Spectrogram)</p>
                            <p className="text-xl font-bold text-purple-300">
                                {formatScore(result.resnet_score)}
                            </p>
                        </div>
                    </div>

                    <div className="pt-2">
                        <div className="flex items-center justify-between mb-3">
                            <h3 className="text-xs font-bold font-mono text-cyan-400 flex items-center gap-2 uppercase tracking-wider">
                                <Activity className="w-4 h-4" /> Temporal Forensic Threat Distribution
                            </h3>
                            {selectedTime !== null && (
                                <span className="text-xs font-mono text-slate-400">
                                    Selected Frame: <strong className="text-cyan-300">{selectedTime}s</strong>
                                </span>
                            )}
                        </div>

                        {result.timeline && result.timeline.length > 0 ? (
                            <div className="h-32 w-full flex items-end gap-1.5 overflow-x-auto pb-2 pt-6 px-3 bg-slate-900/80 rounded-xl border border-slate-800">
                                {result.timeline.map((point, idx) => {
                                    const threatVal = point.threat_score <= 1.0 ? point.threat_score * 100 : point.threat_score;
                                    const isSpoofChunk = threatVal >= 50.0;
                                    const computedHeight = point.is_silent ? '12%' : `${Math.max(threatVal, 12)}%`;

                                    return (
                                        <div
                                            key={idx}
                                            onClick={() => handleBarClick(point.timestamp_sec)}
                                            className="group relative flex-1 min-w-[12px] h-full flex items-end cursor-pointer transition-all hover:opacity-80"
                                            title={`Time: ${point.timestamp_sec}s | Threat: ${threatVal.toFixed(1)}%`}
                                        >
                                            <div className="absolute -top-8 left-1/2 -translate-x-1/2 hidden group-hover:flex flex-col items-center z-30 pointer-events-none">
                                                <div className="bg-black/90 text-cyan-300 text-[10px] font-mono py-0.5 px-1.5 rounded border border-cyan-800 whitespace-nowrap shadow-lg">
                                                    {point.timestamp_sec}s: {threatVal.toFixed(1)}%
                                                </div>
                                            </div>

                                            <div
                                                style={{ height: computedHeight }}
                                                className={`w-full rounded-t transition-all ${point.is_silent
                                                        ? 'bg-slate-700/60'
                                                        : isSpoofChunk
                                                            ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.7)]'
                                                            : 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]'
                                                    }`}
                                            />
                                        </div>
                                    );
                                })}
                            </div>
                        ) : (
                            <div className="h-28 w-full flex items-center justify-center bg-slate-900/40 rounded-xl border border-slate-800/60 font-mono text-xs text-slate-500">
                                NO TIMELINE DATA RECEIVED FROM BACKEND
                            </div>
                        )}

                        {audioUrl && (
                            <div className="mt-4 p-3 bg-slate-900 rounded-xl border border-slate-800 flex items-center gap-3">
                                <button
                                    onClick={togglePlay}
                                    className="p-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white transition-all shrink-0"
                                >
                                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                                </button>
                                <audio
                                    ref={audioRef}
                                    src={audioUrl}
                                    onPlay={() => setIsPlaying(true)}
                                    onPause={() => setIsPlaying(false)}
                                    onEnded={() => setIsPlaying(false)}
                                    controls
                                    className="w-full h-8 accent-cyan-500"
                                />
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default ForensicInspector;