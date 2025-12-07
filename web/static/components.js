// SongGeneration Studio - React Components
// This file contains reusable UI components

var { useState, useEffect, useRef, useCallback } = React;

// Shared utility function for formatting time
var formatTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
};

// Model generation time estimates (in seconds)
var MODEL_BASE_TIMES = {
    'songgeneration_base': 180,
    'songgeneration_base_new': 200,
    'songgeneration_base_full': 300,
    'songgeneration_large': 420,
};

// Custom hook for hover state
var useHover = () => {
    const [isHovered, setIsHovered] = useState(false);
    const handlers = {
        onMouseEnter: () => setIsHovered(true),
        onMouseLeave: () => setIsHovered(false),
    };
    return [isHovered, handlers];
};

// Scrollbar style helper
var getScrollStyle = (isHovered) => ({
    scrollbarWidth: 'thin',
    scrollbarColor: isHovered ? '#3a3a3a transparent' : 'transparent transparent',
});

// Reusable Icon components
var PlayIcon = ({ size = 12, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 12 12" fill={color}>
        <path d="M2 1.5v9l8-4.5-8-4.5z" />
    </svg>
);

var PauseIcon = ({ size = 12, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 12 12" fill={color}>
        <rect x="2" y="1" width="3" height="10" rx="1" />
        <rect x="7" y="1" width="3" height="10" rx="1" />
    </svg>
);

var CloseIcon = ({ size = 12, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M18 6L6 18M6 6l12 12"/>
    </svg>
);

var MusicNoteIcon = ({ size = 24, color = '#666' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M9 18V5l12-2v13"/>
        <circle cx="6" cy="18" r="3"/>
        <circle cx="18" cy="16" r="3"/>
    </svg>
);

var ExpandIcon = ({ size = 12, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
    </svg>
);

var ChevronIcon = ({ size = 16, color = '#888', rotated = false }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transition: 'transform 0.2s', transform: rotated ? 'rotate(180deg)' : 'rotate(0deg)' }}>
        <path d="M6 9l6 6 6-6"/>
    </svg>
);

var VolumeIcon = ({ size = 16, color = '#666' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
        <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>
    </svg>
);

var VolumeFullIcon = ({ size = 18, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M11 5L6 9H2v6h4l5 4V5z"/>
        <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>
    </svg>
);

var VolumeMuteIcon = ({ size = 18, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M11 5L6 9H2v6h4l5 4V5z"/>
        <line x1="23" y1="9" x2="17" y2="15"/>
        <line x1="17" y1="9" x2="23" y2="15"/>
    </svg>
);

var PlusIcon = ({ size = 16, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M12 5v14M5 12h14"/>
    </svg>
);

var TrashIcon = ({ size = 10, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
    </svg>
);

var EditIcon = ({ size = 14, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
    </svg>
);

var SkipBackIcon = ({ size = 20, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
        <path d="M6 6h2v12H6zm3.5 6l8.5 6V6z"/>
    </svg>
);

var SkipForwardIcon = ({ size = 20, color = 'currentColor' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
        <path d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z"/>
    </svg>
);

var PlayLargeIcon = ({ size = 24, color = '#fff', style = {} }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color} style={style}>
        <path d="M8 5v14l11-7z"/>
    </svg>
);

var PauseLargeIcon = ({ size = 24, color = '#fff' }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
        <rect x="6" y="4" width="4" height="16"/>
        <rect x="14" y="4" width="4" height="16"/>
    </svg>
);

var SpinnerIcon = ({ size = 24 }) => (
    <div style={{ width: size, height: size, border: '2px solid #fff', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
);

// Section type configurations
var SECTION_TYPES = {
    'intro': { name: 'Intro', color: '#8B5CF6', hasLyrics: false, hasDuration: true },
    'verse': { name: 'Verse', color: '#3B82F6', hasLyrics: true, hasDuration: false },
    'chorus': { name: 'Chorus', color: '#F59E0B', hasLyrics: true, hasDuration: false },
    'bridge': { name: 'Bridge', color: '#EC4899', hasLyrics: true, hasDuration: false },
    'inst': { name: 'Inst', color: '#10B981', hasLyrics: false, hasDuration: true },
    'outro': { name: 'Outro', color: '#EAB308', hasLyrics: false, hasDuration: true },
};

// Suggestion lists
var GENRE_SUGGESTIONS = ['Pop', 'Rock', 'Metal', 'Jazz', 'R&B', 'Folk', 'Dance', 'Electronic', 'Hip Hop', 'Trap', 'Lo-fi', 'Synthwave', 'Punk', 'Blues', 'Country', 'Ambient', 'Soul', 'Funk', 'Reggae', 'Indie', 'EDM', 'Dubstep', 'House', 'Techno', 'Drum and Bass', 'Classical', 'Opera', 'Gospel', 'Ska', 'Grunge', 'Disco', 'New Wave', 'Shoegaze', 'Post-rock', 'Hardcore', 'Emo'];
var MOOD_SUGGESTIONS = ['happy', 'sad', 'energetic', 'romantic', 'angry', 'peaceful', 'melancholic', 'hopeful', 'aggressive', 'dreamy', 'nostalgic', 'euphoric', 'dark', 'chill', 'epic', 'intense', 'uplifting', 'mysterious', 'playful', 'haunting'];
var TIMBRE_SUGGESTIONS = ['bright', 'dark', 'soft', 'powerful', 'warm', 'clear', 'raspy', 'smooth', 'breathy', 'ethereal', 'gravelly', 'operatic', 'distorted', 'clean', 'raw', 'polished'];
var INSTRUMENT_SUGGESTIONS = ['piano', 'guitar', 'drums', 'bass', 'synth', 'strings', 'violin', 'cello', 'saxophone', 'trumpet', 'flute', 'organ', 'harp', 'percussion', 'choir', 'orchestra'];

// Helper function to parse section types
var fromApiType = (apiType) => {
    const match = apiType.match(/^(.+?)(?:-(short|medium|long))?$/);
    return match ? { base: match[1], duration: match[2] || 'short' } : { base: apiType, duration: null };
};

// Custom dark audio player component
var DarkAudioPlayer = ({ src }) => {
    const audioRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [volume, setVolume] = useState(1);

    const togglePlay = () => {
        if (audioRef.current) {
            if (isPlaying) {
                audioRef.current.pause();
            } else {
                audioRef.current.play();
            }
            setIsPlaying(!isPlaying);
        }
    };

    const handleTimeUpdate = () => {
        if (audioRef.current) {
            setCurrentTime(audioRef.current.currentTime);
        }
    };

    const handleLoadedMetadata = () => {
        if (audioRef.current) {
            setDuration(audioRef.current.duration);
        }
    };

    const handleSeek = (e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        if (audioRef.current) {
            audioRef.current.currentTime = percent * duration;
        }
    };

    const handleEnded = () => {
        setIsPlaying(false);
        setCurrentTime(0);
    };

    const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

    return (
        <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            backgroundColor: '#1a1a1a',
            borderRadius: '8px',
            padding: '10px 14px',
            flex: 1,
        }}>
            <audio
                ref={audioRef}
                src={src}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onEnded={handleEnded}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
            />

            {/* Play/Pause button */}
            <button
                onClick={togglePlay}
                style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '50%',
                    border: 'none',
                    backgroundColor: '#10B981',
                    color: '#fff',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                }}
            >
                {isPlaying ? <PauseIcon /> : <PlayIcon />}
            </button>

            {/* Time */}
            <span style={{ fontSize: '12px', color: '#888', minWidth: '70px' }}>
                {formatTime(currentTime)} / {formatTime(duration)}
            </span>

            {/* Progress bar */}
            <div
                onClick={handleSeek}
                style={{
                    flex: 1,
                    height: '6px',
                    backgroundColor: '#333',
                    borderRadius: '3px',
                    cursor: 'pointer',
                    position: 'relative',
                }}
            >
                <div style={{
                    width: `${progress}%`,
                    height: '100%',
                    backgroundColor: '#10B981',
                    borderRadius: '3px',
                    transition: 'width 0.1s',
                }} />
            </div>

            {/* Volume */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <VolumeIcon />
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={volume}
                    onChange={(e) => {
                        const v = parseFloat(e.target.value);
                        setVolume(v);
                        if (audioRef.current) audioRef.current.volume = v;
                    }}
                    style={{ width: '60px', height: '4px' }}
                />
            </div>
        </div>
    );
};

// Audio Trimmer Component with Waveform Visualization
var AudioTrimmer = ({ onAccept, onClear, onFileLoad }) => {
    const [file, setFile] = useState(null);
    const [clipDuration, setClipDuration] = useState(10);
    const [regionStart, setRegionStart] = useState(0);
    const [totalDuration, setTotalDuration] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [isAccepted, setIsAccepted] = useState(false);
    const [isExpanded, setIsExpanded] = useState(false);
    const [waveformPeaks, setWaveformPeaks] = useState([]);
    const [isDragging, setIsDragging] = useState(false);
    const [error, setError] = useState(null);

    const waveformRef = useRef(null);
    const wavesurferRef = useRef(null);
    const fileInputRef = useRef(null);
    const audioContextRef = useRef(null);
    const audioBufferRef = useRef(null);
    const playIntervalRef = useRef(null);

    // Handle drag/click on waveform to move selection
    const handleWaveformInteraction = (e, containerRef, padding = 12) => {
        if (totalDuration <= 0 || isLoading) return;
        const rect = containerRef.getBoundingClientRect();
        const clickX = e.clientX - rect.left - padding;
        const containerWidth = rect.width - (padding * 2);
        const clickPercent = Math.max(0, Math.min(1, clickX / containerWidth));
        const clickTime = clickPercent * totalDuration;
        const newStart = Math.max(0, Math.min(totalDuration - clipDuration, clickTime - clipDuration / 2));
        setRegionStart(newStart);
    };

    const handleMouseDown = (e, containerRef, padding) => {
        setIsDragging(true);
        handleWaveformInteraction(e, containerRef, padding);
    };

    const handleMouseMove = (e, containerRef, padding) => {
        if (!isDragging) return;
        handleWaveformInteraction(e, containerRef, padding);
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    // Global mouse up listener to stop dragging
    useEffect(() => {
        const handleGlobalMouseUp = () => setIsDragging(false);
        window.addEventListener('mouseup', handleGlobalMouseUp);
        return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
    }, []);

    // Initialize WaveSurfer when file is loaded
    useEffect(() => {
        if (!file || !waveformRef.current) return;

        setIsLoading(true);
        setError(null);

        // Clean up previous instance
        if (wavesurferRef.current) {
            wavesurferRef.current.destroy();
        }

        // Create WaveSurfer instance
        const ws = WaveSurfer.create({
            container: waveformRef.current,
            waveColor: '#4a4a4a',
            progressColor: '#4a4a4a',
            cursorColor: 'transparent',
            cursorWidth: 0,
            height: 80,
            barWidth: 2,
            barGap: 1,
            barRadius: 2,
            normalize: true,
            backend: 'WebAudio',
            interact: false,
        });

        wavesurferRef.current = ws;

        // Load audio file
        const url = URL.createObjectURL(file);
        ws.load(url);

        ws.on('ready', async () => {
            setIsLoading(false);
            const duration = ws.getDuration();
            setTotalDuration(duration);

            const initialClipDuration = Math.min(clipDuration, duration);
            setClipDuration(initialClipDuration);
            setRegionStart(0);

            try {
                if (!audioContextRef.current) {
                    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
                }
                const arrayBuffer = await file.arrayBuffer();
                const decodedBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
                audioBufferRef.current = decodedBuffer;
                console.log(`Audio decoded: ${decodedBuffer.sampleRate}Hz, ${decodedBuffer.numberOfChannels}ch, ${decodedBuffer.length} samples`);

                const numBars = 200;
                const channelData = decodedBuffer.getChannelData(0);
                const samplesPerBar = Math.floor(channelData.length / numBars);
                const peaks = [];
                for (let i = 0; i < numBars; i++) {
                    let max = 0;
                    const start = i * samplesPerBar;
                    for (let j = 0; j < samplesPerBar; j++) {
                        const val = Math.abs(channelData[start + j] || 0);
                        if (val > max) max = val;
                    }
                    peaks.push(max);
                }
                const maxPeak = Math.max(...peaks, 0.01);
                const normalizedPeaks = peaks.map(p => (p / maxPeak) * 100);
                setWaveformPeaks(normalizedPeaks);
            } catch (err) {
                console.error('Failed to decode audio:', err);
                if (ws.getDecodedData()) {
                    audioBufferRef.current = ws.getDecodedData();
                }
            }
        });

        ws.on('error', (err) => {
            setIsLoading(false);
            setError('Failed to load audio file');
            console.error('WaveSurfer error:', err);
        });

        ws.on('finish', () => {
            setIsPlaying(false);
        });

        return () => {
            URL.revokeObjectURL(url);
            if (wavesurferRef.current) {
                wavesurferRef.current.destroy();
                wavesurferRef.current = null;
            }
        };
    }, [file]);

    const handleFileSelect = async (e) => {
        const selectedFile = e.target.files?.[0];
        if (!selectedFile) return;

        const validTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/flac', 'audio/ogg'];
        if (!validTypes.includes(selectedFile.type) && !selectedFile.name.match(/\.(wav|mp3|flac|ogg)$/i)) {
            setError('Please select a valid audio file (WAV, MP3, FLAC, or OGG)');
            return;
        }

        setFile(selectedFile);
        setIsAccepted(false);
        setRegionStart(0);
        if (onFileLoad) onFileLoad(true);
    };

    const stopPlayback = () => {
        if (playIntervalRef.current) {
            clearInterval(playIntervalRef.current);
            playIntervalRef.current = null;
        }
        if (wavesurferRef.current) {
            wavesurferRef.current.pause();
        }
        setIsPlaying(false);
    };

    const handlePreview = () => {
        if (!wavesurferRef.current) return;

        if (isPlaying) {
            stopPlayback();
        } else {
            if (playIntervalRef.current) {
                clearInterval(playIntervalRef.current);
            }

            wavesurferRef.current.setTime(regionStart);
            wavesurferRef.current.play();
            setIsPlaying(true);

            playIntervalRef.current = setInterval(() => {
                if (wavesurferRef.current) {
                    const currentTime = wavesurferRef.current.getCurrentTime();
                    if (currentTime >= regionStart + clipDuration || !wavesurferRef.current.isPlaying()) {
                        stopPlayback();
                    }
                }
            }, 50);
        }
    };

    const audioBufferToWav = (buffer) => {
        const numChannels = buffer.numberOfChannels;
        const sampleRate = buffer.sampleRate;
        const format = 3;
        const bitDepth = 32;

        const bytesPerSample = bitDepth / 8;
        const blockAlign = numChannels * bytesPerSample;
        const byteRate = sampleRate * blockAlign;
        const dataSize = buffer.length * blockAlign;
        const headerSize = 44;
        const totalSize = headerSize + dataSize;

        const arrayBuffer = new ArrayBuffer(totalSize);
        const view = new DataView(arrayBuffer);

        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, totalSize - 8, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(36, 'data');
        view.setUint32(40, dataSize, true);

        const channels = [];
        for (let ch = 0; ch < numChannels; ch++) {
            channels.push(buffer.getChannelData(ch));
        }

        let offset = 44;
        for (let i = 0; i < buffer.length; i++) {
            for (let ch = 0; ch < numChannels; ch++) {
                view.setFloat32(offset, channels[ch][i], true);
                offset += 4;
            }
        }

        return new Blob([arrayBuffer], { type: 'audio/wav' });
    };

    const handleAccept = async () => {
        stopPlayback();

        if (!audioBufferRef.current) {
            setError('Audio not loaded yet');
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            const sourceBuffer = audioBufferRef.current;
            const sampleRate = sourceBuffer.sampleRate;
            const startSample = Math.floor(regionStart * sampleRate);
            const numSamples = Math.floor(clipDuration * sampleRate);

            if (!audioContextRef.current) {
                audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
            }

            const trimmedBuffer = audioContextRef.current.createBuffer(
                sourceBuffer.numberOfChannels,
                numSamples,
                sampleRate
            );

            for (let ch = 0; ch < sourceBuffer.numberOfChannels; ch++) {
                const src = sourceBuffer.getChannelData(ch);
                const dst = trimmedBuffer.getChannelData(ch);
                for (let i = 0; i < numSamples && (startSample + i) < src.length; i++) {
                    dst[i] = src[startSample + i];
                }
            }

            const wavBlob = audioBufferToWav(trimmedBuffer);

            const formData = new FormData();
            formData.append('file', wavBlob, `trimmed_${file.name.replace(/\.[^.]+$/, '')}.wav`);

            const response = await fetch('/api/upload-reference', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(await response.text());
            }

            const data = await response.json();
            setIsAccepted(true);
            setIsLoading(false);

            onAccept({
                id: data.id,
                fileName: file.name,
                clipStart: regionStart,
                clipDuration: clipDuration
            });
        } catch (err) {
            setError(err.message || 'Failed to process audio');
            setIsLoading(false);
        }
    };

    const handleClear = () => {
        if (wavesurferRef.current) {
            wavesurferRef.current.destroy();
            wavesurferRef.current = null;
        }
        setFile(null);
        setIsAccepted(false);
        setRegionStart(0);
        setTotalDuration(0);
        setError(null);
        audioBufferRef.current = null;
        if (onFileLoad) onFileLoad(false);
        onClear();
    };

    const handleDurationChange = (newDuration) => {
        const maxDuration = Math.min(newDuration, totalDuration);
        setClipDuration(maxDuration);

        if (regionStart + maxDuration > totalDuration) {
            setRegionStart(Math.max(0, totalDuration - maxDuration));
        }
    };

    // Render upload state
    if (!file) {
        return (
            <div>
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*"
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                />
                <button
                    onClick={() => fileInputRef.current?.click()}
                    style={{
                        width: '100%',
                        padding: '20px',
                        borderRadius: '10px',
                        border: '2px dashed #3a3a3a',
                        backgroundColor: 'transparent',
                        color: '#666',
                        fontSize: '13px',
                        cursor: 'pointer',
                        transition: 'border-color 0.2s',
                    }}
                    onMouseOver={e => e.target.style.borderColor = '#10B981'}
                    onMouseOut={e => e.target.style.borderColor = '#3a3a3a'}
                >
                    Click to upload reference audio
                </button>
                {error && <div style={{ color: '#EF4444', fontSize: '12px', marginTop: '8px' }}>{error}</div>}
            </div>
        );
    }

    // Render accepted state
    if (isAccepted) {
        return (
            <div style={{
                backgroundColor: '#1e1e1e',
                borderRadius: '10px',
                padding: '14px',
                border: '1px solid #10B981',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span style={{ fontSize: '18px' }}>&#x2713;</span>
                        <div>
                            <div style={{ fontSize: '13px', color: '#e0e0e0' }}>{file.name}</div>
                            <div style={{ fontSize: '11px', color: '#10B981' }}>
                                {formatTime(regionStart)} - {formatTime(regionStart + clipDuration)} ({clipDuration}s clip)
                            </div>
                        </div>
                    </div>
                    <button
                        onClick={handleClear}
                        style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#666',
                            cursor: 'pointer',
                            fontSize: '16px',
                            padding: '4px 8px',
                        }}
                    >x</button>
                </div>
            </div>
        );
    }

    // Render trimmer UI
    return (
        <div>
            {/* File info */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                backgroundColor: '#1e1e1e',
                padding: '10px 14px',
                borderRadius: '10px',
                marginBottom: '12px',
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontSize: '16px' }}>&#x1F3B5;</span>
                    <span style={{ fontSize: '13px', color: '#e0e0e0', maxWidth: '180px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {file.name}
                    </span>
                </div>
                <button
                    onClick={handleClear}
                    style={{
                        background: 'transparent',
                        border: 'none',
                        color: '#666',
                        cursor: 'pointer',
                        fontSize: '16px',
                    }}
                >x</button>
            </div>

            {/* Duration input */}
            <div style={{ marginBottom: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <span style={{ fontSize: '12px', color: '#666' }}>Clip Duration</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <input
                            type="number"
                            min="1"
                            max={Math.floor(totalDuration) || 60}
                            value={clipDuration}
                            onChange={e => handleDurationChange(parseInt(e.target.value) || 1)}
                            className="input-base input-small"
                            style={{ width: '50px', color: '#10B981', textAlign: 'center' }}
                        />
                        <span style={{ fontSize: '12px', color: '#666' }}>seconds</span>
                    </div>
                </div>
            </div>

            {/* Waveform container */}
            <div style={{ position: 'relative' }}>
                {/* Expand button */}
                {totalDuration > 0 && !isLoading && (
                    <button
                        onClick={() => setIsExpanded(true)}
                        style={{
                            position: 'absolute',
                            top: '8px',
                            right: '8px',
                            zIndex: 5,
                            background: 'rgba(0,0,0,0.5)',
                            border: '1px solid #3a3a3a',
                            borderRadius: '6px',
                            padding: '6px 10px',
                            color: '#888',
                            fontSize: '11px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                            transition: 'all 0.2s',
                        }}
                        onMouseOver={e => { e.currentTarget.style.color = '#10B981'; e.currentTarget.style.borderColor = '#10B981'; }}
                        onMouseOut={e => { e.currentTarget.style.color = '#888'; e.currentTarget.style.borderColor = '#3a3a3a'; }}
                    >
                        <ExpandIcon />
                        Expand
                    </button>
                )}

                <div
                    className="waveform-container"
                    style={{
                        position: 'relative',
                        overflow: 'hidden',
                        cursor: totalDuration > 0 ? (isDragging ? 'grabbing' : 'pointer') : 'default',
                        userSelect: 'none',
                    }}
                    onMouseDown={(e) => handleMouseDown(e, e.currentTarget, 12)}
                    onMouseMove={(e) => handleMouseMove(e, e.currentTarget, 12)}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                >
                    {isLoading && (
                        <div style={{
                            position: 'absolute',
                            top: 0, left: 0, right: 0, bottom: 0,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            backgroundColor: 'rgba(26, 26, 26, 0.8)',
                            borderRadius: '8px',
                            zIndex: 10,
                        }}>
                            <span style={{ color: '#10B981', fontSize: '13px' }}>Loading...</span>
                        </div>
                    )}

                    {/* Waveform wrapper with selection overlay */}
                    <div style={{ position: 'relative' }}>
                        <div ref={waveformRef} />

                        {/* Region overlay visualization */}
                        {totalDuration > 0 && !isLoading && (
                            <div style={{
                                position: 'absolute',
                                top: 0,
                                left: `${(regionStart / totalDuration) * 100}%`,
                                width: `${(clipDuration / totalDuration) * 100}%`,
                                height: '100%',
                                backgroundColor: 'rgba(16, 185, 129, 0.25)',
                                borderLeft: '2px solid #10B981',
                                borderRight: '2px solid #10B981',
                                pointerEvents: 'none',
                                boxSizing: 'border-box',
                            }} />
                        )}
                    </div>
                </div>
            </div>

            {/* Expanded waveform popup */}
            {isExpanded && (
                <div
                    style={{
                        position: 'fixed',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        backgroundColor: 'rgba(0, 0, 0, 0.85)',
                        zIndex: 9999,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        padding: '40px',
                        animation: 'fadeIn 0.2s ease-out',
                    }}
                    onClick={(e) => {
                        if (e.target === e.currentTarget) setIsExpanded(false);
                    }}
                >
                    <div style={{
                        width: '100%',
                        maxWidth: '1200px',
                        backgroundColor: '#1e1e1e',
                        borderRadius: '16px',
                        padding: '24px',
                        animation: 'slideUp 0.3s ease-out',
                    }}>
                        {/* Header */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                            <div>
                                <div style={{ fontSize: '16px', fontWeight: '600', color: '#e0e0e0' }}>{file?.name}</div>
                                <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                                    Click on the waveform to position your {clipDuration}s selection
                                </div>
                            </div>
                            <button
                                onClick={() => setIsExpanded(false)}
                                style={{
                                    background: 'transparent',
                                    border: '1px solid #3a3a3a',
                                    borderRadius: '8px',
                                    padding: '8px 16px',
                                    color: '#888',
                                    fontSize: '13px',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '6px',
                                }}
                            >
                                <CloseIcon size={14} />
                                Close
                            </button>
                        </div>

                        {/* Large waveform */}
                        <div
                            style={{
                                backgroundColor: '#1a1a1a',
                                borderRadius: '12px',
                                padding: '20px',
                                cursor: totalDuration > 0 ? (isDragging ? 'grabbing' : 'pointer') : 'default',
                                position: 'relative',
                                userSelect: 'none',
                            }}
                            onMouseDown={(e) => handleMouseDown(e, e.currentTarget, 20)}
                            onMouseMove={(e) => handleMouseMove(e, e.currentTarget, 20)}
                            onMouseUp={handleMouseUp}
                            onMouseLeave={handleMouseUp}
                        >
                            <div style={{ height: '120px', position: 'relative' }}>
                                <div style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    height: '100%',
                                    gap: '1px',
                                }}>
                                    {waveformPeaks.map((peak, i) => {
                                        const pos = i / waveformPeaks.length;
                                        const inSelection = pos >= (regionStart / totalDuration) && pos <= ((regionStart + clipDuration) / totalDuration);
                                        return (
                                            <div
                                                key={i}
                                                style={{
                                                    flex: 1,
                                                    height: `${Math.max(5, peak)}%`,
                                                    backgroundColor: inSelection ? '#10B981' : '#4a4a4a',
                                                    borderRadius: '1px',
                                                }}
                                            />
                                        );
                                    })}
                                </div>

                                {/* Selection overlay */}
                                <div style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: `${(regionStart / totalDuration) * 100}%`,
                                    width: `${(clipDuration / totalDuration) * 100}%`,
                                    height: '100%',
                                    backgroundColor: 'rgba(16, 185, 129, 0.15)',
                                    borderLeft: '3px solid #10B981',
                                    borderRight: '3px solid #10B981',
                                    pointerEvents: 'none',
                                    boxSizing: 'border-box',
                                }} />
                            </div>

                            {/* Time markers */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px' }}>
                                <span style={{ fontSize: '11px', color: '#666' }}>0:00</span>
                                <span style={{ fontSize: '11px', color: '#666' }}>{formatTime(totalDuration / 2)}</span>
                                <span style={{ fontSize: '11px', color: '#666' }}>{formatTime(totalDuration)}</span>
                            </div>
                        </div>

                        {/* Selection info and controls */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '20px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <span style={{ fontSize: '13px', color: '#888' }}>Clip Duration:</span>
                                    <input
                                        type="number"
                                        min="1"
                                        max={Math.floor(totalDuration) || 60}
                                        value={clipDuration}
                                        onChange={e => handleDurationChange(parseInt(e.target.value) || 1)}
                                        className="input-base input-small"
                                        style={{ width: '60px', color: '#10B981', textAlign: 'center', fontSize: '14px' }}
                                    />
                                    <span style={{ fontSize: '13px', color: '#888' }}>sec</span>
                                </div>
                                <div style={{ fontSize: '14px', color: '#10B981', fontWeight: '500' }}>
                                    Selection: {formatTime(regionStart)} - {formatTime(regionStart + clipDuration)}
                                </div>
                            </div>
                            <div style={{ display: 'flex', gap: '12px' }}>
                                <button className="btn btn-secondary" onClick={handlePreview} style={{ color: isPlaying ? '#10B981' : '#e0e0e0', padding: '10px 20px' }}>
                                    {isPlaying ? <><PauseIcon size={14} /> Stop</> : <><PlayIcon size={14} /> Preview</>}
                                </button>
                                <button className="btn btn-primary" onClick={() => { setIsExpanded(false); handleAccept(); }} style={{ padding: '10px 24px' }}>
                                    Accept Selection
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Selection info */}
            {totalDuration > 0 && (
                <div style={{ marginTop: '8px', display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '11px', color: '#888' }}>
                        Selection: {formatTime(regionStart)} - {formatTime(regionStart + clipDuration)}
                    </span>
                    <span style={{ fontSize: '11px', color: '#666' }}>
                        Total: {formatTime(totalDuration)}
                    </span>
                </div>
            )}

            {/* Controls */}
            <div className="flex gap-3" style={{ marginTop: '12px' }}>
                <button
                    className="btn btn-secondary"
                    onClick={handlePreview}
                    disabled={isLoading || totalDuration === 0}
                    style={{ flex: 1, color: isPlaying ? '#10B981' : '#e0e0e0', cursor: isLoading ? 'not-allowed' : 'pointer' }}
                >
                    {isPlaying ? <><PauseIcon /> Stop</> : <><PlayIcon /> Preview</>}
                </button>
                <button
                    className="btn btn-primary"
                    onClick={handleAccept}
                    disabled={isLoading || totalDuration === 0}
                    style={{ flex: 1, cursor: isLoading ? 'not-allowed' : 'pointer', opacity: isLoading ? 0.7 : 1 }}
                >
                    {isLoading ? 'Processing...' : 'Accept'}
                </button>
            </div>

            {error && <div style={{ color: '#EF4444', fontSize: '12px', marginTop: '8px' }}>{error}</div>}
        </div>
    );
};

// Multi-select with horizontal scrolling suggestions
var MultiSelectWithScroll = ({ suggestions, selected, onChange, placeholder }) => {
    const [input, setInput] = useState('');
    const scrollRef = useRef(null);

    const addTag = (tag) => {
        if (tag && !selected.includes(tag)) onChange([...selected, tag]);
        setInput('');
    };

    const removeTag = (tag) => onChange(selected.filter(t => t !== tag));

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && input.trim()) { e.preventDefault(); addTag(input.trim()); }
        else if (e.key === 'Backspace' && !input && selected.length) removeTag(selected[selected.length - 1]);
    };

    return (
        <div>
            <div className="input-base" style={{
                padding: '10px 12px',
                minHeight: '42px',
                display: 'flex',
                flexWrap: 'wrap',
                gap: '6px',
                alignItems: 'center',
            }}>
                {selected.map(tag => (
                    <span key={tag} style={{
                        backgroundColor: '#10B98130',
                        color: '#10B981',
                        padding: '4px 10px',
                        borderRadius: '6px',
                        fontSize: '12px',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        whiteSpace: 'nowrap',
                    }}>
                        {tag}
                        <span style={{ cursor: 'pointer', opacity: 0.7, fontSize: '14px' }} onClick={() => removeTag(tag)}>×</span>
                    </span>
                ))}
                <input
                    type="text"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={selected.length === 0 ? placeholder : 'Type to add...'}
                    style={{
                        border: 'none',
                        background: 'transparent',
                        color: '#e0e0e0',
                        fontSize: '13px',
                        outline: 'none',
                        flex: 1,
                        minWidth: '80px',
                    }}
                />
            </div>
            {/* Horizontal scrolling tags */}
            <div
                ref={scrollRef}
                style={{
                    display: 'flex',
                    gap: '6px',
                    marginTop: '8px',
                    overflowX: 'auto',
                    paddingBottom: '4px',
                    scrollbarWidth: 'thin',
                    scrollbarColor: '#3a3a3a transparent',
                }}
            >
                {suggestions.filter(s => !selected.includes(s)).map(tag => (
                    <button
                        key={tag}
                        onClick={() => addTag(tag)}
                        style={{
                            padding: '5px 12px',
                            borderRadius: '16px',
                            border: '1px solid #3a3a3a',
                            backgroundColor: '#2a2a2a',
                            color: '#999',
                            fontSize: '11px',
                            cursor: 'pointer',
                            whiteSpace: 'nowrap',
                            flexShrink: 0,
                            transition: 'all 0.15s',
                        }}
                        onMouseOver={e => { e.target.style.borderColor = '#10B981'; e.target.style.color = '#10B981'; }}
                        onMouseOut={e => { e.target.style.borderColor = '#3a3a3a'; e.target.style.color = '#999'; }}
                    >
                        + {tag}
                    </button>
                ))}
            </div>
        </div>
    );
};

// Card component
var Card = ({ children }) => (
    <div style={{
        backgroundColor: '#282828',
        borderRadius: '16px',
        padding: '20px',
    }}>
        {children}
    </div>
);

// Card title component
var CardTitle = ({ children }) => (
    <div style={{ fontSize: '13px', fontWeight: '500', color: '#888', marginBottom: '14px' }}>
        {children}
    </div>
);

// Section card component
var SectionCard = ({ section, onUpdate, onRemove }) => {
    const { base } = fromApiType(section.type);
    const cfg = SECTION_TYPES[base] || { name: base, color: '#888', hasLyrics: true };
    const lineCount = Math.max(3, (section.lyrics || '').split('\n').length + 1);

    return (
        <div style={{
            backgroundColor: '#282828',
            borderRadius: '12px',
            borderLeft: `4px solid ${cfg.color}`,
            overflow: 'hidden',
        }}>
            <div style={{
                padding: '14px 18px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
            }}>
                <span style={{ fontSize: '15px', fontWeight: '500', color: cfg.color }}>{cfg.name}</span>
                <button
                    onClick={onRemove}
                    style={{
                        background: 'transparent',
                        border: 'none',
                        color: '#555',
                        cursor: 'pointer',
                        fontSize: '20px',
                        padding: '4px 8px',
                        lineHeight: 1,
                        transition: 'color 0.15s',
                    }}
                    onMouseEnter={e => e.target.style.color = '#888'}
                    onMouseLeave={e => e.target.style.color = '#555'}
                >x</button>
            </div>
            {cfg.hasLyrics && (
                <div style={{ padding: '0 18px 16px 18px' }}>
                    <textarea
                        value={section.lyrics || ''}
                        onChange={e => onUpdate({ lyrics: e.target.value })}
                        placeholder="Enter lyrics..."
                        rows={lineCount}
                        style={{
                            width: '100%',
                            backgroundColor: 'transparent',
                            border: 'none',
                            color: '#999',
                            fontSize: '14px',
                            lineHeight: '1.8',
                            resize: 'none',
                            outline: 'none',
                        }}
                    />
                </div>
            )}
        </div>
    );
};

// Edit Modal component
var EditModal = ({ item, onClose, onSave }) => {
    const [title, setTitle] = useState(item.metadata?.title || item.title || '');
    const [coverPreview, setCoverPreview] = useState(null);
    const [coverFile, setCoverFile] = useState(null);
    const [saving, setSaving] = useState(false);
    const [removeCover, setRemoveCover] = useState(false);  // Track if user wants to remove cover
    // Initialize hasCover from metadata if available for immediate display
    const [hasCover, setHasCover] = useState(() => {
        return !!(item.metadata?.cover);
    });
    // Store existing cover URL to maintain it during edits
    const [existingCoverUrl, setExistingCoverUrl] = useState(() => {
        if (item.metadata?.cover) {
            return `/api/generation/${item.id}/cover?t=${Date.now()}`;
        }
        return null;
    });
    const fileInputRef = useRef(null);

    // Check if cover exists on mount (fallback for cases where metadata doesn't have cover flag)
    useEffect(() => {
        if (!hasCover && item.id) {
            // Try to load the image directly - more reliable than HEAD request
            const img = new Image();
            img.onload = () => {
                setHasCover(true);
                setExistingCoverUrl(`/api/generation/${item.id}/cover?t=${Date.now()}`);
            };
            img.onerror = () => {}; // No cover exists, that's fine
            img.src = `/api/generation/${item.id}/cover?t=${Date.now()}`;
        }
    }, [item.id, hasCover]);

    const handleFileSelect = (e) => {
        const file = e.target.files[0];
        if (file) {
            setCoverFile(file);
            setRemoveCover(false);  // Reset remove flag when new file selected
            const reader = new FileReader();
            reader.onload = (e) => setCoverPreview(e.target.result);
            reader.readAsDataURL(file);
        }
    };

    const handleRemoveCover = () => {
        setRemoveCover(true);
        setCoverFile(null);
        setCoverPreview(null);
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            // Update title
            await fetch(`/api/generation/${item.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title })
            });

            // Handle cover: remove, replace, or keep
            if (removeCover && !coverFile) {
                // User wants to remove cover
                await fetch(`/api/generation/${item.id}/cover`, {
                    method: 'DELETE'
                });
            } else if (coverFile) {
                // Upload new cover
                const formData = new FormData();
                formData.append('file', coverFile);
                await fetch(`/api/generation/${item.id}/cover`, {
                    method: 'POST',
                    body: formData
                });
            }

            onSave && onSave();
            onClose();
        } catch (e) {
            console.error('Save failed:', e);
        } finally {
            setSaving(false);
        }
    };

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.8)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
        }} onClick={onClose}>
            <div style={{
                backgroundColor: '#282828',
                borderRadius: '16px',
                padding: '24px',
                width: '400px',
                maxWidth: '90vw',
                border: '1px solid #3a3a3a',
            }} onClick={e => e.stopPropagation()}>
                <div style={{ fontSize: '18px', fontWeight: '600', color: '#e0e0e0', marginBottom: '20px' }}>
                    Edit Song
                </div>

                {/* Title Input */}
                <div style={{ marginBottom: '20px' }}>
                    <label style={{ display: 'block', fontSize: '12px', color: '#888', marginBottom: '8px' }}>Title</label>
                    <input
                        type="text"
                        value={title}
                        onChange={e => setTitle(e.target.value)}
                        style={{
                            width: '100%',
                            backgroundColor: '#1e1e1e',
                            border: '1px solid #3a3a3a',
                            borderRadius: '8px',
                            padding: '12px',
                            color: '#e0e0e0',
                            fontSize: '14px',
                            outline: 'none',
                            boxSizing: 'border-box',
                        }}
                        placeholder="Enter song title..."
                    />
                </div>

                {/* Album Cover Upload */}
                <div style={{ marginBottom: '24px' }}>
                    <label style={{ display: 'block', fontSize: '12px', color: '#888', marginBottom: '8px' }}>Album Cover</label>
                    <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
                        {/* Preview */}
                        <div style={{
                            width: '100px',
                            height: '100px',
                            borderRadius: '8px',
                            backgroundColor: '#1e1e1e',
                            border: '1px solid #3a3a3a',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            overflow: 'hidden',
                            flexShrink: 0,
                            position: 'relative',
                        }}>
                            {coverPreview ? (
                                <img src={coverPreview} style={{ width: '100%', height: '100%', objectFit: 'cover' }} alt="New cover" />
                            ) : (!removeCover && hasCover && existingCoverUrl) ? (
                                <img src={existingCoverUrl} style={{ width: '100%', height: '100%', objectFit: 'cover' }} alt="Album cover" />
                            ) : (
                                <MusicNoteIcon size={32} color="#444" />
                            )}
                            {/* Trash icon - show when there's an image to remove */}
                            {(coverPreview || (!removeCover && hasCover && existingCoverUrl)) && (
                                <button
                                    onClick={handleRemoveCover}
                                    title="Remove cover"
                                    style={{
                                        position: 'absolute',
                                        top: '4px',
                                        right: '4px',
                                        width: '22px',
                                        height: '22px',
                                        borderRadius: '50%',
                                        backgroundColor: 'rgba(0,0,0,0.7)',
                                        border: 'none',
                                        cursor: 'pointer',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        padding: 0,
                                        transition: 'background-color 0.15s',
                                    }}
                                    onMouseEnter={e => e.target.style.backgroundColor = 'rgba(239,68,68,0.9)'}
                                    onMouseLeave={e => e.target.style.backgroundColor = 'rgba(0,0,0,0.7)'}
                                >
                                    <TrashIcon size={12} color="#fff" />
                                </button>
                            )}
                        </div>
                        {/* Upload Button */}
                        <div style={{ flex: 1 }}>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/*"
                                onChange={handleFileSelect}
                                style={{ display: 'none' }}
                            />
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                style={{
                                    width: '100%',
                                    padding: '10px 16px',
                                    backgroundColor: '#1e1e1e',
                                    border: '1px dashed #3a3a3a',
                                    borderRadius: '8px',
                                    color: '#888',
                                    cursor: 'pointer',
                                    fontSize: '13px',
                                    marginBottom: '8px',
                                }}
                            >
                                Choose Image...
                            </button>
                            <div style={{ fontSize: '11px', color: '#555' }}>
                                JPG, PNG, GIF or WebP. Square recommended.
                            </div>
                        </div>
                    </div>
                </div>

                {/* Buttons */}
                <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
                    <button
                        onClick={onClose}
                        style={{
                            padding: '10px 20px',
                            backgroundColor: '#1e1e1e',
                            border: '1px solid #3a3a3a',
                            borderRadius: '8px',
                            color: '#888',
                            cursor: 'pointer',
                            fontSize: '14px',
                        }}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={saving}
                        style={{
                            padding: '10px 20px',
                            backgroundColor: '#10B981',
                            border: 'none',
                            borderRadius: '8px',
                            color: '#fff',
                            cursor: saving ? 'not-allowed' : 'pointer',
                            fontSize: '14px',
                            fontWeight: '500',
                            opacity: saving ? 0.7 : 1,
                        }}
                    >
                        {saving ? 'Saving...' : 'Save'}
                    </button>
                </div>
            </div>
        </div>
    );
};

// Library item component
var LibraryItem = ({ item, isQueued, isGenerating, queuePosition, onRemoveFromQueue, onStop, onDelete, onPlay, onUpdate, isCurrentlyPlaying, isAudioPlaying, status, elapsedTime, estimatedTime }) => {
    const [expanded, setExpanded] = useState(false);
    const [editing, setEditing] = useState(false);
    const [exportingMp4, setExportingMp4] = useState(false);
    const [exportError, setExportError] = useState(null);
    const meta = (isQueued || isGenerating) ? item : (item.metadata || {});

    // Initialize coverUrl from metadata if available (prevents flash on re-render)
    const [coverUrl, setCoverUrl] = useState(() => {
        if (!isQueued && !isGenerating && item.id && meta.cover) {
            return `/api/generation/${item.id}/cover?t=${Date.now()}`;
        }
        return null;
    });
    // Track if we've already verified the cover
    const coverCheckedRef = useRef(false);

    // Check for cover image - also watch metadata.cover for updates
    useEffect(() => {
        if (!isQueued && !isGenerating && item.id) {
            // If metadata already indicates a cover exists, use it directly
            if (meta.cover) {
                setCoverUrl(`/api/generation/${item.id}/cover?t=${Date.now()}`);
                coverCheckedRef.current = true;
            } else {
                // Cover was removed or doesn't exist - clear the URL
                setCoverUrl(null);
                // Only do image check once per item if metadata doesn't have cover flag
                if (!coverCheckedRef.current) {
                    coverCheckedRef.current = true;
                    const img = new Image();
                    img.onload = () => {
                        setCoverUrl(`/api/generation/${item.id}/cover?t=${Date.now()}`);
                    };
                    img.onerror = () => {}; // No cover, that's fine
                    img.src = `/api/generation/${item.id}/cover?t=${Date.now()}`;
                }
            }
        }
    }, [item.id, isQueued, isGenerating, meta.cover]);

    const formatDate = (dateStr) => {
        if (!dateStr) return 'Unknown';
        try {
            return new Date(dateStr).toLocaleString();
        } catch { return dateStr; }
    };

    const formatDuration = (start, end) => {
        if (!start || !end) return '';
        try {
            const ms = new Date(end) - new Date(start);
            const mins = Math.floor(ms / 60000);
            const secs = Math.floor((ms % 60000) / 1000);
            return `${mins}m ${secs}s`;
        } catch { return ''; }
    };

    const canPlay = !isQueued && !isGenerating && item.status === 'completed' && (item.output_files?.length > 0 || item.output_file);
    const albumClass = isCurrentlyPlaying ? 'playing' : isGenerating ? 'generating' : isQueued ? 'queued' : item.status === 'failed' ? 'failed' : 'default';

    // Export as MP4 video with progress feedback
    const exportMp4 = async () => {
        if (exportingMp4) return;
        setExportingMp4(true);
        setExportError(null);

        try {
            const response = await fetch(`/api/generation/${item.id}/video`);
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || 'Export failed');
            }

            // Get the blob and trigger download
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${item.title || meta.title || 'song'}.mp4`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error('MP4 export error:', err);
            setExportError(err.message);
        } finally {
            setExportingMp4(false);
        }
    };

    return (
        <div className="card-base" style={{
            padding: '12px',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
            border: isCurrentlyPlaying ? '1px solid #10B981' : '1px solid transparent',
        }}>
            {/* Main row */}
            <div className="flex gap-4">
            {/* Album Cover with Play Button */}
            <div
                onClick={() => canPlay && onPlay && onPlay(item)}
                className={`album-cover ${albumClass}`}
                style={{
                    cursor: canPlay ? 'pointer' : 'default',
                    backgroundImage: coverUrl ? `url(${coverUrl})` : 'none',
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                }}
            >
                {isGenerating ? (
                    <SpinnerIcon />
                ) : canPlay ? (
                    coverUrl ? (
                        <div className="album-cover-overlay">
                            {isCurrentlyPlaying && isAudioPlaying ? (
                                <PauseLargeIcon />
                            ) : (
                                <PlayLargeIcon style={{ marginLeft: '3px' }} />
                            )}
                        </div>
                    ) : (
                        isCurrentlyPlaying && isAudioPlaying ? (
                            <PauseLargeIcon />
                        ) : (
                            <PlayLargeIcon style={{ marginLeft: '3px' }} />
                        )
                    )
                ) : !coverUrl ? (
                    <MusicNoteIcon />
                ) : null}
            </div>

            {/* Song Info */}
            <div style={{ flex: 1, minWidth: 0 }} className="flex flex-col justify-center">
                <div className="flex items-center gap-3" style={{ marginBottom: '4px' }}>
                    <div className="text-lg font-medium text-primary truncate">{item.title || 'Untitled'}</div>
                    <span className={`tag ${isGenerating ? 'tag-warning' : isQueued ? 'tag-queue' : item.status === 'completed' ? 'tag-primary' : 'tag-error'}`} style={{ flexShrink: 0 }}>
                        {isGenerating ? 'generating' : (isQueued ? `#${queuePosition}` : item.status)}
                    </span>
                </div>
                <div className="text-sm text-secondary" style={{ marginBottom: '6px' }}>
                    {isGenerating ? (status || 'Generating...') : (isQueued ? `Queued` : formatDate(item.created_at))}
                </div>
                {/* Style tags */}
                <div className="flex gap-1" style={{ flexWrap: 'wrap' }}>
                    {meta.gender && <span className="tag tag-info">{meta.gender}</span>}
                    {meta.genre && <span className="tag tag-purple">{meta.genre}</span>}
                    {meta.emotion && <span className="tag tag-warning">{meta.emotion}</span>}
                    {meta.bpm && <span className="tag tag-primary">{meta.bpm} BPM</span>}
                </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-2 items-center" style={{ flexShrink: 0 }}>
                {canPlay && (
                    <>
                        {['FLAC', 'MP3'].map((fmt) => (
                            <button key={fmt} className="btn-icon btn-success" onClick={() => {
                                const a = document.createElement('a');
                                a.href = `/api/audio/${item.id}/0?format=${fmt.toLowerCase()}`;
                                a.download = `${item.title || 'song'}.${fmt.toLowerCase()}`;
                                a.click();
                            }}>{fmt}</button>
                        ))}
                        <button
                            className="btn-icon btn-success"
                            onClick={exportMp4}
                            disabled={exportingMp4}
                            style={{ opacity: exportingMp4 ? 0.7 : 1, minWidth: '50px' }}
                            title={exportError ? `Error: ${exportError}` : 'Export as MP4 video with waveform visualization'}
                        >
                            {exportingMp4 ? (
                                <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                    <span style={{
                                        width: '10px',
                                        height: '10px',
                                        border: '2px solid #10B981',
                                        borderTopColor: 'transparent',
                                        borderRadius: '50%',
                                        animation: 'spin 1s linear infinite',
                                        display: 'inline-block'
                                    }} />
                                </span>
                            ) : 'MP4'}
                        </button>
                    </>
                )}
                {isGenerating && <button className="btn-icon btn-danger" onClick={onStop}>Stop</button>}
                {isQueued && <button className="btn-icon btn-danger" onClick={onRemoveFromQueue}>Remove</button>}
                {!isQueued && !isGenerating && item.status === 'completed' && (
                    <button className="btn-icon" onClick={() => setEditing(true)} title="Edit"><EditIcon /></button>
                )}
                <button className="btn-icon" onClick={() => setExpanded(!expanded)}>{expanded ? 'Hide' : 'Details'}</button>
                {!isQueued && !isGenerating && (item.status === 'completed' || item.status === 'failed' || item.status === 'stopped') && (
                    <button className="btn-icon" onClick={onDelete} title="Delete"><TrashIcon /></button>
                )}
            </div>
            </div>

            {/* Edit Modal */}
            {editing && (
                <EditModal
                    item={item}
                    onClose={() => setEditing(false)}
                    onSave={() => {
                        // Reset coverCheckedRef so useEffect will re-check after library reloads
                        coverCheckedRef.current = false;
                        onUpdate && onUpdate();
                    }}
                />
            )}

            {/* MP4 Export Error Message */}
            {exportError && (
                <div style={{
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    borderRadius: '8px',
                    padding: '10px 14px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ color: '#EF4444', fontWeight: '500', fontSize: '13px' }}>MP4 Export Failed:</span>
                        <span style={{ color: '#f87171', fontSize: '12px' }}>{exportError}</span>
                    </div>
                    <button
                        onClick={() => setExportError(null)}
                        style={{ background: 'none', border: 'none', color: '#EF4444', cursor: 'pointer', padding: '4px' }}
                    >
                        <CloseIcon size={14} />
                    </button>
                </div>
            )}

            {/* Expanded details */}
            {expanded && (
                <div style={{
                    backgroundColor: '#1a1a1a',
                    borderRadius: '8px',
                    padding: '12px',
                    fontSize: '12px',
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)',
                    gap: '8px 16px',
                }}>
                    <div><span style={{ color: '#666' }}>Model:</span> <span style={{ color: '#aaa' }}>{meta.model || 'unknown'}</span></div>
                    <div><span style={{ color: '#666' }}>Output Mode:</span> <span style={{ color: '#aaa' }}>{meta.output_mode || 'mixed'}</span></div>
                    <div><span style={{ color: '#666' }}>Voice:</span> <span style={{ color: '#aaa' }}>{meta.gender || '-'}</span></div>
                    <div><span style={{ color: '#666' }}>BPM:</span> <span style={{ color: '#aaa' }}>{meta.bpm || '-'}</span></div>
                    <div><span style={{ color: '#666' }}>Genre:</span> <span style={{ color: '#aaa' }}>{meta.genre || '-'}</span></div>
                    <div><span style={{ color: '#666' }}>Mood:</span> <span style={{ color: '#aaa' }}>{meta.emotion || '-'}</span></div>
                    <div><span style={{ color: '#666' }}>Timbre:</span> <span style={{ color: '#aaa' }}>{meta.timbre || '-'}</span></div>
                    <div><span style={{ color: '#666' }}>Instruments:</span> <span style={{ color: '#aaa' }}>{meta.instruments || '-'}</span></div>
                    {meta.custom_style && <div style={{ gridColumn: '1 / -1' }}><span style={{ color: '#666' }}>Custom Style:</span> <span style={{ color: '#aaa' }}>{meta.custom_style}</span></div>}
                    {!isQueued && !isGenerating && <div style={{ gridColumn: '1 / -1' }}><span style={{ color: '#666' }}>Created:</span> <span style={{ color: '#aaa' }}>{formatDate(meta.created_at)}</span></div>}
                    {!isQueued && !isGenerating && meta.completed_at && (
                        <div style={{ gridColumn: '1 / -1' }}><span style={{ color: '#666' }}>Completed:</span> <span style={{ color: '#aaa' }}>{formatDate(meta.completed_at)} ({formatDuration(meta.created_at, meta.completed_at)})</span></div>
                    )}
                    {meta.description && (
                        <div style={{ gridColumn: '1 / -1', marginTop: '8px' }}>
                            <div style={{ color: '#666', marginBottom: '4px' }}>Description sent to model:</div>
                            <div style={{ color: '#10B981', fontFamily: 'monospace', fontSize: '11px', padding: '8px', backgroundColor: '#0a0a0a', borderRadius: '4px' }}>{meta.description}</div>
                        </div>
                    )}
                    {meta.sections && meta.sections.length > 0 && (
                        <div style={{ gridColumn: '1 / -1', marginTop: '8px' }}>
                            <div style={{ color: '#666', marginBottom: '4px' }}>Sections ({meta.sections.length}):</div>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                {meta.sections.map((s, i) => (
                                    <div key={i} style={{ padding: '8px 10px', backgroundColor: '#0a0a0a', borderRadius: '4px', borderLeft: `3px solid ${SECTION_TYPES[s.type]?.color || '#666'}` }}>
                                        <div style={{ color: SECTION_TYPES[s.type]?.color || '#888', fontWeight: '500', marginBottom: s.lyrics ? '6px' : '0' }}>[{s.type}]</div>
                                        {s.lyrics && <div style={{ color: '#999', whiteSpace: 'pre-wrap', lineHeight: '1.5', fontSize: '11px' }}>{s.lyrics}</div>}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                    {(meta.reference_audio || meta.reference_audio_id) && (
                        <div style={{ gridColumn: '1 / -1' }}>
                            <div style={{ color: '#666', marginBottom: '8px' }}>Reference Audio (style cloning):</div>
                            <DarkAudioPlayer src={`/api/reference/${meta.reference_audio_id || meta.reference_audio?.replace(/\\/g, '/').split('/').pop()?.split('_')[0]}`} />
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};
