// SongGeneration Studio - Main App Component
// This file contains the main App component and all its logic

var App = () => {
    const [activeTab, setActiveTab] = useState('create');
    const [sections, setSections] = useState([
        { id: '1', type: 'intro-short', lyrics: '' },
        { id: '2', type: 'verse', lyrics: '' },
        { id: '3', type: 'chorus', lyrics: '' },
        { id: '4', type: 'verse', lyrics: '' },
        { id: '5', type: 'outro-short', lyrics: '' },
    ]);
    const [title, setTitle] = useState('My New Song');
    const [gender, setGender] = useState('female');
    const [genres, setGenres] = useState([]);
    const [moods, setMoods] = useState([]);
    const [timbres, setTimbres] = useState([]);
    const [instruments, setInstruments] = useState([]);
    const [customStyle, setCustomStyle] = useState('');
    const [bpm, setBpm] = useState(120);
    const [outputMode, setOutputMode] = useState('mixed');
    const [memoryMode, setMemoryMode] = useState('auto');
    // Advanced generation parameters
    const [cfgCoef, setCfgCoef] = useState(1.5);
    const [temperature, setTemperature] = useState(0.8);
    const [topK, setTopK] = useState(50);
    const [topP, setTopP] = useState(0.0);
    const [extendStride, setExtendStride] = useState(5);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('songgeneration_base');
    const [generating, setGenerating] = useState(false);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState('');
    const [error, setError] = useState(null);
    const [audio, setAudio] = useState(null);
    const [showAddMenu, setShowAddMenu] = useState(false);
    const [addMenuPos, setAddMenuPos] = useState({ x: 0, y: 0 });
    const [library, setLibrary] = useState([]);
    const [refFile, setRefFile] = useState(null);
    const [refId, setRefId] = useState(null);
    const [useReference, setUseReference] = useState(false);
    const [refFileLoaded, setRefFileLoaded] = useState(false);
    const [genStartTime, setGenStartTime] = useState(null);
    const [estimatedTime, setEstimatedTime] = useState(null);
    const [elapsedTime, setElapsedTime] = useState(0);
    const [gpuInfo, setGpuInfo] = useState(null);
    // Queue system - persist to localStorage
    const [queue, setQueue] = useState(() => {
        try {
            const saved = localStorage.getItem('songgen_queue');
            return saved ? JSON.parse(saved) : [];
        } catch { return []; }
    });
    const [currentGenId, setCurrentGenId] = useState(null);
    const [currentGenPayload, setCurrentGenPayload] = useState(null);
    const pollRef = useRef(null);
    const addBtnRef = useRef(null);
    const timerRef = useRef(null);

    // Activity panel audio / Media player
    const [activityPlayingId, setActivityPlayingId] = useState(null);
    const [activityPlayingItem, setActivityPlayingItem] = useState(null);
    const [isAudioPlaying, setIsAudioPlaying] = useState(false);
    const [audioProgress, setAudioProgress] = useState(0);
    const [audioDuration, setAudioDuration] = useState(0);
    const [audioVolume, setAudioVolume] = useState(1);
    const activityAudioRef = useRef(null);

    // Hover states for scrollbar visibility (using shared useHover hook)
    const [leftHover, leftHoverHandlers] = useHover();
    const [mainHover, mainHoverHandlers] = useHover();
    const [rightHover, rightHoverHandlers] = useHover();
    const [libraryHover, libraryHoverHandlers] = useHover();

    useEffect(() => { loadModels(); loadLibrary(); loadGpuInfo(); }, []);
    useEffect(() => () => {
        pollRef.current && clearInterval(pollRef.current);
        timerRef.current && clearInterval(timerRef.current);
    }, []);

    // Persist queue to localStorage
    useEffect(() => {
        localStorage.setItem('songgen_queue', JSON.stringify(queue));
    }, [queue]);

    // Restore running generation on page load
    const restoredRef = useRef(false);
    useEffect(() => {
        if (restoredRef.current || generating || currentGenId) return;

        const runningGen = library.find(item => item.status === 'generating' || item.status === 'processing');
        if (runningGen) {
            restoredRef.current = true;

            const meta = runningGen.metadata || {};
            const payload = {
                title: meta.title || runningGen.title || 'Untitled',
                model: meta.model || 'songgeneration_base',
                sections: meta.sections || 5,
                ...meta
            };

            setGenerating(true);
            setCurrentGenId(runningGen.id);
            setCurrentGenPayload(payload);
            setStatus(runningGen.message || 'Generating...');
            setProgress(runningGen.progress || 0);

            const startTime = runningGen.created_at ? new Date(runningGen.created_at).getTime() : Date.now();
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            setElapsedTime(elapsed);
            setGenStartTime(startTime);

            const sectionCount = typeof payload.sections === 'number' ? payload.sections : (Array.isArray(payload.sections) ? payload.sections.length : 5);
            const baseTime = MODEL_BASE_TIMES[payload.model] || 240;
            const estimated = baseTime + (sectionCount * 30);
            setEstimatedTime(estimated);

            if (timerRef.current) clearInterval(timerRef.current);
            if (pollRef.current) clearInterval(pollRef.current);

            timerRef.current = setInterval(() => {
                setElapsedTime(prev => prev + 1);
            }, 1000);

            pollRef.current = setInterval(() => poll(runningGen.id), 2000);
        }
    }, [library, generating, currentGenId]);

    // Auto-start queue processing
    const queueProcessedRef = useRef(false);
    useEffect(() => {
        if (!queueProcessedRef.current && queue.length > 0 && !generating && !currentGenId && library.length > 0) {
            const hasRunning = library.some(item => item.status === 'generating' || item.status === 'processing');
            if (hasRunning) return;

            queueProcessedRef.current = true;
            setTimeout(() => {
                setQueue(prev => {
                    if (prev.length > 0 && !generating) {
                        const [next, ...rest] = prev;
                        setGenerating(true);
                        startGeneration(next);
                        return rest;
                    }
                    return prev;
                });
            }, 500);
        }
    }, [queue, generating, currentGenId, library]);

    // Auto-set output mode based on lyrics presence
    useEffect(() => {
        const hasLyrics = sections.some(s => s.lyrics && s.lyrics.trim().length > 0);
        if (hasLyrics && outputMode === 'bgm') {
            setOutputMode('mixed');
        } else if (!hasLyrics && outputMode === 'mixed') {
            setOutputMode('bgm');
        }
    }, [sections]);

    // Auto-select best model based on available VRAM
    useEffect(() => {
        if (!gpuInfo?.gpu?.total_gb || models.length === 0) return;
        const vram = gpuInfo.gpu.total_gb;
        let bestModel = 'songgeneration_base';
        if (vram >= 22 && models.some(m => m.id === 'songgeneration_large')) {
            bestModel = 'songgeneration_large';
        } else if (vram >= 12 && models.some(m => m.id === 'songgeneration_base_full')) {
            bestModel = 'songgeneration_base_full';
        } else if (models.some(m => m.id === 'songgeneration_base_new')) {
            bestModel = 'songgeneration_base_new';
        }
        setSelectedModel(bestModel);
    }, [gpuInfo, models]);

    // Time estimation based on model and lyrics
    const estimateGenerationTime = useCallback((model, sectionsList) => {
        let baseTime = MODEL_BASE_TIMES[model] || 240;

        const totalLyrics = sectionsList.reduce((acc, s) => {
            return acc + (s.lyrics || '').length;
        }, 0);

        const numSections = sectionsList.length;
        const lyricsTimeAdjust = Math.floor(totalLyrics / 500) * 30;
        const sectionsAdjust = Math.max(0, numSections - 3) * 10;

        const durationSections = sectionsList.filter(s =>
            s.type.includes('intro') || s.type.includes('outro') || s.type.includes('inst')
        ).length;
        const durationAdjust = durationSections * 15;

        return baseTime + lyricsTimeAdjust + sectionsAdjust + durationAdjust;
    }, []);

    // Close popup when clicking outside
    useEffect(() => {
        const handleClick = (e) => {
            if (showAddMenu && addBtnRef.current && !addBtnRef.current.contains(e.target)) {
                setShowAddMenu(false);
            }
        };
        document.addEventListener('click', handleClick);
        return () => document.removeEventListener('click', handleClick);
    }, [showAddMenu]);

    const loadModels = async () => {
        try {
            const r = await fetch('/api/models');
            if (r.ok) {
                const d = await r.json();
                setModels(d.models || []);
                if (d.default) setSelectedModel(d.default);
            }
        } catch (e) { console.error(e); }
    };

    const loadGpuInfo = async () => {
        try {
            const r = await fetch('/api/gpu');
            if (r.ok) {
                const d = await r.json();
                setGpuInfo(d);
            }
        } catch (e) { console.error(e); }
    };

    const loadLibrary = async () => {
        try {
            const r = await fetch('/api/generations');
            if (r.ok) {
                const data = await r.json();
                setLibrary(data.sort((a, b) => new Date(b.created_at) - new Date(a.created_at)));
            }
        } catch (e) { console.error(e); }
    };

    const deleteGeneration = async (genId) => {
        try {
            const r = await fetch(`/api/generation/${genId}`, { method: 'DELETE' });
            if (r.ok) {
                setLibrary(prev => prev.filter(item => item.id !== genId));
            }
        } catch (e) { console.error(e); }
    };

    const playActivityAudio = (item) => {
        if (!item.output_file && (!item.output_files || item.output_files.length === 0)) return;
        if (activityPlayingId === item.id && activityAudioRef.current) {
            if (activityAudioRef.current.paused) {
                activityAudioRef.current.play().then(() => {
                    setIsAudioPlaying(true);
                }).catch(e => console.error('Resume error:', e));
            } else {
                activityAudioRef.current.pause();
                setIsAudioPlaying(false);
            }
            return;
        }
        if (activityAudioRef.current) {
            activityAudioRef.current.pause();
            activityAudioRef.current.src = '';
            activityAudioRef.current = null;
        }

        setActivityPlayingId(item.id);
        setActivityPlayingItem(item);
        setAudioProgress(0);
        setAudioDuration(0);
        setIsAudioPlaying(true);

        const audio = new Audio();
        audio.preload = 'auto';
        audio.volume = audioVolume;
        audio.onended = () => {
            setIsAudioPlaying(false);
            setAudioProgress(0);
            playNextSong();
        };
        audio.onerror = (e) => {
            console.error('Audio error:', e);
            setIsAudioPlaying(false);
        };
        audio.ontimeupdate = () => setAudioProgress(audio.currentTime);
        audio.onloadedmetadata = () => setAudioDuration(audio.duration);
        audio.src = `/api/audio/${item.id}/0`;
        activityAudioRef.current = audio;
        audio.play().then(() => {
            setIsAudioPlaying(true);
        }).catch((e) => {
            console.error('Play error:', e);
            setIsAudioPlaying(false);
        });
    };

    const seekAudio = (time) => {
        if (activityAudioRef.current) {
            activityAudioRef.current.currentTime = time;
            setAudioProgress(time);
        }
    };

    const setVolume = (vol) => {
        setAudioVolume(vol);
        if (activityAudioRef.current) {
            activityAudioRef.current.volume = vol;
        }
    };

    const playNextSong = () => {
        if (!activityPlayingItem) return;
        const completedSongs = library.filter(l => l.status === 'completed' && (l.output_file || (l.output_files && l.output_files.length > 0)));
        const currentIndex = completedSongs.findIndex(l => l.id === activityPlayingItem.id);
        if (currentIndex < completedSongs.length - 1) {
            playActivityAudio(completedSongs[currentIndex + 1]);
        }
    };

    const playPrevSong = () => {
        if (!activityPlayingItem) return;
        const completedSongs = library.filter(l => l.status === 'completed' && (l.output_file || (l.output_files && l.output_files.length > 0)));
        const currentIndex = completedSongs.findIndex(l => l.id === activityPlayingItem.id);
        if (currentIndex > 0) {
            playActivityAudio(completedSongs[currentIndex - 1]);
        }
    };

    // Cleanup helper for generation completion/failure
    const cleanupGeneration = useCallback((processQueue = true) => {
        clearInterval(pollRef.current);
        clearInterval(timerRef.current);
        setCurrentGenId(null);
        setCurrentGenPayload(null);
        setGenStartTime(null);
        setEstimatedTime(null);
        setElapsedTime(0);
        if (processQueue) {
            setQueue(prev => {
                if (prev.length > 0) {
                    const [next, ...rest] = prev;
                    setTimeout(() => startGeneration(next), 100);
                    return rest;
                }
                setGenerating(false);
                return prev;
            });
        }
    }, []);

    const poll = useCallback(async (id) => {
        try {
            const r = await fetch(`/api/generation/${id}`);
            if (!r.ok) return;
            const d = await r.json();
            setProgress(d.progress);
            setStatus(d.message);
            if (d.status === 'completed') {
                setAudio({ id, files: d.output_files });
                loadLibrary();
                cleanupGeneration();
            } else if (d.status === 'failed' || d.status === 'stopped') {
                if (d.status === 'failed') setError(d.message);
                cleanupGeneration();
            }
        } catch (e) { console.error(e); }
    }, [cleanupGeneration]);

    const createPayload = () => ({
        title,
        sections: sections.map(s => ({ type: s.type, lyrics: s.lyrics || null })),
        gender,
        genre: genres.join(', '),
        emotion: moods.join(', '),
        timbre: timbres.join(', '),
        instruments: instruments.join(', '),
        custom_style: customStyle,
        bpm,
        output_mode: outputMode,
        model: selectedModel,
        memory_mode: memoryMode,
        reference_audio_id: useReference ? refId : null,
        cfg_coef: cfgCoef,
        temperature: temperature,
        top_k: topK,
        top_p: topP,
        extend_stride: extendStride,
    });

    const startGeneration = async (payload) => {
        setProgress(0);
        setStatus('Starting...');
        setError(null);
        setAudio(null);
        setCurrentGenPayload(payload);

        const estimated = estimateGenerationTime(payload.model, payload.sections);
        setEstimatedTime(estimated);
        setGenStartTime(Date.now());
        setElapsedTime(0);

        timerRef.current = setInterval(() => {
            setElapsedTime(prev => prev + 1);
        }, 1000);

        try {
            const r = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!r.ok) throw new Error(await r.text());
            const { generation_id } = await r.json();
            setCurrentGenId(generation_id);
            pollRef.current = setInterval(() => poll(generation_id), 2000);
        } catch (e) {
            setGenerating(false);
            setCurrentGenId(null);
            setCurrentGenPayload(null);
            setError(e.message);
            setQueue(prev => {
                if (prev.length > 0) {
                    const [next, ...rest] = prev;
                    setTimeout(() => startGeneration(next), 100);
                    return rest;
                }
                return prev;
            });
        }
    };

    const generate = async () => {
        const payload = createPayload();

        if (generating) {
            setQueue(prev => [...prev, payload]);  // Add to end of queue (FIFO)
        } else {
            setGenerating(true);
            startGeneration(payload);
        }
    };

    const stopGeneration = async () => {
        if (!currentGenId) return;
        try {
            await fetch(`/api/stop/${currentGenId}`, { method: 'POST' });
            setStatus('Stopping...');
        } catch (e) {
            console.error('Failed to stop:', e);
        }
    };

    const clearQueue = () => {
        setQueue([]);
    };

    const removeFromQueue = (index) => {
        setQueue(prev => prev.filter((_, i) => i !== index));
    };

    const toggleAddMenu = (e) => {
        e.stopPropagation();
        if (!showAddMenu) {
            const rect = e.currentTarget.getBoundingClientRect();
            const popupHeight = 280;
            const spaceBelow = window.innerHeight - rect.bottom;
            const openUpward = spaceBelow < popupHeight;

            setAddMenuPos({
                x: rect.left,
                y: openUpward ? rect.top : rect.bottom + 8,
                openUpward
            });
        }
        setShowAddMenu(!showAddMenu);
    };

    const addSection = (base) => {
        const cfg = SECTION_TYPES[base];
        const type = cfg?.hasDuration ? `${base}-short` : base;
        setSections([...sections, { id: Date.now().toString(), type, lyrics: '' }]);
        setShowAddMenu(false);
    };

    const removeSection = (id) => setSections(sections.filter(s => s.id !== id));
    const updateSection = (id, updates) => setSections(sections.map(s => s.id === id ? { ...s, ...updates } : s));

    const [dragId, setDragId] = useState(null);
    const [dragOverId, setDragOverId] = useState(null);

    const handleDragStart = (e, id) => {
        e.dataTransfer.setData('text/plain', id);
        e.dataTransfer.effectAllowed = 'move';
        setDragId(id);
    };
    const handleDragOver = (e, targetId) => {
        e.preventDefault();
        if (targetId !== dragOverId) setDragOverId(targetId);
    };
    const handleDragEnd = () => {
        setDragId(null);
        setDragOverId(null);
    };
    const handleDrop = (e, targetId) => {
        e.preventDefault();
        const draggedId = e.dataTransfer.getData('text/plain');
        if (draggedId === targetId) {
            handleDragEnd();
            return;
        }
        const dragIndex = sections.findIndex(s => s.id === draggedId);
        const targetIndex = sections.findIndex(s => s.id === targetId);
        const newSections = [...sections];
        const [removed] = newSections.splice(dragIndex, 1);
        newSections.splice(targetIndex, 0, removed);
        setSections(newSections);
        handleDragEnd();
    };

    const btnStyle = (isActive, activeColor = '#10B981') => ({
        flex: 1,
        padding: '10px 16px',
        borderRadius: '10px',
        border: 'none',
        fontSize: '13px',
        fontWeight: '500',
        cursor: 'pointer',
        transition: 'all 0.15s',
        backgroundColor: isActive ? activeColor : '#1e1e1e',
        color: isActive ? '#fff' : '#777',
    });

    return (
        <div style={{
            height: '100vh',
            backgroundColor: '#1e1e1e',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
        }}>
            {/* Header */}
            <header style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                padding: '20px 24px',
                maxWidth: '1400px',
                margin: '0 auto',
                width: '100%',
                boxSizing: 'border-box',
                flexShrink: 0,
                position: 'relative',
            }}>
                <div style={{ position: 'absolute', left: '24px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <img src="/static/Logo_1.png" alt="SongGeneration" style={{ height: '36px' }} />
                    <div>
                        <div style={{ fontSize: '18px', fontWeight: '600', color: '#e0e0e0' }}>SongGeneration Studio</div>
                        <div style={{ fontSize: '11px', color: '#666' }}>by Tencent AI Lab</div>
                    </div>
                </div>

                {/* Tabs - Centered */}
                <div style={{ display: 'flex', gap: '4px', backgroundColor: '#282828', padding: '4px', borderRadius: '12px' }}>
                    <button
                        onClick={() => setActiveTab('create')}
                        style={{
                            padding: '10px 24px',
                            borderRadius: '10px',
                            border: 'none',
                            fontSize: '14px',
                            fontWeight: '500',
                            cursor: 'pointer',
                            backgroundColor: activeTab === 'create' ? '#10B981' : 'transparent',
                            color: activeTab === 'create' ? '#fff' : '#888',
                            transition: 'all 0.15s',
                        }}
                    >
                        Create
                    </button>
                    <button
                        onClick={() => { setActiveTab('library'); loadLibrary(); }}
                        style={{
                            padding: '10px 24px',
                            borderRadius: '10px',
                            border: 'none',
                            fontSize: '14px',
                            fontWeight: '500',
                            cursor: 'pointer',
                            backgroundColor: activeTab === 'library' ? '#10B981' : 'transparent',
                            color: activeTab === 'library' ? '#fff' : '#888',
                            transition: 'all 0.15s',
                        }}
                    >
                        Library
                    </button>
                </div>

                <div style={{ position: 'absolute', right: '24px', fontSize: '11px', color: '#888', textAlign: 'right' }}>
                    <div>UI by <a href="https://bazed.org" target="_blank" style={{ color: '#10B981', textDecoration: 'none' }}>Bazed.org</a></div>
                </div>
            </header>

            {/* Fixed Content Area */}
            <div style={{
                flex: 1,
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
            }}>
                {/* Create View */}
                <div style={{ display: activeTab === 'create' ? 'flex' : 'none', flex: 1, overflow: 'hidden' }}>
                    <div style={{ display: 'flex', gap: '24px', maxWidth: '1400px', margin: '0 auto', padding: '0 24px 24px 24px', flex: 1, overflow: 'hidden' }}>
                    {/* Sidebar */}
                    <aside
                        {...leftHoverHandlers}
                        style={{ width: '320px', flexShrink: 0, display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto', overflowX: 'hidden', paddingBottom: '100px', paddingRight: '8px', ...getScrollStyle(leftHover) }}>
                        {/* Model */}
                        <Card>
                            <CardTitle>Model</CardTitle>
                            <div style={{ position: 'relative' }}>
                                <select
                                    className="custom-select input-base"
                                    value={selectedModel}
                                    onChange={e => setSelectedModel(e.target.value)}
                                    style={{ paddingRight: '40px', cursor: 'pointer' }}
                                >
                                    {models.map(m => <option key={m.id} value={m.id}>{m.name} - {m.description}</option>)}
                                    {models.length === 0 && <option value="">Loading...</option>}
                                </select>
                                <div style={{ position: 'absolute', right: '14px', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }}>
                                    <ChevronIcon />
                                </div>
                            </div>
                        </Card>

                        {/* Memory Mode */}
                        <Card>
                            <CardTitle>Memory Mode</CardTitle>
                            {gpuInfo && gpuInfo.gpu && (
                                <div style={{ fontSize: '11px', color: '#888', marginBottom: '10px' }}>
                                    {gpuInfo.gpu.name} • {gpuInfo.gpu.free_gb}GB free / {gpuInfo.gpu.total_gb}GB total
                                </div>
                            )}
                            {(() => {
                                if (!gpuInfo || !gpuInfo.gpu) {
                                    return (
                                        <div style={{ display: 'flex', gap: '8px' }}>
                                            {['auto', 'low', 'high'].map(mode => (
                                                <button key={mode} style={{ ...btnStyle(mode === 'auto'), opacity: 0.5 }} disabled>
                                                    {mode.charAt(0).toUpperCase() + mode.slice(1)}
                                                </button>
                                            ))}
                                        </div>
                                    );
                                }

                                const freeVram = gpuInfo.gpu.free_gb || 0;
                                const hasRef = refFileLoaded || !!refId;

                                const modelVramReqs = {
                                    'songgeneration_base': { low: 10, high: 10, highWithRef: 16 },
                                    'songgeneration_base_new': { low: 10, high: 10, highWithRef: 16 },
                                    'songgeneration_base_full': { low: 10, high: 12, highWithRef: 18 },
                                    'songgeneration_large': { low: 10, high: 22, highWithRef: 28 },
                                };

                                const reqs = modelVramReqs[selectedModel] || modelVramReqs['songgeneration_base'];
                                const highVramNeeded = hasRef ? reqs.highWithRef : reqs.high;
                                const canRunLow = freeVram >= reqs.low;
                                const canRunHigh = freeVram >= highVramNeeded;

                                if (!canRunLow) {
                                    return (
                                        <div style={{
                                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                                            border: '1px solid rgba(239, 68, 68, 0.3)',
                                            borderRadius: '8px',
                                            padding: '12px',
                                            textAlign: 'center',
                                        }}>
                                            <div style={{ color: '#EF4444', fontSize: '13px', fontWeight: '500', marginBottom: '4px' }}>
                                                Insufficient VRAM
                                            </div>
                                            <div style={{ color: '#888', fontSize: '11px' }}>
                                                This model requires at least {reqs.low}GB VRAM ({freeVram.toFixed(1)}GB available)
                                            </div>
                                        </div>
                                    );
                                }

                                return (
                                    <div style={{ display: 'flex', gap: '8px' }}>
                                        {['auto', 'low', 'high'].map(mode => {
                                            const isHighMode = mode === 'high';
                                            const isDisabled = isHighMode && !canRunHigh;

                                            let tooltip = '';
                                            if (mode === 'auto') {
                                                tooltip = `Auto-selects based on available VRAM`;
                                            } else if (mode === 'low') {
                                                tooltip = `Optimized for lower VRAM (${reqs.low}GB min)`;
                                            } else if (isDisabled) {
                                                tooltip = `Requires ${highVramNeeded}GB VRAM (${freeVram.toFixed(1)}GB available)`;
                                            } else {
                                                tooltip = `Full quality mode (${highVramNeeded}GB${hasRef ? ' with reference' : ''})`;
                                            }

                                            return (
                                                <button
                                                    key={mode}
                                                    onClick={() => !isDisabled && setMemoryMode(mode)}
                                                    disabled={isDisabled}
                                                    title={tooltip}
                                                    style={{
                                                        ...btnStyle(memoryMode === mode),
                                                        opacity: isDisabled ? 0.4 : 1,
                                                        cursor: isDisabled ? 'not-allowed' : 'pointer',
                                                    }}
                                                >
                                                    {mode.charAt(0).toUpperCase() + mode.slice(1)}
                                                </button>
                                            );
                                        })}
                                    </div>
                                );
                            })()}
                        </Card>

                        {/* Song Settings */}
                        <Card>
                            <CardTitle>Song Settings</CardTitle>

                            <div style={{ marginBottom: '16px' }}>
                                <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Voice</div>
                                <div style={{ display: 'flex', gap: '8px' }}>
                                    <button onClick={() => setGender('female')} style={btnStyle(gender === 'female', '#3B82F6')}>Female</button>
                                    <button onClick={() => setGender('male')} style={btnStyle(gender === 'male', '#3B82F6')}>Male</button>
                                </div>
                            </div>

                            <div style={{ marginBottom: '16px' }}>
                                <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Genre</div>
                                <MultiSelectWithScroll suggestions={GENRE_SUGGESTIONS} selected={genres} onChange={setGenres} placeholder="Select or type genre..." />
                            </div>

                            <div style={{ marginBottom: '16px' }}>
                                <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Mood</div>
                                <MultiSelectWithScroll suggestions={MOOD_SUGGESTIONS} selected={moods} onChange={setMoods} placeholder="Select or type mood..." />
                            </div>

                            <div style={{ marginBottom: '16px' }}>
                                <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Timbre</div>
                                <MultiSelectWithScroll suggestions={TIMBRE_SUGGESTIONS} selected={timbres} onChange={setTimbres} placeholder="Select or type timbre..." />
                            </div>

                            <div style={{ marginBottom: '16px' }}>
                                <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Instruments</div>
                                <MultiSelectWithScroll suggestions={INSTRUMENT_SUGGESTIONS} selected={instruments} onChange={setInstruments} placeholder="Select or type instruments..." />
                            </div>

                            <div style={{ marginBottom: '16px' }}>
                                <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>Custom Style (Advanced)</div>
                                <input
                                    type="text"
                                    value={customStyle}
                                    onChange={e => setCustomStyle(e.target.value)}
                                    placeholder="e.g. dubstep wobble, 808 bass, vinyl crackle..."
                                    style={{
                                        width: '100%',
                                        backgroundColor: '#1e1e1e',
                                        border: '1px solid #3a3a3a',
                                        borderRadius: '10px',
                                        padding: '12px 14px',
                                        color: '#e0e0e0',
                                        fontSize: '13px',
                                        outline: 'none',
                                        boxSizing: 'border-box',
                                    }}
                                />
                                <div style={{ fontSize: '10px', color: '#555', marginTop: '4px' }}>Add any custom style descriptors not in the presets above</div>
                            </div>

                            <div>
                                <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}>
                                    <span>BPM</span><span style={{ color: '#10B981', fontWeight: '500' }}>{bpm}</span>
                                </div>
                                <input type="range" min="60" max="180" value={bpm} onChange={e => setBpm(+e.target.value)} />
                            </div>
                        </Card>

                        {/* Reference Audio */}
                        <Card>
                            <CardTitle>Reference Audio</CardTitle>
                            <AudioTrimmer
                                onAccept={(data) => {
                                    setRefId(data.id);
                                    setRefFile({ name: data.fileName });
                                    setUseReference(true);
                                }}
                                onClear={() => {
                                    setRefId(null);
                                    setRefFile(null);
                                    setUseReference(false);
                                    setRefFileLoaded(false);
                                }}
                                onFileLoad={(loaded) => setRefFileLoaded(loaded)}
                            />
                        </Card>

                        {/* Output */}
                        <Card>
                            <CardTitle>Output</CardTitle>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                                {['mixed', 'vocal', 'bgm', 'separate'].map(mode => (
                                    <button
                                        key={mode}
                                        onClick={() => setOutputMode(mode)}
                                        style={{
                                            padding: '12px',
                                            borderRadius: '10px',
                                            border: outputMode === mode ? '1px solid #10B981' : '1px solid #3a3a3a',
                                            backgroundColor: outputMode === mode ? '#10B98115' : '#1e1e1e',
                                            color: outputMode === mode ? '#10B981' : '#777',
                                            cursor: 'pointer',
                                            fontSize: '13px',
                                            fontWeight: '500',
                                        }}
                                    >
                                        {mode === 'mixed' ? 'Full Song' : mode === 'vocal' ? 'Vocals' : mode === 'bgm' ? 'Instrumental' : 'Separate'}
                                    </button>
                                ))}
                            </div>
                        </Card>

                        {/* Advanced Settings */}
                        <Card>
                            <div onClick={() => setShowAdvanced(!showAdvanced)} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', marginBottom: showAdvanced ? '14px' : '0' }}>
                                <span style={{ fontSize: '13px', fontWeight: '500', color: '#888' }}>Advanced Settings</span>
                                <ChevronIcon rotated={showAdvanced} />
                            </div>
                            {showAdvanced && (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                                    <div>
                                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}><span>Style Strength</span><span style={{ color: '#10B981', fontWeight: '500' }}>{cfgCoef.toFixed(1)}</span></div>
                                        <input type="range" min="0.1" max="3.0" step="0.1" value={cfgCoef} onChange={e => setCfgCoef(+e.target.value)} />
                                        <div style={{ fontSize: '10px', color: '#555', marginTop: '4px' }}>Higher = follows your style more strictly</div>
                                    </div>
                                    <div>
                                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}><span>Creativity</span><span style={{ color: '#10B981', fontWeight: '500' }}>{temperature.toFixed(1)}</span></div>
                                        <input type="range" min="0.1" max="2.0" step="0.1" value={temperature} onChange={e => setTemperature(+e.target.value)} />
                                        <div style={{ fontSize: '10px', color: '#555', marginTop: '4px' }}>Higher = more experimental</div>
                                    </div>
                                    <div>
                                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}><span>Variety</span><span style={{ color: '#10B981', fontWeight: '500' }}>{topK}</span></div>
                                        <input type="range" min="1" max="250" step="1" value={topK} onChange={e => setTopK(+e.target.value)} />
                                        <div style={{ fontSize: '10px', color: '#555', marginTop: '4px' }}>Musical choices per step</div>
                                    </div>
                                    <div>
                                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}><span>Focus (Top-P)</span><span style={{ color: '#10B981', fontWeight: '500' }}>{topP.toFixed(1)}</span></div>
                                        <input type="range" min="0" max="1.0" step="0.1" value={topP} onChange={e => setTopP(+e.target.value)} />
                                        <div style={{ fontSize: '10px', color: '#555', marginTop: '4px' }}>0=off, else limits choices by probability</div>
                                    </div>
                                    <div>
                                        <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px', display: 'flex', justifyContent: 'space-between' }}><span>Extend Stride</span><span style={{ color: '#10B981', fontWeight: '500' }}>{extendStride}</span></div>
                                        <input type="range" min="1" max="15" step="1" value={extendStride} onChange={e => setExtendStride(+e.target.value)} />
                                        <div style={{ fontSize: '10px', color: '#555', marginTop: '4px' }}>Overlap for longer songs</div>
                                    </div>
                                </div>
                            )}
                        </Card>

                        {/* Song Title */}
                        <Card>
                            <CardTitle>Song Title</CardTitle>
                            <input
                                type="text"
                                className="input-base"
                                value={title}
                                onChange={e => setTitle(e.target.value)}
                                placeholder="Enter song title..."
                                style={{ fontSize: '14px' }}
                            />
                        </Card>

                        {/* Generate Button */}
                        <button
                            onClick={generate}
                            style={{
                                width: '100%',
                                backgroundColor: generating ? '#6366F1' : '#10B981',
                                color: '#fff',
                                border: 'none',
                                borderRadius: '12px',
                                padding: '16px 24px',
                                fontSize: '15px',
                                fontWeight: '600',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '8px',
                                transition: 'all 0.15s',
                            }}
                        >
                            {generating ? (
                                <><PlusIcon /> Add to Queue {queue.length > 0 && `(${queue.length})`}</>
                            ) : (
                                <><PlayIcon size={16} /> Generate</>
                            )}
                        </button>
                    </aside>

                    {/* Main Content */}
                    <main
                        {...mainHoverHandlers}
                        style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto', overflowX: 'hidden', paddingBottom: '100px', paddingRight: '8px', ...getScrollStyle(mainHover) }}>
                        {/* Structure */}
                        <Card style={{ overflow: 'hidden' }}>
                            <CardTitle>Structure</CardTitle>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '4px', overflowX: 'auto', paddingBottom: '6px' }}>
                                {sections.map(s => {
                                    const { base, duration } = fromApiType(s.type);
                                    const cfg = SECTION_TYPES[base] || { name: base, color: '#888' };
                                    const isResizable = cfg.hasDuration;
                                    const widths = { short: 52, medium: 66, long: 80 };
                                    const currentWidth = isResizable ? widths[duration] || widths.short : null;

                                    const isDragging = dragId === s.id;
                                    const isDropTarget = dragOverId === s.id && dragId !== s.id;
                                    const dragIndex = dragId ? sections.findIndex(sec => sec.id === dragId) : -1;
                                    const thisIndex = sections.findIndex(sec => sec.id === s.id);
                                    const showLeftIndicator = isDropTarget && dragIndex > thisIndex;
                                    const showRightIndicator = isDropTarget && dragIndex < thisIndex;

                                    return (
                                        <div
                                            key={s.id}
                                            draggable={!isResizable}
                                            onDragStart={e => !isResizable && handleDragStart(e, s.id)}
                                            onDragOver={e => handleDragOver(e, s.id)}
                                            onDrop={e => handleDrop(e, s.id)}
                                            onDragEnd={handleDragEnd}
                                            style={{
                                                position: 'relative',
                                                width: isResizable ? currentWidth : 'auto',
                                                padding: isResizable ? '4px 4px 4px 6px' : '4px 8px',
                                                borderRadius: '5px',
                                                backgroundColor: cfg.color + '20',
                                                border: `1.5px solid ${cfg.color}`,
                                                color: cfg.color,
                                                fontSize: '10px',
                                                fontWeight: '500',
                                                cursor: isResizable ? 'default' : 'grab',
                                                whiteSpace: 'nowrap',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'space-between',
                                                boxSizing: 'border-box',
                                                flexShrink: 0,
                                                opacity: isDragging ? 0.4 : 1,
                                                transition: 'transform 0.15s ease, opacity 0.15s ease',
                                                transform: isDropTarget ? 'scale(1.05)' : 'scale(1)',
                                            }}
                                        >
                                            {showLeftIndicator && (
                                                <div style={{
                                                    position: 'absolute',
                                                    left: '-5px',
                                                    top: '4px',
                                                    bottom: '4px',
                                                    width: '3px',
                                                    backgroundColor: '#10B981',
                                                    borderRadius: '2px',
                                                    boxShadow: '0 0 8px #10B981',
                                                }} />
                                            )}
                                            {showRightIndicator && (
                                                <div style={{
                                                    position: 'absolute',
                                                    right: '-5px',
                                                    top: '4px',
                                                    bottom: '4px',
                                                    width: '3px',
                                                    backgroundColor: '#10B981',
                                                    borderRadius: '2px',
                                                    boxShadow: '0 0 8px #10B981',
                                                }} />
                                            )}
                                            <span
                                                draggable
                                                onDragStart={e => handleDragStart(e, s.id)}
                                                style={{ cursor: 'grab', display: 'flex', alignItems: 'center', gap: '4px' }}
                                            >
                                                {cfg.name}
                                                {isResizable && (
                                                    <span style={{
                                                        fontSize: '9px',
                                                        opacity: 0.6,
                                                        fontWeight: '400',
                                                    }}>
                                                        {duration === 'long' ? 'L' : duration === 'medium' ? 'M' : 'S'}
                                                    </span>
                                                )}
                                            </span>
                                            {isResizable && (
                                                <div
                                                    onMouseDown={(e) => {
                                                        e.preventDefault();
                                                        const startX = e.clientX;
                                                        const startWidth = currentWidth;

                                                        const onMouseMove = (moveE) => {
                                                            const delta = moveE.clientX - startX;
                                                            const newWidth = startWidth + delta;
                                                            const thresholdSM = (widths.short + widths.medium) / 2;
                                                            const thresholdML = (widths.medium + widths.long) / 2;
                                                            const newDuration = newWidth > thresholdML ? 'long' : (newWidth > thresholdSM ? 'medium' : 'short');
                                                            if (newDuration !== duration) {
                                                                updateSection(s.id, { type: `${base}-${newDuration}` });
                                                            }
                                                        };

                                                        const onMouseUp = () => {
                                                            document.removeEventListener('mousemove', onMouseMove);
                                                            document.removeEventListener('mouseup', onMouseUp);
                                                        };

                                                        document.addEventListener('mousemove', onMouseMove);
                                                        document.addEventListener('mouseup', onMouseUp);
                                                    }}
                                                    style={{
                                                        width: '4px',
                                                        height: '10px',
                                                        marginLeft: '3px',
                                                        borderRadius: '2px',
                                                        backgroundColor: cfg.color + '50',
                                                        cursor: 'ew-resize',
                                                        display: 'flex',
                                                        flexDirection: 'column',
                                                        justifyContent: 'center',
                                                        alignItems: 'center',
                                                        gap: '1px',
                                                    }}
                                                    title={`${duration} (drag to resize)`}
                                                >
                                                    <div style={{ width: '2px', height: '2px', borderRadius: '50%', backgroundColor: cfg.color }} />
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                                <button
                                    ref={addBtnRef}
                                    onClick={toggleAddMenu}
                                    style={{
                                        width: '22px',
                                        height: '22px',
                                        borderRadius: '5px',
                                        backgroundColor: '#1e1e1e',
                                        border: '1.5px solid #3a3a3a',
                                        color: '#666',
                                        fontSize: '14px',
                                        cursor: 'pointer',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        flexShrink: 0,
                                    }}
                                >+</button>
                            </div>
                        </Card>

                        {/* Section Cards */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            {sections.map(s => (
                                <SectionCard key={s.id} section={s} onUpdate={u => updateSection(s.id, u)} onRemove={() => removeSection(s.id)} />
                            ))}
                        </div>
                    </main>

                    {/* Right Sidebar - Activity Feed */}
                    <aside style={{
                        width: '280px',
                        flexShrink: 0,
                        display: 'flex',
                        flexDirection: 'column',
                        overflow: 'hidden',
                        paddingBottom: '100px',
                    }}>
                        <div style={{
                            backgroundColor: '#282828',
                            borderRadius: '12px',
                            padding: '16px',
                            border: '1px solid #333',
                            display: 'flex',
                            flexDirection: 'column',
                            flex: 1,
                            overflow: 'hidden',
                        }}>
                            <div style={{
                                fontSize: '14px',
                                fontWeight: '600',
                                color: '#999',
                                marginBottom: '12px',
                                flexShrink: 0,
                            }}>Songs</div>

                            <div
                                {...rightHoverHandlers}
                                style={{
                                flex: 1,
                                overflowY: 'auto',
                                overflowX: 'hidden',
                                marginRight: '-4px',
                                paddingRight: '8px',
                                ...getScrollStyle(rightHover),
                            }}>
                                {!currentGenPayload && queue.length === 0 && library.length === 0 ? (
                                    <div style={{ color: '#555', fontSize: '12px', textAlign: 'center', padding: '20px 0' }}>
                                        No activity yet
                                    </div>
                                ) : (
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', paddingBottom: '80px' }}>
                                        {/* Queued items */}
                                        {queue.map((item, index) => (
                                            <div key={`q-${index}`} style={{
                                                backgroundColor: '#1e1e1e',
                                                borderRadius: '8px',
                                                padding: '10px',
                                                border: '1px solid #3a3a3a',
                                                display: 'flex',
                                                gap: '10px',
                                                alignItems: 'center',
                                            }}>
                                                {/* Album Cover with Queue Number */}
                                                <div style={{
                                                    width: '44px',
                                                    height: '44px',
                                                    borderRadius: '6px',
                                                    backgroundColor: '#6366F1',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    flexShrink: 0,
                                                    fontSize: '16px',
                                                    fontWeight: '600',
                                                    color: '#fff',
                                                }}>{index + 1}</div>
                                                {/* Song Info */}
                                                <div style={{ flex: 1, minWidth: 0, overflow: 'hidden' }}>
                                                    <div className="text-sm font-medium text-primary truncate" style={{ marginBottom: '2px' }}>
                                                        {item.title || 'Untitled'}
                                                    </div>
                                                    <div className="text-xs text-muted">In queue</div>
                                                </div>
                                                <button onClick={(e) => { e.stopPropagation(); removeFromQueue(index); }} style={{ background: 'none', border: 'none', color: '#666', cursor: 'pointer', padding: '4px', display: 'flex', flexShrink: 0 }}>
                                                    <CloseIcon />
                                                </button>
                                            </div>
                                        ))}

                                        {/* Currently generating */}
                                        {currentGenPayload && (
                                            <div style={{
                                                backgroundColor: '#1e1e1e',
                                                borderRadius: '8px',
                                                padding: '10px',
                                                border: '1px solid #10B981',
                                                display: 'flex',
                                                gap: '10px',
                                                alignItems: 'center',
                                                position: 'relative',
                                                overflow: 'hidden',
                                            }}>
                                                {/* Progress background */}
                                                <div style={{
                                                    position: 'absolute',
                                                    top: 0,
                                                    left: 0,
                                                    bottom: 0,
                                                    width: estimatedTime > 0 ? `${Math.min((elapsedTime / estimatedTime) * 100, 100)}%` : '30%',
                                                    background: 'linear-gradient(90deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%)',
                                                    transition: 'width 0.5s ease',
                                                    zIndex: 0,
                                                }} />
                                                {/* Album Cover with Spinner */}
                                                <div style={{
                                                    width: '44px',
                                                    height: '44px',
                                                    borderRadius: '6px',
                                                    backgroundColor: '#10B981',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    flexShrink: 0,
                                                    position: 'relative',
                                                    zIndex: 1,
                                                }}>
                                                    <SpinnerIcon />
                                                </div>
                                                {/* Song Info */}
                                                <div style={{ flex: 1, minWidth: 0, overflow: 'hidden', position: 'relative', zIndex: 1 }}>
                                                    <div className="text-sm font-medium text-primary truncate" style={{ marginBottom: '2px' }}>
                                                        {currentGenPayload.title || 'Untitled'}
                                                    </div>
                                                    <div className="text-xs text-secondary">
                                                        {formatTime(elapsedTime)}{estimatedTime > 0 ? ` / ~${formatTime(estimatedTime)}` : ''}
                                                    </div>
                                                </div>
                                            </div>
                                        )}

                                        {/* Library items (exclude currently generating) */}
                                        {library.filter(item => item.id !== currentGenId).map(item => {
                                            const meta = item.metadata || {};
                                            const isPlaying = activityPlayingId === item.id;
                                            const canPlay = item.status === 'completed' && (item.output_file || (item.output_files && item.output_files.length > 0));
                                            const isFailed = item.status === 'failed' || item.status === 'stopped';
                                            const hasCover = meta.cover;
                                            // Add timestamp for cache busting when library is refreshed
                                            const coverUrl = hasCover ? `/api/generation/${item.id}/cover?v=${meta.cover}` : null;
                                            return (
                                                <div key={item.id}
                                                    className={`activity-item ${isPlaying ? 'active' : ''} ${isFailed ? 'error' : ''}`}
                                                    style={{ cursor: canPlay ? 'pointer' : 'default', display: 'flex', gap: '10px', alignItems: 'center' }}
                                                    onClick={() => canPlay && playActivityAudio(item)}
                                                >
                                                    {/* Album Cover */}
                                                    <div style={{
                                                        width: '44px',
                                                        height: '44px',
                                                        borderRadius: '6px',
                                                        backgroundColor: isPlaying ? '#10B981' : isFailed ? '#EF4444' : '#2a2a2a',
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        justifyContent: 'center',
                                                        flexShrink: 0,
                                                        transition: 'all 0.15s',
                                                        backgroundImage: coverUrl ? `url(${coverUrl})` : 'none',
                                                        backgroundSize: 'cover',
                                                        backgroundPosition: 'center',
                                                        position: 'relative',
                                                    }}>
                                                        {canPlay ? (
                                                            <div style={{
                                                                width: '100%',
                                                                height: '100%',
                                                                display: 'flex',
                                                                alignItems: 'center',
                                                                justifyContent: 'center',
                                                                backgroundColor: coverUrl ? 'rgba(0,0,0,0.4)' : 'transparent',
                                                                borderRadius: '6px',
                                                            }}>
                                                                {isPlaying && isAudioPlaying ? (
                                                                    <PauseLargeIcon size={16} color="#fff" />
                                                                ) : (
                                                                    <PlayLargeIcon size={16} color={coverUrl ? '#fff' : (isPlaying ? '#fff' : '#888')} style={{ marginLeft: '2px' }} />
                                                                )}
                                                            </div>
                                                        ) : isFailed ? (
                                                            <CloseIcon color="#fff" />
                                                        ) : !coverUrl ? (
                                                            <MusicNoteIcon size={16} color="#666" />
                                                        ) : null}
                                                    </div>
                                                    {/* Song Info */}
                                                    <div style={{ flex: 1, minWidth: 0, overflow: 'hidden' }}>
                                                        <div className="text-sm font-medium text-primary truncate" style={{ marginBottom: '2px' }}>
                                                            {meta.title || 'Untitled'}
                                                        </div>
                                                        <div className="text-xs text-secondary truncate" style={{ marginBottom: '2px' }}>
                                                            {[meta.genre, meta.emotion].filter(Boolean).join(' • ') || 'No tags'}
                                                        </div>
                                                        <div className="text-xs text-muted">
                                                            {new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                )}
                            </div>
                        </div>
                    </aside>
                </div>
            </div>

            {/* Library View */}
            <div style={{ display: activeTab === 'library' ? 'flex' : 'none', flex: 1, overflow: 'hidden' }}>
                <div
                    {...libraryHoverHandlers}
                    style={{ maxWidth: '900px', margin: '0 auto', padding: '0 24px 100px 24px', flex: 1, overflowY: 'auto', overflowX: 'hidden', paddingRight: '28px', ...getScrollStyle(libraryHover) }}>
                    <h2 style={{ fontSize: '20px', fontWeight: '600', marginBottom: '20px', color: '#e0e0e0' }}>
                        Your Songs ({library.filter(l => l.status === 'completed').length})
                        {(currentGenPayload || queue.length > 0) && (
                            <span style={{ color: '#6366F1', fontWeight: '400', fontSize: '14px', marginLeft: '12px' }}>
                                + {(currentGenPayload ? 1 : 0) + queue.length} pending
                            </span>
                        )}
                    </h2>
                    {library.length === 0 && queue.length === 0 && !currentGenPayload ? (
                        <div style={{ textAlign: 'center', padding: '60px', color: '#666' }}>
                            No songs generated yet. Start creating!
                        </div>
                    ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            {queue.map((item, index) => (
                                <LibraryItem
                                    key={`queue-${index}`}
                                    item={item}
                                    isQueued={true}
                                    queuePosition={index + 1}
                                    onRemoveFromQueue={() => removeFromQueue(index)}
                                />
                            ))}
                            {currentGenPayload && (
                                <LibraryItem
                                    key="generating"
                                    item={currentGenPayload}
                                    isGenerating={true}
                                    onStop={stopGeneration}
                                    status={status}
                                    elapsedTime={elapsedTime}
                                    estimatedTime={estimatedTime}
                                />
                            )}
                            {library.filter(item => !currentGenId || item.id !== currentGenId).map(item => <LibraryItem key={item.id} item={item} onDelete={() => deleteGeneration(item.id)} onPlay={playActivityAudio} onUpdate={loadLibrary} isCurrentlyPlaying={activityPlayingId === item.id} isAudioPlaying={isAudioPlaying} />)}
                        </div>
                    )}
                </div>
            </div>
            </div>

            {/* Add Section Popup */}
            {showAddMenu && (
                <div
                    className="add-section-popup"
                    style={{
                        left: addMenuPos.x,
                        top: addMenuPos.y,
                        transform: addMenuPos.openUpward ? 'translateY(-100%)' : 'none',
                    }}
                >
                    {Object.entries(SECTION_TYPES)
                        .filter(([key]) => {
                            if (key === 'intro' || key === 'outro') {
                                return !sections.some(s => {
                                    const { base } = fromApiType(s.type);
                                    return base === key;
                                });
                            }
                            return true;
                        })
                        .map(([key, val]) => (
                            <button key={key} onClick={() => addSection(key)} style={{ color: val.color }}>
                                + {val.name}
                            </button>
                        ))}
                </div>
            )}

            {/* Media Player Footer */}
            <footer style={{
                position: 'fixed',
                bottom: '16px',
                left: '50%',
                transform: 'translateX(-50%)',
                width: 'calc(100% - 64px)',
                maxWidth: '900px',
                background: 'linear-gradient(135deg, rgba(25, 25, 25, 0.85) 0%, rgba(35, 35, 35, 0.9) 100%)',
                backdropFilter: 'blur(24px) saturate(180%)',
                WebkitBackdropFilter: 'blur(24px) saturate(180%)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                padding: '12px 20px',
                display: 'flex',
                alignItems: 'center',
                gap: '20px',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.05) inset',
                zIndex: 100,
            }}>
                {/* Song Info */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', minWidth: '200px', maxWidth: '300px' }}>
                    <div style={{
                        width: '48px',
                        height: '48px',
                        borderRadius: '6px',
                        backgroundColor: activityPlayingItem ? '#10B981' : '#3a3a3a',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0,
                    }}>
                        <MusicNoteIcon size={20} color={activityPlayingItem ? '#fff' : '#666'} />
                    </div>
                    <div style={{ overflow: 'hidden' }}>
                        <div style={{ fontSize: '13px', fontWeight: '500', color: activityPlayingItem ? '#fff' : '#666', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {activityPlayingItem ? (activityPlayingItem.metadata?.title || 'Untitled') : 'No song selected'}
                        </div>
                        <div style={{ fontSize: '11px', color: '#888', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {activityPlayingItem ? ([activityPlayingItem.metadata?.genre, activityPlayingItem.metadata?.mood].filter(Boolean).join(' • ') || 'No tags') : 'Select a song to play'}
                        </div>
                    </div>
                </div>

                {/* Center Controls */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                    {/* Playback Controls */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                        <button onClick={playPrevSong} disabled={!activityPlayingItem} style={{ background: 'none', border: 'none', color: activityPlayingItem ? '#888' : '#444', cursor: activityPlayingItem ? 'pointer' : 'not-allowed', padding: '4px' }}>
                            <SkipBackIcon />
                        </button>
                        <button
                            onClick={() => activityPlayingItem && playActivityAudio(activityPlayingItem)}
                            disabled={!activityPlayingItem}
                            style={{
                                background: activityPlayingItem ? '#fff' : '#555',
                                border: 'none',
                                borderRadius: '50%',
                                width: '36px',
                                height: '36px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                cursor: activityPlayingItem ? 'pointer' : 'not-allowed',
                            }}
                        >
                            {isAudioPlaying ? (
                                <PauseLargeIcon size={16} color={activityPlayingItem ? '#000' : '#333'} />
                            ) : (
                                <PlayLargeIcon size={16} color={activityPlayingItem ? '#000' : '#333'} style={{ marginLeft: '2px' }} />
                            )}
                        </button>
                        <button onClick={playNextSong} disabled={!activityPlayingItem} style={{ background: 'none', border: 'none', color: activityPlayingItem ? '#888' : '#444', cursor: activityPlayingItem ? 'pointer' : 'not-allowed', padding: '4px' }}>
                            <SkipForwardIcon />
                        </button>
                    </div>

                    {/* Progress Bar */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', width: '100%', maxWidth: '600px' }}>
                        <span style={{ fontSize: '11px', color: '#888', minWidth: '35px', textAlign: 'right' }}>
                            {formatTime(audioProgress)}
                        </span>
                        <div
                            style={{
                                flex: 1,
                                height: '4px',
                                backgroundColor: '#3a3a3a',
                                borderRadius: '2px',
                                cursor: activityPlayingItem ? 'pointer' : 'default',
                                position: 'relative',
                            }}
                            onClick={(e) => {
                                if (!activityPlayingItem) return;
                                const rect = e.currentTarget.getBoundingClientRect();
                                const percent = (e.clientX - rect.left) / rect.width;
                                seekAudio(percent * audioDuration);
                            }}
                        >
                            <div style={{
                                position: 'absolute',
                                left: 0,
                                top: 0,
                                height: '100%',
                                width: `${audioDuration ? (audioProgress / audioDuration) * 100 : 0}%`,
                                backgroundColor: '#10B981',
                                borderRadius: '2px',
                            }} />
                        </div>
                        <span style={{ fontSize: '11px', color: '#888', minWidth: '35px' }}>
                            {formatTime(audioDuration)}
                        </span>
                    </div>
                </div>

                {/* Volume Control */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: '120px' }}>
                    <button
                        onClick={() => setVolume(audioVolume > 0 ? 0 : 1)}
                        style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', padding: '4px' }}
                    >
                        {audioVolume === 0 ? <VolumeMuteIcon /> : <VolumeFullIcon />}
                    </button>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={audioVolume}
                        onChange={(e) => setVolume(parseFloat(e.target.value))}
                        style={{
                            width: '80px',
                            accentColor: '#10B981',
                        }}
                    />
                </div>
            </footer>
        </div>
    );
};

// Render the app
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
