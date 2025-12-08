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
    const [hasReadyModel, setHasReadyModel] = useState(false);  // Start false, assume no models until API responds
    const [recommendedModel, setRecommendedModel] = useState(null);
    const autoDownloadTriggeredRef = useRef(false);  // Use ref to avoid React 18 StrictMode double-render issues
    const [downloadPolling, setDownloadPolling] = useState(false);
    const [isInitializing, setIsInitializing] = useState(true);  // Track initial load state
    const [autoDownloadStarting, setAutoDownloadStarting] = useState(false);  // Track auto-download in progress
    const [showModelManager, setShowModelManager] = useState(false);
    const [allModels, setAllModels] = useState([]);
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
    const [timingStats, setTimingStats] = useState(null);  // Historical timing data for smart estimates
    // Queue system - server-side storage (shared across all clients)
    const [queue, setQueue] = useState([]);
    const [currentGenId, setCurrentGenId] = useState(null);
    const [currentGenPayload, setCurrentGenPayload] = useState(null);
    const pollRef = useRef(null);
    const addBtnRef = useRef(null);
    const timerRef = useRef(null);
    const idlePollRef = useRef(null);  // Legacy ref, kept for cleanup compatibility

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

    // SSE connection for real-time updates
    const sseRef = useRef(null);
    const sseReconnectRef = useRef(null);

    // Connect to Server-Sent Events for real-time updates
    useEffect(() => {
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 10;
        const baseReconnectDelay = 1000;

        const connectSSE = () => {
            if (sseRef.current) {
                sseRef.current.close();
            }

            console.log('[SSE] Connecting to /api/events...');
            const eventSource = new EventSource('/api/events');
            sseRef.current = eventSource;

            eventSource.onopen = () => {
                console.log('[SSE] Connected');
                reconnectAttempts = 0;
            };

            eventSource.onerror = (e) => {
                console.error('[SSE] Connection error:', e);
                eventSource.close();
                sseRef.current = null;

                // Reconnect with exponential backoff
                if (reconnectAttempts < maxReconnectAttempts) {
                    const delay = baseReconnectDelay * Math.pow(2, reconnectAttempts);
                    reconnectAttempts++;
                    console.log(`[SSE] Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
                    sseReconnectRef.current = setTimeout(connectSSE, delay);
                } else {
                    console.warn('[SSE] Max reconnect attempts reached, falling back to polling');
                }
            };

            // Handle queue updates
            eventSource.addEventListener('queue', (e) => {
                try {
                    const data = JSON.parse(e.data);
                    if (data.queue) {
                        setQueue(data.queue);
                    }
                } catch (err) {
                    console.error('[SSE] Error parsing queue event:', err);
                }
            });

            // Handle generation updates
            eventSource.addEventListener('generation', (e) => {
                try {
                    const data = JSON.parse(e.data);
                    if (data.generation && data.id) {
                        const gen = data.generation;
                        // Update progress if this is the current generation
                        if (data.id === currentGenId) {
                            setProgress(gen.progress || 0);
                            setStatus(gen.message || '');
                            if (typeof gen.elapsed_seconds === 'number') {
                                setElapsedTime(gen.elapsed_seconds);
                            }
                        }
                        // Handle completion
                        if (gen.status === 'completed' || gen.status === 'failed' || gen.status === 'stopped') {
                            // Refresh library to get full updated data
                            loadLibrary();
                        }
                    }
                } catch (err) {
                    console.error('[SSE] Error parsing generation event:', err);
                }
            });

            // Handle library updates
            eventSource.addEventListener('library', (e) => {
                try {
                    const data = JSON.parse(e.data);
                    if (data.generations) {
                        // Check if any generation is now active that we should track
                        const activeGen = data.generations.find(g => g.status === 'pending' || g.status === 'processing');
                        if (activeGen && !currentGenId && !generating) {
                            console.log('[SSE] Detected active generation:', activeGen.id);
                            // Refresh library to get full data and trigger restoration
                            loadLibrary();
                        }
                        // Check if current generation completed
                        if (currentGenId) {
                            const current = data.generations.find(g => g.id === currentGenId);
                            if (current && (current.status === 'completed' || current.status === 'failed' || current.status === 'stopped')) {
                                loadLibrary();
                            }
                        }
                    }
                } catch (err) {
                    console.error('[SSE] Error parsing library event:', err);
                }
            });

            // Handle model updates (download complete, etc)
            eventSource.addEventListener('models', (e) => {
                try {
                    const data = JSON.parse(e.data);
                    console.log('[SSE] Models update received:', data);
                    if (data.models) {
                        const all = data.models;
                        const ready = data.ready_models || all.filter(m => m.status === 'ready');
                        setAllModels(all);
                        setModels(ready);
                        setHasReadyModel(data.has_ready_model);
                        // Stop download polling if no downloads in progress
                        const hasDownloading = all.some(m => m.status === 'downloading');
                        setDownloadPolling(hasDownloading);
                    }
                } catch (err) {
                    console.error('[SSE] Error parsing models event:', err);
                }
            });
        };

        connectSSE();

        return () => {
            if (sseRef.current) {
                sseRef.current.close();
                sseRef.current = null;
            }
            if (sseReconnectRef.current) {
                clearTimeout(sseReconnectRef.current);
            }
        };
    }, [currentGenId, generating]);

    // Load queue from server
    const loadQueue = async () => {
        try {
            const r = await fetch('/api/queue');
            if (r.ok) {
                const data = await r.json();
                setQueue(data);
            }
        } catch (e) { console.error('Error loading queue:', e); }
    };

    // Add item to server-side queue
    const addToQueue = async (payload) => {
        try {
            const r = await fetch('/api/queue', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (r.ok) {
                await loadQueue(); // Refresh queue from server
            }
        } catch (e) { console.error('Error adding to queue:', e); }
    };

    // Remove item from server-side queue
    const removeFromQueue = async (itemId) => {
        try {
            await fetch(`/api/queue/${itemId}`, { method: 'DELETE' });
            await loadQueue(); // Refresh queue from server
        } catch (e) { console.error('Error removing from queue:', e); }
    };

    // NOTE: popFromQueue removed - queue processing is handled server-side

    useEffect(() => { loadModels(true); loadLibrary(); loadGpuInfo(); loadQueue(); loadTimingStats(); }, []);
    useEffect(() => () => {
        pollRef.current && clearInterval(pollRef.current);
        timerRef.current && clearInterval(timerRef.current);
    }, []);

    // Poll for model download progress (reduced frequency - SSE handles real-time updates)
    useEffect(() => {
        if (!downloadPolling) return;
        const interval = setInterval(() => {
            loadModels();
        }, 5000);  // Poll every 5 seconds instead of 2
        return () => clearInterval(interval);
    }, [downloadPolling]);

    // Fallback periodic sync (only when SSE is not connected)
    // SSE handles real-time updates, this is just a safety net
    useEffect(() => {
        // Only poll when NOT currently tracking a generation (those have their own polling)
        if (currentGenId) return;

        const syncInterval = setInterval(async () => {
            // Only sync if SSE is disconnected
            if (!sseRef.current || sseRef.current.readyState !== EventSource.OPEN) {
                console.log('[Fallback] SSE disconnected, polling...');
                await Promise.all([loadQueue(), loadLibrary()]);
            }
        }, 10000);  // Fallback poll every 10 seconds

        return () => clearInterval(syncInterval);
    }, [currentGenId]);

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

            // Use server-provided elapsed_seconds if available, otherwise calculate from started_at
            // The server's started_at is when processing began (not when queued)
            if (typeof runningGen.elapsed_seconds === 'number') {
                setElapsedTime(runningGen.elapsed_seconds);
            } else if (runningGen.started_at) {
                const startTime = new Date(runningGen.started_at).getTime();
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                setElapsedTime(Math.max(0, elapsed));
            } else if (runningGen.created_at) {
                // Fallback to created_at (less accurate)
                const startTime = new Date(runningGen.created_at).getTime();
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                setElapsedTime(Math.max(0, elapsed));
            }

            // Use smart estimation (will use history if available, otherwise fallback)
            const sectionsList = Array.isArray(payload.sections) ? payload.sections : [];
            const hasReference = Boolean(payload.reference_audio_id || meta.reference_audio_id);
            const estimated = estimateGenerationTime(payload.model, sectionsList, hasReference);
            setEstimatedTime(estimated);

            if (timerRef.current) clearInterval(timerRef.current);
            if (pollRef.current) clearInterval(pollRef.current);

            // Use server time as source of truth - poll updates elapsedTime from server's elapsed_seconds
            // Local timer provides smooth increments between polls for better UX
            timerRef.current = setInterval(() => {
                setElapsedTime(prev => prev + 1);
            }, 1000);

            // Poll every 2 seconds - this syncs elapsed time from server
            pollRef.current = setInterval(() => poll(runningGen.id), 2000);
        }
    }, [library, generating, currentGenId]);

    // NOTE: Queue processing is now handled entirely server-side
    // The server auto-starts the next queued item when a generation completes
    // Clients detect new generations via the library polling in cleanupGeneration

    // Auto-set output mode based on lyrics presence
    useEffect(() => {
        const hasLyrics = sections.some(s => s.lyrics && s.lyrics.trim().length > 0);
        if (hasLyrics && outputMode === 'bgm') {
            setOutputMode('mixed');
        } else if (!hasLyrics && outputMode === 'mixed') {
            setOutputMode('bgm');
        }
    }, [sections]);

    // Auto-select best model based on available (free) VRAM
    // Priority: prefer larger models if VRAM allows, but ALWAYS pick a ready model
    useEffect(() => {
        if (models.length === 0) return;

        const freeVram = gpuInfo?.gpu?.free_gb || 0;
        const readyModels = models.filter(m => m.status === 'ready');

        // If no models ready, don't change selection
        if (readyModels.length === 0) return;

        let bestModel = null;

        // VRAM requirements (conservative estimates for high mode + reference):
        // LARGE: 22GB+, FULL: 12GB+ (18GB with ref), NEW/BASE: 10GB+ (16GB with ref)
        // Use lower thresholds since low/auto memory modes can work with less
        if (freeVram >= 20 && readyModels.some(m => m.id === 'songgeneration_large')) {
            bestModel = 'songgeneration_large';
        } else if (freeVram >= 12 && readyModels.some(m => m.id === 'songgeneration_base_full')) {
            bestModel = 'songgeneration_base_full';
        } else if (readyModels.some(m => m.id === 'songgeneration_base_new')) {
            bestModel = 'songgeneration_base_new';
        } else if (readyModels.some(m => m.id === 'songgeneration_base')) {
            bestModel = 'songgeneration_base';
        }

        // Fallback: if no preferred model found, select first ready model
        if (!bestModel && readyModels.length > 0) {
            bestModel = readyModels[0].id;
        }

        if (bestModel) {
            setSelectedModel(bestModel);
        }
    }, [gpuInfo, models]);

    // Smart time estimation based on model, lyrics, and historical data
    const estimateGenerationTime = useCallback((model, sectionsList, hasReference = false) => {
        const numSections = sectionsList.length;
        const totalLyrics = sectionsList.reduce((acc, s) => acc + (s.lyrics || '').length, 0);
        const hasLyrics = totalLyrics > 0;

        // Try to use historical data first
        if (timingStats?.has_history && timingStats.models?.[model]) {
            const modelStats = timingStats.models[model];

            // Priority 1: Exact section count match
            const sectionKey = String(numSections);
            if (modelStats.by_sections?.[sectionKey]) {
                let estimate = modelStats.by_sections[sectionKey];

                // Adjust for lyrics vs no lyrics (if we have that data)
                if (hasLyrics && modelStats.avg_with_lyrics && modelStats.avg_without_lyrics) {
                    const lyricsRatio = modelStats.avg_with_lyrics / modelStats.avg_without_lyrics;
                    if (!hasLyrics) estimate = Math.round(estimate / lyricsRatio);
                }

                // Adjust for reference audio
                if (hasReference && modelStats.avg_with_reference && modelStats.avg_without_reference) {
                    const refRatio = modelStats.avg_with_reference / modelStats.avg_without_reference;
                    estimate = Math.round(estimate * refRatio);
                }

                console.log(`[TIMING] Estimate from history (sections=${numSections}): ${estimate}s`);
                return estimate;
            }

            // Priority 2: Use lyrics/no-lyrics average
            let baseEstimate = hasLyrics ? modelStats.avg_with_lyrics : modelStats.avg_without_lyrics;
            if (!baseEstimate) baseEstimate = modelStats.avg_time;

            if (baseEstimate) {
                // Adjust for section count (use average sections as baseline, ~5)
                const avgSections = 5;
                const sectionMultiplier = 1 + ((numSections - avgSections) * 0.08);
                let estimate = Math.round(baseEstimate * sectionMultiplier);

                // Adjust for reference audio
                if (hasReference && modelStats.avg_with_reference && modelStats.avg_without_reference) {
                    const refRatio = modelStats.avg_with_reference / modelStats.avg_without_reference;
                    estimate = Math.round(estimate * refRatio);
                }

                console.log(`[TIMING] Estimate from history (avg): ${estimate}s`);
                return Math.max(60, estimate);  // Minimum 1 minute
            }
        }

        // Fallback: Static estimation (no historical data)
        let baseTime = MODEL_BASE_TIMES[model] || 240;

        // Adjust for lyrics length
        const lyricsTimeAdjust = hasLyrics ? Math.floor(totalLyrics / 500) * 30 : -30;

        // Adjust for section count
        const sectionsAdjust = Math.max(0, numSections - 3) * 15;

        // Adjust for instrumental sections (longer duration)
        const durationSections = sectionsList.filter(s =>
            s.type.includes('intro') || s.type.includes('outro') || s.type.includes('inst')
        ).length;
        const durationAdjust = durationSections * 20;

        // Adjust for reference audio
        const referenceAdjust = hasReference ? 60 : 0;

        const estimate = baseTime + lyricsTimeAdjust + sectionsAdjust + durationAdjust + referenceAdjust;
        console.log(`[TIMING] Estimate from static: ${estimate}s (base=${baseTime}, lyrics=${lyricsTimeAdjust}, sections=${sectionsAdjust}, duration=${durationAdjust}, ref=${referenceAdjust})`);

        return Math.max(60, estimate);
    }, [timingStats]);

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

    const loadModels = async (triggerAutoDownload = false) => {
        try {
            console.log('[MODELS] Loading models, triggerAutoDownload:', triggerAutoDownload);
            const r = await fetch('/api/models');
            if (r.ok) {
                const d = await r.json();
                console.log('[MODELS] API response:', { has_ready_model: d.has_ready_model, recommended: d.recommended, modelCount: d.models?.length });

                const all = d.models || [];
                const ready = d.ready_models || all.filter(m => m.status === 'ready');

                setAllModels(all);
                setModels(ready);
                setHasReadyModel(d.has_ready_model);
                setRecommendedModel(d.recommended);

                if (d.default) setSelectedModel(d.default);

                // Check if any models are downloading
                const hasDownloading = all.some(m => m.status === 'downloading');
                setDownloadPolling(hasDownloading);

                // Auto-download recommended model on first launch if no models ready
                // Don't auto-download if already downloading something
                if (triggerAutoDownload && !d.has_ready_model && d.recommended && !autoDownloadTriggeredRef.current && !hasDownloading) {
                    console.log('[AUTO-DOWNLOAD] Triggering download of recommended model:', d.recommended);
                    autoDownloadTriggeredRef.current = true;  // Set ref immediately to prevent double-trigger
                    setAutoDownloadStarting(true);  // Show immediate feedback in UI

                    // Start download with error handling - inline to ensure it runs
                    try {
                        const downloadResult = await fetch(`/api/models/${d.recommended}/download`, { method: 'POST' });
                        if (downloadResult.ok) {
                            console.log('[AUTO-DOWNLOAD] Download started successfully');
                            setDownloadPolling(true);
                            setAutoDownloadStarting(false);
                            loadModels();  // Refresh to show download progress
                        } else {
                            const errorText = await downloadResult.text();
                            console.error('[AUTO-DOWNLOAD] Failed to start download:', errorText);
                            autoDownloadTriggeredRef.current = false;  // Reset so user can try again
                            setAutoDownloadStarting(false);
                        }
                    } catch (downloadError) {
                        console.error('[AUTO-DOWNLOAD] Error starting download:', downloadError);
                        autoDownloadTriggeredRef.current = false;  // Reset so user can try again
                        setAutoDownloadStarting(false);
                    }
                } else if (triggerAutoDownload) {
                    console.log('[AUTO-DOWNLOAD] Conditions not met:', {
                        hasReadyModel: d.has_ready_model,
                        recommended: d.recommended,
                        alreadyTriggered: autoDownloadTriggeredRef.current,
                        hasDownloading
                    });
                }

                // Mark initialization complete after first load
                setIsInitializing(false);
            } else {
                console.error('[MODELS] Failed to load models, status:', r.status);
                setIsInitializing(false);
            }
        } catch (e) {
            console.error('[MODELS] Error loading models:', e);
            setIsInitializing(false);
        }
    };

    const startModelDownload = async (modelId) => {
        try {
            const r = await fetch(`/api/models/${modelId}/download`, { method: 'POST' });
            if (r.ok) {
                setDownloadPolling(true);
                loadModels();
            }
        } catch (e) { console.error(e); }
    };

    const cancelModelDownload = async (modelId) => {
        try {
            const r = await fetch(`/api/models/${modelId}/download`, { method: 'DELETE' });
            if (r.ok) {
                // Immediately update local state to show cancellation
                setAllModels(prev => {
                    const updated = prev.map(m =>
                        m.id === modelId ? { ...m, status: 'not_downloaded', progress: 0 } : m
                    );
                    // Stop polling if no other downloads active
                    const stillDownloading = updated.some(m => m.status === 'downloading');
                    if (!stillDownloading) {
                        setDownloadPolling(false);
                    }
                    return updated;
                });
            }
            // Refresh from server to get accurate state
            await loadModels();
        } catch (e) { console.error(e); }
    };

    const deleteModel = async (modelId) => {
        if (!confirm(`Delete model ${modelId}? This will free up disk space but you'll need to download it again to use it.`)) return;
        try {
            const r = await fetch(`/api/models/${modelId}`, { method: 'DELETE' });
            if (r.ok) {
                loadModels();
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

    const loadTimingStats = async () => {
        try {
            const r = await fetch('/api/timing-stats');
            if (r.ok) {
                const d = await r.json();
                setTimingStats(d);
                console.log('[TIMING] Loaded timing stats:', d);
            }
        } catch (e) { console.error('[TIMING] Error loading timing stats:', e); }
    };

    const loadLibrary = async () => {
        try {
            const r = await fetch('/api/generations');
            if (r.ok) {
                const data = await r.json();
                const sorted = data.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
                setLibrary(sorted);

                // Update activityPlayingItem if it exists in the new data (to get updated cover, title, etc.)
                setActivityPlayingItem(current => {
                    if (current) {
                        const updatedItem = sorted.find(item => item.id === current.id);
                        return updatedItem || current;
                    }
                    return current;
                });
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
    // SSE handles detecting new server-started generations automatically
    const cleanupGeneration = useCallback(async () => {
        clearInterval(pollRef.current);
        clearInterval(timerRef.current);
        clearInterval(idlePollRef.current);
        idlePollRef.current = null;
        setCurrentGenId(null);
        setCurrentGenPayload(null);
        setGenStartTime(null);
        setEstimatedTime(null);
        setElapsedTime(0);
        setGenerating(false);

        // Reset restoredRef so the useEffect can detect new server-started generations
        restoredRef.current = false;

        // Refresh library and queue from server
        await loadLibrary();
        await loadQueue();

        // Refresh timing stats to incorporate the just-completed generation
        await loadTimingStats();

        // SSE will automatically notify us when a new generation starts
        console.log('[Cleanup] Generation cleanup complete, SSE will handle new generation detection');
    }, []);

    const poll = useCallback(async (id) => {
        try {
            // Poll generation status - SSE handles queue and library updates
            const genResponse = await fetch(`/api/generation/${id}`);

            if (!genResponse.ok) {
                // 404 means generation doesn't exist - stop polling
                if (genResponse.status === 404) {
                    console.warn(`Generation ${id} not found (404), stopping poll`);
                    cleanupGeneration();
                }
                return;
            }
            const d = await genResponse.json();
            setProgress(d.progress);
            setStatus(d.message);

            // Use server-provided elapsed time for consistency across all clients
            if (typeof d.elapsed_seconds === 'number') {
                setElapsedTime(d.elapsed_seconds);
            }

            if (d.status === 'completed') {
                setAudio({ id, files: d.output_files });
                cleanupGeneration();
            } else if (d.status === 'failed' || d.status === 'stopped') {
                if (d.status === 'failed') setError(d.message);
                cleanupGeneration();
            }
        } catch (e) { console.error(e); }
    }, [cleanupGeneration]);

    // NOTE: idlePoll, startIdlePolling, stopIdlePolling removed - SSE handles real-time updates now

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

        const hasReference = Boolean(payload.reference_audio_id);
        const estimated = estimateGenerationTime(payload.model, payload.sections, hasReference);
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
            clearInterval(timerRef.current);
            // Handle 409 conflict - another generation is already running
            if (e.message.includes('409') || e.message.includes('already in progress')) {
                console.log('Generation conflict - server already running a generation');
                // SSE will automatically detect and track the server's generation
                // Just refresh library to show current state
                loadLibrary();
            } else {
                setError(e.message);
            }
        }
    };

    const generate = async () => {
        // Validate any model is ready before generating
        if (!hasReadyModel || models.length === 0) {
            setError("No models downloaded. Please download a model first before generating.");
            return;
        }

        // Auto-correct model selection if selected model isn't ready
        let modelToUse = selectedModel;
        const selectedModelReady = models.some(m => m.id === selectedModel && m.status === 'ready');
        if (!selectedModelReady) {
            // Find first ready model
            const firstReady = models.find(m => m.status === 'ready');
            if (firstReady) {
                modelToUse = firstReady.id;
                setSelectedModel(modelToUse);
                console.log(`[GENERATE] Auto-corrected model: ${selectedModel} -> ${modelToUse}`);
            } else {
                setError("No models ready. Please wait for download to complete.");
                return;
            }
        }

        const payload = createPayload();
        payload.model = modelToUse;  // Ensure we use the corrected model

        if (generating) {
            await addToQueue(payload);  // Add to server-side queue
        } else {
            setGenerating(true);
            startGeneration(payload);
        }
    };

    const stopGeneration = async (genId = null) => {
        const idToStop = genId || currentGenId;
        if (!idToStop) return;
        try {
            await fetch(`/api/stop/${idToStop}`, { method: 'POST' });
            if (idToStop === currentGenId) {
                setStatus('Stopping...');
            }
            // Immediately refresh library and queue to show updated status
            await Promise.all([loadLibrary(), loadQueue()]);
        } catch (e) {
            console.error('Failed to stop:', e);
        }
    };

    const clearQueueHandler = async () => {
        try {
            await fetch('/api/queue', { method: 'DELETE' });
            await loadQueue();
        } catch (e) { console.error('Error clearing queue:', e); }
    };

    const removeFromQueueHandler = async (itemId) => {
        await removeFromQueue(itemId);
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
                            {/* Model selector - only ready models */}
                            {hasReadyModel ? (
                                <div style={{ position: 'relative', marginBottom: '12px' }}>
                                    <select
                                        className="custom-select input-base"
                                        value={selectedModel}
                                        onChange={e => setSelectedModel(e.target.value)}
                                        style={{ paddingRight: '40px', cursor: 'pointer' }}
                                    >
                                        {models.map(m => (
                                            <option key={m.id} value={m.id}>{m.name}</option>
                                        ))}
                                    </select>
                                    <div style={{ position: 'absolute', right: '14px', top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none' }}>
                                        <ChevronIcon />
                                    </div>
                                </div>
                            ) : isInitializing || autoDownloadStarting ? (
                                <div style={{
                                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                                    border: '1px solid rgba(99, 102, 241, 0.3)',
                                    borderRadius: '8px',
                                    padding: '12px',
                                    marginBottom: '12px',
                                    textAlign: 'center',
                                }}>
                                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                                        <SpinnerIcon />
                                        <span style={{ color: '#6366F1', fontSize: '13px', fontWeight: '500' }}>
                                            {autoDownloadStarting ? 'Starting download...' : 'Loading models...'}
                                        </span>
                                    </div>
                                </div>
                            ) : allModels.some(m => m.status === 'downloading') ? (
                                <div style={{
                                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                                    border: '1px solid rgba(245, 158, 11, 0.3)',
                                    borderRadius: '8px',
                                    padding: '12px',
                                    marginBottom: '12px',
                                    textAlign: 'center',
                                }}>
                                    <div style={{ color: '#F59E0B', fontSize: '13px', fontWeight: '500' }}>
                                        Downloading model...
                                    </div>
                                </div>
                            ) : (
                                <div style={{
                                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                                    border: '1px solid rgba(245, 158, 11, 0.3)',
                                    borderRadius: '8px',
                                    padding: '12px',
                                    marginBottom: '12px',
                                    textAlign: 'center',
                                }}>
                                    <div style={{ color: '#F59E0B', fontSize: '13px', fontWeight: '500', marginBottom: '4px' }}>
                                        No Models Downloaded
                                    </div>
                                    <div style={{ color: '#888', fontSize: '11px' }}>
                                        Click "Manage Models" to download
                                    </div>
                                </div>
                            )}

                            {/* Download progress indicator (if any model is downloading) */}
                            {allModels.filter(m => m.status === 'downloading').map(m => (
                                <div key={m.id} style={{
                                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                                    border: '1px solid rgba(245, 158, 11, 0.3)',
                                    borderRadius: '8px',
                                    padding: '10px 12px',
                                    marginBottom: '12px',
                                }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                                        <span style={{ fontSize: '12px', color: '#F59E0B', fontWeight: '500' }}>
                                            Downloading {m.name}
                                        </span>
                                        <span style={{ fontSize: '11px', color: '#888' }}>{m.progress || 0}%</span>
                                    </div>
                                    <div style={{
                                        height: '6px',
                                        backgroundColor: 'rgba(245, 158, 11, 0.2)',
                                        borderRadius: '3px',
                                        overflow: 'hidden',
                                    }}>
                                        <div style={{
                                            width: `${m.progress || 0}%`,
                                            height: '100%',
                                            backgroundColor: '#F59E0B',
                                            borderRadius: '3px',
                                            transition: 'width 0.3s ease',
                                        }} />
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '6px' }}>
                                        <span style={{ fontSize: '10px', color: '#666' }}>
                                            {m.downloaded_gb ? `${m.downloaded_gb.toFixed(1)}GB / ${m.size_gb}GB` : `~${m.size_gb}GB`}
                                        </span>
                                        <button
                                            onClick={() => cancelModelDownload(m.id)}
                                            style={{
                                                background: 'none',
                                                border: 'none',
                                                color: '#EF4444',
                                                cursor: 'pointer',
                                                padding: '2px 6px',
                                                fontSize: '10px',
                                            }}
                                        >
                                            Cancel
                                        </button>
                                    </div>
                                </div>
                            ))}

                            {/* Manage Models button */}
                            <button
                                onClick={() => setShowModelManager(true)}
                                style={{
                                    width: '100%',
                                    padding: '10px',
                                    backgroundColor: '#2a2a2a',
                                    border: '1px solid #444',
                                    borderRadius: '8px',
                                    color: '#fff',
                                    fontSize: '12px',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    gap: '8px',
                                    transition: 'all 0.2s ease',
                                }}
                                onMouseEnter={e => {
                                    e.target.style.backgroundColor = '#333';
                                    e.target.style.borderColor = '#555';
                                }}
                                onMouseLeave={e => {
                                    e.target.style.backgroundColor = '#2a2a2a';
                                    e.target.style.borderColor = '#444';
                                }}
                            >
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <circle cx="12" cy="12" r="3"/>
                                    <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
                                </svg>
                                Manage Models
                            </button>
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
                            disabled={!hasReadyModel}
                            title={!hasReadyModel ? 'Download a model first' : ''}
                            style={{
                                width: '100%',
                                backgroundColor: !hasReadyModel ? '#444' : generating ? '#6366F1' : '#10B981',
                                color: '#fff',
                                border: 'none',
                                borderRadius: '12px',
                                padding: '16px 24px',
                                fontSize: '15px',
                                fontWeight: '600',
                                cursor: !hasReadyModel ? 'not-allowed' : 'pointer',
                                opacity: !hasReadyModel ? 0.6 : 1,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: '8px',
                                transition: 'all 0.15s',
                            }}
                        >
                            {!hasReadyModel ? (
                                'Download Model to Generate'
                            ) : generating ? (
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
                                                position: 'relative',
                                            }}>
                                                {/* X button top right to remove from queue */}
                                                <button
                                                    onClick={(e) => { e.stopPropagation(); removeFromQueue(item.id); }}
                                                    style={{
                                                        position: 'absolute',
                                                        top: '4px',
                                                        right: '4px',
                                                        background: 'none',
                                                        border: 'none',
                                                        color: '#666',
                                                        cursor: 'pointer',
                                                        padding: '2px',
                                                        zIndex: 2,
                                                        display: 'flex',
                                                    }}
                                                    onMouseEnter={(e) => e.currentTarget.style.color = '#EF4444'}
                                                    onMouseLeave={(e) => e.currentTarget.style.color = '#666'}
                                                    title="Remove from queue"
                                                >
                                                    <CloseIcon size={10} />
                                                </button>
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
                                            </div>
                                        ))}

                                        {/* Currently generating */}
                                        {currentGenPayload && (
                                            <div style={{
                                                backgroundColor: '#1e1e1e',
                                                borderRadius: '8px',
                                                padding: '10px',
                                                border: '1px solid #3a3a3a',
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
                                                    background: 'linear-gradient(90deg, rgba(245, 158, 11, 0.15) 0%, rgba(245, 158, 11, 0.05) 100%)',
                                                    transition: 'width 0.5s ease',
                                                    zIndex: 0,
                                                }} />
                                                {/* X button top right to stop */}
                                                <button
                                                    onClick={(e) => { e.stopPropagation(); stopGeneration(); }}
                                                    style={{
                                                        position: 'absolute',
                                                        top: '4px',
                                                        right: '4px',
                                                        background: 'none',
                                                        border: 'none',
                                                        color: '#666',
                                                        cursor: 'pointer',
                                                        padding: '2px',
                                                        zIndex: 2,
                                                        display: 'flex',
                                                    }}
                                                    onMouseEnter={(e) => e.currentTarget.style.color = '#EF4444'}
                                                    onMouseLeave={(e) => e.currentTarget.style.color = '#666'}
                                                    title="Stop generation"
                                                >
                                                    <CloseIcon size={10} />
                                                </button>
                                                {/* Album Cover with Spinner */}
                                                <div style={{
                                                    width: '44px',
                                                    height: '44px',
                                                    borderRadius: '6px',
                                                    backgroundColor: '#F59E0B',
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
                                            const isProcessing = item.status === 'processing' || item.status === 'generating' || item.status === 'pending';
                                            const canDelete = item.status === 'completed' || item.status === 'failed' || item.status === 'stopped';
                                            const hasCover = meta.cover;
                                            // Add timestamp for cache busting when library is refreshed
                                            const coverUrl = hasCover ? `/api/generation/${item.id}/cover?v=${meta.cover}` : null;
                                            return (
                                                <div key={item.id}
                                                    className={`activity-item ${isPlaying ? 'active' : ''} ${isFailed ? 'error' : ''} ${isProcessing ? 'processing' : ''}`}
                                                    style={{ cursor: canPlay ? 'pointer' : 'default', display: 'flex', gap: '10px', alignItems: 'center', position: 'relative' }}
                                                    onClick={() => canPlay && playActivityAudio(item)}
                                                >
                                                    {/* X button top right to stop (for processing) */}
                                                    {isProcessing && (
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); stopGeneration(item.id); }}
                                                            style={{
                                                                position: 'absolute',
                                                                top: '4px',
                                                                right: '4px',
                                                                background: 'none',
                                                                border: 'none',
                                                                color: '#666',
                                                                cursor: 'pointer',
                                                                padding: '2px',
                                                                zIndex: 2,
                                                                display: 'flex',
                                                            }}
                                                            onMouseEnter={(e) => e.currentTarget.style.color = '#EF4444'}
                                                            onMouseLeave={(e) => e.currentTarget.style.color = '#666'}
                                                            title="Stop generation"
                                                        >
                                                            <CloseIcon size={10} />
                                                        </button>
                                                    )}
                                                    {/* Trash button top right (for completed/failed/stopped) */}
                                                    {canDelete && (
                                                        <button
                                                            onClick={(e) => { e.stopPropagation(); deleteGeneration(item.id); }}
                                                            style={{
                                                                position: 'absolute',
                                                                top: '4px',
                                                                right: '4px',
                                                                background: 'none',
                                                                border: 'none',
                                                                color: '#555',
                                                                cursor: 'pointer',
                                                                padding: '2px',
                                                                zIndex: 2,
                                                                display: 'flex',
                                                            }}
                                                            onMouseEnter={(e) => e.currentTarget.style.color = '#EF4444'}
                                                            onMouseLeave={(e) => e.currentTarget.style.color = '#555'}
                                                            title="Delete"
                                                        >
                                                            <TrashIcon size={10} />
                                                        </button>
                                                    )}
                                                    {/* Album Cover */}
                                                    <div style={{
                                                        width: '44px',
                                                        height: '44px',
                                                        borderRadius: '6px',
                                                        backgroundColor: isProcessing ? '#F59E0B' : isPlaying ? '#10B981' : isFailed ? '#EF4444' : '#2a2a2a',
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
                                                        {isProcessing ? (
                                                            <SpinnerIcon />
                                                        ) : canPlay ? (
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
                                                            {isProcessing ? 'Generating...' : ([meta.genre, meta.emotion].filter(Boolean).join(' • ') || 'No tags')}
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
                                    key={`queue-${item.id}`}
                                    item={item}
                                    isQueued={true}
                                    queuePosition={index + 1}
                                    onRemoveFromQueue={() => removeFromQueue(item.id)}
                                />
                            ))}
                            {currentGenPayload && (
                                <LibraryItem
                                    key="generating"
                                    item={currentGenPayload}
                                    isGenerating={true}
                                    onStop={() => stopGeneration()}
                                    status={status}
                                    elapsedTime={elapsedTime}
                                    estimatedTime={estimatedTime}
                                />
                            )}
                            {library.filter(item => !currentGenId || item.id !== currentGenId).map(item => <LibraryItem key={item.id} item={item} onDelete={() => deleteGeneration(item.id)} onPlay={playActivityAudio} onUpdate={loadLibrary} onStop={() => stopGeneration(item.id)} isCurrentlyPlaying={activityPlayingId === item.id} isAudioPlaying={isAudioPlaying} />)}
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
                    {(() => {
                        const playerCoverUrl = activityPlayingItem?.metadata?.cover
                            ? `/api/generation/${activityPlayingItem.id}/cover?v=${activityPlayingItem.metadata.cover}`
                            : null;
                        return (
                            <div style={{
                                width: '48px',
                                height: '48px',
                                borderRadius: '6px',
                                backgroundColor: activityPlayingItem ? (playerCoverUrl ? 'transparent' : '#10B981') : '#3a3a3a',
                                backgroundImage: playerCoverUrl ? `url(${playerCoverUrl})` : 'none',
                                backgroundSize: 'cover',
                                backgroundPosition: 'center',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                flexShrink: 0,
                            }}>
                                {!playerCoverUrl && <MusicNoteIcon size={20} color={activityPlayingItem ? '#fff' : '#666'} />}
                            </div>
                        );
                    })()}
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

            {/* Model Manager Modal */}
            {showModelManager && (
                <div style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 1000,
                }} onClick={() => setShowModelManager(false)}>
                    <div
                        style={{
                            backgroundColor: '#1a1a1a',
                            borderRadius: '16px',
                            padding: '24px',
                            width: '500px',
                            maxWidth: '90vw',
                            maxHeight: '80vh',
                            overflow: 'auto',
                            border: '1px solid #333',
                            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
                        }}
                        onClick={e => e.stopPropagation()}
                    >
                        {/* Header */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                            <h2 style={{ margin: 0, fontSize: '18px', fontWeight: '600', color: '#fff' }}>
                                Model Manager
                            </h2>
                            <button
                                onClick={() => setShowModelManager(false)}
                                style={{
                                    background: 'none',
                                    border: 'none',
                                    color: '#888',
                                    cursor: 'pointer',
                                    fontSize: '20px',
                                    padding: '4px 8px',
                                    lineHeight: 1,
                                }}
                            >
                                x
                            </button>
                        </div>

                        {/* GPU Info */}
                        {gpuInfo && gpuInfo.gpu && (
                            <div style={{
                                backgroundColor: '#252525',
                                borderRadius: '8px',
                                padding: '12px',
                                marginBottom: '16px',
                                fontSize: '12px',
                            }}>
                                <div style={{ color: '#fff', fontWeight: '500', marginBottom: '4px' }}>
                                    {gpuInfo.gpu.name}
                                </div>
                                <div style={{ color: '#888' }}>
                                    {gpuInfo.gpu.free_gb}GB available / {gpuInfo.gpu.total_gb}GB total
                                </div>
                            </div>
                        )}

                        {/* Model List */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            {allModels.map(m => (
                                <div key={m.id} style={{
                                    backgroundColor: '#252525',
                                    borderRadius: '12px',
                                    padding: '16px',
                                    border: m.status === 'ready' ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid #333',
                                }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                                        <div>
                                            <div style={{ fontSize: '14px', fontWeight: '500', color: '#fff', marginBottom: '4px' }}>
                                                {m.name}
                                                {m.id === recommendedModel && (
                                                    <span style={{
                                                        marginLeft: '8px',
                                                        fontSize: '10px',
                                                        backgroundColor: 'rgba(16, 185, 129, 0.2)',
                                                        color: '#10B981',
                                                        padding: '2px 6px',
                                                        borderRadius: '4px',
                                                    }}>
                                                        Recommended
                                                    </span>
                                                )}
                                            </div>
                                            <div style={{ fontSize: '12px', color: '#888' }}>{m.description}</div>
                                        </div>
                                        <div style={{ textAlign: 'right', fontSize: '11px', color: '#666' }}>
                                            <div>{m.size_gb}GB</div>
                                            <div>{m.vram_required}GB VRAM</div>
                                        </div>
                                    </div>

                                    {/* Status & Actions */}
                                    {m.status === 'ready' && (
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                            <span style={{ fontSize: '12px', color: '#10B981', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                                                    <polyline points="20 6 9 17 4 12"/>
                                                </svg>
                                                Ready to use
                                            </span>
                                            <button
                                                onClick={() => deleteModel(m.id)}
                                                style={{
                                                    padding: '6px 12px',
                                                    fontSize: '11px',
                                                    backgroundColor: 'transparent',
                                                    color: '#EF4444',
                                                    border: '1px solid rgba(239, 68, 68, 0.3)',
                                                    borderRadius: '6px',
                                                    cursor: 'pointer',
                                                }}
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    )}

                                    {m.status === 'downloading' && (
                                        <div>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                                <span style={{ fontSize: '12px', color: '#F59E0B' }}>
                                                    Downloading... {m.progress || 0}%
                                                </span>
                                                <button
                                                    onClick={() => cancelModelDownload(m.id)}
                                                    style={{
                                                        padding: '4px 10px',
                                                        fontSize: '11px',
                                                        backgroundColor: 'transparent',
                                                        color: '#EF4444',
                                                        border: '1px solid rgba(239, 68, 68, 0.3)',
                                                        borderRadius: '6px',
                                                        cursor: 'pointer',
                                                    }}
                                                >
                                                    Cancel
                                                </button>
                                            </div>
                                            <div style={{
                                                height: '6px',
                                                backgroundColor: '#333',
                                                borderRadius: '3px',
                                                overflow: 'hidden',
                                            }}>
                                                <div style={{
                                                    width: `${m.progress || 0}%`,
                                                    height: '100%',
                                                    backgroundColor: '#F59E0B',
                                                    borderRadius: '3px',
                                                    transition: 'width 0.3s ease',
                                                }} />
                                            </div>
                                            {m.speed_mbps > 0 && (
                                                <div style={{ fontSize: '10px', color: '#666', marginTop: '6px' }}>
                                                    {m.speed_mbps.toFixed(1)} MB/s
                                                    {m.eta_seconds > 0 && ` - ~${Math.ceil(m.eta_seconds / 60)} min remaining`}
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {m.status === 'not_downloaded' && (
                                        <button
                                            onClick={() => startModelDownload(m.id)}
                                            style={{
                                                width: '100%',
                                                padding: '10px',
                                                fontSize: '12px',
                                                backgroundColor: '#10B981',
                                                color: '#fff',
                                                border: 'none',
                                                borderRadius: '8px',
                                                cursor: 'pointer',
                                                fontWeight: '500',
                                            }}
                                        >
                                            Download ({m.size_gb}GB)
                                        </button>
                                    )}
                                </div>
                            ))}
                        </div>

                        {/* Footer note */}
                        <div style={{ marginTop: '16px', fontSize: '11px', color: '#666', textAlign: 'center' }}>
                            Models are stored locally. Delete unused models to free disk space.
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

// Render the app
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
