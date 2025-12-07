"""
SongGeneration Studio - Backend API
AI Song Generation with Full Style Control

PATCHED: Fixed style not being applied - auto_prompt_audio_type was overriding descriptions
"""

import os
import sys
import json
import uuid
import asyncio
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ============================================================================
# GPU/VRAM Detection
# ============================================================================

def get_gpu_info() -> dict:
    """Detect GPU and available VRAM."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpus.append({
                        'name': parts[0],
                        'total_mb': int(parts[1]),
                        'free_mb': int(parts[2]),
                        'used_mb': int(parts[3]),
                        'total_gb': round(int(parts[1]) / 1024, 1),
                        'free_gb': round(int(parts[2]) / 1024, 1),
                    })
            if gpus:
                gpu = gpus[0]  # Primary GPU
                if gpu['free_gb'] >= 24:
                    recommended = 'full'
                else:
                    recommended = 'low'
                return {
                    'available': True,
                    'gpu': gpu,
                    'recommended_mode': recommended,
                    'can_run_full': gpu['free_gb'] >= 24,
                    'can_run_low': gpu['free_gb'] >= 10,
                }
    except Exception as e:
        print(f"[GPU] Detection error: {e}")

    return {
        'available': False,
        'gpu': None,
        'recommended_mode': 'low',
        'can_run_full': False,
        'can_run_low': False,
    }

# Detect GPU on startup
gpu_info = get_gpu_info()
if gpu_info['available']:
    print(f"[GPU] Detected: {gpu_info['gpu']['name']}")
    print(f"[GPU] VRAM: {gpu_info['gpu']['free_gb']}GB free / {gpu_info['gpu']['total_gb']}GB total")
    print(f"[GPU] Recommended mode: {gpu_info['recommended_mode']}")
else:
    print("[GPU] No NVIDIA GPU detected or nvidia-smi not available")

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent
DEFAULT_MODEL = "songgeneration_base"
OUTPUT_DIR = BASE_DIR / "output"
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "web" / "static"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
(BASE_DIR / "web" / "static").mkdir(parents=True, exist_ok=True)

def get_available_models() -> List[dict]:
    """Detect available model folders in BASE_DIR."""
    models = []
    model_patterns = [
        ("songgeneration_base", "Base (2m30s)", "Chinese + English, 10GB VRAM, max 2m30s"),
        ("songgeneration_base_new", "Base New (2m30s)", "Updated base model, max 2m30s"),
        ("songgeneration_base_full", "Base Full (4m30s)", "Full duration up to 4m30s, 12GB VRAM"),
        ("songgeneration_large", "Large (4m30s)", "Best quality, 22GB VRAM, max 4m30s"),
    ]

    for folder_name, display_name, description in model_patterns:
        folder_path = BASE_DIR / folder_name
        if folder_path.exists():
            # Check for various model file patterns
            has_model = (
                (folder_path / "model.pt").exists() or
                (folder_path / "model.safetensors").exists() or
                any(folder_path.glob("*.pt")) or
                any(folder_path.glob("*.safetensors")) or
                any(folder_path.glob("*.bin")) or
                (folder_path / "config.yaml").exists()  # Some models only have config
            )
            has_config = (folder_path / "config.yaml").exists()

            if has_model or has_config:
                models.append({
                    "id": folder_name,
                    "name": display_name,
                    "description": description,
                    "path": str(folder_path),
                    "has_config": has_config,
                    "status": "ready"
                })

    return models

available_models = get_available_models()
print(f"[CONFIG] Base dir: {BASE_DIR}")
print(f"[CONFIG] Output dir: {OUTPUT_DIR}")
print(f"[CONFIG] Available models: {[m['id'] for m in available_models]}")

# ============================================================================
# Data Models
# ============================================================================

class Section(BaseModel):
    type: str
    lyrics: Optional[str] = None

class SongRequest(BaseModel):
    title: str = "Untitled"
    sections: List[Section]
    gender: str = "female"
    timbre: str = ""
    genre: str = ""
    emotion: str = ""
    instruments: str = ""
    custom_style: Optional[str] = None  # Additional free-text style descriptors
    bpm: int = 120
    output_mode: str = "mixed"
    auto_prompt_type: Optional[str] = None
    reference_audio_id: Optional[str] = None
    model: str = "songgeneration_base"
    memory_mode: str = "auto"
    # Advanced generation parameters
    cfg_coef: float = 1.5          # Classifier-free guidance (0.1-3.0)
    temperature: float = 0.8       # Sampling randomness (0.1-2.0)
    top_k: int = 50                # Top-K sampling (1-250)
    top_p: float = 0.0             # Nucleus sampling, 0 = disabled (0.0-1.0)
    extend_stride: int = 5         # Extension stride for longer songs

# ============================================================================
# State
# ============================================================================

generations: dict[str, dict] = {}
running_processes: dict[str, asyncio.subprocess.Process] = {}  # Track running processes for stop functionality

def restore_library():
    """Restore completed generations from output directory on startup."""
    global generations
    restored = 0
    
    print(f"[LIBRARY] Scanning output directory: {OUTPUT_DIR}")
    
    if not OUTPUT_DIR.exists():
        return
    
    subdirs = list(OUTPUT_DIR.iterdir())
    for subdir in subdirs:
        if not subdir.is_dir():
            continue
        
        gen_id = subdir.name
        
        # Look for audio files
        audio_files = []
        for search_dir in [subdir, subdir / "audios"]:
            if search_dir.exists():
                audio_files.extend(search_dir.glob("*.flac"))
                audio_files.extend(search_dir.glob("*.wav"))
                audio_files.extend(search_dir.glob("*.mp3"))
        
        if not audio_files:
            continue
        
        audio_files = sorted(set(audio_files))
        
        # Look for metadata file
        metadata_path = subdir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"[LIBRARY] Error loading metadata for {gen_id}: {e}")
        
        # Fallback dates
        try:
            file_mtime = datetime.fromtimestamp(audio_files[0].stat().st_mtime).isoformat()
        except:
            file_mtime = datetime.now().isoformat()

        # Check for cover image if not in metadata
        if not metadata.get("cover"):
            for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                cover_path = subdir / f"cover{ext}"
                if cover_path.exists():
                    metadata["cover"] = f"cover{ext}"
                    # Update metadata file on disk too
                    try:
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"[LIBRARY] Warning: Could not update metadata for {gen_id}: {e}")
                    break

        generations[gen_id] = {
            "id": gen_id,
            "status": "completed",
            "progress": 100,
            "message": "Complete",
            "title": metadata.get("title", "Untitled"),
            "model": metadata.get("model", "unknown"),
            "created_at": metadata.get("created_at", file_mtime),
            "completed_at": metadata.get("completed_at", file_mtime),
            "output_files": [str(f) for f in audio_files],
            "audio_files": [f.name for f in audio_files],
            "output_dir": str(subdir),
            "metadata": metadata if metadata else {
                "title": "Untitled",
                "model": "unknown",
                "created_at": file_mtime,
            }
        }
        restored += 1
    
    print(f"[LIBRARY] Restored {restored} generation(s)")

# Restore library on import
restore_library()

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="SongGeneration Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Helper Functions
# ============================================================================

def build_lyrics_string(sections: List[Section]) -> str:
    """Build the lyrics string in SongGeneration format."""
    parts = []
    for section in sections:
        tag = f"[{section.type}]"
        if section.lyrics:
            lyrics = section.lyrics.replace('\n', '. ').replace('..', '.').strip()
            if not lyrics.endswith('.'):
                lyrics += '.'
            parts.append(f"{tag} {lyrics}")
        else:
            parts.append(tag)
    return " ; ".join(parts)

def build_description(request: SongRequest, exclude_genre: bool = False) -> str:
    """Build the description string for style control.
    
    Args:
        request: The song request
        exclude_genre: If True, excludes genre from description (use when auto_prompt handles genre)
    """
    parts = []

    if request.gender and request.gender != "auto":
        parts.append(request.gender)

    if request.timbre:
        parts.append(request.timbre)

    # Only include genre in description if not handled by auto_prompt
    if not exclude_genre and request.genre and request.genre != "Auto":
        parts.append(request.genre)

    if request.emotion:
        parts.append(request.emotion)

    if request.instruments:
        parts.append(request.instruments)

    # Add custom style descriptors if provided
    if request.custom_style:
        parts.append(request.custom_style)

    if request.bpm:
        parts.append(f"the bpm is {request.bpm}")

    return ", ".join(parts) + "." if parts else ""


# Genre to auto_prompt_audio_type mapping
GENRE_TO_AUTO_PROMPT = {
    "pop": "Pop", 
    "r&b": "R&B", 
    "rnb": "R&B",
    "dance": "Dance", 
    "electronic": "Dance", 
    "edm": "Dance",
    "jazz": "Jazz", 
    "folk": "Folk", 
    "acoustic": "Folk",
    "rock": "Rock", 
    "metal": "Metal", 
    "heavy metal": "Metal",
    "reggae": "Reggae", 
    "chinese": "Chinese Style",
    "chinese style": "Chinese Style",
    "chinese tradition": "Chinese Tradition",
    "chinese opera": "Chinese Opera",
}

async def run_generation(gen_id: str, request: SongRequest, reference_path: Optional[str]):
    """Run the actual SongGeneration inference."""
    global generations

    try:
        print(f"[GEN {gen_id}] Starting generation...")
        generations[gen_id]["status"] = "processing"
        generations[gen_id]["message"] = "Preparing input..."
        generations[gen_id]["progress"] = 5

        # Validate model
        model_id = request.model or DEFAULT_MODEL
        model_path = BASE_DIR / model_id

        if not model_path.exists():
            raise Exception(f"Model not found: {model_id}")

        print(f"[GEN {gen_id}] Using model: {model_id}")
        generations[gen_id]["model"] = model_id

        # Create input JSONL
        input_file = UPLOADS_DIR / f"{gen_id}_input.jsonl"
        output_subdir = OUTPUT_DIR / gen_id
        output_subdir.mkdir(exist_ok=True)

        lyrics = build_lyrics_string(request.sections)

        input_data = {
            "idx": gen_id,
            "gt_lyric": lyrics,
        }

        description = ""

        # =======================================================================
        # FIX: Style control logic
        # The model documentation states: "Avoid providing both prompt_audio_path 
        # and descriptions at the same time."
        # 
        # The same applies to auto_prompt_audio_type - it loads pre-trained style
        # embeddings that can OVERRIDE text descriptions!
        #
        # Solution: Only use auto_prompt_audio_type when there are NO descriptions.
        # When descriptions are provided, let them control the style entirely.
        # =======================================================================
        
        if reference_path:
            # User provided reference audio - use it exclusively
            input_data["prompt_audio_path"] = reference_path
            print(f"[GEN {gen_id}] Using reference audio for style (no text descriptions to avoid conflicts)")
        else:
            # =======================================================================
            # STYLE CONTROL STRATEGY:
            # 
            # The model works best with BOTH:
            # 1. auto_prompt_audio_type - loads pre-trained audio embeddings for the genre
            # 2. descriptions - text conditioning for gender, timbre, instruments, BPM
            #
            # Looking at official demo, they use genre -> auto_prompt but empty description.
            # However, we want to also control gender/timbre/etc, so we use descriptions
            # BUT exclude the genre from descriptions to avoid conflict.
            # =======================================================================
            
            # Determine auto_prompt_audio_type from genre
            auto_type = "Auto"  # Default
            genre_for_auto_prompt = None
            
            if request.genre:
                # Extract first genre from comma-separated list
                first_genre = request.genre.split(',')[0].strip().lower()
                if first_genre in GENRE_TO_AUTO_PROMPT:
                    auto_type = GENRE_TO_AUTO_PROMPT[first_genre]
                    genre_for_auto_prompt = first_genre
            
            # Always set auto_prompt_audio_type - this provides the core musical style
            input_data["auto_prompt_audio_type"] = auto_type
            print(f"[GEN {gen_id}] Using auto_prompt_audio_type: {auto_type}")
            
            # Build descriptions for OTHER attributes (gender, timbre, emotion, instruments, BPM)
            # Exclude genre if it's being handled by auto_prompt to avoid conflict
            exclude_genre = genre_for_auto_prompt is not None
            description = build_description(request, exclude_genre=exclude_genre)
            
            if description:
                input_data["descriptions"] = description
                print(f"[GEN {gen_id}] Additional descriptions: {description}")
            else:
                print(f"[GEN {gen_id}] No additional descriptions (using auto_prompt style only)")

        print(f"[GEN {gen_id}] Lyrics: {lyrics[:200]}...")
        print(f"[GEN {gen_id}] Input data: {json.dumps(input_data, indent=2)}")

        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(input_data, f, ensure_ascii=False)
            f.write('\n')

        generations[gen_id]["message"] = f"Loading model ({model_id})..."
        generations[gen_id]["progress"] = 10

        # Build command
        cmd = [
            sys.executable, "generate.py",
            "--ckpt_path", model_id,
            "--input_jsonl", str(input_file),
            "--save_dir", str(output_subdir),
        ]

        # Memory mode handling
        memory_mode = request.memory_mode
        if memory_mode == "auto":
            current_gpu = get_gpu_info()
            if current_gpu['available'] and current_gpu['can_run_full']:
                memory_mode = "full"
                print(f"[GEN {gen_id}] Auto-selected FULL mode ({current_gpu['gpu']['free_gb']}GB free)")
            else:
                memory_mode = "low"
                free_gb = current_gpu['gpu']['free_gb'] if current_gpu['available'] else 'unknown'
                print(f"[GEN {gen_id}] Auto-selected LOW memory mode ({free_gb}GB free)")

        generations[gen_id]["memory_mode"] = memory_mode
        print(f"[GEN {gen_id}] Memory mode: {memory_mode}")

        if memory_mode == "low":
            cmd.append("--low_mem")

        # Advanced generation parameters - pass via environment variables
        adv_params = {
            "SONGGEN_CFG_COEF": str(request.cfg_coef),
            "SONGGEN_TEMPERATURE": str(request.temperature),
            "SONGGEN_TOP_K": str(request.top_k),
            "SONGGEN_TOP_P": str(request.top_p),
            "SONGGEN_EXTEND_STRIDE": str(request.extend_stride),
        }

        print(f"[GEN {gen_id}] Advanced params: cfg={request.cfg_coef}, temp={request.temperature}, top_k={request.top_k}, top_p={request.top_p}")

        # Output mode - use --generate_type argument
        if request.output_mode and request.output_mode != "mixed":
            cmd.extend(["--generate_type", request.output_mode])

        print(f"[GEN {gen_id}] Command: {' '.join(cmd)}")

        generations[gen_id]["message"] = "Starting inference..."
        generations[gen_id]["progress"] = 15

        # Set up environment with correct PYTHONPATH (use os.pathsep for cross-platform)
        flow_vae_dir = BASE_DIR / "codeclm" / "tokenizer" / "Flow1dVAE"
        env = os.environ.copy()
        pathsep = os.pathsep  # ; on Windows, : on Unix
        env["PYTHONPATH"] = f"{BASE_DIR}{pathsep}{flow_vae_dir}{pathsep}{env.get('PYTHONPATH', '')}"
        env["PYTHONUTF8"] = "1"
        # Add advanced params to environment
        env.update(adv_params)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(BASE_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=1024*1024
        )

        # Store process reference for stop functionality
        running_processes[gen_id] = process

        generations[gen_id]["message"] = "Generating..."
        generations[gen_id]["progress"] = 20

        all_stderr = []
        stopped = False
        while True:
            # Check if generation was stopped
            if generations[gen_id].get("status") == "stopped":
                stopped = True
                print(f"[GEN {gen_id}] Stop requested, terminating process...")
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                break

            try:
                line = await asyncio.wait_for(process.stderr.readline(), timeout=1.0)
                if not line:
                    break
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str:
                    log_line = line_str[:200] + '...' if len(line_str) > 200 else line_str
                    all_stderr.append(log_line)
                    print(f"[GEN {gen_id}] {log_line}")

                    if "%" in line_str:
                        match = re.search(r'(\d+)%', line_str)
                        if match:
                            pct = int(match.group(1))
                            progress = min(95, 20 + (pct * 0.75))
                            generations[gen_id]["progress"] = progress
            except asyncio.TimeoutError:
                # Timeout on readline, just continue to check for stop
                continue
            except ValueError:
                chunk = await process.stderr.read(8192)
                if not chunk:
                    break
                chunk_str = chunk.decode('utf-8', errors='ignore')
                if "%" in chunk_str:
                    match = re.search(r'(\d+)%', chunk_str)
                    if match:
                        pct = int(match.group(1))
                        progress = min(95, 20 + (pct * 0.75))
                        generations[gen_id]["progress"] = progress

        # Clean up process reference
        running_processes.pop(gen_id, None)

        # If stopped (either by flag or by status), don't continue
        if stopped or generations[gen_id]["status"] == "stopped":
            generations[gen_id]["status"] = "stopped"
            generations[gen_id]["message"] = "Generation stopped by user"
            input_file.unlink(missing_ok=True)
            return

        await process.wait()
        stdout = await process.stdout.read()

        print(f"[GEN {gen_id}] Process finished with code {process.returncode}")

        # Check again if stopped (process may have been terminated)
        if generations[gen_id]["status"] == "stopped":
            generations[gen_id]["message"] = "Generation stopped by user"
            input_file.unlink(missing_ok=True)
            return

        if process.returncode != 0:
            # Filter out common warnings from stderr to get actual error
            error_lines = [line for line in all_stderr[-20:] if not any(w in line for w in [
                'is_flash_attn_available',
                'deprecated',
                'FutureWarning',
                'UserWarning',
            ])]
            stderr_text = '\n'.join(error_lines).strip() if error_lines else f"Process exited with code {process.returncode}"
            raise Exception(f"Generation failed: {stderr_text}")

        # Find output files
        audios_dir = output_subdir / "audios"
        output_files = []

        for search_dir in [audios_dir, output_subdir]:
            if search_dir.exists():
                output_files.extend(search_dir.glob("*.wav"))
                output_files.extend(search_dir.glob("*.flac"))
                output_files.extend(search_dir.glob("*.mp3"))

        output_files = sorted(set(output_files))
        print(f"[GEN {gen_id}] Found {len(output_files)} audio files: {[f.name for f in output_files]}")

        if not output_files:
            print(f"[GEN {gen_id}] Contents of {output_subdir}:")
            for item in output_subdir.rglob("*"):
                print(f"[GEN {gen_id}]   - {item.relative_to(output_subdir)}")
            raise Exception("No output file generated")

        generations[gen_id]["status"] = "completed"
        generations[gen_id]["progress"] = 100
        generations[gen_id]["message"] = "Song generated successfully!"
        generations[gen_id]["output_files"] = [str(f) for f in output_files]
        generations[gen_id]["output_file"] = str(output_files[0])
        generations[gen_id]["completed_at"] = datetime.now().isoformat()

        # Save complete metadata for library restoration
        try:
            metadata_path = output_subdir / "metadata.json"
            metadata = {
                "id": gen_id,
                "title": request.title,
                "model": model_id,
                "created_at": generations[gen_id].get("created_at", datetime.now().isoformat()),
                "completed_at": datetime.now().isoformat(),
                "gender": request.gender,
                "timbre": request.timbre,
                "genre": request.genre,
                "emotion": request.emotion,
                "instruments": request.instruments,
                "custom_style": request.custom_style,
                "bpm": request.bpm,
                "output_mode": request.output_mode,
                "memory_mode": request.memory_mode,
                "sections": [{"type": s.type, "lyrics": s.lyrics} for s in request.sections],
                "description": description,
                "reference_audio": reference_path if reference_path else None,
                "reference_audio_id": request.reference_audio_id,
                "audio_files": [f.name for f in output_files],
                # Advanced parameters
                "cfg_coef": request.cfg_coef,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p,
                "extend_stride": request.extend_stride,
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            generations[gen_id]["metadata"] = metadata
            
        except Exception as meta_err:
            print(f"[GEN {gen_id}] Warning: Could not save metadata: {meta_err}")

        input_file.unlink(missing_ok=True)

    except Exception as e:
        print(f"[GEN {gen_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        generations[gen_id]["status"] = "failed"
        generations[gen_id]["message"] = str(e)

# ============================================================================
# API Routes
# ============================================================================

@app.get("/")
async def root():
    """Serve the main UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        response = FileResponse(index_path)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return {"message": "SongGeneration Studio API", "status": "running"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    global available_models, gpu_info
    available_models = get_available_models()
    gpu_info = get_gpu_info()
    return {
        "status": "ok",
        "models": available_models,
        "default_model": DEFAULT_MODEL,
        "gpu": gpu_info
    }

@app.get("/api/gpu")
async def get_gpu_status():
    """Get current GPU status and VRAM."""
    global gpu_info
    gpu_info = get_gpu_info()
    return gpu_info

@app.get("/api/models")
async def list_models():
    """List all available models."""
    global available_models
    available_models = get_available_models()
    return {
        "models": available_models,
        "default": DEFAULT_MODEL
    }

@app.post("/api/upload-reference")
async def upload_reference(file: UploadFile = File(...)):
    """Upload a reference audio file."""
    allowed_ext = ('.wav', '.mp3', '.flac', '.ogg')
    if not file.filename.lower().endswith(allowed_ext):
        raise HTTPException(400, f"Invalid file type. Allowed: {allowed_ext}")

    file_id = str(uuid.uuid4())
    file_path = UPLOADS_DIR / f"{file_id}_{file.filename}"

    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)

    print(f"[UPLOAD] Saved reference: {file_path} ({len(content)} bytes)")

    return {
        "id": file_id,
        "filename": file.filename,
        "path": str(file_path)
    }

@app.get("/api/reference/{ref_id}")
async def get_reference_audio(ref_id: str):
    """Stream a reference audio file."""
    ref_files = list(UPLOADS_DIR.glob(f"{ref_id}_*"))
    if not ref_files:
        raise HTTPException(404, "Reference audio not found")

    file_path = ref_files[0]

    # Determine media type
    ext = file_path.suffix.lower()
    media_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg'
    }
    media_type = media_types.get(ext, 'audio/wav')

    return FileResponse(file_path, media_type=media_type)

@app.post("/api/generate")
async def generate_song(request: SongRequest, background_tasks: BackgroundTasks):
    """Start a new song generation."""
    gen_id = str(uuid.uuid4())[:8]

    print(f"[API] New generation request: {gen_id}")
    print(f"[API] Title: {request.title}")
    print(f"[API] Model: {request.model}")
    print(f"[API] Sections: {len(request.sections)}")
    print(f"[API] Style: {request.genre}, {request.emotion}, {request.timbre}, Voice: {request.gender}")

    reference_path = None
    if request.reference_audio_id:
        ref_files = list(UPLOADS_DIR.glob(f"{request.reference_audio_id}_*"))
        if ref_files:
            reference_path = str(ref_files[0])

    generations[gen_id] = {
        "id": gen_id,
        "title": request.title,
        "model": request.model,
        "status": "pending",
        "progress": 0,
        "message": "Queued for generation...",
        "output_file": None,
        "output_files": [],
        "created_at": datetime.now().isoformat(),
        "completed_at": None
    }

    background_tasks.add_task(run_generation, gen_id, request, reference_path)

    return {"generation_id": gen_id}

@app.get("/api/generation/{gen_id}")
async def get_generation_status(gen_id: str):
    """Get the status of a generation."""
    if gen_id not in generations:
        raise HTTPException(404, "Generation not found")
    return generations[gen_id]

@app.post("/api/stop/{gen_id}")
async def stop_generation(gen_id: str):
    """Stop a running generation."""
    if gen_id not in generations:
        raise HTTPException(404, "Generation not found")

    gen = generations[gen_id]
    if gen["status"] not in ("pending", "processing"):
        raise HTTPException(400, f"Cannot stop generation with status: {gen['status']}")

    print(f"[API] Stop requested for generation: {gen_id}")

    # Set status to stopped - the run_generation loop will pick this up
    generations[gen_id]["status"] = "stopped"
    generations[gen_id]["message"] = "Stopping..."

    # If process exists, try to terminate it directly too
    if gen_id in running_processes:
        try:
            running_processes[gen_id].terminate()
        except Exception as e:
            print(f"[API] Error terminating process: {e}")

    return {"status": "stopped", "message": "Generation stop requested"}

@app.delete("/api/generation/{gen_id}")
async def delete_generation(gen_id: str):
    """Delete a generation and its output files."""
    if gen_id not in generations:
        raise HTTPException(404, "Generation not found")

    gen = generations[gen_id]
    if gen["status"] in ("pending", "processing"):
        raise HTTPException(400, "Cannot delete a generation that is still running. Stop it first.")

    print(f"[API] Delete requested for generation: {gen_id}")

    # Delete output files
    output_subdir = OUTPUT_DIR / gen_id
    if output_subdir.exists():
        import shutil
        try:
            shutil.rmtree(output_subdir)
            print(f"[API] Deleted output directory: {output_subdir}")
        except Exception as e:
            print(f"[API] Error deleting output directory: {e}")

    # Delete input file if exists
    input_file = UPLOADS_DIR / f"{gen_id}_input.jsonl"
    if input_file.exists():
        try:
            input_file.unlink()
            print(f"[API] Deleted input file: {input_file}")
        except Exception as e:
            print(f"[API] Error deleting input file: {e}")

    # Remove from generations dict
    del generations[gen_id]

    return {"status": "deleted", "message": "Generation deleted successfully"}

class UpdateGenerationRequest(BaseModel):
    title: Optional[str] = None

@app.put("/api/generation/{gen_id}")
async def update_generation(gen_id: str, request: UpdateGenerationRequest):
    """Update generation metadata (title, etc.)."""
    output_subdir = OUTPUT_DIR / gen_id
    metadata_path = output_subdir / "metadata.json"

    # Check if generation exists (in memory or on disk)
    if gen_id not in generations and not output_subdir.exists():
        raise HTTPException(404, "Generation not found")

    # Load existing metadata
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"[API] Error loading metadata: {e}")

    # Update fields
    if request.title is not None:
        metadata["title"] = request.title
        # Update in-memory if exists
        if gen_id in generations:
            generations[gen_id]["title"] = request.title
            if "metadata" in generations[gen_id]:
                generations[gen_id]["metadata"]["title"] = request.title

    # Save updated metadata
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[API] Error saving metadata: {e}")
        raise HTTPException(500, f"Failed to save metadata: {e}")

    print(f"[API] Updated generation {gen_id}: title='{request.title}'")
    return {"status": "updated", "metadata": metadata}

@app.post("/api/generation/{gen_id}/cover")
async def upload_cover(gen_id: str, file: UploadFile = File(...)):
    """Upload an album cover image for a generation."""
    output_subdir = OUTPUT_DIR / gen_id

    # Check if generation exists (in memory or on disk)
    if gen_id not in generations and not output_subdir.exists():
        raise HTTPException(404, "Generation not found")

    allowed_ext = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
    if not file.filename.lower().endswith(allowed_ext):
        raise HTTPException(400, f"Invalid file type. Allowed: {allowed_ext}")

    if not output_subdir.exists():
        output_subdir.mkdir(parents=True, exist_ok=True)

    # Save cover image (always as cover.jpg/png based on upload)
    ext = Path(file.filename).suffix.lower()
    cover_path = output_subdir / f"cover{ext}"

    # Remove any existing cover files
    for old_cover in output_subdir.glob("cover.*"):
        old_cover.unlink()

    content = await file.read()
    with open(cover_path, 'wb') as f:
        f.write(content)

    # Update metadata
    metadata_path = output_subdir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except:
            pass

    metadata["cover"] = f"cover{ext}"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Update in-memory generation if exists
    if gen_id in generations:
        if "metadata" not in generations[gen_id]:
            generations[gen_id]["metadata"] = {}
        generations[gen_id]["metadata"]["cover"] = f"cover{ext}"

    print(f"[API] Uploaded cover for {gen_id}: {cover_path} ({len(content)} bytes)")
    return {"status": "uploaded", "cover": f"cover{ext}"}

@app.get("/api/generation/{gen_id}/cover")
async def get_cover(gen_id: str):
    """Get the album cover image for a generation."""
    output_subdir = OUTPUT_DIR / gen_id

    # Check if generation exists (in memory or on disk)
    if gen_id not in generations and not output_subdir.exists():
        raise HTTPException(404, "Generation not found")

    # Look for cover file
    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        cover_path = output_subdir / f"cover{ext}"
        if cover_path.exists():
            media_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            return FileResponse(cover_path, media_type=media_types.get(ext, 'image/jpeg'))

    raise HTTPException(404, "No cover image found")

@app.delete("/api/generation/{gen_id}/cover")
async def delete_cover(gen_id: str):
    """Delete the album cover image for a generation."""
    output_subdir = OUTPUT_DIR / gen_id

    # Check if generation exists (in memory or on disk)
    if gen_id not in generations and not output_subdir.exists():
        raise HTTPException(404, "Generation not found")

    # Find and delete cover file
    deleted = False
    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        cover_path = output_subdir / f"cover{ext}"
        if cover_path.exists():
            cover_path.unlink()
            deleted = True
            print(f"[API] Deleted cover for {gen_id}: {cover_path}")
            break

    if not deleted:
        raise HTTPException(404, "No cover image found")

    # Update metadata
    metadata_path = output_subdir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if "cover" in metadata:
                del metadata["cover"]
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[API] Warning: Could not update metadata: {e}")

    # Update in-memory generation if exists
    if gen_id in generations:
        if "metadata" in generations[gen_id] and "cover" in generations[gen_id]["metadata"]:
            del generations[gen_id]["metadata"]["cover"]

    return {"status": "deleted", "message": "Cover image deleted"}

@app.get("/api/generation/{gen_id}/video")
async def export_video(gen_id: str, background_tasks: BackgroundTasks):
    """Export generation as MP4 video with waveform visualization."""
    import subprocess
    import tempfile

    output_subdir = OUTPUT_DIR / gen_id

    # Check if generation exists (in memory or on disk)
    if gen_id in generations:
        gen = generations[gen_id]
        if gen["status"] != "completed":
            raise HTTPException(400, "Generation not completed")
    elif output_subdir.exists():
        # Load from disk - check for metadata
        metadata_path = output_subdir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                gen = {"metadata": json.load(f), "status": "completed"}
        else:
            gen = {"metadata": {}, "status": "completed"}
    else:
        raise HTTPException(404, "Generation not found")

    # Find the audio file (check both main dir and audios/ subdirectory)
    audio_file = None
    search_dirs = [output_subdir, output_subdir / "audios"]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for ext in ['.wav', '.mp3', '.flac']:
            for f in search_dir.glob(f'*{ext}'):
                audio_file = f
                break
            if audio_file:
                break
        if audio_file:
            break

    if not audio_file:
        raise HTTPException(404, f"Audio file not found in {output_subdir}")

    # Find cover image or use default
    cover_path = None
    has_custom_cover = False  # Track if user has a custom album cover
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        cp = output_subdir / f"cover{ext}"
        if cp.exists():
            cover_path = cp
            has_custom_cover = True
            break

    # Create temp directory for video export
    temp_dir = Path(tempfile.gettempdir()) / "songgen_videos"
    temp_dir.mkdir(exist_ok=True)

    if not cover_path:
        # Use default background - check multiple possible locations
        possible_defaults = [
            Path(__file__).parent / "web" / "static" / "default.jpg",
            Path(__file__).parent.parent / "web" / "static" / "default.jpg",
            Path(__file__).parent / "static" / "default.jpg",
        ]
        for dp in possible_defaults:
            if dp.exists():
                cover_path = dp
                break

        if not cover_path:
            # Generate a simple black background if no default found
            print("[API] Warning: No default background image found, generating a black background")
            temp_bg = temp_dir / f"{gen_id}_bg.png"
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=1080x1080:d=1',
                    '-frames:v', '1', str(temp_bg)
                ], capture_output=True, timeout=30)
                if temp_bg.exists():
                    cover_path = temp_bg
            except Exception as e:
                print(f"[API] Failed to generate fallback background: {e}")

        if not cover_path:
            raise HTTPException(500, "Default background image not found and could not generate fallback")

    # Output video path
    video_path = temp_dir / f"{gen_id}.mp4"
    waveform_path = temp_dir / f"{gen_id}_waveform.png"

    # Get audio duration
    duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_file)]
    try:
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=30)
        if duration_result.returncode != 0 or not duration_result.stdout.strip():
            print(f"[API] ffprobe failed: {duration_result.stderr}")
            raise HTTPException(500, f"Failed to get audio duration. Is ffmpeg/ffprobe installed? Error: {duration_result.stderr}")
        duration = float(duration_result.stdout.strip())
    except ValueError as e:
        print(f"[API] Failed to parse duration: {duration_result.stdout}")
        raise HTTPException(500, f"Failed to parse audio duration: {e}")
    except FileNotFoundError:
        raise HTTPException(500, "ffprobe not found. Please ensure ffmpeg is installed and in PATH.")

    print(f"[API] Exporting video for {gen_id}, duration: {duration}s")

    # Step 1: Generate bright waveform image (square format: 1080x1080)
    waveform_cmd = [
        'ffmpeg', '-y', '-i', str(audio_file),
        '-filter_complex',
        'showwavespic=s=1040x120:colors=#10B981|#10B981:scale=sqrt',
        '-frames:v', '1',
        str(waveform_path)
    ]
    result = subprocess.run(waveform_cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"[API] Waveform generation failed: {result.stderr}")
        raise HTTPException(500, f"Waveform generation failed: {result.stderr}")

    # Step 2: Create video with progressive waveform reveal (1080x1080 square)
    # IMPORTANT: drawbox does NOT support 't' variable for animation!
    # Must use color source + overlay filter instead, as overlay DOES support 't'

    # Different layouts based on whether user has a custom album cover
    # - With custom cover: waveform at bottom (to not obscure the artwork)
    # - Without custom cover: waveform centered in middle (cleaner look)

    if has_custom_cover:
        # Layout: Waveform at bottom with dark bar
        waveform_y = "H-140"      # 20px from bottom
        bar_y = "ih-160"          # Dark bar at bottom
        line_y = "H-142"          # Progress line position
        print(f"[API] Using bottom waveform layout (custom cover)")

        filter_complex = (
            # Scale background image to square 1080x1080
            f"[1:v]scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080[bg];"
            # Add semi-transparent dark bar at bottom
            f"[bg]drawbox=x=0:y={bar_y}:w=iw:h=160:color=black@0.7:t=fill[bg2];"
            # Overlay bright waveform
            f"[bg2][2:v]overlay=20:{waveform_y}[v1];"
            # Create dark overlay rectangle for unplayed portion
            f"color=c=0x0d1f17:s=1040x120:r=30,format=rgba,colorchannelmixer=aa=0.75[dark];"
            # Animate dark overlay - moves right over time
            f"[v1][dark]overlay=x='20+(t/{duration})*1040':y={waveform_y}:shortest=1[v2];"
            # Create white progress line (4px wide)
            f"color=c=white:s=4x124:r=30[line];"
            # Animate progress line
            f"[v2][line]overlay=x='18+(t/{duration})*1040':y={line_y}:shortest=1[v3];"
            # Create glow effect
            f"color=c=white:s=12x124:r=30,format=rgba,colorchannelmixer=aa=0.2[glow];"
            # Animate glow
            f"[v3][glow]overlay=x='12+(t/{duration})*1040':y={line_y}:shortest=1[vout]"
        )
    else:
        # Layout: Waveform centered in middle (no dark bar needed for clean look)
        # Video is 1080x1080, waveform is 1040x120
        # Center Y = (1080 - 120) / 2 = 480
        waveform_y = 480
        line_y = waveform_y - 2   # Progress line slightly above waveform
        print(f"[API] Using centered waveform layout (no custom cover)")

        filter_complex = (
            # Scale background image to square 1080x1080
            f"[1:v]scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080[bg];"
            # Add semi-transparent dark bar in the middle (centered around waveform)
            f"[bg]drawbox=x=0:y={waveform_y - 20}:w=iw:h=160:color=black@0.5:t=fill[bg2];"
            # Overlay bright waveform (centered)
            f"[bg2][2:v]overlay=20:{waveform_y}[v1];"
            # Create dark overlay rectangle for unplayed portion
            f"color=c=0x0d1f17:s=1040x120:r=30,format=rgba,colorchannelmixer=aa=0.75[dark];"
            # Animate dark overlay - moves right over time
            f"[v1][dark]overlay=x='20+(t/{duration})*1040':y={waveform_y}:shortest=1[v2];"
            # Create white progress line (4px wide)
            f"color=c=white:s=4x124:r=30[line];"
            # Animate progress line
            f"[v2][line]overlay=x='18+(t/{duration})*1040':y={line_y}:shortest=1[v3];"
            # Create glow effect
            f"color=c=white:s=12x124:r=30,format=rgba,colorchannelmixer=aa=0.2[glow];"
            # Animate glow
            f"[v3][glow]overlay=x='12+(t/{duration})*1040':y={line_y}:shortest=1[vout]"
        )

    video_cmd = [
        'ffmpeg', '-y',
        '-i', str(audio_file),                # Input 0: audio
        '-loop', '1', '-i', str(cover_path),  # Input 1: background image
        '-loop', '1', '-i', str(waveform_path),  # Input 2: waveform
        '-filter_complex', filter_complex,
        '-map', '[vout]',
        '-map', '0:a',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-pix_fmt', 'yuv420p',
        '-shortest',
        '-t', str(duration),
        str(video_path)
    ]

    print(f"[API] Running FFmpeg video export...")
    result = subprocess.run(video_cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"[API] Video export failed: {result.stderr}")
        raise HTTPException(500, f"Video export failed: {result.stderr}")

    print(f"[API] Video exported successfully: {video_path}")

    # Clean up waveform image
    if waveform_path.exists():
        waveform_path.unlink()

    title = gen.get("title", gen.get("metadata", {}).get("title", "song"))
    return FileResponse(
        video_path,
        media_type='video/mp4',
        filename=f"{title}.mp4"
    )

def convert_audio(input_path: Path, output_format: str) -> Path:
    """Convert audio file to the specified format using ffmpeg."""
    import subprocess
    import tempfile

    output_format = output_format.lower().lstrip('.')
    if output_format not in ('mp3', 'flac', 'wav'):
        raise ValueError(f"Unsupported format: {output_format}")

    # Create output path in temp directory
    temp_dir = Path(tempfile.gettempdir()) / "songgen_conversions"
    temp_dir.mkdir(exist_ok=True)

    # Use hash of input path + format for caching
    cache_key = f"{input_path.stem}_{hash(str(input_path))}_{output_format}"
    output_path = temp_dir / f"{cache_key}.{output_format}"

    # Return cached file if exists
    if output_path.exists():
        return output_path

    # Convert using ffmpeg
    cmd = ['ffmpeg', '-y', '-i', str(input_path)]

    if output_format == 'mp3':
        cmd.extend(['-codec:a', 'libmp3lame', '-qscale:a', '2'])  # High quality MP3
    elif output_format == 'flac':
        cmd.extend(['-codec:a', 'flac', '-compression_level', '8'])
    elif output_format == 'wav':
        cmd.extend(['-codec:a', 'pcm_s16le'])

    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

    return output_path


@app.get("/api/audio/{gen_id}/{track_idx}")
async def get_audio_track(gen_id: str, track_idx: int, format: Optional[str] = None):
    """Stream a specific track from a generation.

    Args:
        gen_id: Generation ID
        track_idx: Track index (0-based)
        format: Optional output format (mp3, flac, wav). If not specified, returns original format.
    """
    if gen_id not in generations:
        raise HTTPException(404, "Generation not found")

    gen = generations[gen_id]
    if gen["status"] != "completed":
        raise HTTPException(400, "Generation not complete")

    output_files = gen.get("output_files", [])
    if track_idx >= len(output_files):
        raise HTTPException(404, f"Track {track_idx} not found")

    output_path = Path(output_files[track_idx])
    if not output_path.exists():
        raise HTTPException(404, "Audio file not found")

    media_types = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg"}

    # Convert format if requested and different from source
    if format:
        format = format.lower().lstrip('.')
        source_ext = output_path.suffix.lower().lstrip('.')

        if format not in ('mp3', 'flac', 'wav'):
            raise HTTPException(400, f"Unsupported format: {format}. Use mp3, flac, or wav.")

        if format != source_ext:
            try:
                output_path = convert_audio(output_path, format)
            except Exception as e:
                raise HTTPException(500, f"Audio conversion failed: {str(e)}")

    ext = output_path.suffix.lower()

    return FileResponse(
        output_path,
        media_type=media_types.get(ext, "audio/wav"),
        filename=f"{gen.get('title', gen_id)}_track{track_idx + 1}{ext}"
    )

@app.get("/api/generations")
async def list_generations():
    """List all generations."""
    return list(generations.values())

@app.get("/api/presets")
async def get_presets():
    """Get available style presets."""
    return {
        "genres": ["Pop", "Rock", "Metal", "Jazz", "R&B", "Folk", "Dance", "Reggae", "Chinese Style", "Electronic"],
        "emotions": ["happy", "sad", "energetic", "romantic", "angry", "peaceful", "melancholic", "hopeful"],
        "timbres": ["bright", "dark", "soft", "powerful", "warm", "clear", "raspy", "smooth"],
        "genders": ["female", "male"],
        "auto_prompts": ["Auto", "Pop", "Rock", "Metal", "Jazz", "Folk", "Dance", "R&B", "Reggae"]
    }

# Serve static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SongGeneration Studio API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  SongGeneration Studio")
    print(f"  Open http://{args.host}:{args.port} in your browser")
    print("=" * 60)
    print()

    uvicorn.run(app, host=args.host, port=args.port)
