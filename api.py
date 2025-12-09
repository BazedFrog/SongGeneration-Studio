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
import subprocess
import shutil
import threading
import time
import queue as queue_module
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
# Audio Duration Helper
# ============================================================================

def get_audio_duration(audio_path: Path) -> Optional[float]:
    """Get audio duration in seconds using ffprobe."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        print(f"[AUDIO] Failed to get duration for {audio_path}: {e}")
    return None

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent
DEFAULT_MODEL = "songgeneration_base"
OUTPUT_DIR = BASE_DIR / "output"
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "web" / "static"
QUEUE_FILE = BASE_DIR / "queue.json"
VERIFIED_MODELS_FILE = BASE_DIR / "verified_models.json"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
(BASE_DIR / "web" / "static").mkdir(parents=True, exist_ok=True)

# Cache for verified model sizes (persisted to disk)
# Format: {model_id: {"verified": True, "model_pt_size": bytes, "verified_at": timestamp}}
verified_models_cache: Dict[str, dict] = {}

def load_verified_models() -> dict:
    """Load verified models cache from disk"""
    try:
        if VERIFIED_MODELS_FILE.exists():
            with open(VERIFIED_MODELS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_verified_models(cache: dict):
    """Save verified models cache to disk"""
    try:
        with open(VERIFIED_MODELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

def mark_model_verified(model_id: str, model_pt_size: int):
    """Mark a model as verified (size checked against HuggingFace)"""
    global verified_models_cache
    verified_models_cache[model_id] = {
        "verified": True,
        "model_pt_size": model_pt_size,
        "verified_at": datetime.now().isoformat()
    }
    save_verified_models(verified_models_cache)

def is_model_verified(model_id: str) -> bool:
    """Check if model has been verified (no need to call HuggingFace again)"""
    return model_id in verified_models_cache and verified_models_cache[model_id].get("verified")

def get_verified_model_size(model_id: str) -> int:
    """Get verified model.pt size if available"""
    if model_id in verified_models_cache:
        return verified_models_cache[model_id].get("model_pt_size", 0)
    return 0

# Load verified models cache on startup
verified_models_cache = load_verified_models()

# ============================================================================
# Server-side Queue Storage
# ============================================================================

def load_queue() -> list:
    """Load queue from file"""
    try:
        if QUEUE_FILE.exists():
            with open(QUEUE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[QUEUE] Error loading queue: {e}")
    return []

def save_queue(queue: list):
    """Save queue to file"""
    try:
        with open(QUEUE_FILE, 'w', encoding='utf-8') as f:
            json.dump(queue, f, indent=2)
    except Exception as e:
        print(f"[QUEUE] Error saving queue: {e}")

# ============================================================================
# Timing History for Smart Estimates
# ============================================================================

TIMING_FILE = BASE_DIR / "timing_history.json"
MAX_TIMING_RECORDS = 100  # Keep last 100 successful generations

def load_timing_history() -> list:
    """Load timing history from file"""
    try:
        if TIMING_FILE.exists():
            with open(TIMING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[TIMING] Error loading timing history: {e}")
    return []

def save_timing_history(history: list):
    """Save timing history to file"""
    try:
        with open(TIMING_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[TIMING] Error saving timing history: {e}")

def save_timing_record(metadata: dict):
    """Save a timing record from a completed generation"""
    if metadata.get("generation_time_seconds", 0) <= 0:
        return  # Don't save if no valid timing

    record = {
        "model": metadata.get("model"),
        "num_sections": metadata.get("num_sections", 0),
        "total_lyrics_length": metadata.get("total_lyrics_length", 0),
        "has_lyrics": metadata.get("has_lyrics", False),
        "output_mode": metadata.get("output_mode", "mixed"),
        "has_reference": bool(metadata.get("reference_audio_id")),
        "generation_time_seconds": metadata.get("generation_time_seconds"),
        "completed_at": metadata.get("completed_at"),
    }

    history = load_timing_history()
    history.append(record)

    # Keep only last N records
    if len(history) > MAX_TIMING_RECORDS:
        history = history[-MAX_TIMING_RECORDS:]

    save_timing_history(history)
    print(f"[TIMING] Saved record: {record['model']}, {record['num_sections']} sections, {record['generation_time_seconds']}s")

def get_timing_stats() -> dict:
    """Calculate timing statistics from history for smart estimates"""
    history = load_timing_history()

    if not history:
        return {"has_history": False, "models": {}}

    # Group by model and calculate stats
    model_stats = {}
    for record in history:
        model = record.get("model", "unknown")
        if model not in model_stats:
            model_stats[model] = {
                "times": [],
                "with_lyrics": [],
                "without_lyrics": [],
                "with_reference": [],
                "without_reference": [],
                "by_sections": {},
            }

        time_sec = record.get("generation_time_seconds", 0)
        if time_sec <= 0:
            continue

        model_stats[model]["times"].append(time_sec)

        # Track by lyrics presence
        if record.get("has_lyrics"):
            model_stats[model]["with_lyrics"].append(time_sec)
        else:
            model_stats[model]["without_lyrics"].append(time_sec)

        # Track by reference presence
        if record.get("has_reference"):
            model_stats[model]["with_reference"].append(time_sec)
        else:
            model_stats[model]["without_reference"].append(time_sec)

        # Track by section count
        sections = record.get("num_sections", 0)
        sections_key = str(sections)
        if sections_key not in model_stats[model]["by_sections"]:
            model_stats[model]["by_sections"][sections_key] = []
        model_stats[model]["by_sections"][sections_key].append(time_sec)

    # Calculate averages
    result = {"has_history": True, "models": {}}
    for model, stats in model_stats.items():
        if not stats["times"]:
            continue

        result["models"][model] = {
            "avg_time": int(sum(stats["times"]) / len(stats["times"])),
            "min_time": min(stats["times"]),
            "max_time": max(stats["times"]),
            "count": len(stats["times"]),
            "avg_with_lyrics": int(sum(stats["with_lyrics"]) / len(stats["with_lyrics"])) if stats["with_lyrics"] else None,
            "avg_without_lyrics": int(sum(stats["without_lyrics"]) / len(stats["without_lyrics"])) if stats["without_lyrics"] else None,
            "avg_with_reference": int(sum(stats["with_reference"]) / len(stats["with_reference"])) if stats["with_reference"] else None,
            "avg_without_reference": int(sum(stats["without_reference"]) / len(stats["without_reference"])) if stats["without_reference"] else None,
            "by_sections": {
                k: int(sum(v) / len(v)) for k, v in stats["by_sections"].items() if v
            },
        }

    return result

# ============================================================================
# Model Registry & Download Manager
# ============================================================================

MODEL_REGISTRY: Dict[str, dict] = {
    "songgeneration_base": {
        "name": "SongGeneration - Base (2m30s)",
        "description": "Chinese + English, 10GB VRAM, max 2m30s",
        "vram_required": 10,
        "hf_repo": "lglg666/SongGeneration-base",
        "size_gb": 11.3,
        "priority": 1,  # Lower = recommended first for lower VRAM
    },
    "songgeneration_base_new": {
        "name": "SongGeneration - Base New (2m30s)",
        "description": "Updated base model, 10GB VRAM, max 2m30s",
        "vram_required": 10,
        "hf_repo": "lglg666/SongGeneration-base-new",
        "size_gb": 11.3,
        "priority": 2,
    },
    "songgeneration_base_full": {
        "name": "SongGeneration - Base Full (4m30s)",
        "description": "Full duration up to 4m30s, 12GB VRAM",
        "vram_required": 12,
        "hf_repo": "lglg666/SongGeneration-base-full",
        "size_gb": 11.3,
        "priority": 3,
    },
    "songgeneration_large": {
        "name": "SongGeneration - Large (4m30s)",
        "description": "Best quality, 22GB VRAM, max 4m30s",
        "vram_required": 22,
        "hf_repo": "lglg666/SongGeneration-large",
        "size_gb": 20.5,
        "priority": 4,
    },
}

# Download state tracking
download_states: Dict[str, dict] = {}
download_threads: Dict[str, threading.Thread] = {}
download_processes: Dict[str, subprocess.Popen] = {}  # Store subprocess for killing
download_cancel_flags: Dict[str, threading.Event] = {}  # Use Event for thread-safe cancellation

# Cache for expected file sizes from HuggingFace (model_id -> {filename: size_bytes})
expected_file_sizes_cache: Dict[str, dict] = {}

def get_expected_file_sizes(model_id: str) -> dict:
    """Get expected file sizes for a model, using cache or fetching from HuggingFace.

    Returns dict mapping filename -> size_in_bytes, or empty dict if unavailable.
    """
    global expected_file_sizes_cache

    # Check cache first
    if model_id in expected_file_sizes_cache:
        return expected_file_sizes_cache[model_id]

    # Get from HuggingFace API
    if model_id not in MODEL_REGISTRY:
        return {}

    hf_repo = MODEL_REGISTRY[model_id]["hf_repo"]
    file_sizes = get_repo_file_sizes_from_hf(hf_repo)

    if file_sizes:
        expected_file_sizes_cache[model_id] = file_sizes

    return file_sizes

def get_model_status(model_id: str) -> str:
    """Get the status of a model: ready, downloading, not_downloaded

    A model is only 'ready' if:
    1. The folder exists
    2. The model file (model.pt) exists
    3. The model file size matches expected (verified cache or HuggingFace)
    """
    # Check if currently downloading
    if model_id in download_states and download_states[model_id].get("status") == "downloading":
        return "downloading"

    folder_path = BASE_DIR / model_id
    if not folder_path.exists():
        return "not_downloaded"

    if model_id not in MODEL_REGISTRY:
        return "not_downloaded"

    # Check model.pt specifically (the main model file)
    model_file = folder_path / "model.pt"
    if model_file.exists():
        try:
            actual_size = model_file.stat().st_size

            # FAST PATH: If model was previously verified, just check size matches
            if is_model_verified(model_id):
                verified_size = get_verified_model_size(model_id)
                if verified_size > 0 and actual_size == verified_size:
                    return "ready"
                # Size changed - need to re-verify

            # SLOW PATH: Check against HuggingFace API (only on first check or size change)
            expected_sizes = get_expected_file_sizes(model_id)
            expected_size = expected_sizes.get('model.pt', 0)

            if expected_size > 0:
                # We have expected size from HuggingFace - allow 0.1% tolerance
                size_diff_pct = abs(actual_size - expected_size) / expected_size * 100
                if size_diff_pct < 0.1:
                    # Mark as verified so future checks are instant
                    mark_model_verified(model_id, actual_size)
                    return "ready"
                else:
                    actual_gb = actual_size / (1024 * 1024 * 1024)
                    expected_gb = expected_size / (1024 * 1024 * 1024)
                    pct = (actual_size / expected_size) * 100 if expected_size else 0
                    print(f"[MODEL] {model_id}/model.pt incomplete: {actual_size:,} / {expected_size:,} bytes ({pct:.1f}%) - {actual_gb:.2f}GB / {expected_gb:.2f}GB")
            else:
                # No expected size from HuggingFace - fallback to registry size
                expected_size_gb = MODEL_REGISTRY[model_id]["size_gb"]
                min_required_bytes = int(expected_size_gb * 0.95 * 1000 * 1000 * 1000)
                if actual_size >= min_required_bytes:
                    mark_model_verified(model_id, actual_size)
                    return "ready"
                else:
                    actual_gb = actual_size / (1000 * 1000 * 1000)
                    print(f"[MODEL] {model_id}/model.pt incomplete (fallback check): {actual_gb:.2f}GB / {expected_size_gb}GB expected")

        except (OSError, IOError) as e:
            print(f"[MODEL] Error checking {model_id}/model.pt: {e}")

    # Also check for safetensors format
    model_file_st = folder_path / "model.safetensors"
    if model_file_st.exists():
        try:
            actual_size = model_file_st.stat().st_size

            # Fast path for verified models
            if is_model_verified(model_id):
                verified_size = get_verified_model_size(model_id)
                if verified_size > 0 and actual_size == verified_size:
                    return "ready"

            expected_sizes = get_expected_file_sizes(model_id)
            expected_size = expected_sizes.get('model.safetensors', 0)

            if expected_size > 0:
                size_diff_pct = abs(actual_size - expected_size) / expected_size * 100
                if size_diff_pct < 0.1:
                    mark_model_verified(model_id, actual_size)
                    return "ready"
            else:
                expected_size_gb = MODEL_REGISTRY[model_id]["size_gb"]
                min_required_bytes = int(expected_size_gb * 0.95 * 1000 * 1000 * 1000)
                if actual_size >= min_required_bytes:
                    mark_model_verified(model_id, actual_size)
                    return "ready"
        except (OSError, IOError):
            pass

    return "not_downloaded"

def get_download_progress(model_id: str) -> dict:
    """Get download progress for a model"""
    if model_id not in download_states:
        return {"status": "not_started", "progress": 0}
    return download_states[model_id]

def is_model_ready_quick(model_id: str) -> bool:
    """Quick check if model is likely ready (no HuggingFace API call).

    This is a fast check for queue operations. It checks:
    1. If model was previously verified (instant)
    2. If model folder and file exist with reasonable size (fast)

    Returns True if model appears ready, False otherwise.
    """
    if model_id not in MODEL_REGISTRY:
        return False

    # Fast path: already verified
    if is_model_verified(model_id):
        folder_path = BASE_DIR / model_id
        model_file = folder_path / "model.pt"
        if model_file.exists():
            return True

    # Check if folder and model file exist with reasonable size
    folder_path = BASE_DIR / model_id
    if not folder_path.exists():
        return False

    # Check for model.pt or model.safetensors
    model_file = folder_path / "model.pt"
    model_file_st = folder_path / "model.safetensors"

    expected_size_gb = MODEL_REGISTRY[model_id]["size_gb"]
    min_size_bytes = int(expected_size_gb * 0.9 * 1000 * 1000 * 1000)  # 90% of expected

    if model_file.exists():
        try:
            if model_file.stat().st_size >= min_size_bytes:
                return True
        except:
            pass

    if model_file_st.exists():
        try:
            if model_file_st.stat().st_size >= min_size_bytes:
                return True
        except:
            pass

    return False

def get_model_status_quick(model_id: str) -> str:
    """Fast model status check (no HuggingFace API calls).

    Returns: 'ready', 'downloading', or 'not_downloaded'

    This is optimized for UI display - it uses the verified cache and
    local file size checks only. For exact verification, use get_model_status().
    """
    # Check if currently downloading
    if model_id in download_states and download_states[model_id].get("status") == "downloading":
        return "downloading"

    if model_id not in MODEL_REGISTRY:
        return "not_downloaded"

    folder_path = BASE_DIR / model_id
    if not folder_path.exists():
        return "not_downloaded"

    # Fast path: already verified
    if is_model_verified(model_id):
        model_file = folder_path / "model.pt"
        model_file_st = folder_path / "model.safetensors"
        if model_file.exists() or model_file_st.exists():
            return "ready"

    # Check for model file with reasonable size (90% of expected)
    expected_size_gb = MODEL_REGISTRY[model_id]["size_gb"]
    min_size_bytes = int(expected_size_gb * 0.9 * 1000 * 1000 * 1000)

    model_file = folder_path / "model.pt"
    if model_file.exists():
        try:
            if model_file.stat().st_size >= min_size_bytes:
                return "ready"
        except:
            pass

    model_file_st = folder_path / "model.safetensors"
    if model_file_st.exists():
        try:
            if model_file_st.stat().st_size >= min_size_bytes:
                return "ready"
        except:
            pass

    return "not_downloaded"

def get_repo_file_sizes_from_hf(repo_id: str) -> dict:
    """Get exact file sizes in bytes from HuggingFace API.

    Returns dict mapping filename -> size_in_bytes
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)

        file_sizes = {}
        total_bytes = 0
        for sibling in repo_info.siblings:
            if hasattr(sibling, 'size') and sibling.size:
                # rfilename is the relative filename
                filename = sibling.rfilename
                size_bytes = sibling.size
                file_sizes[filename] = size_bytes
                total_bytes += size_bytes

        file_sizes['__total__'] = total_bytes

        # Log the main model file size
        model_size = file_sizes.get('model.pt', 0)
        if model_size:
            print(f"[DOWNLOAD] HuggingFace API: {repo_id}/model.pt = {model_size:,} bytes ({model_size / (1024**3):.2f} GB)")
        print(f"[DOWNLOAD] HuggingFace API: {repo_id} total = {total_bytes:,} bytes ({total_bytes / (1024**3):.2f} GB)")

        return file_sizes
    except Exception as e:
        print(f"[DOWNLOAD] Could not get file sizes from HF API: {e}")
        return {}

def get_repo_size_from_hf(repo_id: str) -> float:
    """Get total repository size from HuggingFace API in GB"""
    file_sizes = get_repo_file_sizes_from_hf(repo_id)
    total_bytes = file_sizes.get('__total__', 0)
    return total_bytes / (1024 * 1024 * 1024) if total_bytes else 0

def get_directory_size(path: Path) -> int:
    """Get total size of all files in a directory in bytes"""
    if not path.exists():
        return 0
    total = 0
    try:
        for f in path.rglob('*'):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except (OSError, IOError):
                    pass
    except Exception:
        pass
    return total

def run_model_download(model_id: str):
    """Run model download in background thread with robust progress tracking.

    Uses file size polling instead of parsing stdout (much more reliable).
    Stores subprocess reference for proper cancellation.
    """
    global download_states, download_cancel_flags, download_processes

    if model_id not in MODEL_REGISTRY:
        download_states[model_id] = {"status": "error", "error": "Unknown model"}
        return

    model_info = MODEL_REGISTRY[model_id]
    hf_repo = model_info["hf_repo"]
    local_dir = BASE_DIR / model_id

    print(f"[DOWNLOAD] Starting download of {model_id} from {hf_repo}")

    # Get exact file sizes from HuggingFace API (and cache them for later verification)
    expected_sizes = get_expected_file_sizes(model_id)
    total_bytes = expected_sizes.get('__total__', 0)
    actual_size_gb = total_bytes / (1024 * 1024 * 1024) if total_bytes else 0

    if actual_size_gb <= 0:
        actual_size_gb = model_info["size_gb"]  # Fallback to registry value
        print(f"[DOWNLOAD] Using fallback size: {actual_size_gb} GB")
    else:
        model_pt_size = expected_sizes.get('model.pt', 0)
        if model_pt_size:
            print(f"[DOWNLOAD] Expected model.pt: {model_pt_size:,} bytes ({model_pt_size / (1024**3):.2f} GB)")

    # Initialize download state
    download_states[model_id] = {
        "status": "downloading",
        "progress": 0,
        "downloaded_gb": 0,
        "total_gb": round(actual_size_gb, 2),
        "speed_mbps": 0,
        "eta_seconds": 0,
        "started_at": time.time(),
    }

    # Create cancellation event
    cancel_event = threading.Event()
    download_cancel_flags[model_id] = cancel_event

    process = None
    try:
        # Start download subprocess
        cmd = [
            sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
            "download", hf_repo,
            "--local-dir", str(local_dir),
            "--local-dir-use-symlinks", "False"
        ]

        # On Windows, use CREATE_NEW_PROCESS_GROUP for proper termination
        creationflags = 0
        if sys.platform == 'win32':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            creationflags=creationflags
        )

        # Store process reference for cancellation
        download_processes[model_id] = process

        print(f"[DOWNLOAD] Started subprocess PID {process.pid}")

        # Progress tracking via file size polling
        total_bytes = int(actual_size_gb * 1024 * 1024 * 1024)
        last_downloaded = 0
        last_time = time.time()
        poll_interval = 1.0  # Poll every second

        while process.poll() is None:
            # Check for cancellation
            if cancel_event.is_set():
                print(f"[DOWNLOAD] Cancel requested for {model_id}")
                try:
                    if sys.platform == 'win32':
                        # On Windows, use taskkill for reliable termination
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                                      capture_output=True, timeout=10)
                    else:
                        process.terminate()
                        process.wait(timeout=5)
                except Exception as e:
                    print(f"[DOWNLOAD] Error terminating process: {e}")
                    try:
                        process.kill()
                    except:
                        pass

                download_states[model_id] = {"status": "cancelled", "progress": 0}

                # Clean up partial download
                if local_dir.exists():
                    print(f"[DOWNLOAD] Cleaning up partial download at {local_dir}")
                    shutil.rmtree(local_dir, ignore_errors=True)
                return

            # Poll file sizes for progress
            current_time = time.time()
            current_downloaded = get_directory_size(local_dir)

            # Calculate progress
            if total_bytes > 0:
                progress = min(99, int((current_downloaded / total_bytes) * 100))
            else:
                progress = 0

            downloaded_gb = current_downloaded / (1024 * 1024 * 1024)

            # Calculate speed (MB/s)
            time_diff = current_time - last_time
            if time_diff > 0:
                bytes_diff = current_downloaded - last_downloaded
                speed_mbps = (bytes_diff / time_diff) / (1024 * 1024)

                # Calculate ETA
                if speed_mbps > 0:
                    remaining_bytes = total_bytes - current_downloaded
                    eta_seconds = remaining_bytes / (speed_mbps * 1024 * 1024)
                else:
                    eta_seconds = 0

                # Update state (check if still exists - cancel may have removed it)
                if model_id in download_states:
                    download_states[model_id].update({
                        "progress": progress,
                        "downloaded_gb": round(downloaded_gb, 2),
                        "speed_mbps": round(speed_mbps, 1),
                        "eta_seconds": int(eta_seconds),
                    })

            last_downloaded = current_downloaded
            last_time = current_time

            # Wait before next poll
            time.sleep(poll_interval)

        # Process finished - FIRST check if it was cancelled
        # This handles the race condition where taskkill kills the process
        # before we check cancel_event in the while loop
        if cancel_event.is_set():
            print(f"[DOWNLOAD] Download was cancelled for {model_id}")
            # Cancel function may have already cleaned up - just exit
            return

        # Process finished naturally - check result
        exit_code = process.returncode
        print(f"[DOWNLOAD] Process finished with exit code {exit_code}")

        # Read any remaining output for debugging
        try:
            stdout, _ = process.communicate(timeout=5)
            if stdout:
                output = stdout.decode('utf-8', errors='ignore')
                if 'error' in output.lower() or 'exception' in output.lower():
                    print(f"[DOWNLOAD] Process output: {output[-500:]}")
        except:
            pass

        # Verify the download by checking EXACT file sizes
        # Get expected sizes (should already be cached from start of download)
        expected_sizes = get_expected_file_sizes(model_id)
        expected_model_size = expected_sizes.get('model.pt', 0) or expected_sizes.get('model.safetensors', 0)

        # Check if model file exists and has correct size
        model_file = local_dir / "model.pt"
        if not model_file.exists():
            model_file = local_dir / "model.safetensors"

        download_complete = False
        actual_model_size = 0

        if model_file.exists():
            try:
                actual_model_size = model_file.stat().st_size
                if expected_model_size > 0:
                    # Exact comparison
                    download_complete = (actual_model_size == expected_model_size)
                    if download_complete:
                        print(f"[DOWNLOAD] Verified: {model_file.name} = {actual_model_size:,} bytes (exact match)")
                    else:
                        pct = (actual_model_size / expected_model_size) * 100
                        print(f"[DOWNLOAD] Size mismatch: {actual_model_size:,} / {expected_model_size:,} bytes ({pct:.1f}%)")
                else:
                    # Fallback: check if size is reasonable (at least 99% of registry size)
                    expected_size_gb = model_info["size_gb"]
                    min_bytes = int(expected_size_gb * 0.99 * 1024 * 1024 * 1024)
                    download_complete = (actual_model_size >= min_bytes)
            except (OSError, IOError) as e:
                print(f"[DOWNLOAD] Error checking model file: {e}")

        final_size = get_directory_size(local_dir)
        final_gb = final_size / (1024 * 1024 * 1024)

        if download_complete:
            print(f"[DOWNLOAD] Completed: {final_gb:.2f} GB downloaded, model file verified")

            download_states[model_id] = {
                "status": "completed",
                "progress": 100,
                "downloaded_gb": round(final_gb, 2),
                "total_gb": round(actual_size_gb, 2),
            }
            print(f"[DOWNLOAD] Successfully downloaded {model_id}")

            # Notify all clients that models changed (so they can auto-select)
            notify_models_update()

            # Update cache with verified sizes
            expected_file_sizes_cache[model_id] = expected_sizes
        else:
            # Download failed or incomplete
            actual_gb = actual_model_size / (1024 * 1024 * 1024) if actual_model_size else 0
            expected_gb = expected_model_size / (1024 * 1024 * 1024) if expected_model_size else actual_size_gb

            download_states[model_id] = {
                "status": "error",
                "error": f"Download incomplete: model.pt is {actual_model_size:,} bytes, expected {expected_model_size:,} bytes",
                "progress": download_states[model_id].get("progress", 0) if model_id in download_states else 0,
            }
            print(f"[DOWNLOAD] Failed: {model_id} model file is {actual_gb:.2f}GB, expected {expected_gb:.2f}GB")

            # Clean up incomplete download
            if local_dir.exists() and final_gb < 0.5:  # Less than 500MB
                print(f"[DOWNLOAD] Cleaning up failed download at {local_dir}")
                try:
                    shutil.rmtree(local_dir, ignore_errors=True)
                except:
                    pass

    except Exception as e:
        print(f"[DOWNLOAD] Error downloading {model_id}: {e}")
        import traceback
        traceback.print_exc()
        download_states[model_id] = {"status": "error", "error": str(e)}
    finally:
        # Cleanup
        download_cancel_flags.pop(model_id, None)
        download_processes.pop(model_id, None)
        download_threads.pop(model_id, None)

# Lock to prevent race conditions when starting downloads
download_start_lock = threading.Lock()

def start_model_download(model_id: str) -> dict:
    """Start downloading a model in the background"""
    if model_id not in MODEL_REGISTRY:
        print(f"[DOWNLOAD] Rejected: unknown model {model_id}")
        return {"error": "Unknown model"}

    # Use lock to prevent race condition between check and start
    with download_start_lock:
        current_status = get_model_status(model_id)
        if current_status == "ready":
            print(f"[DOWNLOAD] Rejected: {model_id} already downloaded")
            return {"error": "Model already downloaded"}
        if current_status == "downloading":
            print(f"[DOWNLOAD] Rejected: {model_id} is already downloading")
            return {"error": "Model is already downloading"}

        print(f"[DOWNLOAD] Starting download for {model_id}")

        # Set download state IMMEDIATELY to prevent race conditions
        # This ensures concurrent requests will see "downloading" status
        download_states[model_id] = {
            "status": "downloading",
            "progress": 0,
            "speed": None,
            "eta": None,
            "message": "Initializing download..."
        }

        # Start download thread (cancel flag will be created in run_model_download)
        thread = threading.Thread(target=run_model_download, args=(model_id,), daemon=True)
        download_threads[model_id] = thread
        thread.start()

    return {"status": "started", "model_id": model_id}

def cancel_model_download(model_id: str) -> dict:
    """Cancel an ongoing download"""
    # Check if there's an active download
    if model_id not in download_states or download_states[model_id].get("status") != "downloading":
        return {"error": "No active download for this model"}

    print(f"[DOWNLOAD] Cancel requested for {model_id}")

    # Signal the download thread to stop
    if model_id in download_cancel_flags:
        download_cancel_flags[model_id].set()

    # Kill the process directly for immediate effect
    if model_id in download_processes:
        process = download_processes[model_id]
        try:
            if sys.platform == 'win32':
                # On Windows, use taskkill to kill the entire process tree
                result = subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                                       capture_output=True, timeout=10)
                print(f"[DOWNLOAD] taskkill result: {result.returncode}")
            else:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        except Exception as e:
            print(f"[DOWNLOAD] Error killing process: {e}")
            try:
                process.kill()
            except:
                pass

    # Immediately set state to cancelled (don't wait for thread)
    download_states[model_id] = {"status": "cancelled", "progress": 0}

    # Clean up partial download immediately
    local_dir = BASE_DIR / model_id
    if local_dir.exists():
        print(f"[DOWNLOAD] Cleaning up partial download at {local_dir}")
        try:
            shutil.rmtree(local_dir, ignore_errors=True)
        except Exception as e:
            print(f"[DOWNLOAD] Cleanup error: {e}")

    # Clean up tracking state
    download_processes.pop(model_id, None)
    download_threads.pop(model_id, None)
    # Remove from download_states entirely so get_model_status returns clean state
    download_states.pop(model_id, None)

    print(f"[DOWNLOAD] Cancelled and cleaned up {model_id}")
    return {"status": "cancelled", "model_id": model_id}

def delete_model(model_id: str) -> dict:
    """Delete a downloaded model"""
    if model_id not in MODEL_REGISTRY:
        return {"error": "Unknown model"}

    status = get_model_status(model_id)
    if status == "downloading":
        return {"error": "Cannot delete model while downloading. Cancel first."}
    if status == "not_downloaded":
        return {"error": "Model is not downloaded"}

    folder_path = BASE_DIR / model_id
    try:
        shutil.rmtree(folder_path)
        print(f"[MODEL] Deleted model {model_id}")
        return {"status": "deleted", "model_id": model_id}
    except Exception as e:
        return {"error": f"Failed to delete: {e}"}

def get_recommended_model() -> Optional[str]:
    """Get the recommended model based on available (free) VRAM, not total.

    Uses cached GPU info for speed - GPU info is refreshed separately when needed.
    """
    global gpu_info
    # Use cached GPU info for speed (don't refresh every time)
    # GPU info is refreshed on startup and via /api/gpu endpoint

    if not gpu_info['available']:
        # No GPU, recommend smallest model
        return "songgeneration_base"

    # Use FREE VRAM, not total - other apps may be using GPU memory
    vram = gpu_info['gpu']['free_gb']
    print(f"[MODEL] Recommending based on {vram}GB available VRAM")

    # Find the best model that fits in available VRAM, prioritizing by quality (higher priority = better)
    suitable_models = []
    for model_id, info in MODEL_REGISTRY.items():
        if info['vram_required'] <= vram:
            suitable_models.append((model_id, info['priority']))

    if not suitable_models:
        return "songgeneration_base"  # Fallback

    # Sort by priority (higher = better) and return the best one
    suitable_models.sort(key=lambda x: x[1], reverse=True)
    return suitable_models[0][0]

# ============================================================================
# Model Server (Persistent Model in VRAM) - Helper functions
# ============================================================================

MODEL_SERVER_PORT = 42100
MODEL_SERVER_URL = f"http://127.0.0.1:{MODEL_SERVER_PORT}"
model_server_process: Optional[subprocess.Popen] = None

def is_model_server_running() -> bool:
    """Check if model server is running and responsive."""
    try:
        resp = requests.get(f"{MODEL_SERVER_URL}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False

def kill_process_on_port(port: int) -> bool:
    """Kill any process using the specified port."""
    try:
        if sys.platform == "win32":
            # Windows: use netstat to find PID, then taskkill
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        print(f"[MODEL_SERVER] Killing process {pid} on port {port}", flush=True)
                        subprocess.run(["taskkill", "/F", "/PID", pid],
                                      capture_output=True, timeout=10)
                        time.sleep(2)  # Wait for port to be released
                        return True
        else:
            # Linux/Mac: use lsof or fuser
            try:
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True, text=True, timeout=10
                )
                if result.stdout.strip():
                    pid = result.stdout.strip().split('\n')[0]
                    print(f"[MODEL_SERVER] Killing process {pid} on port {port}", flush=True)
                    subprocess.run(["kill", "-9", pid], capture_output=True, timeout=10)
                    time.sleep(2)
                    return True
            except FileNotFoundError:
                # lsof not available, try fuser
                subprocess.run(["fuser", "-k", f"{port}/tcp"],
                              capture_output=True, timeout=10)
                time.sleep(2)
                return True
    except Exception as e:
        print(f"[MODEL_SERVER] Failed to kill process on port {port}: {e}", flush=True)
    return False

def get_model_server_status() -> dict:
    """Get model server status including loaded model info."""
    try:
        resp = requests.get(f"{MODEL_SERVER_URL}/status", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            return data
    except requests.exceptions.Timeout:
        pass  # Server not responding quickly
    except requests.exceptions.ConnectionError:
        pass  # Server not running
    except Exception as e:
        print(f"[MODEL_SERVER] Status check error: {e}")
    return {"loaded": False, "running": False}

# Model status tracking - check actual model server status
def get_model_warmth(model_id: str) -> str:
    """Get model status: 'loaded' (in VRAM), 'loading', 'not_loaded'"""
    # Check model server status
    server_status = get_model_server_status()

    if server_status.get("loading"):
        return "loading"

    if server_status.get("loaded") and server_status.get("model_id") == model_id:
        # Check if currently generating
        if 'generations' in globals():
            for gen in generations.values():
                if gen.get("status") in ("processing", "pending") and gen.get("model") == model_id:
                    return "generating"
        return "loaded"

    return "not_loaded"

def get_all_models() -> List[dict]:
    """Get all models with their current status and warmth"""
    # Get model server status ONCE (avoid multiple slow HTTP calls)
    server_status = get_model_server_status()

    models = []
    for model_id, info in MODEL_REGISTRY.items():
        # Use fast status check - no HuggingFace API calls
        status = get_model_status_quick(model_id)
        # Inline warmth check using cached server_status
        if status == "ready":
            if server_status.get("loading"):
                warmth = "loading"
            elif server_status.get("loaded") and server_status.get("model_id") == model_id:
                # Check if currently generating
                is_generating = False
                if 'generations' in globals():
                    for gen in generations.values():
                        if gen.get("status") in ("processing", "pending") and gen.get("model") == model_id:
                            is_generating = True
                            break
                warmth = "generating" if is_generating else "loaded"
            else:
                warmth = "not_loaded"
        else:
            warmth = "cold"
        model_data = {
            "id": model_id,
            "name": info["name"],
            "description": info["description"],
            "vram_required": info["vram_required"],
            "size_gb": info["size_gb"],
            "status": status,
            "warmth": warmth,  # 'hot', 'warm', or 'cold'
        }

        # Add download progress if downloading
        if status == "downloading":
            progress = get_download_progress(model_id)
            model_data.update({
                "progress": progress.get("progress", 0),
                "downloaded_gb": progress.get("downloaded_gb", 0),
                "speed_mbps": progress.get("speed_mbps", 0),
                "eta_seconds": progress.get("eta_seconds", 0),
            })

        models.append(model_data)

    return models

# Legacy function for compatibility
def get_available_models() -> List[dict]:
    """Get only ready models (for backwards compatibility)"""
    return [m for m in get_all_models() if m["status"] == "ready"]

print(f"[CONFIG] Base dir: {BASE_DIR}")
print(f"[CONFIG] Output dir: {OUTPUT_DIR}")

# Cleanup stale download states on startup
# These are in-memory only and should be empty, but clear just in case
download_states.clear()
download_threads.clear()
download_processes.clear()
download_cancel_flags.clear()
print(f"[CONFIG] Cleared stale download states")

# Quick check for partial downloads (fast - no HuggingFace API calls)
for model_id in MODEL_REGISTRY.keys():
    folder_path = BASE_DIR / model_id
    if folder_path.exists():
        # Use fast status check on startup
        status = get_model_status_quick(model_id)
        if status == "not_downloaded":
            # Folder exists but model is incomplete
            print(f"[CONFIG] Found incomplete model folder: {model_id} - will need re-download")

ready_models = get_available_models()
print(f"[CONFIG] Available models: {[m['id'] for m in ready_models]}")
if not ready_models:
    recommended = get_recommended_model()
    print(f"[CONFIG] No models downloaded. Recommended: {recommended}")

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
# Model Server (Persistent Model in VRAM) - Remaining functions
# ============================================================================

async def start_model_server(preload_model: str = None) -> bool:
    """Start the model server process (async - doesn't block event loop)."""
    global model_server_process

    if is_model_server_running():
        print("[MODEL_SERVER] Already running", flush=True)
        return True

    # Kill any orphaned process on the port before starting
    # This handles cases where previous server hung/crashed but port is still bound
    kill_process_on_port(MODEL_SERVER_PORT)

    print("[MODEL_SERVER] Starting model server...", flush=True)

    # Find Python executable in virtual environment
    if sys.platform == "win32":
        python_exe = BASE_DIR / "env" / "Scripts" / "python.exe"
    else:
        python_exe = BASE_DIR / "env" / "bin" / "python"

    if not python_exe.exists():
        python_exe = sys.executable

    print(f"[MODEL_SERVER] Using Python: {python_exe}", flush=True)

    cmd = [str(python_exe), str(BASE_DIR / "model_server.py"),
           "--port", str(MODEL_SERVER_PORT), "--host", "127.0.0.1"]

    if preload_model:
        cmd.extend(["--preload", preload_model])

    # Set up environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    flow_vae_dir = BASE_DIR / "codeclm" / "tokenizer" / "Flow1dVAE"
    pathsep = os.pathsep
    env["PYTHONPATH"] = f"{BASE_DIR}{pathsep}{flow_vae_dir}{pathsep}{env.get('PYTHONPATH', '')}"

    print(f"[MODEL_SERVER] Command: {' '.join(cmd)}", flush=True)
    print(f"[MODEL_SERVER] PYTHONPATH: {env['PYTHONPATH']}", flush=True)

    try:
        # Don't capture stdout - let output print directly to console for debugging
        # This prevents pipe buffer from filling up and blocking the subprocess
        model_server_process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )

        # Wait for server to be ready (async - doesn't block event loop)
        for i in range(60):  # Wait up to 60 seconds (model loading takes time)
            await asyncio.sleep(1)  # Non-blocking sleep

            # Check if process died
            if model_server_process.poll() is not None:
                print(f"[MODEL_SERVER] Process exited with code {model_server_process.returncode}", flush=True)
                return False

            if is_model_server_running():
                print(f"[MODEL_SERVER] Server started successfully after {i+1}s", flush=True)
                return True

            if i % 10 == 9:
                print(f"[MODEL_SERVER] Still waiting... ({i+1}s)", flush=True)

        print("[MODEL_SERVER] Server failed to start in time (60s timeout)", flush=True)
        try:
            model_server_process.terminate()
        except:
            pass
        return False

    except Exception as e:
        print(f"[MODEL_SERVER] Failed to start: {e}")
        import traceback
        traceback.print_exc()
        return False

def stop_model_server():
    """Stop the model server process."""
    global model_server_process

    if model_server_process:
        model_server_process.terminate()
        try:
            model_server_process.wait(timeout=5)
        except:
            model_server_process.kill()
        model_server_process = None
        print("[MODEL_SERVER] Server stopped")

def load_model_on_server(model_id: str) -> dict:
    """Request model server to load a model."""
    try:
        resp = requests.post(f"{MODEL_SERVER_URL}/load",
                           json={"model_id": model_id}, timeout=5)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def generate_via_server(input_jsonl: str, save_dir: str, gen_type: str = "mixed") -> dict:
    """Send generation request to model server."""
    try:
        resp = requests.post(f"{MODEL_SERVER_URL}/generate",
                           json={
                               "input_jsonl": input_jsonl,
                               "save_dir": save_dir,
                               "gen_type": gen_type
                           }, timeout=1800)  # 30 min timeout for generation (some GPUs are slow)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# State
# ============================================================================

generations: dict[str, dict] = {}
running_processes: dict[str, asyncio.subprocess.Process] = {}  # Track running processes for stop functionality
generation_lock = threading.Lock()  # Prevents multiple simultaneous generations

def is_generation_active() -> bool:
    """Check if there's currently an active generation running."""
    for gen in generations.values():
        if gen.get("status") in ("pending", "processing"):
            return True
    return False

def get_active_generation_id() -> Optional[str]:
    """Get the ID of the currently active generation, if any."""
    for gen_id, gen in generations.items():
        if gen.get("status") in ("pending", "processing"):
            return gen_id
    return None

async def process_queue_item():
    """Process the next item in the queue if no generation is active.

    This function is called automatically when a generation completes.
    Must be async because it calls run_generation which is async.
    It runs in the same background thread as the generation.
    """
    global generations

    with generation_lock:
        # Double-check no generation is active
        if is_generation_active():
            print("[QUEUE-PROC] Skipping - generation already active", flush=True)
            return

        queue = load_queue()
        if not queue:
            print("[QUEUE-PROC] Queue is empty", flush=True)
            return

        # Check if ANY model is ready before processing
        all_models = get_all_models()
        ready_models = [m for m in all_models if m["status"] == "ready"]
        if not ready_models:
            print("[QUEUE-PROC] No models ready - waiting for download to complete", flush=True)
            return  # Don't pop item, just wait

        # Pop the next item
        item = queue.pop(0)
        save_queue(queue)
        print(f"[QUEUE-PROC] Auto-starting next item: {item.get('title', 'Untitled')}", flush=True)

        # Notify clients that queue changed
        notify_queue_update()

        # Validate model - auto-correct if needed
        model_id = item.get('model') or DEFAULT_MODEL
        model_status = get_model_status(model_id)
        if model_status != "ready":
            # Auto-correct to first available ready model
            original_model = model_id
            model_id = ready_models[0]["id"]
            item['model'] = model_id
            print(f"[QUEUE-PROC] Auto-corrected model: {original_model} -> {model_id}", flush=True)

        # Create generation request from queue item
        gen_id = str(uuid.uuid4())[:8]

        # Convert sections from dicts to Section objects
        sections = [Section(type=s.get('type', 'verse'), lyrics=s.get('lyrics'))
                    for s in item.get('sections', [{'type': 'verse'}])]

        try:
            request = SongRequest(
                title=item.get('title', 'Untitled'),
                sections=sections,
                gender=item.get('gender', 'female'),
                timbre=item.get('timbre', ''),
                genre=item.get('genre', ''),
                emotion=item.get('emotion', ''),
                instruments=item.get('instruments', ''),
                custom_style=item.get('custom_style'),
                bpm=item.get('bpm', 120),
                output_mode=item.get('output_mode', 'mixed'),
                auto_prompt_type=item.get('auto_prompt_type'),
                reference_audio_id=item.get('reference_audio_id'),
                model=item.get('model', 'songgeneration_base'),
                memory_mode=item.get('memory_mode', 'auto'),
                cfg_coef=item.get('cfg_coef', 1.5),
                temperature=item.get('temperature', 0.8),
                top_k=item.get('top_k', 50),
                top_p=item.get('top_p', 0.0),
                extend_stride=item.get('extend_stride', 5),
            )
        except Exception as e:
            print(f"[QUEUE-PROC] Error creating request from queue item: {e}", flush=True)
            return

        # Handle reference audio
        reference_path = None
        if request.reference_audio_id:
            ref_files = list(UPLOADS_DIR.glob(f"{request.reference_audio_id}_*"))
            if ref_files:
                reference_path = str(ref_files[0])

        # Register the generation (still inside lock)
        generations[gen_id] = {
            "id": gen_id,
            "title": request.title,
            "model": request.model,
            "status": "pending",
            "progress": 0,
            "message": "Starting from queue...",
            "output_file": None,
            "output_files": [],
            "created_at": datetime.now().isoformat(),
            "completed_at": None
        }

    # Run generation outside the lock
    # Since this function is now async, we can directly await run_generation
    print(f"[QUEUE-PROC] Starting generation: {gen_id}", flush=True)
    try:
        await run_generation(gen_id, request, reference_path)
    except Exception as e:
        print(f"[QUEUE-PROC] Error running generation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        generations[gen_id]["status"] = "failed"
        generations[gen_id]["message"] = str(e)

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

        # Get audio duration if not in metadata
        duration = metadata.get("duration")
        if duration is None and audio_files:
            duration = get_audio_duration(audio_files[0])
            if duration is not None:
                metadata["duration"] = duration
                # Update metadata file on disk
                try:
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[LIBRARY] Warning: Could not save duration for {gen_id}: {e}")

        generations[gen_id] = {
            "id": gen_id,
            "status": "completed",
            "progress": 100,
            "message": "Complete",
            "title": metadata.get("title", "Untitled"),
            "model": metadata.get("model", "unknown"),
            "created_at": metadata.get("created_at", file_mtime),
            "completed_at": metadata.get("completed_at", file_mtime),
            "duration": duration,
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
# Server-Sent Events (SSE) for Real-time Updates
# ============================================================================

# SSE clients - each client gets their own queue for events
sse_clients: List[queue_module.Queue] = []
sse_lock = threading.Lock()

def broadcast_event(event_type: str, data: dict):
    """Broadcast an event to all connected SSE clients."""
    event_data = json.dumps({"type": event_type, **data})
    message = f"event: {event_type}\ndata: {event_data}\n\n"

    with sse_lock:
        dead_clients = []
        for client_queue in sse_clients:
            try:
                client_queue.put_nowait(message)
            except queue_module.Full:
                dead_clients.append(client_queue)
        # Remove dead clients
        for dead in dead_clients:
            sse_clients.remove(dead)

def notify_queue_update():
    """Notify all clients that queue changed."""
    broadcast_event("queue", {"queue": load_queue()})

def notify_generation_update(gen_id: str, gen_data: dict):
    """Notify all clients about generation status change."""
    broadcast_event("generation", {"id": gen_id, "generation": gen_data})

def notify_library_update():
    """Notify all clients that library changed."""
    # Send list of generation IDs and their statuses (not full data to keep message small)
    summary = [{"id": g["id"], "status": g.get("status"), "progress": g.get("progress", 0)}
               for g in generations.values()]
    broadcast_event("library", {"generations": summary})

def notify_models_update():
    """Notify all clients that model status changed (download complete, etc)."""
    all_models = get_all_models()
    ready_models = [m for m in all_models if m["status"] == "ready"]
    broadcast_event("models", {
        "models": all_models,
        "ready_models": ready_models,
        "has_ready_model": len(ready_models) > 0
    })

# ============================================================================
# Background Queue Processor
# ============================================================================

queue_processor_running = False
queue_processor_task = None

async def background_queue_processor():
    """Background task that continuously monitors and processes the queue.

    This runs as a persistent background task, checking for queue items
    and processing them when no generation is active.
    """
    global queue_processor_running
    queue_processor_running = True
    print("[QUEUE-WORKER] Background queue processor started", flush=True)

    while queue_processor_running:
        try:
            # Check if there's work to do
            if not is_generation_active():
                queue = load_queue()
                if queue:
                    print(f"[QUEUE-WORKER] Found {len(queue)} item(s), processing next...", flush=True)
                    await process_queue_item()

            # Sleep briefly before checking again
            await asyncio.sleep(2.0)

        except asyncio.CancelledError:
            print("[QUEUE-WORKER] Cancelled, shutting down...", flush=True)
            break
        except Exception as e:
            print(f"[QUEUE-WORKER] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Continue running even after errors
            await asyncio.sleep(5.0)

    print("[QUEUE-WORKER] Background queue processor stopped", flush=True)

# ============================================================================
# FastAPI App with Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app):
    """Application lifespan handler - runs on startup and shutdown."""
    global queue_processor_task

    # Startup: Start the background queue processor
    print("[STARTUP] Starting background queue processor...", flush=True)
    queue_processor_task = asyncio.create_task(background_queue_processor())

    queue = load_queue()
    if queue:
        print(f"[STARTUP] {len(queue)} item(s) in queue, will be processed automatically", flush=True)
    else:
        print("[STARTUP] Queue is empty", flush=True)
    sys.stdout.flush()

    yield  # App runs here

    # Shutdown: Stop background processor
    global queue_processor_running
    queue_processor_running = False
    if queue_processor_task:
        queue_processor_task.cancel()
        try:
            await queue_processor_task
        except asyncio.CancelledError:
            pass
    print("[SHUTDOWN] Server shutting down...", flush=True)

app = FastAPI(title="SongGeneration Studio", version="1.0.0", lifespan=lifespan)

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

# Configuration: Use model server for persistent VRAM loading
USE_MODEL_SERVER = True  # Set to False to use old subprocess method

async def run_generation(gen_id: str, request: SongRequest, reference_path: Optional[str]):
    """Run the actual SongGeneration inference."""
    global generations

    try:
        print(f"[GEN {gen_id}] Starting generation...")
        generations[gen_id]["status"] = "processing"
        generations[gen_id]["started_at"] = datetime.now().isoformat()  # Track actual start time
        generations[gen_id]["message"] = "Initializing..."
        generations[gen_id]["progress"] = 0

        # Notify model status changed (now in use)
        notify_models_update()

        # Calculate estimated time from history
        model_id = request.model or DEFAULT_MODEL
        num_sections = len(request.sections) if request.sections else 5
        timing_stats = get_timing_stats()
        estimated_seconds = 180  # Default: 3 minutes
        if timing_stats.get("has_history") and model_id in timing_stats.get("models", {}):
            model_timing = timing_stats["models"][model_id]
            # Try to get estimate by section count first, then fall back to average
            by_sections = model_timing.get("by_sections", {})
            if str(num_sections) in by_sections:
                estimated_seconds = by_sections[str(num_sections)]
            else:
                estimated_seconds = model_timing.get("avg_time", 180)
        generations[gen_id]["estimated_seconds"] = estimated_seconds
        print(f"[GEN {gen_id}] Estimated time: {estimated_seconds}s (based on timing history)")

        # Notify clients of status change
        notify_generation_update(gen_id, generations[gen_id])
        notify_library_update()

        # Validate model
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

        generations[gen_id]["message"] = "Loading Model..."
        generations[gen_id]["progress"] = 10
        notify_generation_update(gen_id, generations[gen_id])

        # =====================================================================
        # MODEL SERVER PATH - Persistent model in VRAM for fast subsequent generations
        # =====================================================================
        if USE_MODEL_SERVER:
            print(f"[GEN {gen_id}] Using model server for persistent VRAM...")

            # Ensure model server is running
            if not is_model_server_running():
                generations[gen_id]["message"] = "Starting model server..."
                notify_generation_update(gen_id, generations[gen_id])
                if not await start_model_server():
                    raise Exception("Failed to start model server")

            # Check if correct model is loaded
            server_status = get_model_server_status()
            if not server_status.get("loaded") or server_status.get("model_id") != model_id:
                generations[gen_id]["message"] = "Loading Model..."
                generations[gen_id]["progress"] = 15
                notify_generation_update(gen_id, generations[gen_id])

                # Request model load
                load_result = load_model_on_server(model_id)
                if "error" in load_result:
                    raise Exception(f"Failed to load model: {load_result['error']}")

                # Wait for model to load (up to 300 seconds / 5 minutes)
                for i in range(300):
                    await asyncio.sleep(1)
                    server_status = get_model_server_status()
                    if server_status.get("loaded") and server_status.get("model_id") == model_id:
                        print(f"[GEN {gen_id}] Model loaded in VRAM")
                        break
                    if server_status.get("error"):
                        raise Exception(f"Model load failed: {server_status['error']}")
                    # Update progress during loading
                    generations[gen_id]["progress"] = min(30, 15 + i // 4)
                    notify_generation_update(gen_id, generations[gen_id])
                else:
                    raise Exception("Model load timeout")
            else:
                print(f"[GEN {gen_id}] Model already loaded in VRAM - skipping load!")

            # Generate via model server
            generations[gen_id]["message"] = "Generating music..."
            generations[gen_id]["progress"] = 35
            generations[gen_id]["stage"] = "generating"
            notify_generation_update(gen_id, generations[gen_id])
            notify_models_update()

            gen_type = request.output_mode or "mixed"
            start_time = time.time()

            # Run generation in thread pool to not block
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                generate_via_server,
                str(input_file),
                str(output_subdir),
                gen_type
            )

            if "error" in result:
                raise Exception(f"Generation failed: {result['error']}")

            gen_time = time.time() - start_time
            print(f"[GEN {gen_id}] Model server generation completed in {gen_time:.1f}s")

            # Find output files
            audios_dir = output_subdir / "audios"
            output_files = list(audios_dir.glob("*.flac")) if audios_dir.exists() else []
            if not output_files:
                output_files = list(output_subdir.glob("*.flac"))

            if not output_files:
                raise Exception("No output files generated")

            # Get audio duration
            audio_duration = None
            if output_files:
                audio_duration = get_audio_duration(output_files[0])

            # Mark as complete - skip to success handling below
            generations[gen_id]["status"] = "completed"
            generations[gen_id]["progress"] = 100
            generations[gen_id]["message"] = "Song generated successfully!"
            generations[gen_id]["output_files"] = [str(f) for f in output_files]
            generations[gen_id]["output_file"] = str(output_files[0])
            generations[gen_id]["completed_at"] = datetime.now().isoformat()
            generations[gen_id]["duration"] = audio_duration

            # Notify model status changed (no longer generating)
            notify_models_update()

            # Calculate and save timing
            generation_time_seconds = int(gen_time)
            total_lyrics_length = sum(len(s.lyrics or '') for s in request.sections)
            num_sections = len(request.sections)
            has_lyrics = total_lyrics_length > 0

            # Save metadata
            try:
                metadata_path = output_subdir / "metadata.json"
                metadata = {
                    "id": gen_id,
                    "title": request.title,
                    "model": model_id,
                    "created_at": generations[gen_id].get("created_at", datetime.now().isoformat()),
                    "completed_at": generations[gen_id]["completed_at"],
                    "generation_time_seconds": generation_time_seconds,
                    "duration": audio_duration,
                    "sections": [s.model_dump() for s in request.sections],
                    "genre": request.genre,
                    "gender": request.gender,
                    "timbre": request.timbre,
                    "output_mode": request.output_mode,
                    "num_sections": num_sections,
                    "has_lyrics": has_lyrics,
                    "total_lyrics_length": total_lyrics_length,
                    "used_model_server": True,
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                generations[gen_id]["metadata"] = metadata
                save_timing_record(metadata)
            except Exception as meta_err:
                print(f"[GEN {gen_id}] Warning: Could not save metadata: {meta_err}")

            input_file.unlink(missing_ok=True)
            notify_generation_update(gen_id, generations[gen_id])
            notify_library_update()
            return  # Done with model server path

        # =====================================================================
        # SUBPROCESS PATH - Original method (fallback)
        # =====================================================================
        print(f"[GEN {gen_id}] Using subprocess method...")

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
        env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output for real-time logging
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

        generations[gen_id]["message"] = "Initializing model..."
        generations[gen_id]["progress"] = 0
        generations[gen_id]["stage"] = "init"  # Track current stage

        all_stderr = []
        # =====================================================================
        # PURE TIME-BASED PROGRESS MONITORING
        # Progress = elapsed_time / estimated_time * 100
        # Phase detection via simple keyword matching (no tqdm parsing)
        # =====================================================================
        stopped = False
        last_progress_update = -1
        current_stage = "init"
        start_time = time.time()
        estimated_seconds = generations[gen_id].get("estimated_seconds", 180)

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
                # Read stderr with short timeout for responsive progress
                chunk = await asyncio.wait_for(process.stderr.read(4096), timeout=0.5)
                if not chunk:
                    break

                chunk_str = chunk.decode('utf-8', errors='ignore')

                # Process lines for logging and phase detection
                for line_str in chunk_str.split('\n'):
                    line_str = line_str.strip()
                    if not line_str:
                        continue

                    # Check tqdm lines for phase detection BEFORE skipping them
                    is_tqdm = '%|' in line_str or 'it/s]' in line_str

                    if is_tqdm:
                        # Phase detection from tqdm progress bars
                        if current_stage == "init":
                            # Detect generation starting (tqdm shows /7000 or /6000 for LM steps)
                            if '/7000' in line_str or '/6000' in line_str or '/5000' in line_str:
                                current_stage = "generating"
                                generations[gen_id]["message"] = "Generating music..."
                                generations[gen_id]["stage"] = "generating"
                                print(f"[GEN {gen_id}] Entered generation stage (detected tqdm)")
                                notify_generation_update(gen_id, generations[gen_id])

                        if current_stage == "generating":
                            # Detect diffusion stage (tqdm shows /50 for diffusion steps)
                            if '/50]' in line_str or '/50 ' in line_str:
                                current_stage = "finalizing"
                                generations[gen_id]["message"] = "Finalizing audio..."
                                generations[gen_id]["stage"] = "finalizing"
                                print(f"[GEN {gen_id}] Entered finalization stage (detected tqdm)")
                                notify_generation_update(gen_id, generations[gen_id])
                        continue  # Don't log tqdm lines

                    # Log non-tqdm lines
                    log_line = line_str[:200] + '...' if len(line_str) > 200 else line_str
                    all_stderr.append(log_line)
                    print(f"[GEN {gen_id}] {log_line}")

                    # Phase detection from regular log lines
                    line_lower = line_str.lower()
                    if current_stage == "init":
                        # Detect model loaded, generation about to start
                        if 'use generate' in line_lower:
                            current_stage = "generating"
                            generations[gen_id]["message"] = "Generating music..."
                            generations[gen_id]["stage"] = "generating"
                            print(f"[GEN {gen_id}] Entered generation stage (detected 'use generate')")
                            notify_generation_update(gen_id, generations[gen_id])

                    if current_stage != "finalizing":
                        # Detect completion (lm cost appears after diffusion)
                        if 'lm cost' in line_lower and 'diffusion cost' in line_lower:
                            current_stage = "finalizing"
                            generations[gen_id]["message"] = "Saving audio..."
                            generations[gen_id]["stage"] = "finalizing"
                            print(f"[GEN {gen_id}] Entered finalization stage (detected timing log)")
                            notify_generation_update(gen_id, generations[gen_id])

            except asyncio.TimeoutError:
                # Timeout - check if process ended
                if process.returncode is not None:
                    break

            # Calculate time-based progress
            elapsed = time.time() - start_time
            if estimated_seconds > 0:
                progress = min(95, (elapsed / estimated_seconds) * 100)  # Cap at 95% until done
            else:
                progress = min(95, elapsed / 180 * 100)  # Default 3 min estimate

            # Update progress if changed significantly
            if progress > last_progress_update + 1:
                generations[gen_id]["progress"] = int(progress)
                last_progress_update = progress
                notify_generation_update(gen_id, generations[gen_id])

        # Clean up process reference
        running_processes.pop(gen_id, None)

        # If stopped (either by flag or by status), don't continue
        if stopped or generations[gen_id]["status"] == "stopped":
            generations[gen_id]["status"] = "stopped"
            generations[gen_id]["message"] = "Generation stopped by user"
            input_file.unlink(missing_ok=True)
            # Notify clients
            notify_generation_update(gen_id, generations[gen_id])
            notify_library_update()
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

        # Get audio duration
        audio_duration = None
        if output_files:
            audio_duration = get_audio_duration(output_files[0])

        generations[gen_id]["status"] = "completed"
        generations[gen_id]["progress"] = 100
        generations[gen_id]["message"] = "Song generated successfully!"
        generations[gen_id]["output_files"] = [str(f) for f in output_files]
        generations[gen_id]["output_file"] = str(output_files[0])
        generations[gen_id]["completed_at"] = datetime.now().isoformat()
        generations[gen_id]["duration"] = audio_duration

        # Notify model status changed (no longer in use)
        notify_models_update()

        # Calculate actual generation time
        generation_time_seconds = 0
        if generations[gen_id].get("started_at"):
            try:
                started = datetime.fromisoformat(generations[gen_id]["started_at"])
                generation_time_seconds = int((datetime.now() - started).total_seconds())
            except:
                pass

        # Calculate lyrics length for timing stats
        total_lyrics_length = sum(len(s.lyrics or '') for s in request.sections)
        num_sections = len(request.sections)
        has_lyrics = total_lyrics_length > 0

        # Save complete metadata for library restoration
        try:
            metadata_path = output_subdir / "metadata.json"
            metadata = {
                "id": gen_id,
                "title": request.title,
                "model": model_id,
                "created_at": generations[gen_id].get("created_at", datetime.now().isoformat()),
                "completed_at": datetime.now().isoformat(),
                "generation_time_seconds": generation_time_seconds,
                "duration": audio_duration,
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
                "num_sections": num_sections,
                "total_lyrics_length": total_lyrics_length,
                "has_lyrics": has_lyrics,
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

            # Also save to timing history for future estimates
            save_timing_record(metadata)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            generations[gen_id]["metadata"] = metadata

        except Exception as meta_err:
            print(f"[GEN {gen_id}] Warning: Could not save metadata: {meta_err}")

        input_file.unlink(missing_ok=True)

        # Notify clients of completion
        notify_generation_update(gen_id, generations[gen_id])
        notify_library_update()

    except Exception as e:
        print(f"[GEN {gen_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        generations[gen_id]["status"] = "failed"
        generations[gen_id]["message"] = str(e)

        # Notify clients of failure
        notify_generation_update(gen_id, generations[gen_id])
        notify_library_update()
        notify_models_update()  # Model no longer in use

    # Background queue processor will automatically handle the next item

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

@app.get("/api/timing-stats")
async def get_timing_statistics():
    """Get timing statistics for smart generation time estimates.

    Returns historical data about generation times grouped by model,
    with breakdowns by lyrics presence, reference audio, and section count.
    """
    return get_timing_stats()

@app.get("/api/models")
async def list_models():
    """List all models with their status (ready, downloading, not_downloaded)."""
    all_models = get_all_models()
    ready_models = [m for m in all_models if m["status"] == "ready"]
    recommended = get_recommended_model()

    return {
        "models": all_models,
        "ready_models": ready_models,
        "default": ready_models[0]["id"] if ready_models else None,
        "recommended": recommended,
        "has_ready_model": len(ready_models) > 0,
    }

@app.get("/api/models/recommend")
async def recommend_model():
    """Get the recommended model based on available VRAM."""
    recommended = get_recommended_model()
    model_info = MODEL_REGISTRY.get(recommended, {})
    return {
        "recommended": recommended,
        "name": model_info.get("name", "Unknown"),
        "description": model_info.get("description", ""),
        "vram_required": model_info.get("vram_required", 0),
        "size_gb": model_info.get("size_gb", 0),
        "gpu": gpu_info,
    }

@app.post("/api/models/{model_id}/download")
async def download_model(model_id: str):
    """Start downloading a model."""
    result = start_model_download(model_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result

@app.get("/api/models/{model_id}/download/status")
async def get_download_status(model_id: str):
    """Get the download status and progress for a model."""
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(404, "Unknown model")

    status = get_model_status(model_id)
    progress = get_download_progress(model_id)

    return {
        "model_id": model_id,
        "status": status,
        **progress
    }

@app.delete("/api/models/{model_id}/download")
async def cancel_download(model_id: str):
    """Cancel an ongoing model download."""
    result = cancel_model_download(model_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result

@app.delete("/api/models/{model_id}")
async def remove_model(model_id: str):
    """Delete a downloaded model to free up space."""
    result = delete_model(model_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result

# ============================================================================
# Model Server Endpoints (Persistent VRAM)
# ============================================================================

@app.get("/api/model-server/status")
async def model_server_status():
    """Get model server status."""
    running = is_model_server_running()
    status = get_model_server_status() if running else {"loaded": False}
    return {
        "running": running,
        **status
    }

@app.post("/api/model-server/start")
async def start_server():
    """Start the model server."""
    if is_model_server_running():
        return {"status": "already_running"}
    success = await start_model_server()
    return {"status": "started" if success else "failed"}

@app.post("/api/model-server/stop")
async def stop_server():
    """Stop the model server and free VRAM."""
    stop_model_server()
    return {"status": "stopped"}

@app.post("/api/model-server/load/{model_id}")
async def load_model_endpoint(model_id: str):
    """Load a model into VRAM via model server."""
    if not is_model_server_running():
        if not await start_model_server():
            raise HTTPException(500, "Failed to start model server")
    result = load_model_on_server(model_id)
    return result

@app.post("/api/model-server/unload")
async def unload_model_endpoint():
    """Unload model from VRAM."""
    try:
        resp = requests.post(f"{MODEL_SERVER_URL}/unload", timeout=10)
        return resp.json()
    except Exception as e:
        raise HTTPException(500, str(e))

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

    # Validate model exists BEFORE accepting request
    model_id = request.model or DEFAULT_MODEL
    model_status = get_model_status(model_id)

    # If requested model isn't ready, try to auto-select a ready model
    if model_status != "ready":
        # Get all ready models
        all_models = get_all_models()
        ready_models = [m for m in all_models if m["status"] == "ready"]

        if ready_models:
            # Auto-correct to first available ready model
            original_model = model_id
            model_id = ready_models[0]["id"]
            request.model = model_id
            print(f"[API] Auto-corrected model: {original_model} -> {model_id} (original not ready)")
        else:
            # No models ready at all
            print(f"[API] Rejected generation - no models ready (requested: {model_id})")
            raise HTTPException(400, "No models downloaded. Please download a model first before generating.")

    # Acquire lock to prevent race conditions between multiple clients
    with generation_lock:
        # Check if there's already an active generation
        active_id = get_active_generation_id()
        if active_id:
            print(f"[API] Rejected generation request - already generating: {active_id}")
            raise HTTPException(409, f"Generation already in progress: {active_id}")

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

        # Register generation INSIDE lock to prevent race
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

    # Start background task outside lock
    background_tasks.add_task(run_generation, gen_id, request, reference_path)

    return {"generation_id": gen_id}

@app.get("/api/generation/{gen_id}")
async def get_generation_status(gen_id: str):
    """Get the status of a generation."""
    if gen_id not in generations:
        raise HTTPException(404, "Generation not found")

    gen = generations[gen_id].copy()  # Return a copy with additional computed fields

    # Calculate elapsed_seconds from started_at (if processing/generating)
    if gen.get("started_at"):
        try:
            started = datetime.fromisoformat(gen["started_at"])
            elapsed = (datetime.now() - started).total_seconds()
            gen["elapsed_seconds"] = int(elapsed)
        except:
            gen["elapsed_seconds"] = 0
    else:
        gen["elapsed_seconds"] = 0

    return gen

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

    # Notify clients
    notify_generation_update(gen_id, generations[gen_id])
    notify_library_update()
    notify_models_update()  # Model no longer in use

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

    # Log the incoming file details for debugging
    print(f"[API] Cover upload for {gen_id}: filename='{file.filename}', content_type='{file.content_type}'")

    # Determine file extension from filename or content-type
    allowed_ext = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.jfif', '.heic', '.avif')
    ext = Path(file.filename).suffix.lower() if file.filename else ''

    # If no valid extension from filename, try to determine from content-type
    if not ext or ext not in allowed_ext:
        content_type_to_ext = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'image/bmp': '.bmp',
            'image/tiff': '.tiff',
            'image/heic': '.heic',
            'image/avif': '.avif',
        }
        ext = content_type_to_ext.get(file.content_type, ext)

    # Normalize some extensions
    if ext in ('.jfif', '.bmp', '.tiff', '.heic', '.avif'):
        ext = '.jpg'  # Convert to jpg for compatibility

    if not ext or ext not in ('.jpg', '.jpeg', '.png', '.gif', '.webp'):
        print(f"[API] Cover upload rejected: invalid file type ext='{ext}'")
        raise HTTPException(400, f"Invalid file type. Got extension '{ext}', content-type '{file.content_type}'. Allowed: jpg, png, gif, webp")

    if not output_subdir.exists():
        output_subdir.mkdir(parents=True, exist_ok=True)

    # Normalize extension for storage
    if ext == '.jpeg':
        ext = '.jpg'
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

    # Use timestamp as cache buster (changes every upload, forcing browser to fetch new image)
    cover_timestamp = int(time.time() * 1000)
    metadata["cover"] = cover_timestamp
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Update in-memory generation if exists
    if gen_id in generations:
        if "metadata" not in generations[gen_id]:
            generations[gen_id]["metadata"] = {}
        generations[gen_id]["metadata"]["cover"] = cover_timestamp

    print(f"[API] Uploaded cover for {gen_id}: {cover_path} ({len(content)} bytes)")
    return {"status": "uploaded", "cover": cover_timestamp}

@app.get("/api/generation/{gen_id}/cover")
async def get_cover(gen_id: str):
    """Get the album cover image for a generation.
    Returns 204 No Content if no cover exists (instead of 404 to reduce log noise).
    """
    output_subdir = OUTPUT_DIR / gen_id

    # Check if generation exists (in memory or on disk)
    if gen_id not in generations and not output_subdir.exists():
        # Return 204 for non-existent generations too (reduces 404 spam)
        return Response(status_code=204)

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
            response = FileResponse(cover_path, media_type=media_types.get(ext, 'image/jpeg'))
            # Allow caching but require revalidation (URL has timestamp cache buster)
            response.headers["Cache-Control"] = "public, max-age=31536000"
            return response

    # Return 204 No Content instead of 404 to reduce log noise
    # Frontend handles missing covers gracefully
    return Response(status_code=204)

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
    """List all generations with computed elapsed_seconds for running generations."""
    result = []
    for gen in generations.values():
        gen_copy = gen.copy()
        # Calculate elapsed_seconds for running generations
        if gen_copy.get("status") in ("processing", "generating", "pending") and gen_copy.get("started_at"):
            try:
                started = datetime.fromisoformat(gen_copy["started_at"])
                elapsed = (datetime.now() - started).total_seconds()
                gen_copy["elapsed_seconds"] = int(elapsed)
            except:
                gen_copy["elapsed_seconds"] = 0
        result.append(gen_copy)
    return result

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

# ============================================================================
# Queue API - Server-side queue storage (shared across all clients)
# ============================================================================

@app.get("/api/queue")
async def get_queue():
    """Get the current generation queue."""
    return load_queue()

@app.post("/api/queue")
async def add_to_queue(payload: dict):
    """Add an item to the generation queue."""
    # Validate model using FAST check (no HuggingFace API call)
    model_id = payload.get('model') or DEFAULT_MODEL

    if not is_model_ready_quick(model_id):
        # Model not ready - try to find another ready model (also using quick check)
        ready_model = None
        for mid in MODEL_REGISTRY.keys():
            if is_model_ready_quick(mid):
                ready_model = mid
                break

        if ready_model:
            original_model = model_id
            model_id = ready_model
            payload['model'] = model_id
            print(f"[QUEUE] Auto-corrected model: {original_model} -> {model_id}")
        else:
            # No models ready - still add to queue (will process when model is downloaded)
            print(f"[QUEUE] Added with unavailable model: {model_id} (will process when model is ready)")

    queue = load_queue()
    # Add unique ID and timestamp
    item = {
        "id": str(uuid.uuid4()),
        "added_at": datetime.now().isoformat(),
        **payload
    }
    queue.insert(0, item)  # Add at top of queue
    save_queue(queue)
    print(f"[QUEUE] Added item at top: {item.get('title', 'Untitled')}")

    # Notify all clients
    notify_queue_update()

    return {"status": "added", "item": item, "queue_length": len(queue)}

@app.delete("/api/queue/{item_id}")
async def remove_from_queue(item_id: str):
    """Remove an item from the queue by ID."""
    queue = load_queue()
    original_len = len(queue)
    queue = [item for item in queue if item.get("id") != item_id]
    if len(queue) < original_len:
        save_queue(queue)
        print(f"[QUEUE] Removed item: {item_id}")
        notify_queue_update()
        return {"status": "removed", "queue_length": len(queue)}
    raise HTTPException(404, "Item not found in queue")

@app.delete("/api/queue")
async def clear_queue():
    """Clear the entire queue."""
    save_queue([])
    print("[QUEUE] Cleared queue")
    notify_queue_update()
    return {"status": "cleared"}

@app.post("/api/queue/next")
async def pop_queue():
    """Get and remove the next item from the queue.

    This endpoint also checks if there's an active generation to prevent
    multiple clients from starting simultaneous generations.
    """
    with generation_lock:
        # Check if there's already an active generation
        active_id = get_active_generation_id()
        if active_id:
            print(f"[QUEUE] Rejected pop - generation active: {active_id}")
            raise HTTPException(409, f"Generation already in progress: {active_id}")

        queue = load_queue()
        if not queue:
            raise HTTPException(404, "Queue is empty")
        item = queue.pop(0)
        save_queue(queue)
        print(f"[QUEUE] Popped item: {item.get('title', 'Untitled')}")
        notify_queue_update()
        return {"item": item, "remaining": len(queue)}

# ============================================================================
# Server-Sent Events (SSE) Endpoint
# ============================================================================

async def sse_event_generator(client_queue: queue_module.Queue):
    """Generate SSE events for a connected client."""
    try:
        # Send initial state
        yield f"event: connected\ndata: {json.dumps({'type': 'connected'})}\n\n"

        # Send current queue
        yield f"event: queue\ndata: {json.dumps({'type': 'queue', 'queue': load_queue()})}\n\n"

        # Send current generations summary
        summary = [{"id": g["id"], "status": g.get("status"), "progress": g.get("progress", 0)}
                   for g in generations.values()]
        yield f"event: library\ndata: {json.dumps({'type': 'library', 'generations': summary})}\n\n"

        while True:
            try:
                # Wait for events with timeout
                message = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: client_queue.get(timeout=30)
                )
                yield message
            except queue_module.Empty:
                # Send heartbeat to keep connection alive
                yield f": heartbeat\n\n"
    except asyncio.CancelledError:
        pass
    except GeneratorExit:
        pass

@app.get("/api/events")
async def sse_endpoint():
    """Server-Sent Events endpoint for real-time updates."""
    client_queue = queue_module.Queue(maxsize=100)

    with sse_lock:
        sse_clients.append(client_queue)

    async def cleanup():
        with sse_lock:
            if client_queue in sse_clients:
                sse_clients.remove(client_queue)

    response = StreamingResponse(
        sse_event_generator(client_queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
    response.background = cleanup
    return response

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
