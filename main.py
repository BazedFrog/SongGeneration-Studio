"""
SongGeneration Studio - Main Application
FastAPI app, routes, and entry point.
"""

import os
import sys
import uuid
import json
import asyncio
import argparse
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Local imports
from config import (
    BASE_DIR, DEFAULT_MODEL, OUTPUT_DIR, UPLOADS_DIR, STATIC_DIR,
    load_queue, save_queue, log_startup_info
)
from gpu import gpu_info, refresh_gpu_info, log_gpu_info
from schemas import Section, SongRequest, UpdateGenerationRequest
from timing import get_timing_stats
from models import (
    MODEL_REGISTRY, get_model_status, get_model_status_quick,
    get_download_progress, get_recommended_model, get_best_ready_model,
    get_available_models_sync, start_model_download, cancel_model_download,
    delete_model, cleanup_download_states, is_model_ready_quick
)
from model_server import (
    is_model_server_running_async, start_model_server, stop_model_server,
    get_model_server_status_async, load_model_on_server_async,
    unload_model_on_server
)
from sse import (
    notify_queue_update, notify_generation_update as sse_notify_gen,
    notify_library_update as sse_notify_lib, notify_models_update,
    notify_models_update_sync
)
from generation import (
    generations, generation_lock, is_generation_active, get_active_generation_id,
    restore_library, run_generation
)

# ============================================================================
# Startup Initialization
# ============================================================================

log_gpu_info()
log_startup_info()
cleanup_download_states()
restore_library()

ready_models = get_available_models_sync()
print(f"[CONFIG] Available models: {[m['id'] for m in ready_models]}")
if not ready_models:
    recommended = get_recommended_model()
    print(f"[CONFIG] No models downloaded. Recommended: {recommended}")

# ============================================================================
# Helper Wrappers for SSE Notifications
# ============================================================================

def notify_gen(gen_id: str, gen_data: dict):
    sse_notify_gen(gen_id, gen_data)

def notify_lib(gens=None):
    sse_notify_lib(gens or generations)

async def notify_models():
    await notify_models_update(get_all_models)

# ============================================================================
# Model Info Functions
# ============================================================================

async def get_all_models():
    """Get all models with their current status and warmth"""
    server_status = await get_model_server_status_async()

    models = []
    for model_id, info in MODEL_REGISTRY.items():
        status = get_model_status_quick(model_id)
        
        warmth = "cold"
        if status == "ready":
            if server_status.get("loading"):
                warmth = "loading"
            elif server_status.get("loaded") and server_status.get("model_id") == model_id:
                is_generating = any(
                    gen.get("status") in ("processing", "pending") and gen.get("model") == model_id
                    for gen in generations.values()
                )
                warmth = "generating" if is_generating else "loaded"
            else:
                warmth = "not_loaded"

        model_data = {
            "id": model_id,
            "name": info["name"],
            "description": info["description"],
            "vram_required": info["vram_required"],
            "size_gb": info["size_gb"],
            "status": status,
            "warmth": warmth,
        }

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

# ============================================================================
# Background Queue Processor
# ============================================================================

queue_processor_running = False
queue_processor_task = None

async def process_queue_item():
    """Process the next item in the queue."""
    with generation_lock:
        if is_generation_active():
            return

        queue = load_queue()
        if not queue:
            return

        all_models = await get_all_models()
        ready_models = [m for m in all_models if m["status"] == "ready"]
        if not ready_models:
            return

        item = queue.pop(0)
        save_queue(queue)
        notify_queue_update()

        model_id = item.get('model') or DEFAULT_MODEL
        if get_model_status(model_id) != "ready":
            model_id = ready_models[0]["id"]
            item['model'] = model_id

        # Use queue item's ID so frontend can match queue items with generations
        gen_id = item.get("id") or str(uuid.uuid4())[:8]

        sections = [Section(type=s.get('type', 'verse'), lyrics=s.get('lyrics'))
                    for s in item.get('sections', [{'type': 'verse'}])]

        try:
            request = SongRequest(
                title=item.get('title', 'Untitled'),
                sections=sections,
                gender=item.get('gender', 'female'),
                genre=item.get('genre', ''),
                emotion=item.get('emotion', ''),
                timbre=item.get('timbre', ''),
                instruments=item.get('instruments', ''),
                custom_style=item.get('custom_style'),
                bpm=item.get('bpm', 120),
                model=item.get('model', DEFAULT_MODEL),
                output_mode=item.get('output_mode', 'mixed'),
                reference_audio_id=item.get('reference_audio_id'),
                # Advanced generation parameters
                cfg_coef=item.get('cfg_coef', 1.5),
                temperature=item.get('temperature', 0.8),
                top_k=item.get('top_k', 50),
                top_p=item.get('top_p', 0.0),
                extend_stride=item.get('extend_stride', 5),
            )
        except Exception as e:
            print(f"[QUEUE-PROC] Error creating request: {e}")
            return

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
            "message": "Starting from queue...",
            "created_at": datetime.now().isoformat(),
        }

    try:
        await run_generation(gen_id, request, reference_path, notify_gen, notify_lib, notify_models)
    except Exception as e:
        print(f"[QUEUE-PROC] Error: {e}")
        generations[gen_id]["status"] = "failed"
        generations[gen_id]["message"] = str(e)


async def background_queue_processor():
    """Background task that processes the queue."""
    global queue_processor_running
    queue_processor_running = True

    while queue_processor_running:
        try:
            if not is_generation_active():
                queue = load_queue()
                if queue:
                    await process_queue_item()
            await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[QUEUE-WORKER] Error: {e}")
            await asyncio.sleep(5.0)

# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app):
    global queue_processor_task
    queue_processor_task = asyncio.create_task(background_queue_processor())
    yield
    global queue_processor_running
    queue_processor_running = False
    if queue_processor_task:
        queue_processor_task.cancel()
        try:
            await queue_processor_task
        except asyncio.CancelledError:
            pass

app = FastAPI(title="SongGeneration Studio", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API Routes
# ============================================================================

@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        response = FileResponse(index_path)
        response.headers["Cache-Control"] = "no-cache"
        return response
    return {"message": "SongGeneration Studio API", "status": "running"}


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "gpu": gpu_info}


@app.get("/api/gpu")
async def get_gpu_status():
    return refresh_gpu_info()


@app.get("/api/timing-stats")
async def get_timing_statistics():
    return get_timing_stats()


# Simple SSE test endpoint
@app.get("/api/test-sse")
async def test_sse(request: Request):
    """Simple SSE test - counts to 10 then stops."""
    async def generate():
        for i in range(10):
            if await request.is_disconnected():
                print(f"[TEST-SSE] Client disconnected at {i}", flush=True)
                break
            yield f"data: count {i}\n\n"
            print(f"[TEST-SSE] Sent count {i}", flush=True)
            await asyncio.sleep(1)
        yield f"data: done\n\n"
        print("[TEST-SSE] Done", flush=True)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/models")
async def list_models():
    print("[API] /api/models called", flush=True)
    try:
        all_models = await get_all_models()
        print(f"[API] Got {len(all_models)} models", flush=True)
        ready_models = [m for m in all_models if m["status"] == "ready"]

        # Get fresh GPU info and determine best model (single source of truth)
        recommended = get_recommended_model(refresh=True)
        best_ready = get_best_ready_model(refresh=False)  # GPU already refreshed above

        result = {
            "models": all_models,
            "ready_models": ready_models,
            "default": best_ready,  # Best ready model based on current VRAM
            "recommended": recommended,  # Best model to download if none ready
            "has_ready_model": len(ready_models) > 0,
        }
        print(f"[API] /api/models returning: has_ready_model={result['has_ready_model']}, default={result['default']}, count={len(all_models)}", flush=True)
        return result
    except Exception as e:
        print(f"[API] /api/models ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


@app.post("/api/models/{model_id}/download")
async def download_model_route(model_id: str):
    result = start_model_download(model_id, notify_models_update_sync)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.delete("/api/models/{model_id}/download")
async def cancel_download_route(model_id: str):
    result = cancel_model_download(model_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.delete("/api/models/{model_id}")
async def remove_model_route(model_id: str):
    result = delete_model(model_id)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/api/model-server/status")
async def model_server_status():
    running = await is_model_server_running_async()
    status = await get_model_server_status_async() if running else {"loaded": False}
    return {"running": running, **status}


@app.post("/api/model-server/start")
async def start_server():
    if await is_model_server_running_async():
        return {"status": "already_running"}
    success = await start_model_server()
    return {"status": "started" if success else "failed"}


@app.post("/api/model-server/stop")
async def stop_server():
    await asyncio.to_thread(stop_model_server)
    return {"status": "stopped"}


@app.post("/api/model-server/load/{model_id}")
async def load_model(model_id: str):
    """Load a model onto the model server."""
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(404, "Model not found")

    if get_model_status_quick(model_id) != "ready":
        raise HTTPException(400, "Model not downloaded")

    if not await is_model_server_running_async():
        success = await start_model_server()
        if not success:
            raise HTTPException(500, "Failed to start model server")

    result = await load_model_on_server_async(model_id)
    if "error" in result:
        raise HTTPException(500, result["error"])

    return result


@app.post("/api/model-server/unload")
async def unload_model():
    """Unload the currently loaded model from VRAM."""
    result = await asyncio.to_thread(unload_model_on_server)
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


@app.post("/api/upload-reference")
async def upload_reference(file: UploadFile = File(...)):
    allowed_ext = ('.wav', '.mp3', '.flac', '.ogg')
    if not file.filename.lower().endswith(allowed_ext):
        raise HTTPException(400, f"Invalid file type. Allowed: {allowed_ext}")

    file_id = str(uuid.uuid4())
    file_path = UPLOADS_DIR / f"{file_id}_{file.filename}"

    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)

    return {"id": file_id, "filename": file.filename}


@app.post("/api/upload-and-trim-reference")
async def upload_and_trim_reference(
    file: UploadFile = File(...),
    trim_start: float = Form(0.0),
    trim_duration: float = Form(10.0)
):
    """Upload audio and trim it server-side using ffmpeg to preserve quality."""
    allowed_ext = ('.wav', '.mp3', '.flac', '.ogg')
    if not file.filename.lower().endswith(allowed_ext):
        raise HTTPException(400, f"Invalid file type. Allowed: {allowed_ext}")

    file_id = str(uuid.uuid4())

    # Save original file temporarily
    original_ext = Path(file.filename).suffix.lower()
    temp_original = UPLOADS_DIR / f"temp_{file_id}{original_ext}"

    content = await file.read()
    with open(temp_original, 'wb') as f:
        f.write(content)

    try:
        # Output as WAV PCM16 48kHz - consistent format for model input
        # SongGeneration only uses first 10 seconds, so cap duration
        actual_duration = min(trim_duration, 10.0)
        trimmed_filename = f"trimmed_{Path(file.filename).stem}.wav"
        trimmed_path = UPLOADS_DIR / f"{file_id}_{trimmed_filename}"

        # Use ffmpeg to trim and convert to WAV PCM16 48kHz
        # -ss before -i for fast seeking, -t for duration
        # -ar 48000 for 48kHz sample rate, -c:a pcm_s16le for 16-bit PCM
        # Preserve original channels (no -ac flag)
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(trim_start),
            '-t', str(actual_duration),
            '-i', str(temp_original),
            '-ar', '48000',
            '-c:a', 'pcm_s16le',
            str(trimmed_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            print(f"[API] ffmpeg trim failed: {result.stderr}")
            raise HTTPException(500, f"Failed to trim audio: {result.stderr}")

        return {"id": file_id, "filename": trimmed_filename}

    finally:
        # Clean up temp file
        if temp_original.exists():
            temp_original.unlink()


@app.get("/api/reference/{ref_id}")
async def get_reference_audio(ref_id: str):
    ref_files = list(UPLOADS_DIR.glob(f"{ref_id}_*"))
    if not ref_files:
        raise HTTPException(404, "Reference audio not found")
    return FileResponse(ref_files[0], media_type='audio/wav')


@app.post("/api/generate")
async def generate_song(request: SongRequest, background_tasks: BackgroundTasks):
    model_id = request.model or DEFAULT_MODEL
    model_status = get_model_status_quick(model_id)

    if model_status != "ready":
        all_models = await get_all_models()
        ready_models = [m for m in all_models if m["status"] == "ready"]

        if ready_models:
            request.model = ready_models[0]["id"]
        else:
            raise HTTPException(400, "No models downloaded.")

    with generation_lock:
        active_id = get_active_generation_id()
        if active_id:
            raise HTTPException(409, f"Generation already in progress: {active_id}")

        gen_id = str(uuid.uuid4())[:8]

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
            "message": "Queued...",
            "created_at": datetime.now().isoformat(),
        }

    background_tasks.add_task(
        run_generation, gen_id, request, reference_path,
        notify_gen, notify_lib, notify_models
    )

    return {"generation_id": gen_id}


@app.get("/api/generation/{gen_id}")
async def get_generation_status(gen_id: str):
    if gen_id not in generations:
        raise HTTPException(404, "Generation not found")

    gen = generations[gen_id].copy()
    if gen.get("started_at"):
        try:
            started = datetime.fromisoformat(gen["started_at"])
            gen["elapsed_seconds"] = int((datetime.now() - started).total_seconds())
        except:
            gen["elapsed_seconds"] = 0
    return gen


@app.post("/api/stop/{gen_id}")
async def stop_generation(gen_id: str):
    """Stop a generation. Only works for queued (pending) generations, not actively processing ones."""
    import shutil

    if gen_id not in generations:
        return {"status": "stopped", "message": "Generation not found (already stopped or deleted)"}

    gen = generations[gen_id]

    # Cannot stop a generation that's actively processing (PyTorch inference can't be interrupted)
    if gen["status"] == "processing":
        raise HTTPException(400, "Cannot stop a generation in progress. Please wait for it to complete.")

    if gen["status"] not in ("pending",):
        # Already in a terminal state
        return {"status": gen["status"], "message": f"Generation already {gen['status']}"}

    print(f"[STOP] Stopping pending generation {gen_id}", flush=True)

    # Clean up files
    output_subdir = OUTPUT_DIR / gen_id
    if output_subdir.exists():
        try:
            shutil.rmtree(output_subdir)
        except Exception as e:
            print(f"[STOP] Failed to delete output dir: {e}")

    del generations[gen_id]

    notify_lib()
    await notify_models()

    return {"status": "stopped"}


@app.delete("/api/generation/{gen_id}")
async def delete_generation(gen_id: str):
    """Delete a generation. Cannot delete actively processing generations."""
    import shutil

    if gen_id not in generations:
        raise HTTPException(404, "Generation not found")

    gen = generations[gen_id]

    # Cannot delete a generation that's actively processing
    if gen["status"] == "processing":
        raise HTTPException(400, "Cannot delete a generation in progress. Please wait for it to complete.")

    # If pending, just delete it
    if gen["status"] == "pending":
        print(f"[DELETE] Deleting pending generation {gen_id}", flush=True)

    output_subdir = OUTPUT_DIR / gen_id
    if output_subdir.exists():
        shutil.rmtree(output_subdir)

    del generations[gen_id]

    notify_lib()
    await notify_models()

    return {"status": "deleted"}


@app.put("/api/generation/{gen_id}")
async def update_generation(gen_id: str, request: UpdateGenerationRequest):
    """Update generation metadata (title, etc.)."""
    output_subdir = OUTPUT_DIR / gen_id
    metadata_path = output_subdir / "metadata.json"

    if gen_id not in generations and not output_subdir.exists():
        raise HTTPException(404, "Generation not found")

    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"[API] Error loading metadata: {e}")

    if request.title is not None:
        metadata["title"] = request.title
        if gen_id in generations:
            generations[gen_id]["title"] = request.title
            if "metadata" in generations[gen_id]:
                generations[gen_id]["metadata"]["title"] = request.title

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
    import time as time_module
    output_subdir = OUTPUT_DIR / gen_id

    if gen_id not in generations and not output_subdir.exists():
        raise HTTPException(404, "Generation not found")

    print(f"[API] Cover upload for {gen_id}: filename='{file.filename}', content_type='{file.content_type}'")

    allowed_ext = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.jfif', '.heic', '.avif')
    ext = Path(file.filename).suffix.lower() if file.filename else ''

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

    if ext in ('.jfif', '.bmp', '.tiff', '.heic', '.avif'):
        ext = '.jpg'

    if not ext or ext not in ('.jpg', '.jpeg', '.png', '.gif', '.webp'):
        print(f"[API] Cover upload rejected: invalid file type ext='{ext}'")
        raise HTTPException(400, f"Invalid file type. Got extension '{ext}', content-type '{file.content_type}'. Allowed: jpg, png, gif, webp")

    if not output_subdir.exists():
        output_subdir.mkdir(parents=True, exist_ok=True)

    if ext == '.jpeg':
        ext = '.jpg'
    cover_path = output_subdir / f"cover{ext}"

    for old_cover in output_subdir.glob("cover.*"):
        old_cover.unlink()

    content = await file.read()
    with open(cover_path, 'wb') as f:
        f.write(content)

    metadata_path = output_subdir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except:
            pass

    cover_timestamp = int(time_module.time() * 1000)
    metadata["cover"] = cover_timestamp
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if gen_id in generations:
        if "metadata" not in generations[gen_id]:
            generations[gen_id]["metadata"] = {}
        generations[gen_id]["metadata"]["cover"] = cover_timestamp

    print(f"[API] Uploaded cover for {gen_id}: {cover_path} ({len(content)} bytes)")
    return {"status": "uploaded", "cover": cover_timestamp}


@app.get("/api/generation/{gen_id}/cover")
async def get_cover(gen_id: str):
    """Get the album cover image for a generation."""
    output_subdir = OUTPUT_DIR / gen_id

    if gen_id not in generations and not output_subdir.exists():
        return Response(status_code=204)

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
            response.headers["Cache-Control"] = "public, max-age=31536000"
            return response

    return Response(status_code=204)


@app.delete("/api/generation/{gen_id}/cover")
async def delete_cover(gen_id: str):
    """Delete the album cover image for a generation."""
    output_subdir = OUTPUT_DIR / gen_id

    if gen_id not in generations and not output_subdir.exists():
        raise HTTPException(404, "Generation not found")

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

    if gen_id in generations:
        if "metadata" in generations[gen_id] and "cover" in generations[gen_id]["metadata"]:
            del generations[gen_id]["metadata"]["cover"]

    return {"status": "deleted", "message": "Cover image deleted"}


@app.get("/api/generation/{gen_id}/video")
async def export_video(gen_id: str, background_tasks: BackgroundTasks):
    """Export generation as MP4 video with waveform visualization."""
    output_subdir = OUTPUT_DIR / gen_id

    if gen_id in generations:
        gen = generations[gen_id]
        if gen["status"] != "completed":
            raise HTTPException(400, "Generation not completed")
    elif output_subdir.exists():
        metadata_path = output_subdir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                gen = {"metadata": json.load(f), "status": "completed"}
        else:
            gen = {"metadata": {}, "status": "completed"}
    else:
        raise HTTPException(404, "Generation not found")

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

    cover_path = None
    has_custom_cover = False
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        cp = output_subdir / f"cover{ext}"
        if cp.exists():
            cover_path = cp
            has_custom_cover = True
            break

    temp_dir = Path(tempfile.gettempdir()) / "songgen_videos"
    temp_dir.mkdir(exist_ok=True)

    if not cover_path:
        possible_defaults = [
            STATIC_DIR / "default.jpg",
            BASE_DIR / "web" / "static" / "default.jpg",
        ]
        for dp in possible_defaults:
            if dp.exists():
                cover_path = dp
                break

        if not cover_path:
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

    video_path = temp_dir / f"{gen_id}.mp4"
    waveform_path = temp_dir / f"{gen_id}_waveform.png"

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

    if has_custom_cover:
        waveform_y = "H-140"
        bar_y = "ih-160"
        line_y = "H-142"
        print(f"[API] Using bottom waveform layout (custom cover)")

        filter_complex = (
            f"[1:v]scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080[bg];"
            f"[bg]drawbox=x=0:y={bar_y}:w=iw:h=160:color=black@0.7:t=fill[bg2];"
            f"[bg2][2:v]overlay=20:{waveform_y}[v1];"
            f"color=c=0x0d1f17:s=1040x120:r=30,format=rgba,colorchannelmixer=aa=0.75[dark];"
            f"[v1][dark]overlay=x='20+(t/{duration})*1040':y={waveform_y}:shortest=1[v2];"
            f"color=c=white:s=4x124:r=30[line];"
            f"[v2][line]overlay=x='18+(t/{duration})*1040':y={line_y}:shortest=1[v3];"
            f"color=c=white:s=12x124:r=30,format=rgba,colorchannelmixer=aa=0.2[glow];"
            f"[v3][glow]overlay=x='12+(t/{duration})*1040':y={line_y}:shortest=1[vout]"
        )
    else:
        waveform_y = 480
        line_y = waveform_y - 2
        print(f"[API] Using centered waveform layout (no custom cover)")

        filter_complex = (
            f"[1:v]scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080[bg];"
            f"[bg]drawbox=x=0:y={waveform_y - 20}:w=iw:h=160:color=black@0.5:t=fill[bg2];"
            f"[bg2][2:v]overlay=20:{waveform_y}[v1];"
            f"color=c=0x0d1f17:s=1040x120:r=30,format=rgba,colorchannelmixer=aa=0.75[dark];"
            f"[v1][dark]overlay=x='20+(t/{duration})*1040':y={waveform_y}:shortest=1[v2];"
            f"color=c=white:s=4x124:r=30[line];"
            f"[v2][line]overlay=x='18+(t/{duration})*1040':y={line_y}:shortest=1[v3];"
            f"color=c=white:s=12x124:r=30,format=rgba,colorchannelmixer=aa=0.2[glow];"
            f"[v3][glow]overlay=x='12+(t/{duration})*1040':y={line_y}:shortest=1[vout]"
        )

    video_cmd = [
        'ffmpeg', '-y',
        '-i', str(audio_file),
        '-loop', '1', '-i', str(cover_path),
        '-loop', '1', '-i', str(waveform_path),
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

    if waveform_path.exists():
        waveform_path.unlink()

    title = gen.get("title", gen.get("metadata", {}).get("title", "song"))
    return FileResponse(
        video_path,
        media_type='video/mp4',
        filename=f"{title}.mp4"
    )


@app.get("/api/generations")
async def list_generations():
    result = []
    for gen in generations.values():
        gen_copy = gen.copy()
        if gen_copy.get("status") in ("processing", "pending") and gen_copy.get("started_at"):
            try:
                started = datetime.fromisoformat(gen_copy["started_at"])
                gen_copy["elapsed_seconds"] = int((datetime.now() - started).total_seconds())
            except:
                gen_copy["elapsed_seconds"] = 0
        result.append(gen_copy)
    return result


@app.get("/api/audio/{gen_id}/{track_idx}")
async def get_audio_track(gen_id: str, track_idx: int, format: Optional[str] = None):
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
    return FileResponse(output_path, media_type=media_types.get(output_path.suffix.lower(), "audio/wav"))


# Queue endpoints
@app.get("/api/queue")
async def get_queue():
    return load_queue()


@app.post("/api/queue")
async def add_to_queue(payload: dict):
    model_id = payload.get('model') or DEFAULT_MODEL
    if not is_model_ready_quick(model_id):
        for mid in MODEL_REGISTRY.keys():
            if is_model_ready_quick(mid):
                payload['model'] = mid
                break

    queue = load_queue()
    item = {
        "id": str(uuid.uuid4())[:8],  # Use 8-char ID (same format as generations)
        "added_at": datetime.now().isoformat(),
        **payload
    }
    queue.append(item)  # Add to end (FIFO order)
    save_queue(queue)
    notify_queue_update()
    return {"status": "added", "item": item}


@app.delete("/api/queue/{item_id}")
async def remove_from_queue(item_id: str):
    queue = load_queue()
    original_len = len(queue)
    queue = [item for item in queue if item.get("id") != item_id]
    if len(queue) < original_len:
        save_queue(queue)
        notify_queue_update()
        return {"status": "removed"}
    raise HTTPException(404, "Item not found")


@app.delete("/api/queue")
async def clear_queue():
    save_queue([])
    notify_queue_update()
    return {"status": "cleared"}


# SSE endpoint - Disabled, return 410 to stop EventSource reconnection
@app.api_route("/api/events", methods=["GET", "HEAD"])
async def sse_endpoint():
    from fastapi.responses import Response
    return Response(status_code=410, content="SSE disabled")


# Static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SongGeneration Studio API Server")
    parser.add_argument("--host", default="127.0.0.1")
    # Accept string so templating leftovers like "{{port}}" don't crash argparse
    parser.add_argument("--port", type=str, default=None)
    args = parser.parse_args()

    # Resolve port: prefer env PORT (set by Pinokio), then CLI, else default
    port_raw = os.getenv("PORT") or args.port or "7860"

    # Robust port parsing with sane fallback
    try:
        port = int(port_raw)
    except (TypeError, ValueError):
        port = 7860

    if port < 1 or port > 65535:
        print(f"[WARN] Invalid port '{port_raw}', falling back to 7860")
        port = 7860

    print()
    print("=" * 60)
    print("  SongGeneration Studio")
    print(f"  Open http://{args.host}:{port} in your browser")
    print("=" * 60)
    print()

    uvicorn.run(app, host=args.host, port=port)
