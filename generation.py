"""
SongGeneration Studio - Generation Logic
Song generation, lyrics building, and style control.
"""

import re
import json
import asyncio
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

from config import (
    BASE_DIR, DEFAULT_MODEL, OUTPUT_DIR, UPLOADS_DIR,
    USE_MODEL_SERVER
)
from gpu import get_audio_duration
from schemas import Section, SongRequest
from timing import save_timing_record, get_timing_stats

# ============================================================================
# State
# ============================================================================

generations: Dict[str, dict] = {}
generation_lock = threading.Lock()
model_server_busy = False  # Track if model server is running a generation

# ============================================================================
# Genre to Auto-Prompt Mapping
# ============================================================================

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

# ============================================================================
# Lyrics Normalization (matches official HuggingFace Gradio app)
# ============================================================================

# Regex to filter lyrics - keeps only:
# - Word chars (\w), whitespace (\s), brackets [], hyphen -
# - CJK Chinese (u4e00-u9fff), Japanese Hiragana (u3040-u309f),
#   Japanese Katakana (u30a0-u30ff), Korean (uac00-ud7af)
# - Extended Latin (u00c0-u017f)
LYRICS_FILTER_REGEX = re.compile(
    r"[^\w\s\[\]\-\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u00c0-\u017f]"
)

# Section types that have vocals (lyrics should be cleaned)
VOCAL_SECTION_TYPES = {"verse", "chorus", "bridge", "prechorus"}


def clean_lyrics_line(line: str) -> str:
    """Clean a single line of lyrics by removing unwanted punctuation."""
    cleaned = LYRICS_FILTER_REGEX.sub("", line)
    return cleaned.strip()


# ============================================================================
# Helper Functions
# ============================================================================

def is_generation_active() -> bool:
    """Check if there's currently an active generation running."""
    # Check if any generation is in active state
    for gen in generations.values():
        if gen.get("status") in ("pending", "processing"):
            return True
    # Also check if model server is still busy (flag set when generation starts)
    if model_server_busy:
        return True
    return False


def get_active_generation_id() -> Optional[str]:
    """Get the ID of the currently active generation, if any."""
    for gen_id, gen in generations.items():
        if gen.get("status") in ("pending", "processing"):
            return gen_id
    return None


def build_lyrics_string(sections: List[Section]) -> str:
    """Build the lyrics string in SongGeneration format.

    Normalizes lyrics to match the official HuggingFace Gradio app:
    - Filters out special punctuation (keeps letters, numbers, CJK, hyphens)
    - Joins lines with '.' for vocal sections
    - Joins all sections with ' ; '
    """
    parts = []
    for section in sections:
        # Extract base type (e.g., "verse" from "verse" or "intro" from "intro-short")
        base_type = section.type.split('-')[0].lower()
        tag = f"[{section.type}]"

        if section.lyrics and base_type in VOCAL_SECTION_TYPES:
            # Vocal section with lyrics - clean each line
            lines = section.lyrics.strip().split('\n')
            cleaned_lines = []
            for line in lines:
                cleaned = clean_lyrics_line(line)
                if cleaned:
                    cleaned_lines.append(cleaned)

            if cleaned_lines:
                # Join cleaned lines with '.' as per official app
                lyrics_str = '.'.join(cleaned_lines)
                parts.append(f"{tag} {lyrics_str}")
            else:
                parts.append(tag)
        else:
            # Instrumental section or no lyrics - just the tag
            parts.append(tag)

    return " ; ".join(parts)


def build_description(request: SongRequest, exclude_genre: bool = False) -> str:
    """Build the description string for style control."""
    parts = []

    if request.gender and request.gender != "auto":
        parts.append(request.gender)

    if request.timbre:
        parts.append(request.timbre)

    if not exclude_genre and request.genre and request.genre != "Auto":
        parts.append(request.genre)

    if request.emotion:
        parts.append(request.emotion)

    if request.instruments:
        parts.append(request.instruments)

    if request.custom_style:
        parts.append(request.custom_style)

    if request.bpm:
        parts.append(f"the bpm is {request.bpm}")

    return ", ".join(parts) + "." if parts else ""


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

        audio_files = []
        for search_dir in [subdir, subdir / "audios"]:
            if search_dir.exists():
                audio_files.extend(search_dir.glob("*.flac"))
                audio_files.extend(search_dir.glob("*.wav"))
                audio_files.extend(search_dir.glob("*.mp3"))

        if not audio_files:
            continue

        audio_files = sorted(set(audio_files))

        metadata_path = subdir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"[LIBRARY] Error loading metadata for {gen_id}: {e}")

        try:
            file_mtime = datetime.fromtimestamp(audio_files[0].stat().st_mtime).isoformat()
        except:
            file_mtime = datetime.now().isoformat()

        if not metadata.get("cover"):
            for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                cover_path = subdir / f"cover{ext}"
                if cover_path.exists():
                    metadata["cover"] = f"cover{ext}"
                    try:
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"[LIBRARY] Warning: Could not update metadata for {gen_id}: {e}")
                    break

        duration = metadata.get("duration")
        if duration is None and audio_files:
            duration = get_audio_duration(audio_files[0])
            if duration is not None:
                metadata["duration"] = duration
                try:
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[LIBRARY] Warning: Could not save duration for {gen_id}: {e}")

        # Determine output_mode and track_labels from files
        output_mode = metadata.get("output_mode", "mixed")
        track_labels = []
        ordered_files = list(audio_files)

        # Check if this is a separate mode generation (has vocal and bgm files)
        file_names = [f.stem for f in audio_files]
        has_vocal = any(n.endswith('_vocal') for n in file_names)
        has_bgm = any(n.endswith('_bgm') for n in file_names)

        if has_vocal and has_bgm:
            # This is a separate mode generation - order the files
            main_file = None
            vocal_file = None
            bgm_file = None
            for f in audio_files:
                if f.stem.endswith('_vocal'):
                    vocal_file = f
                elif f.stem.endswith('_bgm'):
                    bgm_file = f
                else:
                    main_file = f
            ordered_files = [f for f in [main_file, vocal_file, bgm_file] if f]
            track_labels = ["Full Song", "Vocals", "Instrumental"]
            output_mode = "separate"
        else:
            track_labels = ["Full Song"] if output_mode == 'mixed' else ["Vocals" if output_mode == 'vocal' else "Instrumental"]

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
            "output_files": [str(f) for f in ordered_files],
            "audio_files": [f.name for f in ordered_files],
            "output_dir": str(subdir),
            "track_labels": track_labels,
            "output_mode": output_mode,
            "metadata": metadata if metadata else {
                "title": "Untitled",
                "model": "unknown",
                "created_at": file_mtime,
            }
        }
        restored += 1

    print(f"[LIBRARY] Restored {restored} generation(s)")


async def run_generation(
    gen_id: str,
    request: SongRequest,
    reference_path: Optional[str],
    notify_generation_update,
    notify_library_update,
    notify_models_update
):
    """Run the actual SongGeneration inference."""
    global generations

    # Import here to avoid circular imports
    from model_server import (
        is_model_server_running_async, start_model_server,
        get_model_server_status_async, load_model_on_server_async,
        generate_via_server_async
    )

    try:
        print(f"[GEN {gen_id}] Starting generation...")
        generations[gen_id]["status"] = "processing"
        generations[gen_id]["started_at"] = datetime.now().isoformat()
        generations[gen_id]["message"] = "Initializing..."
        generations[gen_id]["progress"] = 0

        await notify_models_update()

        model_id = request.model or DEFAULT_MODEL
        num_sections = len(request.sections) if request.sections else 5
        timing_stats = get_timing_stats()
        estimated_seconds = 180

        if timing_stats.get("has_history") and model_id in timing_stats.get("models", {}):
            model_timing = timing_stats["models"][model_id]
            by_sections = model_timing.get("by_sections", {})
            if str(num_sections) in by_sections:
                estimated_seconds = by_sections[str(num_sections)]
            else:
                estimated_seconds = model_timing.get("avg_time", 180)

        generations[gen_id]["estimated_seconds"] = estimated_seconds
        notify_generation_update(gen_id, generations[gen_id])
        notify_library_update(generations)

        model_path = BASE_DIR / model_id
        if not model_path.exists():
            raise Exception(f"Model not found: {model_id}")

        print(f"[GEN {gen_id}] Using model: {model_id}")
        generations[gen_id]["model"] = model_id

        input_file = UPLOADS_DIR / f"{gen_id}_input.jsonl"
        output_subdir = OUTPUT_DIR / gen_id
        output_subdir.mkdir(exist_ok=True)

        lyrics = build_lyrics_string(request.sections)

        input_data = {
            "idx": gen_id,
            "gt_lyric": lyrics,
        }

        description = ""

        if reference_path:
            input_data["prompt_audio_path"] = reference_path
        else:
            auto_type = "Auto"
            genre_for_auto_prompt = None

            if request.genre:
                first_genre = request.genre.split(',')[0].strip().lower()
                if first_genre in GENRE_TO_AUTO_PROMPT:
                    auto_type = GENRE_TO_AUTO_PROMPT[first_genre]
                    genre_for_auto_prompt = first_genre

            input_data["auto_prompt_audio_type"] = auto_type
            exclude_genre = genre_for_auto_prompt is not None
            description = build_description(request, exclude_genre=exclude_genre)

            if description:
                input_data["descriptions"] = description

        # Add advanced generation parameters
        input_data["cfg_coef"] = request.cfg_coef
        input_data["temperature"] = request.temperature
        input_data["top_k"] = request.top_k
        input_data["top_p"] = request.top_p
        input_data["extend_stride"] = request.extend_stride

        print(f"[GEN {gen_id}] Lyrics: {lyrics[:200]}...")
        print(f"[GEN {gen_id}] Input data: {json.dumps(input_data, indent=2)}")

        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(input_data, f, ensure_ascii=False)
            f.write('\n')

        generations[gen_id]["message"] = "Loading Model..."
        generations[gen_id]["progress"] = 10
        notify_generation_update(gen_id, generations[gen_id])

        # Model Server Path
        if USE_MODEL_SERVER:
            print(f"[GEN {gen_id}] Using model server for persistent VRAM...")

            # Check if already stopped before we even start
            if generations.get(gen_id, {}).get("status") == "stopped" or gen_id not in generations:
                print(f"[GEN {gen_id}] Generation already stopped/deleted, aborting")
                return

            if not await is_model_server_running_async():
                generations[gen_id]["message"] = "Starting model server..."
                notify_generation_update(gen_id, generations[gen_id])
                if not await start_model_server():
                    raise Exception("Failed to start model server")

            server_status = await get_model_server_status_async()
            if not server_status.get("loaded") or server_status.get("model_id") != model_id:
                generations[gen_id]["message"] = "Loading Model..."
                generations[gen_id]["progress"] = 15
                notify_generation_update(gen_id, generations[gen_id])

                load_result = await load_model_on_server_async(model_id)
                if "error" in load_result:
                    raise Exception(f"Failed to load model: {load_result['error']}")

                for i in range(300):
                    await asyncio.sleep(1)
                    # Check if stopped/deleted during model loading
                    if gen_id not in generations or generations.get(gen_id, {}).get("status") == "stopped":
                        print(f"[GEN {gen_id}] Generation stopped/deleted during model loading")
                        return
                    server_status = await get_model_server_status_async()
                    if server_status.get("loaded") and server_status.get("model_id") == model_id:
                        print(f"[GEN {gen_id}] Model loaded in VRAM")
                        break
                    if server_status.get("error"):
                        raise Exception(f"Model load failed: {server_status['error']}")
                    generations[gen_id]["progress"] = min(30, 15 + i // 4)
                    notify_generation_update(gen_id, generations[gen_id])
                else:
                    raise Exception("Model load timeout")

            # Check if stopped/deleted before starting generation
            if gen_id not in generations or generations.get(gen_id, {}).get("status") == "stopped":
                print(f"[GEN {gen_id}] Generation stopped/deleted before model server call")
                return

            generations[gen_id]["message"] = "Generating music..."
            generations[gen_id]["progress"] = 35
            generations[gen_id]["stage"] = "generating"
            notify_generation_update(gen_id, generations[gen_id])
            await notify_models_update()

            gen_type = request.output_mode or "mixed"
            start_time = time.time()

            global model_server_busy
            model_server_busy = True
            try:
                result = await generate_via_server_async(
                    str(input_file),
                    str(output_subdir),
                    gen_type
                )
            finally:
                model_server_busy = False

            # Check if cancelled, stopped, or deleted
            if result.get("status") == "cancelled" or gen_id not in generations or generations.get(gen_id, {}).get("status") == "stopped":
                print(f"[GEN {gen_id}] Generation was cancelled/stopped/deleted")
                return

            if "error" in result:
                raise Exception(f"Generation failed: {result['error']}")

            gen_time = time.time() - start_time
            print(f"[GEN {gen_id}] Model server generation completed in {gen_time:.1f}s")

            # Check again if stopped/deleted during post-processing
            if gen_id not in generations or generations.get(gen_id, {}).get("status") == "stopped":
                print(f"[GEN {gen_id}] Generation stopped/deleted during post-processing")
                return

            audios_dir = output_subdir / "audios"
            output_files = list(audios_dir.glob("*.flac")) if audios_dir.exists() else []
            if not output_files:
                output_files = list(output_subdir.glob("*.flac"))

            if not output_files:
                raise Exception("No output files generated")

            # For separate mode, order files and create labels
            gen_type = request.output_mode or "mixed"
            track_labels = []
            if gen_type == 'separate' and len(output_files) >= 3:
                # Sort files: main file first, then vocal, then bgm
                main_file = None
                vocal_file = None
                bgm_file = None
                for f in output_files:
                    name = f.stem
                    if name.endswith('_vocal'):
                        vocal_file = f
                    elif name.endswith('_bgm'):
                        bgm_file = f
                    else:
                        main_file = f
                output_files = [f for f in [main_file, vocal_file, bgm_file] if f]
                track_labels = ["Full Song", "Vocals", "Instrumental"]
            else:
                # Single file mode
                track_labels = ["Full Song"] if gen_type == 'mixed' else ["Vocals" if gen_type == 'vocal' else "Instrumental"]

            audio_duration = None
            if output_files:
                audio_duration = get_audio_duration(output_files[0])

            generations[gen_id]["status"] = "completed"
            generations[gen_id]["progress"] = 100
            generations[gen_id]["message"] = "Song generated successfully!"
            generations[gen_id]["output_files"] = [str(f) for f in output_files]
            generations[gen_id]["output_file"] = str(output_files[0])
            generations[gen_id]["track_labels"] = track_labels
            generations[gen_id]["output_mode"] = gen_type
            generations[gen_id]["completed_at"] = datetime.now().isoformat()
            generations[gen_id]["duration"] = audio_duration

            await notify_models_update()

            generation_time_seconds = int(gen_time)
            total_lyrics_length = sum(len(s.lyrics or '') for s in request.sections)
            num_sections = len(request.sections)
            has_lyrics = total_lyrics_length > 0

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
                    "emotion": request.emotion,
                    "bpm": request.bpm,
                    "instruments": request.instruments,
                    "output_mode": request.output_mode,
                    "num_sections": num_sections,
                    "has_lyrics": has_lyrics,
                    "total_lyrics_length": total_lyrics_length,
                    "used_model_server": True,
                    "reference_audio_id": request.reference_audio_id,
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                generations[gen_id]["metadata"] = metadata
                save_timing_record(metadata)
            except Exception as meta_err:
                print(f"[GEN {gen_id}] Warning: Could not save metadata: {meta_err}")

            input_file.unlink(missing_ok=True)
            notify_generation_update(gen_id, generations[gen_id])
            notify_library_update(generations)
            return

        # Subprocess fallback path would go here (omitted for brevity)
        raise Exception("Subprocess mode not implemented in refactored version")

    except Exception as e:
        print(f"[GEN {gen_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        generations[gen_id]["status"] = "failed"
        generations[gen_id]["message"] = str(e)

        notify_generation_update(gen_id, generations[gen_id])
        notify_library_update(generations)
        await notify_models_update()
