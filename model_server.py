"""
Persistent Model Server for SongGeneration
Keeps the model loaded in VRAM between generations to avoid 30s reload time.
Uses FastAPI (already installed).
"""

import sys
import os
import json
import time
import threading
import traceback
from pathlib import Path

print("[MODEL_SERVER] Script starting...", flush=True)

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "codeclm" / "tokenizer" / "Flow1dVAE"))

print(f"[MODEL_SERVER] BASE_DIR: {BASE_DIR}", flush=True)
print(f"[MODEL_SERVER] Python: {sys.executable}", flush=True)

try:
    import torch
    print(f"[MODEL_SERVER] PyTorch loaded, CUDA available: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"[MODEL_SERVER] Failed to import torch: {e}", flush=True)
    sys.exit(1)

try:
    import torchaudio
    import numpy as np
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import Optional
    import uvicorn
    from omegaconf import OmegaConf
    print("[MODEL_SERVER] All imports successful", flush=True)
except Exception as e:
    print(f"[MODEL_SERVER] Import error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

app = FastAPI(title="SongGeneration Model Server")

# Global state
model_state = {
    "loaded": False,
    "loading": False,
    "model_id": None,
    "error": None,
    "load_time": None,
    "last_used": None,
    "generations_count": 0,
}

# Model components (kept in memory)
loaded_model = None
loaded_separator = None
loaded_audio_tokenizer = None
loaded_seperate_tokenizer = None
loaded_cfg = None
loaded_auto_prompt = None
model_lock = threading.Lock()

auto_prompt_type = ['Pop', 'R&B', 'Dance', 'Jazz', 'Folk', 'Rock', 'Chinese Style', 'Chinese Tradition', 'Metal', 'Reggae', 'Chinese Opera', 'Auto']


class LoadRequest(BaseModel):
    model_id: str = "songgeneration_base"
    use_flash_attn: bool = False
    low_mem: bool = True


class GenerateRequest(BaseModel):
    input_jsonl: str
    save_dir: str
    gen_type: str = "mixed"


def load_model_impl(model_id: str, use_flash_attn: bool = False, low_mem: bool = True):
    """Load model into VRAM. Called once and kept resident."""
    global loaded_model, loaded_separator, loaded_audio_tokenizer
    global loaded_seperate_tokenizer, loaded_cfg, loaded_auto_prompt, model_state

    # Import heavy modules only when needed
    from codeclm.models import builders
    from codeclm.models import CodecLM
    from third_party.demucs.models.pretrained import get_model_from_yaml

    class Separator:
        """Audio separator using Demucs"""
        def __init__(self, dm_model_path='third_party/demucs/ckpt/htdemucs.pth',
                     dm_config_path='third_party/demucs/ckpt/htdemucs.yaml', gpu_id=0):
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                self.device = torch.device(f"cuda:{gpu_id}")
            else:
                self.device = torch.device("cpu")
            model = get_model_from_yaml(dm_config_path, dm_model_path)
            model.to(self.device)
            model.eval()
            self.demucs_model = model

        def load_audio(self, f):
            a, fs = torchaudio.load(f)
            if fs != 48000:
                a = torchaudio.functional.resample(a, fs, 48000)
            if a.shape[-1] >= 48000 * 10:
                a = a[..., :48000 * 10]
            return a[:, 0:48000 * 10]

        def run(self, audio_path, output_dir='tmp', ext=".flac"):
            os.makedirs(output_dir, exist_ok=True)
            name, _ = os.path.splitext(os.path.split(audio_path)[-1])
            output_paths = []
            for stem in self.demucs_model.sources:
                output_path = os.path.join(output_dir, f"{name}_{stem}{ext}")
                if os.path.exists(output_path):
                    output_paths.append(output_path)
            if len(output_paths) == 1:
                vocal_path = output_paths[0]
            else:
                drums_path, bass_path, other_path, vocal_path = self.demucs_model.separate(
                    audio_path, output_dir, device=self.device)
                for path in [drums_path, bass_path, other_path]:
                    os.remove(path)
            full_audio = self.load_audio(audio_path)
            vocal_audio = self.load_audio(vocal_path)
            bgm_audio = full_audio - vocal_audio
            return full_audio, vocal_audio, bgm_audio

    with model_lock:
        if model_state["loading"]:
            return {"error": "Model is already loading"}
        model_state["loading"] = True
        model_state["error"] = None

    try:
        print(f"[MODEL_SERVER] Loading model: {model_id}", flush=True)
        start_time = time.time()

        ckpt_dir = BASE_DIR / model_id
        cfg_path = ckpt_dir / 'config.yaml'
        ckpt_path = ckpt_dir / 'model.pt'

        if not cfg_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(f"Model files not found at {ckpt_dir}")

        # Register OmegaConf resolvers (required for config interpolations)
        try:
            OmegaConf.register_new_resolver("eval", lambda x: eval(x))
            OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
            OmegaConf.register_new_resolver("get_fname", lambda: 'default')
            OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))
        except ValueError:
            pass  # Resolvers already registered

        # Load config
        cfg = OmegaConf.load(str(cfg_path))
        cfg.lm.use_flash_attn_2 = use_flash_attn
        cfg.mode = 'inference'
        loaded_cfg = cfg

        max_duration = cfg.max_dur

        print(f"[MODEL_SERVER] Loading audio tokenizer...", flush=True)
        loaded_audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
        loaded_audio_tokenizer = loaded_audio_tokenizer.eval().cuda()

        print(f"[MODEL_SERVER] Loading auto prompt...", flush=True)
        loaded_auto_prompt = torch.load(str(BASE_DIR / 'tools/new_prompt.pt'))

        # Load separate tokenizer if available
        if "audio_tokenizer_checkpoint_sep" in cfg.keys():
            print(f"[MODEL_SERVER] Loading separate tokenizer...", flush=True)
            loaded_seperate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg)
            loaded_seperate_tokenizer = loaded_seperate_tokenizer.eval().cuda()
        else:
            loaded_seperate_tokenizer = None

        print(f"[MODEL_SERVER] Loading LM model (this takes ~30s)...", flush=True)
        audiolm = builders.get_lm_model(cfg)
        checkpoint = torch.load(str(ckpt_path), map_location='cpu')
        audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
        audiolm.load_state_dict(audiolm_state_dict, strict=False)
        audiolm = audiolm.eval()
        audiolm = audiolm.cuda().to(torch.float16)

        print(f"[MODEL_SERVER] Creating CodecLM wrapper...", flush=True)
        loaded_model = CodecLM(
            name="persistent_model",
            lm=audiolm,
            audiotokenizer=None,
            max_duration=max_duration,
            seperate_tokenizer=loaded_seperate_tokenizer,
        )
        loaded_model.max_duration = max_duration

        load_time = time.time() - start_time

        with model_lock:
            model_state["loaded"] = True
            model_state["loading"] = False
            model_state["model_id"] = model_id
            model_state["load_time"] = load_time
            model_state["last_used"] = time.time()

        print(f"[MODEL_SERVER] Model loaded in {load_time:.1f}s", flush=True)
        return {"status": "loaded", "model_id": model_id, "load_time": load_time}

    except Exception as e:
        traceback.print_exc()
        with model_lock:
            model_state["loading"] = False
            model_state["error"] = str(e)
        return {"error": str(e)}


def generate_impl(input_jsonl: str, save_dir: str, gen_type: str = 'mixed'):
    """Generate a song using the loaded model."""
    global loaded_model, loaded_audio_tokenizer, loaded_seperate_tokenizer
    global loaded_cfg, loaded_auto_prompt, model_state

    if not model_state["loaded"]:
        return {"error": "Model not loaded"}

    try:
        print(f"[MODEL_SERVER] Starting generation...", flush=True)
        start_time = time.time()

        cfg = loaded_cfg
        max_duration = loaded_model.max_duration

        with open(input_jsonl, "r") as fp:
            lines = fp.readlines()

        new_items = []
        for line in lines:
            item = json.loads(line)
            target_wav_name = f"{save_dir}/audios/{item['idx']}.flac"

            # Handle auto_prompt_audio_type
            if "auto_prompt_audio_type" in item:
                prompt_token = loaded_auto_prompt[item["auto_prompt_audio_type"]][
                    np.random.randint(0, len(loaded_auto_prompt[item["auto_prompt_audio_type"]]))]
                pmt_wav = prompt_token[:, [0], :]
                vocal_wav = prompt_token[:, [1], :]
                bgm_wav = prompt_token[:, [2], :]
                melody_is_wav = False
            else:
                pmt_wav = None
                vocal_wav = None
                bgm_wav = None
                melody_is_wav = True

            item['pmt_wav'] = pmt_wav
            item['vocal_wav'] = vocal_wav
            item['bgm_wav'] = bgm_wav
            item['melody_is_wav'] = melody_is_wav
            item["idx"] = f"{item['idx']}"
            item["wav_path"] = target_wav_name
            new_items.append(item)

        # Set generation params
        loaded_model.set_generation_params(
            duration=max_duration,
            extend_stride=5,
            temperature=0.9,
            cfg_coef=1.5,
            top_k=50,
            top_p=0.0,
            record_tokens=True,
            record_window=50
        )

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir + "/audios", exist_ok=True)
        os.makedirs(save_dir + "/jsonl", exist_ok=True)

        for item in new_items:
            lyric = item["gt_lyric"]
            descriptions = item.get("descriptions")
            target_wav_name = item["wav_path"]

            generate_inp = {
                'lyrics': [lyric.replace("  ", " ")],
                'descriptions': [descriptions],
                'melody_wavs': item['pmt_wav'],
                'vocal_wavs': item['vocal_wav'],
                'bgm_wavs': item['bgm_wav'],
                'melody_is_wav': item['melody_is_wav'],
            }

            gen_start = time.time()
            print(f"[MODEL_SERVER] Generating tokens...", flush=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    tokens = loaded_model.generate(**generate_inp, return_tokens=True)
            mid_time = time.time()

            print(f"[MODEL_SERVER] Generating audio...", flush=True)
            with torch.no_grad():
                wav_seperate = loaded_model.generate_audio(tokens, chunked=True, gen_type=gen_type)

            end_time = time.time()
            torchaudio.save(target_wav_name, wav_seperate[0].cpu().float(), cfg.sample_rate)

            print(f"[MODEL_SERVER] process{item['idx']}, lm cost {mid_time - gen_start:.1f}s, "
                  f"diffusion cost {end_time - mid_time:.1f}s", flush=True)

        # Save output jsonl
        src_jsonl_name = os.path.split(input_jsonl)[-1]
        with open(f"{save_dir}/jsonl/{src_jsonl_name}.jsonl", "w", encoding='utf-8') as fw:
            for item in new_items:
                clean_item = {k: v for k, v in item.items() if k not in ['pmt_wav', 'vocal_wav', 'bgm_wav', 'melody_is_wav']}
                fw.writelines(json.dumps(clean_item, ensure_ascii=False) + "\n")

        total_time = time.time() - start_time

        with model_lock:
            model_state["last_used"] = time.time()
            model_state["generations_count"] += 1

        print(f"[MODEL_SERVER] Generation complete in {total_time:.1f}s", flush=True)
        return {
            "status": "completed",
            "output_files": [item["wav_path"] for item in new_items],
            "generation_time": total_time
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/status")
async def status():
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        }
    return {**model_state, "gpu_memory": gpu_mem}


@app.post("/load")
async def load(req: LoadRequest):
    def do_load():
        load_model_impl(req.model_id, req.use_flash_attn, req.low_mem)
    thread = threading.Thread(target=do_load)
    thread.start()
    return {"status": "loading", "model_id": req.model_id}


@app.post("/unload")
async def unload():
    global loaded_model, loaded_separator, loaded_audio_tokenizer
    global loaded_seperate_tokenizer, loaded_cfg, loaded_auto_prompt, model_state

    with model_lock:
        loaded_model = None
        loaded_separator = None
        loaded_audio_tokenizer = None
        loaded_seperate_tokenizer = None
        loaded_cfg = None
        loaded_auto_prompt = None
        torch.cuda.empty_cache()
        model_state["loaded"] = False
        model_state["model_id"] = None

    return {"status": "unloaded"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not model_state["loaded"]:
        raise HTTPException(400, "Model not loaded")
    result = generate_impl(req.input_jsonl, req.save_dir, req.gen_type)
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=42100)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--preload', type=str, help='Model to preload on startup')
    args = parser.parse_args()

    print(f"[MODEL_SERVER] Starting on {args.host}:{args.port}", flush=True)

    if args.preload:
        print(f"[MODEL_SERVER] Preloading model: {args.preload}", flush=True)
        load_model_impl(args.preload)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
