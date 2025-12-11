module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // 1. Clone the SongGeneration repo (contains codeclm/, requirements.txt, web/, etc.)
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/tencent-ailab/SongGeneration app"
        ]
      }
    },
    // 2. Download model weights from HuggingFace (ckpt/, third_party/)
    {
      method: "hf.download",
      params: {
        path: "app",
        _: ["lglg666/SongGeneration-Runtime"],
        "local-dir": "."
      }
    },
    // 3. Install PyTorch
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    // 4. Install Python dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
          "uv pip install -r requirements_nodeps.txt --no-deps",
          "uv pip install fastapi uvicorn python-multipart aiofiles",
          "uv pip install transformers==4.37.2 tokenizers==0.15.2 --force-reinstall"
        ]
      }
    },
    // 5. Sync custom Python files from root to app/
    { method: "fs.copy", params: { src: "main.py", dest: "app/main.py" } },
    { method: "fs.copy", params: { src: "generation.py", dest: "app/generation.py" } },
    { method: "fs.copy", params: { src: "model_server.py", dest: "app/model_server.py" } },
    { method: "fs.copy", params: { src: "models.py", dest: "app/models.py" } },
    { method: "fs.copy", params: { src: "gpu.py", dest: "app/gpu.py" } },
    { method: "fs.copy", params: { src: "config.py", dest: "app/config.py" } },
    { method: "fs.copy", params: { src: "schemas.py", dest: "app/schemas.py" } },
    { method: "fs.copy", params: { src: "sse.py", dest: "app/sse.py" } },
    { method: "fs.copy", params: { src: "timing.py", dest: "app/timing.py" } },
    // 6. Sync custom web files from root to app/
    { method: "fs.copy", params: { src: "web/static/index.html", dest: "app/web/static/index.html" } },
    { method: "fs.copy", params: { src: "web/static/styles.css", dest: "app/web/static/styles.css" } },
    { method: "fs.copy", params: { src: "web/static/app.js", dest: "app/web/static/app.js" } },
    { method: "fs.copy", params: { src: "web/static/components.js", dest: "app/web/static/components.js" } },
    { method: "fs.copy", params: { src: "web/static/hooks.js", dest: "app/web/static/hooks.js" } },
    { method: "fs.copy", params: { src: "web/static/api.js", dest: "app/web/static/api.js" } },
    { method: "fs.copy", params: { src: "web/static/constants.js", dest: "app/web/static/constants.js" } },
    { method: "fs.copy", params: { src: "web/static/icons.js", dest: "app/web/static/icons.js" } },
    { method: "fs.copy", params: { src: "web/static/Logo_1.png", dest: "app/web/static/Logo_1.png" } },
    { method: "fs.copy", params: { src: "web/static/default.jpg", dest: "app/web/static/default.jpg" } },
    // 7. Apply flash attention fix for Windows compatibility (disables flash attn if not available)
    { method: "fs.copy", params: { src: "patches/builders.py", dest: "app/codeclm/models/builders.py" } }
  ]
}
