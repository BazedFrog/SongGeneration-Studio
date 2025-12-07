module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // 1. Clone the repository
    {
      method: "shell.run",
      params: {
        message: "git clone https://github.com/tencent-ailab/SongGeneration.git app"
      }
    },
    // 2. Install PyTorch
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
    // 3. Install Python dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
          "uv pip install -r requirements_nodeps.txt --no-deps",
          "uv pip install fastapi uvicorn python-multipart aiofiles"
        ]
      }
    },
    // 4. Download runtime files (ckpt + third_party) directly to app/
    {
      method: "hf.download",
      params: {
        path: "app",
        _: ["lglg666/SongGeneration-Runtime"],
        "local-dir": "."
      }
    },
    // 5. Download base model (~24GB)
    {
      method: "hf.download",
      params: {
        path: "app",
        _: ["lglg666/SongGeneration-base"],
        "local-dir": "songgeneration_base"
      }
    },
    // 6. Download base-new model
    {
      method: "hf.download",
      params: {
        path: "app",
        _: ["lglg666/SongGeneration-base-new"],
        "local-dir": "songgeneration_base_new"
      }
    },
    // 7. Download base-full model
    {
      method: "hf.download",
      params: {
        path: "app",
        _: ["lglg666/SongGeneration-base-full"],
        "local-dir": "songgeneration_base_full"
      }
    },
    // 8. Copy API file
    {
      method: "fs.copy",
      params: {
        src: "api.py",
        dest: "app/api.py"
      }
    },
    // 9. Copy web files (all static files)
    {
      method: "fs.copy",
      params: {
        src: "web/static/index.html",
        dest: "app/web/static/index.html"
      }
    },
    {
      method: "fs.copy",
      params: {
        src: "web/static/styles.css",
        dest: "app/web/static/styles.css"
      }
    },
    {
      method: "fs.copy",
      params: {
        src: "web/static/components.js",
        dest: "app/web/static/components.js"
      }
    },
    {
      method: "fs.copy",
      params: {
        src: "web/static/app.js",
        dest: "app/web/static/app.js"
      }
    },
    {
      method: "fs.copy",
      params: {
        src: "web/static/Logo_1.png",
        dest: "app/web/static/Logo_1.png"
      }
    },
    {
      method: "fs.copy",
      params: {
        src: "web/static/default.jpg",
        dest: "app/web/static/default.jpg"
      }
    }
  ]
}
