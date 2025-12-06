module.exports = {
  run: [
    // Update the launcher scripts
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    // Update the SongGeneration app
    {
      method: "shell.run",
      params: {
        path: "app",
        message: "git pull"
      }
    },
    // Re-copy custom files (api.py, web/) to app folder
    {
      method: "fs.copy",
      params: {
        src: "api.py",
        dest: "app/api.py"
      }
    },
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
    }
  ]
}
