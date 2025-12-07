module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          // Use platform-specific path separator: semicolon for Windows, colon for Unix
          PYTHONPATH: "{{platform === 'win32' ? cwd + '/app;' + cwd + '/app/codeclm/tokenizer/Flow1dVAE' : cwd + '/app:' + cwd + '/app/codeclm/tokenizer/Flow1dVAE'}}",
          PYTHONUTF8: "1"
        },
        path: "app",
        message: "python api.py --host 127.0.0.1 --port {{port}}",
        on: [{
          event: "/(http:\/\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
