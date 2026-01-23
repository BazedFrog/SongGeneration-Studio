module.exports = {
  requires: {
    bundle: "ai"
  },
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          // Use platform-specific path separator: semicolon for Windows, colon for Unix
          PYTHONPATH: "{{platform === 'win32' ? cwd + '/app;' + cwd + '/app/codeclm/tokenizer/Flow1dVAE' : cwd + '/app:' + cwd + '/app/codeclm/tokenizer/Flow1dVAE'}}",
          PYTHONUTF8: "1",
          // Pinokio fills PORT; main.py will fallback internally if empty
          PORT: "{{port}}"
        },
        path: "app",
        // Let main.py resolve the port from env or its own default to avoid CLI parse issues
        message: "python main.py --host 127.0.0.1",
        on: [{
          event: "/http:\\/\\/[^\\s\\/]+:\\d{2,5}(?=[^\\w]|$)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"
      }
    }
  ]
}
