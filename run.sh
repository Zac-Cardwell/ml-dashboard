#!/usr/bin/env bash
# Start the ML Training Dashboard on the local network.
#   ./run.sh                 -> runs in this terminal (keep it open; Ctrl+C stops it)
#   ./run.sh --background    -> runs detached; logs in data/server.log; stop with ./run.sh --stop
#   ./run.sh --stop          -> stop a background dashboard
#   ./run.sh --rebuild       -> force a frontend rebuild
#   PORT=9000 ./run.sh       -> another port (default 8000)
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8000}"
ENV_NAME="${MLTRAIN_ENV:-mltrain}"
PIDFILE="data/server.pid"
MODE="foreground"
REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --background|-b) MODE="background" ;;
    --stop) MODE="stop" ;;
    --rebuild) REBUILD=1 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

if [[ "$MODE" == "stop" ]]; then
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill -TERM "$(cat "$PIDFILE")" && rm -f "$PIDFILE" && echo "Dashboard stopped."
  else
    echo "No background dashboard is running."; rm -f "$PIDFILE"
  fi
  exit 0
fi

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "A dashboard is already running in the background (pid $(cat "$PIDFILE")). Stop it with: ./run.sh --stop"
  exit 1
fi
if ss -ltn 2>/dev/null | grep -q ":${PORT}\b"; then
  echo "Port ${PORT} is already in use by another program. Try: PORT=9000 ./run.sh"
  exit 1
fi

# activate the conda env
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# build the frontend if missing / sources changed / requested
if [[ "$REBUILD" == 1 || ! -f frontend/dist/index.html || -n "$(find frontend/src frontend/index.html -newer frontend/dist/index.html 2>/dev/null | head -1)" ]]; then
  echo "Building frontend…"
  (cd frontend && { [[ -d node_modules ]] || npm install; } && npm run build)
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
export PYTHONPATH="$PWD/backend:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MLTRAIN_PORT="$PORT"   # used for links in notifications
mkdir -p data
CMD=(python -m uvicorn mltrain.api.app:app --host 0.0.0.0 --port "$PORT" --log-level info --no-access-log)

banner() {
  echo
  echo "  Dashboard:  http://${IP:-localhost}:${PORT}   (this machine: http://localhost:${PORT})"
  echo "  API docs:   http://${IP:-localhost}:${PORT}/docs"
  echo "  Password:   in data/config.json  (also printed in the log on first start)"
  echo
}

if [[ "$MODE" == "background" ]]; then
  nohup "${CMD[@]}" >> data/server.log 2>&1 &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 30); do
    if curl -s -o /dev/null "http://127.0.0.1:${PORT}/api/auth/status"; then
      banner
      echo "  Running in the background (pid $(cat "$PIDFILE")). Logs: data/server.log   Stop: ./run.sh --stop"
      exit 0
    fi
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break
    sleep 1
  done
  echo "The dashboard failed to start; last log lines:"; tail -20 data/server.log; rm -f "$PIDFILE"; exit 1
fi

banner
echo "  Keep this terminal open — closing it or pressing Ctrl+C stops the dashboard."
echo "  (Use ./run.sh --background to run it without a terminal.)"
echo
exec "${CMD[@]}"
