#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs/training"
ACTION="${1:-status}"

case "$ACTION" in
  status)
    shopt -s nullglob
    pid_files=("$LOG_DIR"/*.pid)
    if ((${#pid_files[@]} == 0)); then
      echo "No active training PID file found."
      [[ -e "$LOG_DIR/latest.log" ]] && echo "Latest log: $LOG_DIR/latest.log"
      exit 0
    fi
    for pid_file in "${pid_files[@]}"; do
      pid="$(cat "$pid_file")"
      run="$(basename "$pid_file" .pid)"
      if kill -0 "$pid" 2>/dev/null; then
        echo "$run: RUNNING (PID $pid)"
      else
        echo "$run: STOPPED (PID $pid)"
      fi
    done
    ;;
  tail)
    if [[ ! -e "$LOG_DIR/latest.log" ]]; then
      echo "No training log found at $LOG_DIR/latest.log" >&2
      exit 1
    fi
    tail -n 80 -F "$LOG_DIR/latest.log"
    ;;
  *)
    echo "Usage: $0 [status|tail]" >&2
    exit 2
    ;;
esac
