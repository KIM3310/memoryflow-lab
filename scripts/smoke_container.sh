#!/usr/bin/env bash
set -euo pipefail

image="${MEMORYFLOW_IMAGE:-memoryflow-lab:local}"
container="${MEMORYFLOW_CONTAINER:-memoryflow-lab-smoke}"
port="${MEMORYFLOW_PORT:-18080}"
base_url="http://127.0.0.1:${port}"
response_file="$(mktemp)"

cleanup() {
  exit_code=$?
  trap - EXIT
  if [[ $exit_code -ne 0 ]]; then
    docker logs "$container" 2>/dev/null || true
  fi
  docker rm --force "$container" >/dev/null 2>&1 || true
  rm -f "$response_file"
  exit "$exit_code"
}
trap cleanup EXIT

docker rm --force "$container" >/dev/null 2>&1 || true

if [[ "${MEMORYFLOW_SKIP_BUILD:-0}" != "1" ]]; then
  docker build --tag "$image" .
fi

docker run \
  --detach \
  --name "$container" \
  --init \
  --read-only \
  --tmpfs /tmp:size=16m,mode=1777 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --publish "127.0.0.1:${port}:8000" \
  "$image" >/dev/null

for attempt in $(seq 1 30); do
  if curl --silent --fail --max-time 2 "$base_url/health" >"$response_file"; then
    break
  fi
  if [[ $attempt -eq 30 ]]; then
    echo "container health endpoint did not become ready" >&2
    exit 1
  fi
  sleep 1
done

python3 - "$response_file" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload == {"status": "ok", "model": "analytical-first-order"}, payload
PY

if [[ "$(docker exec "$container" id -u)" != "10001" ]]; then
  echo "container must run as UID 10001" >&2
  exit 1
fi

if ! curl --silent --fail --max-time 5 "$base_url/" | grep --fixed-strings --quiet "MemoryFlow Lab"; then
  echo "dashboard identity check failed" >&2
  exit 1
fi

curl --silent --fail --max-time 15 \
  --header "Content-Type: application/json" \
  --data-binary @scenarios/7b-long-context-tiered.json \
  "$base_url/v1/simulations" >"$response_file"

python3 - "$response_file" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["feasible"] is True, payload
assert len(payload["steps"]) == 64, len(payload["steps"])
assert payload["bottleneck"] == "remote_transfer", payload["bottleneck"]
PY

for attempt in $(seq 1 15); do
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}}' "$container")"
  if [[ "$health" == "healthy" ]]; then
    break
  fi
  if [[ $attempt -eq 15 ]]; then
    echo "image health check did not report healthy: $health" >&2
    exit 1
  fi
  sleep 1
done

echo "container smoke passed: image=$image uid=10001 health=healthy simulation_steps=64"
