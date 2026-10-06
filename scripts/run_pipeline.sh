#!/bin/sh
# Offline rebuild after a (re)crawl: cluster -> chunk -> index, then serve the dev UI.
# Output goes to the terminal; redirect it yourself if you want a log (e.g. `| tee /tmp/run.log`).
set -e
cd "$(dirname "$0")/.."
uv run python -m medrag.cli cluster
uv run python -m medrag.cli chunk
uv run python -m medrag.cli index
exec uv run uvicorn medrag.app.api:app --port 8000
