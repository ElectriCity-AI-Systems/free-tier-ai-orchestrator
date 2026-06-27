#!/usr/bin/env bash
# Convenience launcher for the OpenRouter Free Orchestrator.
#   ./run.sh "your complex goal"
#   ./run.sh --list-models
#   ./run.sh --self-test
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 -m orchestrator "$@"
