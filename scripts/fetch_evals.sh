#!/usr/bin/env bash
# Mirror only the eval summary.json files from stormtrooper's runs/ (incl. runs/verda_backup/)
# into a local directory. Read-only on the remote side.
# Usage: scripts/fetch_evals.sh [DEST]
set -euo pipefail
DEST="${1:-/tmp/claude-1000/-home-henry-Documents-PythonProjects-on-policy-subliminal-learning/e570c98f-ef13-4a60-91b3-7806837f73ab/scratchpad/evals}"
REMOTE="${REMOTE:-stormtrooper:Documents/PythonProjects/on_policy_subliminal_learning/runs/}"
mkdir -p "$DEST"
rsync -am --include='*/' --include='*/summary.json' --exclude='*' "$REMOTE" "$DEST/"
echo "fetched $(find "$DEST" -name summary.json | wc -l) summary.json files into $DEST"
