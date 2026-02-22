#!/bin/bash
# Run the full BMAD pipeline: create-story → dev-story → code-review
# Stops immediately if any step fails.
set -e

REPO_DIR="/home/dhilg/git/NCAA_eval"
cd "$REPO_DIR"

echo "=== Step 1/3: Create Story (Opus 4.6) ==="
claude --model claude-opus-4-6 --yolo -p "/bmad-bmm-create-story"

echo "=== Step 2/3: Dev Story (Opus 4.6) ==="
claude --model claude-opus-4-6 --yolo -p "/bmad-bmm-dev-story"

echo "=== Step 3/3: Code Review (Sonnet 4.6) ==="
claude --model claude-sonnet-4-6 --yolo -p "/bmad-bmm-code-review"

echo "=== Pipeline complete ==="
