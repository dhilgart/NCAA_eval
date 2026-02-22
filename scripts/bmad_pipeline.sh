#!/bin/bash
# Run the full BMAD pipeline: create-story → dev-story → code-review
# Stops immediately if any step fails.
set -e

REPO_DIR="/home/dhilg/git/NCAA_eval"
cd "$REPO_DIR"

echo "=== Step 1/4: Create Story (Opus 4.6) === $(TZ='America/New_York' date '+%Y-%m-%d %H:%M:%S %Z')"
claude --model claude-opus-4-6 --verbose --dangerously-skip-permissions -p "/bmad-bmm-create-story --yolo"

echo "=== Step 2/4: Dev Story (Opus 4.6) === $(TZ='America/New_York' date '+%Y-%m-%d %H:%M:%S %Z')"
claude --model claude-opus-4-6 --verbose --dangerously-skip-permissions -p "/bmad-bmm-dev-story --yolo"

echo "=== Step 3/4: Code Review 1 (Sonnet 4.6) === $(TZ='America/New_York' date '+%Y-%m-%d %H:%M:%S %Z')"
claude --model claude-sonnet-4-6 --verbose --dangerously-skip-permissions -p "/bmad-bmm-code-review --yolo"

echo "=== Step 4/4: Code Review 2 (Sonnet 4.6) === $(TZ='America/New_York' date '+%Y-%m-%d %H:%M:%S %Z')"
claude --model claude-sonnet-4-6 --verbose --dangerously-skip-permissions -p "/bmad-bmm-code-review --yolo"

echo "=== Pipeline complete === $(TZ='America/New_York' date '+%Y-%m-%d %H:%M:%S %Z')"
