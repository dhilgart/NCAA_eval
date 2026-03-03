#!/bin/bash
# Run the full BMAD pipeline: create-story → dev-story → code-review
# Stops immediately if any step fails.
set -e

REPO_DIR="/home/dhilg/git/NCAA_eval"
cd "$REPO_DIR"


# claude --model claude-opus-4-6 --verbose --dangerously-skip-permissions -p "/bmad-party-mode I want to perform a complete analysis of the current status of the repo. I want all agents to weigh in with the shortcomings they find. Make sure to identify code smells, poor code quality, and where the implementation does not match the requirements. Develop a report documenting all issues identified, both concrete and abstract. Separate the issues into three categories: 1) Definitely requires human PO direction, 2) Might require human judgment, and 3) Flaws so obvious that no human insight is needed to verify that it needs to be fixed. After this document has been populated, conduct a second pass through with all of the agents, getting their input on the document AND using what other agents have put into the document as inspiration as they look through the code base a second time. Then Develop an Epic, the first stories in the epic should be from category 3. Then after those are complete, there should be a story to seek human input on the issues in category 1 and 2. --yolo"

# Do not stop tasks early; the context will be automatically compacted, allowing you to continue. Continue working systematically until the task is fully complete, regardless of the remaining context budget. However, also save current progress and state to memory frequently.

# load prompt text from file instead of embedding it here
PROMPT=$(<"$REPO_DIR/scripts/prompt_party_fix.txt")

# run the full BMAD pipeline…
claude --model claude-opus-4-6 --verbose --dangerously-skip-permissions -p "$PROMPT"
