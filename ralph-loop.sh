#!/usr/bin/env bash
# ralph-loop.sh — Run Claude Code in a Ralph loop for the theta-alpha shift project
# Usage: ./ralph-loop.sh [max_iterations]
# Default: 20 iterations

set -euo pipefail

MAX_ITERATIONS=${1:-20}
ANCHOR=".ralph/ralph_task.md"
PROMPT="Read CLAUDE.md for project context. Then read .ralph/ralph_task.md, .ralph/progress.md, and .ralph/guardrails.md. Follow every guardrail sign. Pick up the current focus from progress.md. Work on the next unchecked item in ralph_task.md. IMPORTANT: Before writing new code, check the parent HBN project for existing utilities and style conventions to match. Run tests after changes. Before exiting, update .ralph/progress.md (what you did, what is next), check off completed items in .ralph/ralph_task.md, and append a summary to .ralph/activity.log. If you hit a repeated error, add a sign to .ralph/guardrails.md."

if [ ! -f "$ANCHOR" ]; then
    echo "ERROR: $ANCHOR not found. Are you in the project root?"
    exit 1
fi

echo "Starting Ralph loop (max $MAX_ITERATIONS iterations)"
echo "Anchor: $ANCHOR"
echo "Watch progress: tail -f .ralph/activity.log"
echo "---"

for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo ""
    echo "=== Ralph iteration $i / $MAX_ITERATIONS — $(date +%H:%M:%S) ==="

    if ! grep -q '\[ \]' "$ANCHOR"; then
        echo "All checkboxes complete. Stopping at iteration $i."
        exit 0
    fi

    REMAINING=$(grep -c '\[ \]' "$ANCHOR" || true)
    echo "Unchecked tasks remaining: $REMAINING"

    claude --print \
        -p "$PROMPT" \
        --dangerously-skip-permissions \
        2>&1 | tee -a .ralph/ralph_output_iter${i}.log

    echo "--- Iteration $i complete ---"
    sleep 2
done

echo ""
echo "=== Ralph loop finished after $MAX_ITERATIONS iterations ==="
if grep -q '\[ \]' "$ANCHOR"; then
    REMAINING=$(grep -c '\[ \]' "$ANCHOR" || true)
    echo "WARNING: $REMAINING unchecked tasks remain."
else
    echo "All checkboxes complete."
fi
