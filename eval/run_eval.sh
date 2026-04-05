#!/bin/bash
# Run the full eval (all providers × all test cases)
set -euo pipefail

cd "$(dirname "$0")"

# Source env vars
if [ -f ../.env ]; then
  set -a
  source ../.env
  set +a
fi

echo "=== Running Thoven Pedagogy Eval ==="
echo "Providers: A (Gemma neutral), B (Gemma Socratic), D (Opus neutral)"
echo "Test cases: 23 × 11 dimensions"
echo ""

npx promptfoo eval \
  --config promptfoo-pedagogy.yaml \
  --output "results/baseline_$(date +%Y-%m-%d).json" \
  "$@"

echo ""
echo "=== Results saved to results/baseline_$(date +%Y-%m-%d).json ==="
echo "Run 'npx promptfoo view' to see the dashboard"
