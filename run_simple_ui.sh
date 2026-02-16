#!/bin/bash

echo "=========================================="
echo "  ACE-Step Simple UI"
echo "  Streamlined interface for executives"
echo "=========================================="
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv to run Simple UI..."
    uv run acestep-simple "$@"
else
    echo "Using Python directly..."
    python -m acestep.simple_ui "$@"
fi
