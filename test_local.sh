#!/bin/bash

# Delete venv if exists
if [ -d "venv" ]; then
    echo "Deleting existing venv..."
    rm -rf venv
fi

# Create venv
echo "Creating venv..."
python -m venv venv

# Activate venv
echo "Activating venv..."
source venv/bin/activate

# Install uv if not already present
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install dependencies using uv
echo "Installing dependencies with uv..."
uv pip install runpod openai-whisper pyannote.audio git+https://github.com/yinruiqing/pyannote-whisper librosa soundfile

# Run the code
echo "Running rp_handler.py..."
python rp_handler.py

deactivate