FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /

# Install Python 3.11 and essential system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install PyTorch 2.4.1 with CUDA 12.4 support
RUN uv pip install --system --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Pin numpy<2.0 for pyannote.audio compatibility
RUN uv pip install --system --no-cache-dir "numpy<2.0"

# Install application dependencies
# Pin huggingface_hub to version that supports use_auth_token for pyannote compatibility
RUN uv pip install --system --no-cache-dir \
    "huggingface_hub<0.23.0" \
    runpod \
    openai-whisper \
    pyannote.audio \
    librosa \
    soundfile

# Copy handler
COPY rp_handler.py /

# Copy .runpod directory
COPY .runpod /.runpod

CMD ["python3", "-u", "rp_handler.py"]
