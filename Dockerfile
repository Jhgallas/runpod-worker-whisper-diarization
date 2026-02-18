FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

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

# Install PyTorch 2.8.0 with CUDA 12.6 support (required by pyannote.audio 4.0)
RUN uv pip install --system --no-cache-dir \
    torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Pin numpy<2.0 for pyannote.audio compatibility
RUN uv pip install --system --no-cache-dir "numpy<2.0"

# Install application dependencies
# pyannote.audio 4.0+ uses community-1 model with VBx clustering (faster + more accurate)
RUN uv pip install --system --no-cache-dir \
    "pyannote.audio>=4.0" \
    "torchcodec>=0.7" \
    runpod \
    faster-whisper \
    librosa \
    soundfile \
    requests

# Copy handler
COPY rp_handler.py /

# Copy .runpod directory
COPY .runpod /.runpod

CMD ["python3", "-u", "rp_handler.py"]
