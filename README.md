# RunPod Whisper Diarization Worker

A serverless worker for audio transcription with speaker diarization using OpenAI's Whisper model on RunPod.

## Repository

[https://github.com/navopw/runpod-worker-whisper-diarization](https://github.com/navopw/runpod-worker-whisper-diarization)

## Features

- 🎯 Automatic GPU/CPU detection
- 🔄 Base64 audio input support
- 📝 Returns transcription with segments and language detection
- ⚡ Fast deployment with Docker

## Local Development

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install runpod

# Test locally (uses test_input.json)
python3 rp_handler.py
```

## Deployment

### Docker Hub

Docker Image: `navopw/runpod-worker-whisper-diarization`

### Build and Push

```bash
# Build for linux/amd64 platform
docker build -t navopw/runpod-worker-whisper-diarization:latest --platform linux/amd64 .

# Tag with version
docker tag navopw/runpod-worker-whisper-diarization:latest navopw/runpod-worker-whisper-diarization:v1.0.0

# Push to Docker Hub
docker push navopw/runpod-worker-whisper-diarization:latest
docker push navopw/runpod-worker-whisper-diarization:v1.0.0
```

### Pull from Docker Hub

```bash
docker pull navopw/runpod-whisper-diarization:latest
```

## API Input

```json
{
  "input": {
    "file": "base64_encoded_audio",
    "ext": "mp3"
  }
}
```

## Output

```json
{
  "text": "Transcribed text...",
  "segments": [...],
  "language": "en"
}
```
