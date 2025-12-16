# RunPod Whisper Diarization Worker

Serverless worker for audio transcription with speaker diarization using Whisper and pyannote.

## Features

- Automatic GPU/CPU detection
- Base64 audio input support
- Speaker diarization with timestamps

## Deployment

TODO

## API

**Input:**
```json
{
  "input": {
    "file": "base64_encoded_audio",
    "ext": "mp3"
  }
}
```

**Output:**
```json
[
  {
    "start": 0.0,
    "end": 5.2,
    "speaker": "SPEAKER_00",
    "text": "Transcribed text segment..."
  }
]
```
