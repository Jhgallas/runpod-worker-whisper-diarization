# RunPod Whisper Diarization Worker

Serverless worker for audio transcription with speaker diarization using Whisper and pyannote.

## Features

- Automatic GPU/CPU detection
- Base64 audio input support
- Speaker diarization with timestamps

## Future improvements
- Switch to WhisperX for batched inference (wav2vec2 alignment, faster-whisper etc.)
- Configurable min_speakers and max_speakers
- VAC (Voice Activity Detection)
- Try out more diarization frameworks such as DiariZen

## Deployment

### Create new serverless endpoint
<img width="851" height="363" alt="CleanShot 2025-12-16 at 18 59 23" src="https://github.com/user-attachments/assets/0461374d-f3dd-40f9-894c-ae60551e1311" />

### Select repo
<img width="1376" height="261" alt="CleanShot 2025-12-16 at 18 59 58" src="https://github.com/user-attachments/assets/5fa6cfe4-cc9f-449a-943e-19cc426889e0" />

### Click on Next
<img width="1388" height="426" alt="CleanShot 2025-12-16 at 19 00 16" src="https://github.com/user-attachments/assets/e9bd8d85-8551-4a0b-a53d-b961bb1234cd" />

### Select GPU's
16- or 24GB is enough for this workload.
<img width="1378" height="1094" alt="CleanShot 2025-12-16 at 19 00 31" src="https://github.com/user-attachments/assets/ce54845b-ea13-49d0-b39b-eb58d1020ae6" />

### Environment variables
<img width="1374" height="351" alt="CleanShot 2025-12-16 at 19 03 26" src="https://github.com/user-attachments/assets/fe0bbf31-29e4-4521-89bb-23e4ee237310" />

You should accept these huggingface user conditions:
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/speaker-diarization-community-1

### Click on Deploy Endpoint
<img width="1371" height="66" alt="CleanShot 2025-12-16 at 19 03 36" src="https://github.com/user-attachments/assets/c5172185-bbbd-4443-a980-507577c151db" />

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
