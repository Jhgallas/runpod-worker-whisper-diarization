import base64
import os
import subprocess
import tempfile
import time

import requests as http_requests
import runpod
import torch
import torchaudio

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Segment


# ── Model loading at startup (once per worker, not per job) ──────────────

def _resolve_runtime():
    if torch.cuda.is_available():
        print("CUDA detected. Using GPU execution.")
        # int8_float16: quantized weights (fast) with float16 activations (accurate)
        # Best balance for RTX 4090 — faster than float16 with negligible quality loss
        return "cuda", "large-v3-turbo", "int8_float16"
    print("No CUDA detected. Using CPU execution.")
    return "cpu", "tiny", "int8"

DEVICE, MODEL_SIZE, COMPUTE_TYPE = _resolve_runtime()

print(f"Loading faster-whisper model '{MODEL_SIZE}' on '{DEVICE}' with {COMPUTE_TYPE}...")
WHISPER_MODEL = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Whisper model loaded.")

print("Loading pyannote diarization pipeline...")
DIARIZATION_PIPELINE = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.environ["HUGGINGFACE_ACCESS_TOKEN"]
)
if DEVICE == "cuda":
    DIARIZATION_PIPELINE.to(torch.device("cuda"))
print("Diarization pipeline loaded.")


# ── Preprocessing with ffmpeg (no RAM loading) ──────────────────────────

def preprocess_audio_ffmpeg(input_path, output_path):
    """Convert any audio to 16kHz mono WAV using ffmpeg. Streams without loading into RAM."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg preprocessing failed: {result.stderr[-300:]}")


# ── Diarize text (inlined from pyannote-whisper) ────────────────────────

def diarize_text(segments, diarization_result):
    """Merge whisper segments with diarization speaker labels."""
    timestamp_texts = []
    for item in segments:
        timestamp_texts.append((Segment(item['start'], item['end']), item['text']))

    spk_text = []
    for seg, text in timestamp_texts:
        spk = diarization_result.crop(seg).argmax()
        spk_text.append((seg, spk, text))

    PUNC_SENT_END = ['.', '?', '!']
    merged = []
    pre_spk = None
    cache = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(cache) > 0:
            sentence = ''.join([c[-1] for c in cache])
            merged.append((Segment(cache[0][0].start, cache[-1][0].end), cache[0][1], sentence))
            cache = [(seg, spk, text)]
            pre_spk = spk
        elif text and len(text) > 0 and text[-1] in PUNC_SENT_END:
            cache.append((seg, spk, text))
            sentence = ''.join([c[-1] for c in cache])
            merged.append((Segment(cache[0][0].start, cache[-1][0].end), cache[0][1], sentence))
            cache = []
            pre_spk = spk
        else:
            cache.append((seg, spk, text))
            pre_spk = spk
    if len(cache) > 0:
        sentence = ''.join([c[-1] for c in cache])
        merged.append((Segment(cache[0][0].start, cache[-1][0].end), cache[0][1], sentence))
    return merged


# ── Handler (called per job) ─────────────────────────────────────────────

def handler(event):
    job_start = time.time()
    print("Job Start")

    if not event or "input" not in event:
        raise ValueError("Event must contain 'input' field.")

    request_input = event["input"]

    audio_url = request_input.get("audio_url")
    base64file = request_input.get("file")
    extension = request_input.get("ext", "wav")
    language = request_input.get("language", None)
    results_upload_url = request_input.get("results_upload_url", None)

    if not audio_url and not base64file:
        raise ValueError("Either 'audio_url' or 'file' (base64) input is required.")

    # Get audio to a temporary file
    temp_file_path = None
    if audio_url:
        print(f"Downloading audio from URL...")
        t0 = time.time()
        response = http_requests.get(audio_url, stream=True, timeout=600)
        response.raise_for_status()
        if "." in audio_url.split("?")[0].split("/")[-1]:
            extension = audio_url.split("?")[0].split("/")[-1].split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        print(f"Downloaded in {time.time()-t0:.1f}s: {temp_file_path} ({os.path.getsize(temp_file_path) / (1024*1024):.1f} MB)")
    else:
        print("Decoding base64...")
        audio_data = base64.b64decode(base64file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        print(f"Decoded file saved to: {temp_file_path}")

    # Preprocess audio for pyannote using ffmpeg (streams, no RAM loading)
    print("Preprocessing audio for diarization (ffmpeg)...")
    t0 = time.time()
    diarization_file_path = temp_file_path.replace(f".{extension}", "_diarization.wav")
    preprocess_audio_ffmpeg(temp_file_path, diarization_file_path)
    print(f"Preprocessing done in {time.time()-t0:.1f}s ({os.path.getsize(diarization_file_path) / (1024*1024):.1f} MB)")

    transcribe_kwargs = {
        "beam_size": 1,                      # greedy decoding — ~2-3x faster, negligible quality loss
        "vad_filter": True,                   # skip silence
        "condition_on_previous_text": False,  # prevents hallucination cascading on long audio
    }
    if language:
        transcribe_kwargs["language"] = language

    # Run transcription first, then diarization sequentially
    # (parallel execution causes OOM on long recordings — 46GB+ RAM needed)
    print(f"Transcribing with faster-whisper (language={language})...")
    t0 = time.time()
    segs_iter, info = WHISPER_MODEL.transcribe(temp_file_path, **transcribe_kwargs)
    segments_list = [{"start": s.start, "end": s.end, "text": s.text} for s in segs_iter]
    print(f"Transcription done in {time.time()-t0:.1f}s — {len(segments_list)} segments, detected language: {info.language} ({info.language_probability:.0%})")

    # Free whisper GPU memory before diarization
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("Running diarization pipeline...")
    t0 = time.time()
    # Preload with torchaudio to bypass torchcodec dependency
    # (pyannote 4.0 requires torchcodec for file paths, but torchcodec is incompatible with PyTorch 2.8.0)
    waveform, sample_rate = torchaudio.load(diarization_file_path)
    speaker_segments = DIARIZATION_PIPELINE({"waveform": waveform, "sample_rate": sample_rate})
    # pyannote 4.0+ returns DiarizeOutput; extract the Annotation for merging
    if hasattr(speaker_segments, 'speaker_diarization'):
        speaker_segments = speaker_segments.speaker_diarization
    print(f"Diarization done in {time.time()-t0:.1f}s")

    # Merge transcription and diarization results
    print("Merging transcription and diarization results...")
    t0 = time.time()
    merged_segments = diarize_text(segments_list, speaker_segments)
    print(f"Merging done in {time.time()-t0:.1f}s — {len(merged_segments)} merged segments")

    # Count unique speakers
    speakers = set()
    for _, spk, _ in merged_segments:
        speakers.add(spk)
    print(f"Speakers detected: {len(speakers)}")

    # Cleanup files
    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    if diarization_file_path and os.path.exists(diarization_file_path):
        os.remove(diarization_file_path)

    # Build output JSON
    merged_segments_json = []
    for segment, speaker, text in merged_segments:
        merged_segments_json.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker,
            "text": text.strip()
        })

    total_time = time.time() - job_start
    print(f"Job complete in {total_time:.1f}s ({total_time/60:.1f} min)")

    # Upload results to GCS if a signed upload URL was provided
    if results_upload_url:
        print("Uploading results to GCS...")
        try:
            import json
            results_json = json.dumps(merged_segments_json, indent=2)
            upload_resp = http_requests.put(
                results_upload_url,
                data=results_json,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            upload_resp.raise_for_status()
            print(f"Results uploaded to GCS ({len(results_json)} bytes)")
        except Exception as e:
            print(f"WARNING: Failed to upload results to GCS: {e}")

    return merged_segments_json

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
