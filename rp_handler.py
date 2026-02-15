import base64
import os
import tempfile
import time

import requests as http_requests
import runpod
import torch
import soundfile as sf
import librosa

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Segment


# ── Model loading at startup (once per worker, not per job) ──────────────

def _resolve_runtime():
    if torch.cuda.is_available():
        print("CUDA detected. Using GPU execution.")
        return "cuda", "large-v2", "float16"
    print("No CUDA detected. Using CPU execution.")
    return "cpu", "tiny", "int8"

DEVICE, MODEL_SIZE, COMPUTE_TYPE = _resolve_runtime()

print(f"Loading faster-whisper model '{MODEL_SIZE}' on '{DEVICE}' with {COMPUTE_TYPE}...")
WHISPER_MODEL = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Whisper model loaded.")

print("Loading pyannote diarization pipeline...")
DIARIZATION_PIPELINE = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ["HUGGINGFACE_ACCESS_TOKEN"]
)
if DEVICE == "cuda":
    DIARIZATION_PIPELINE.to(torch.device("cuda"))
print("Diarization pipeline loaded.")


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

    # Validate event structure
    if not event or "input" not in event:
        raise ValueError("Event must contain 'input' field.")

    request_input = event["input"]

    # Support two input modes: base64 or URL
    audio_url = request_input.get("audio_url")
    base64file = request_input.get("file")
    extension = request_input.get("ext", "wav")
    language = request_input.get("language", None)

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

    # Preprocess audio for pyannote: load and resample to 16kHz
    print("Preprocessing audio for diarization...")
    t0 = time.time()
    audio, original_sr = librosa.load(temp_file_path, sr=None)
    target_sr = 16000
    if original_sr != target_sr:
        print(f"Resampling audio from {original_sr}Hz to {target_sr}Hz...")
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    diarization_file_path = temp_file_path.replace(f".{extension}", "_diarization.wav")
    sf.write(diarization_file_path, audio, target_sr)
    print(f"Preprocessing done in {time.time()-t0:.1f}s")

    # Transcribe with faster-whisper
    print(f"Transcribing with faster-whisper (language={language})...")
    t0 = time.time()
    transcribe_kwargs = {"beam_size": 5, "vad_filter": True}
    if language:
        transcribe_kwargs["language"] = language
    segments_iter, info = WHISPER_MODEL.transcribe(temp_file_path, **transcribe_kwargs)

    # Convert faster-whisper segments to dict format
    segments_list = []
    for seg in segments_iter:
        segments_list.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        })
    print(f"Transcription done in {time.time()-t0:.1f}s — {len(segments_list)} segments, detected language: {info.language} ({info.language_probability:.0%})")

    # Build transcribe_res format for diarize_text
    transcribe_res = {"segments": segments_list}

    # Run diarization pipeline
    print("Running diarization pipeline...")
    t0 = time.time()
    speaker_segments = DIARIZATION_PIPELINE(diarization_file_path)
    print(f"Diarization done in {time.time()-t0:.1f}s")

    for turn, _, speaker in speaker_segments.itertracks(yield_label=True):
        print(f"Speaker {speaker}: {turn.start:.2f}s - {turn.end:.2f}s")

    # Merge transcription and diarization results
    print("Merging transcription and diarization results...")
    merged_segments = diarize_text(transcribe_res, speaker_segments)

    for segment, speaker, text in merged_segments:
        print(f"{segment.start:.1f} - {segment.end:.1f}: Speaker_{speaker} {text}")

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

    return merged_segments_json

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
