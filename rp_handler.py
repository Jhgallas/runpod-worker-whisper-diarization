import base64
import os
import tempfile

import requests as http_requests
import runpod
import torch
import whisper
import soundfile as sf
import librosa

from pyannote.audio import Pipeline
from pyannote.core import Segment


# Inlined from pyannote-whisper (repo install is broken)
def diarize_text(transcribe_res, diarization_result):
    timestamp_texts = []
    for item in transcribe_res['segments']:
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

def _resolve_runtime():
    env = os.environ.get("ENV", "").lower()
    if torch.cuda.is_available():
        print("CUDA detected. Using GPU execution.")
        return "cuda", "turbo"

    if env == "prod":
        print("WARNING: Production mode requested but CUDA not available. Falling back to CPU.")
        return "cpu", "turbo"

    print("No CUDA detected. Using CPU execution.")
    return "cpu", "tiny"


def handler(event):
    print("Worker Start")
    
    # Validate event structure
    if not event or "input" not in event:
        raise ValueError("Event must contain 'input' field.")
    
    request_input = event["input"]

    # Support two input modes: base64 or URL
    audio_url = request_input.get("audio_url")
    base64file = request_input.get("file")
    extension = request_input.get("ext", "wav")

    if not audio_url and not base64file:
        raise ValueError("Either 'audio_url' or 'file' (base64) input is required.")

    device, transcribe_model = _resolve_runtime()

    # Load whisper model
    print(f"Loading whisper model '{transcribe_model}' on device '{device}'...")
    whisper_model = whisper.load_model(transcribe_model, device=device)

    # Init pyannote pipeline
    print("Loading pyannote pipeline...")
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HUGGINGFACE_ACCESS_TOKEN"])

    # Get audio to a temporary file
    temp_file_path = None
    if audio_url:
        # Download from URL
        print(f"Downloading audio from URL...")
        response = http_requests.get(audio_url, stream=True, timeout=600)
        response.raise_for_status()
        # Guess extension from URL if not provided
        if "." in audio_url.split("?")[0].split("/")[-1]:
            extension = audio_url.split("?")[0].split("/")[-1].split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        print(f"Downloaded to: {temp_file_path} ({os.path.getsize(temp_file_path) / (1024*1024):.1f} MB)")
    else:
        # Decode base64
        print("Decoding base64...")
        audio_data = base64.b64decode(base64file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        print(f"Decoded file saved to: {temp_file_path}")

    # Preprocess audio for pyannote: load and resample to ensure compatibility
    print("Preprocessing audio for diarization...")
    audio, original_sr = librosa.load(temp_file_path, sr=None)

    # Resample to 16kHz which is standard for pyannote
    target_sr = 16000
    if original_sr != target_sr:
        print(f"Resampling audio from {original_sr}Hz to {target_sr}Hz...")
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    
    # Save preprocessed audio as WAV for pyannote
    diarization_file_path = temp_file_path.replace(f".{extension}", "_diarization.wav")
    sf.write(diarization_file_path, audio, target_sr)
    print(f"Preprocessed audio saved to: {diarization_file_path}")

    # Transcribe (use original file, whisper handles various formats)
    print("Transcribing...")
    transcription_result = whisper_model.transcribe(temp_file_path)

    # Run diarization pipeline (use preprocessed WAV file)
    print("Running diarization pipeline...")
    speaker_segments = diarization_pipeline(diarization_file_path)

    # Print diarization results
    for turn, _, speaker in speaker_segments.itertracks(yield_label=True):
        print(f"Speaker {speaker}: {turn.start:.2f}s - {turn.end:.2f}s")

    # Merge transcription and diarization results
    print("Merging transcription and diarization results...")
    merged_segments = diarize_text(transcription_result, speaker_segments)

    # Print merged segments
    for segment, speaker, text in merged_segments:
        print(f"{segment.start:.1f} - {segment.end:.1f}: Speaker_{speaker} {text}")

    # Cleanup files
    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
        print(f"Cleaned up temporary file: {temp_file_path}")
    if diarization_file_path and os.path.exists(diarization_file_path):
        os.remove(diarization_file_path)
        print(f"Cleaned up diarization file: {diarization_file_path}")

    # Map merged segments to a json
    merged_segments_json = []
    for segment, speaker, text in merged_segments:
        trimmed_text = text.strip()
        merged_segments_json.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker,
            "text": trimmed_text
        })

    return merged_segments_json

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
