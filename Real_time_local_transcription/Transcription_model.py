# === Real-Time Transcription & Speaker Diarization Backend ===
# This FastAPI backend receives audio chunks via WebSocket, transcribes them in real time using Whisper,
# and performs speaker diarization using SpeechBrain. Results are streamed back to the frontend as JSON.
# Audio is expected as raw float32 PCM, mono, 16kHz.

import torchaudio
import torch
import numpy as np
from faster_whisper import WhisperModel
from speechbrain.pretrained import SpeakerRecognition
import tempfile
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_SIZE = "small"        # Whisper model size; use "medium" or "large" for better accuracy (requires more resources)
WINDOW_SIZE = 3.5           # Length (seconds) of each audio window to process
WINDOW_STRIDE = 3.25        # Step (seconds) to slide window forward (overlap for no missed speech)
MIN_CHUNK_SEC = 1.0         # Minimum segment length (seconds) for speaker verification
SR = 16000                  # Audio sample rate (Hz); must match frontend
# -----------------------------

torchaudio.set_audio_backend("soundfile")

print("Loading Whisper model...")
model = WhisperModel(MODEL_SIZE)

# This part "warms up" the Whisper model by transcribing a short, silent audio file. 
# This ensures that the first actual transcription is fast, as the model's resources are already loaded into memory.
print("Warming up Whisper model...")
dummy = np.zeros((SR,), dtype=np.float32)
with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
    torchaudio.save(tmp.name, torch.from_numpy(dummy).unsqueeze(0), SR)
    model.transcribe(tmp.name, beam_size=1)

print("Loading SpeechBrain speaker embedding model...")
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

app = FastAPI()

def merge_text(text1, text2):
    if text2 in text1:
        return text1
    return (text1 + " " + text2).strip()

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.
    Receives audio chunks from the frontend, buffers them, and processes in sliding windows.
    Sends back transcription segments (with speaker info) as soon as available.
    """
    await websocket.accept()
    speaker_map = {}         # Maps speaker label to their reference audio (for diarization)
    speaker_counter = 1      # Counter for assigning new speaker labels
    all_segments = []        # Stores all segments for optional final merging
    audio_buffer_list = []   # List of received audio chunks (for efficient concatenation)
    start_sample = 0         # Tracks position in the audio stream (for timestamps)

    try:
        while True:
            # --- 1. Receive audio chunk from frontend ---
            data = await websocket.receive_bytes()
            # Expecting float32 PCM mono audio (from frontend)
            chunk = np.frombuffer(data, dtype=np.float32).reshape(-1, 1)
            audio_buffer_list.append(chunk)

            # --- 2. Concatenate all buffered audio ---
            audio_buffer = np.concatenate(audio_buffer_list, axis=0)

            # --- 3. Process if enough audio for a window ---
            # Window-based processing ensures real-time, low-latency transcription and diarization.
            if audio_buffer.shape[0] >= int(WINDOW_SIZE * SR):
                chunk_to_process = audio_buffer[:int(WINDOW_SIZE * SR)].T  # (1, samples)

                with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                    torchaudio.save(tmp.name, torch.from_numpy(chunk_to_process), SR)
                    # Whisper transcribes the current window and returns segments with timestamps.
                    segments, _ = model.transcribe(tmp.name, beam_size=1)

                output_segments = []
                for seg in segments:
                    # Calculate absolute start/end times for this segment in the stream
                    seg_start = start_sample / SR + seg.start
                    seg_end = start_sample / SR + seg.end
                    seg_text = seg.text.strip()

                    # Extract the audio for this segment (for speaker verification)
                    seg_audio_start = int(seg.start * SR)
                    seg_audio_end = int(seg.end * SR)
                    seg_chunk = chunk_to_process[:, seg_audio_start:seg_audio_end]

                    # Skip very short segments (not enough for reliable speaker verification)
                    if seg_chunk.shape[1] < SR * MIN_CHUNK_SEC:
                        continue

                    # --- 4. Speaker Diarization ---
                    # Compare this segment's audio to all known speakers using SpeechBrain.
                    # If a match is found, assign to that speaker; otherwise, create a new speaker.
                    assigned = None
                    for spk, ref_chunk in speaker_map.items():
                        ref_chunk_tensor = torch.from_numpy(ref_chunk) if isinstance(ref_chunk, np.ndarray) else ref_chunk
                        seg_chunk_tensor = torch.from_numpy(seg_chunk) if isinstance(seg_chunk, np.ndarray) else seg_chunk

                        score, prediction = verification.verify_batch(ref_chunk_tensor, seg_chunk_tensor)
                        if prediction[0].item():
                            assigned = spk
                            # Update reference audio for this speaker (concatenate for robustness)
                            speaker_map[spk] = torch.cat([ref_chunk_tensor, seg_chunk_tensor], dim=1)
                            break

                    if assigned is None:
                        assigned = f"Speaker {speaker_counter}"
                        speaker_counter += 1
                        speaker_map[assigned] = seg_chunk

                    segment_dict = {
                        "start": seg_start,
                        "end": seg_end,
                        "text": seg_text,
                        "speaker": assigned
                    }
                    all_segments.append(segment_dict)
                    output_segments.append(segment_dict)

                # --- 5. Send new segments to frontend as JSON ---
                # Only send new segments for this window (not the full transcript)
                if output_segments:
                    await websocket.send_json(output_segments)

                # --- 6. Slide the window forward ---
                # Remove WINDOW_STRIDE seconds from the start of the buffer (overlap for no missed speech)
                slide_amount = int(WINDOW_STRIDE * SR)
                audio_buffer = audio_buffer[slide_amount:]
                audio_buffer_list = [audio_buffer] if audio_buffer.shape[0] > 0 else []
                start_sample += slide_amount

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print("WebSocket closed with error:", e)

    # --- Optional: Merge overlapping segments for a clean final transcript ---
    merged_segments = []
    for seg in all_segments:
        if not merged_segments:
            merged_segments.append(seg)
            continue
        last = merged_segments[-1]
        if seg["start"] < last["end"]:
            last["end"] = max(last["end"], seg["end"])
            last["text"] = merge_text(last["text"], seg["text"])
            if last["speaker"] != seg["speaker"]:
                last["speaker"] = f"{last['speaker']}/{seg['speaker']}"
        else:
            merged_segments.append(seg)

    print("\n=== Final Merged Transcript ===")
    for seg in merged_segments:
        print(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']}: {seg['text']}")
