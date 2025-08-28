# Real-Time Local Transcription & Speaker Diarization
This project provides a real-time speech transcription and speaker diarization system using a FastAPI backend (Python) and a modern React frontend.

# Create environment:
   ```sh
   conda env create -f environment.yml
   conda activate transcribe
   ```
All required Python packages are listed in the README below.

# Backend/Frontend
start backend (Transcription_model.py) with: uvicorn Transcription_model:app --reload

frontend: node modules was not uploaded due to its size. 
Use npm install

start frontend react app (npm start)


# Dependencies:
  - python=3.11
  - pip
  - pip:
    - faster-whisper
    - speechbrain
    - torchaudio
    - sounddevice
    fastapi uvicorn websockets

# Notes:
To significantly improve performance, run the faster-whisper (medium / large) model using GPU acceleration in the Transcription_model.py backend. Current model setting were setup for a M1 pro chip - 32GB memory.

Make sure the backend is running before starting the frontend.
If you change the backend port or address, update the WebSocket URL in the frontend accordingly.

# Features
Real-time transcription using faster-whisper.
Speaker diarization using SpeechBrain.

Modern React frontend with:
Start/Stop recording
Live transcript display (with speaker labels and timestamps)
Save transcript as JSON

# Audio format
The frontend streams raw Float32 PCM audio (mono, 16kHz) to the backend via WebSocket.
If you change the frontend, ensure the audio format matches the backend expectations.

# Structure
Transcription_model.py — FastAPI backend for audio streaming, transcription, and diarization.
my-react-app — React frontend (see transcriber.js for main logic).
environment.yml — Conda environment specification.
README.md — This file.
