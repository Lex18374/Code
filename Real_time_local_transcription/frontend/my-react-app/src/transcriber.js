import React, { useRef, useState } from "react";
import {
  Button,
  Box,
  Typography,
  Paper,
  Stack,
  CircularProgress,
} from "@mui/material";

// INFO
// This React component streams microphone audio to a FastAPI backend via WebSocket.
// The backend must be running and accessible at ws://localhost:8000/ws/audio.
// Audio is sent as raw Float32 PCM, mono, 16kHz (see float32ToPCM).
// The backend responds with JSON arrays of transcription segments (with speaker info).
// The "Save Transcript" button downloads the full transcript as a JSON file.
// Dependencies: @mui/material, @emotion/react, @emotion/styled

// Converts a Float32Array to an ArrayBuffer for sending over WebSocket.
// The backend expects raw Float32 PCM audio.
function float32ToPCM(float32Array) {
  return float32Array.buffer;
}

// Utility to trigger a download of a JSON file in the browser.
function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function Transcriber() {
  // State for recording status, transcript data, and loading spinner
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState([]);
  const [loading, setLoading] = useState(false);

  // Refs to hold WebSocket, media stream, audio context, and processor node
  const wsRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);

  // Start recording: open WebSocket, get mic, stream audio to backend
  const handleStart = async () => {
    setTranscript([]); // Clear previous transcript
    setLoading(true);  // Show loading spinner

    // Open WebSocket connection to backend
    wsRef.current = new WebSocket("ws://localhost:8000/ws/audio");
    wsRef.current.binaryType = "arraybuffer";
    wsRef.current.onopen = () => setLoading(false);
    wsRef.current.onmessage = (event) => {
      // Receive JSON array of segments from backend and append to transcript
      try {
        const segments = JSON.parse(event.data);
        setTranscript((prev) => [...prev, ...segments]);
      } catch (e) {
        // Ignore parse errors
      }
    };

    // Request microphone access
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStreamRef.current = stream;

    // Create audio context at 16kHz sample rate
    const audioContext = new (window.AudioContext ||
      window.webkitAudioContext)({ sampleRate: 16000 });
    audioContextRef.current = audioContext;
    const source = audioContext.createMediaStreamSource(stream);

    // Create a ScriptProcessorNode to access raw PCM audio
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    // On each audio process event, send audio buffer to backend
    processor.onaudioprocess = (e) => {
      if (
        wsRef.current &&
        wsRef.current.readyState === 1 // OPEN
      ) {
        const float32 = e.inputBuffer.getChannelData(0);
        wsRef.current.send(float32ToPCM(float32));
      }
    };

    // Connect audio nodes
    source.connect(processor);
    processor.connect(audioContext.destination);

    setRecording(true);
  };

  // Stop recording: clean up audio and WebSocket resources
  const handleStop = () => {
    setRecording(false);
    if (processorRef.current) {
      processorRef.current.disconnect();
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
  };

  // Download the full transcript as a JSON file
  const handleSave = () => {
    if (transcript.length === 0) return;
    downloadJSON(transcript, "transcript.json");
  };

  // UI rendering
  return (
    <Box
      sx={{
        maxWidth: 700,
        mx: "auto",
        mt: 6,
        p: 3,
        bgcolor: "#f7f7fa",
        borderRadius: 3,
        boxShadow: 2,
      }}
    >
      <Typography variant="h4" gutterBottom>
        Real-Time Transcription
      </Typography>
      <Stack direction="row" spacing={2} mb={3}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleStart}
          disabled={recording || loading}
        >
          {loading ? <CircularProgress size={24} /> : "Start Recording"}
        </Button>
        <Button
          variant="outlined"
          color="secondary"
          onClick={handleStop}
          disabled={!recording}
        >
          Stop Recording
        </Button>
        <Button
          variant="outlined"
          color="success"
          onClick={handleSave}
          disabled={transcript.length === 0}
        >
          Save Transcript
        </Button>
      </Stack>
      <Paper
        elevation={0}
        sx={{
          minHeight: 180,
          p: 2,
          bgcolor: "#fff",
          fontFamily: "monospace",
          whiteSpace: "pre-wrap",
        }}
      >
        {transcript.length === 0 ? (
          <Typography color="text.secondary">
            Transcript will appear here...
          </Typography>
        ) : (
          transcript.map((seg, idx) => (
            <div key={idx}>
              <b>
                [{seg.start?.toFixed(2)}-{seg.end?.toFixed(2)}] {seg.speaker}:
              </b>{" "}
              {seg.text}
            </div>
          ))
        )}
      </Paper>
    </Box>
  );
}