# Backend

FastAPI backend for CortexFlow.

This backend is designed to avoid fake/demo scoring:
- Uses deterministic feature extraction from actual text/pause input.
- Emits confidence and quality notes when evidence is weak.
- Applies non-diagnostic safety language by default.
- Uses Groq models only for narrative generation and safety rewriting, never to invent numeric metrics.

## Why this is safer

1. Numeric scores are computed directly from measured input features.
2. If the sample is too short, the API returns a reliability error instead of guessing.
3. High-risk labels are confidence-gated to reduce false panic from low-quality samples.
4. Every report includes an explicit non-diagnostic disclaimer.

## Recommended Groq models for this use case

- Primary reasoning model: `openai/gpt-oss-120b`
- Production fallback: `llama-3.3-70b-versatile`
- Fast fallback: `openai/gpt-oss-20b`
- Safety rewrite model: `openai/gpt-oss-safeguard-20b` (if available)
- Speech transcription model (frontend route): `whisper-large-v3-turbo`

Model availability can vary by project/permissions. The backend auto-discovers models from your key and picks the best available candidate at runtime.

## Local run

```bash
cd backend
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload --port 8000
```

## Endpoints

- `GET /health`
- `GET /models/recommended`
- `POST /analyze`

`POST /analyze` input body:

```json
{
  "input_value": "optional text input",
  "transcript": "optional transcript input",
  "pause_map": [0.32, 0.45],
  "audio_duration": 24.8,
  "session_id": "optional"
}
```

Response is streamed NDJSON with `step`, `error`, and final `end` events.
