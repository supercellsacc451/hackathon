# Hackathon Project

Frontend-first AI analysis workspace inspired by NeuroTrace, rebuilt for your own stack with Groq-based APIs.

## Current Scope

- Frontend imported and adapted from NeuroTrace-style architecture
- Brand + theme refreshed for a cleaner premium look
- Local transcription route migrated to Groq OpenAI-compatible endpoint
- Backend implemented with deterministic metric extraction + Groq narrative/safety layers
- Frontend set up for Vercel deployment
- Backend ready for Render and connected via BACKEND_URL

## Repository Layout

- frontend/: Next.js app (active build target right now)
- backend/: FastAPI analysis service (NDJSON streaming)

## Backend Quick Start

```bash
cd backend
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload --port 8000
```

Recommended Groq model strategy for this use case:

- Primary reasoning: `openai/gpt-oss-120b`
- Fallback reasoning: `llama-3.3-70b-versatile`
- Fast fallback: `openai/gpt-oss-20b`
- Safety rewrite: `openai/gpt-oss-safeguard-20b` (if available)
- STT in frontend route: `whisper-large-v3-turbo`

The backend auto-discovers available models from the same Groq key and picks the best available candidate in order.

## Accuracy & Safety Guardrails

- Numeric scores are computed from real text/pause features, not LLM-generated values.
- Very short samples are rejected for reliability instead of returning guessed scores.
- Confidence and quality notes are included in each report.
- Non-diagnostic wording is enforced to reduce panic and false interpretation.

## Frontend Quick Start

```bash
cd frontend
npm ci
cp .env.example .env.local
npm run dev
```

Required env vars in frontend/.env.local:

```env
GROQ_API_KEY=your_groq_key
GROQ_TRANSCRIBE_MODEL=whisper-large-v3-turbo
BACKEND_URL=http://localhost:8000
```

## Deployment Plan

1. Deploy frontend on Vercel.
2. Deploy backend on Render.
3. Set Vercel env BACKEND_URL to the Render backend URL.
4. Keep GROQ_API_KEY in Vercel environment variables.

## Build Notes

- Keep componentized panel layout for fast hackathon iteration.
- Keep visual polish by changing token variables in src/app/globals.css, not scattered inline colors.
- For small changes, commit and push quickly to keep remote always up to date.
