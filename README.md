# Hackathon Project

Frontend-first AI analysis workspace inspired by NeuroTrace, rebuilt for your own stack with Groq-based APIs.

## Current Scope

- Frontend imported and adapted from NeuroTrace-style architecture
- Brand + theme refreshed for a cleaner premium look
- Local transcription route migrated to Groq OpenAI-compatible endpoint
- Frontend set up for Vercel deployment
- Backend expected on Render and connected via BACKEND_URL

## Repository Layout

- frontend/: Next.js app (active build target right now)

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
