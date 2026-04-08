# Frontend (Next.js)

Frontend for the hackathon app. This project is inspired by NeuroTrace UI patterns and updated to use Groq-first transcription in the local API route.

## Stack

- Next.js App Router
- React + TypeScript
- Tailwind CSS v4
- Recharts + Three.js
- Groq OpenAI-compatible transcription API

## Local Setup

1. Install dependencies:

```bash
npm ci
```

2. Create local env file:

```bash
cp .env.example .env.local
```

3. Fill env vars in `.env.local`:

```env
GROQ_API_KEY=your_groq_key
GROQ_TRANSCRIBE_MODEL=whisper-large-v3-turbo
BACKEND_URL=http://localhost:8000
```

4. Start dev server:

```bash
npm run dev
```

## API Routes

- `POST /api/transcribe`: uploads audio and calls Groq transcription.
- `POST /api/analyze`: proxies analysis payload to backend (`BACKEND_URL`).

## Deployment

### Vercel (frontend)

Set these environment variables in Vercel project settings:

- `GROQ_API_KEY`
- `GROQ_TRANSCRIBE_MODEL` (optional, default is `whisper-large-v3-turbo`)
- `BACKEND_URL` (public URL of the backend hosted on Render)

### Render (backend)

Deploy backend separately on Render and copy the Render service URL into Vercel as `BACKEND_URL`.

## Notes

- Keep API secrets only in environment variables.
- The frontend is designed to be iterated quickly during the hackathon: preserve the reusable panel architecture and adjust only tokens/content where needed.
