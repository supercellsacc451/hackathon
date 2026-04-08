import { type NextRequest, NextResponse } from "next/server";

const GROQ_API_KEY = process.env.GROQ_API_KEY ?? "";
const GROQ_TRANSCRIBE_MODEL = process.env.GROQ_TRANSCRIBE_MODEL ?? "whisper-large-v3-turbo";
const MAX_AUDIO_BYTES = 25 * 1024 * 1024;

/**
 * Extract pause map from Whisper response with word-level timestamps
 * Returns array of silence durations (in seconds) between words
 */
function extractPauseMap(wordTimestamps: Array<{ word: string; start: number; end: number }>): number[] {
  const pauses: number[] = [];
  for (let i = 0; i < wordTimestamps.length - 1; i++) {
    const gap = wordTimestamps[i + 1].start - wordTimestamps[i].end;
    if (gap > 0.1) {
      // Only record pauses > 100ms
      pauses.push(gap);
    }
  }
  return pauses;
}

export async function POST(req: NextRequest) {
  try {
    if (!GROQ_API_KEY) {
      return NextResponse.json({ error: "Missing GROQ_API_KEY" }, { status: 500 });
    }

    const formData = await req.formData();
    const audioFile = formData.get("audio") as File;

    if (!audioFile) {
      return NextResponse.json({ error: "No audio file provided" }, { status: 400 });
    }

    if (audioFile.size <= 0) {
      return NextResponse.json({ error: "Uploaded file is empty" }, { status: 400 });
    }

    if (audioFile.size > MAX_AUDIO_BYTES) {
      return NextResponse.json(
        { error: "Audio file is too large. Please upload a file under 25MB." },
        { status: 413 }
      );
    }

    // Call Groq's OpenAI-compatible transcription API
    const whisperFormData = new FormData();
    whisperFormData.append("file", audioFile);
    whisperFormData.append("model", GROQ_TRANSCRIBE_MODEL);
    whisperFormData.append("response_format", "verbose_json");
    whisperFormData.append("timestamp_granularities[]", "word");

    const whisperRes = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${GROQ_API_KEY}`,
      },
      body: whisperFormData,
      signal: AbortSignal.timeout(60_000),
    });

    if (!whisperRes.ok) {
      const error = await whisperRes.text();
      return NextResponse.json({ error: `Whisper API error: ${error}` }, { status: whisperRes.status });
    }

    const whisperData = await whisperRes.json();
    const transcript = whisperData.text || "";
    const wordTimestamps = whisperData.words || [];

    // Extract pause map from word timestamps
    const pauseMap = extractPauseMap(wordTimestamps);

    return NextResponse.json({
      transcript,
      pauseMap,
      wordTimestamps,
      duration: whisperData.duration,
    });
  } catch (error) {
    console.error("Transcription error:", error);
    return NextResponse.json(
      { error: "Failed to transcribe audio" },
      { status: 500 }
    );
  }
}
