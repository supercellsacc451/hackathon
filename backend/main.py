import asyncio
import json
import os
import re
import statistics
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="CortexFlow Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "40"))
MODEL_DISCOVERY_TTL_SECONDS = int(os.getenv("MODEL_DISCOVERY_TTL_SECONDS", "900"))

PREFERRED_REASONING_MODELS = [
    m.strip()
    for m in os.getenv(
        "GROQ_REASONING_CANDIDATES",
        "openai/gpt-oss-120b,llama-3.3-70b-versatile,openai/gpt-oss-20b,llama-3.1-8b-instant",
    ).split(",")
    if m.strip()
]
PREFERRED_SAFETY_MODELS = [
    m.strip()
    for m in os.getenv(
        "GROQ_SAFETY_CANDIDATES",
        "openai/gpt-oss-safeguard-20b,openai/gpt-oss-20b,llama-3.1-8b-instant",
    ).split(",")
    if m.strip()
]

OVERRIDE_REASONING_MODEL = os.getenv("GROQ_REASONING_MODEL", "").strip()
OVERRIDE_SAFETY_MODEL = os.getenv("GROQ_SAFETY_MODEL", "").strip()

MIN_WORDS_REQUIRED = int(os.getenv("MIN_WORDS_REQUIRED", "25"))

STEP_NAMES = [
    "STT preprocessor",
    "Lexical agent",
    "Semantic agent",
    "Prosody agent",
    "Syntax agent",
    "Biomarker mapper",
    "Report composer",
]

DOMAIN_REGION = {
    "lexical": "Broca's area",
    "semantic": "Wernicke's area",
    "prosody": "SMA",
    "syntax": "DLPFC",
    "affective": "Amygdala",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "of", "to", "in", "on", "at", "for",
    "with", "without", "by", "from", "as", "is", "am", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she", "we", "they", "them",
    "my", "your", "our", "their", "me", "him", "her", "us", "do", "does", "did", "have", "has", "had",
    "not", "no", "yes", "so", "because", "about", "into", "out", "up", "down", "can", "could", "would",
    "should", "will", "just", "very", "really", "also",
}

FILLERS = {
    "um", "uh", "erm", "hmm", "like", "you", "know", "actually", "basically", "literally", "sort", "kind", "maybe",
}

POSITIVE_WORDS = {
    "good", "better", "great", "calm", "confident", "clear", "focused", "stable", "happy", "optimistic", "safe", "steady",
}
NEGATIVE_WORDS = {
    "bad", "worse", "anxious", "scared", "panic", "panicked", "confused", "sad", "depressed", "angry", "overwhelmed", "stressed",
}
AROUSAL_WORDS = {
    "urgent", "immediately", "intense", "extreme", "critical", "afraid", "panic", "terrified", "racing", "shaking", "worried",
}
HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "probably", "sort", "kind", "might", "could", "guess", "unsure", "not sure",
}
SUBORDINATORS = {
    "because", "although", "though", "while", "unless", "until", "since", "whereas", "however", "therefore", "moreover", "which", "that",
}


class AnalyzeRequest(BaseModel):
    input_value: Optional[str] = None
    transcript: Optional[str] = None
    pause_map: Optional[list[float]] = None
    audio_duration: Optional[float] = None
    session_id: Optional[str] = None


@dataclass
class DomainScore:
    overall: float
    details: dict[str, float]


@dataclass
class AnalysisState:
    scores: dict[str, DomainScore]
    overall_load: float
    confidence: float
    quality_notes: list[str]
    metrics: dict[str, Any]


_MODEL_CACHE: dict[str, Any] = {"updated": 0.0, "models": []}
_MODEL_CACHE_LOCK = asyncio.Lock()


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def mean(values: list[float], default: float = 0.0) -> float:
    return float(statistics.mean(values)) if values else default


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def content_words(tokens: list[str]) -> list[str]:
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union else 0.0


def scale_linear(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp01((value - low) / (high - low))


def scale_inverse(value: float, good: float, poor: float) -> float:
    if poor >= good:
        return 0.0
    return clamp01((good - value) / (good - poor))


def safe_step_event(name: str, status: str, detail: Optional[str] = None) -> bytes:
    payload: dict[str, Any] = {"type": "step", "step": {"name": name, "status": status}}
    if detail:
        payload["step"]["detail"] = detail
    return (json.dumps(payload) + "\n").encode()


def ensure_nonempty_text(req: AnalyzeRequest) -> str:
    text = (req.input_value or req.transcript or "").strip()
    words = tokenize_words(text)
    if not text:
        raise HTTPException(status_code=400, detail="No input text provided")
    if len(words) < MIN_WORDS_REQUIRED:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {MIN_WORDS_REQUIRED} words for reliable analysis. Received {len(words)} words.",
        )
    return text


# -----------------------------------------------------------------------------
# Deterministic analysis pipeline
# -----------------------------------------------------------------------------


def lexical_domain(tokens: list[str], content: list[str]) -> tuple[DomainScore, dict[str, float]]:
    total = max(len(tokens), 1)
    unique = len(set(tokens))
    filler_hits = sum(1 for t in tokens if t in FILLERS)

    ttr = unique / total
    density = len(content) / total
    filler_rate = (filler_hits / total) * 100.0

    s_ttr = clamp01(abs(ttr - 0.52) / 0.30)
    s_density = clamp01(abs(density - 0.58) / 0.25)
    s_filler = scale_linear(filler_rate, 2.0, 14.0)

    overall = clamp01((0.4 * s_ttr) + (0.35 * s_density) + (0.25 * s_filler))

    details = {
        "ttr": round(s_ttr, 4),
        "density": round(s_density, 4),
        "filler_rate": round(s_filler, 4),
    }
    raw = {
        "ttr": round(ttr, 4),
        "lexical_density": round(density, 4),
        "filler_rate_per_100w": round(filler_rate, 2),
    }
    return DomainScore(round(overall, 4), details), raw


def semantic_domain(sentences: list[str]) -> tuple[DomainScore, dict[str, float]]:
    if len(sentences) < 2:
        coherence = 0.16
        idea_density = 0.45
        tangentiality = 0.55
    else:
        sentence_content = [set(content_words(tokenize_words(s))) for s in sentences]
        pairwise = [jaccard(sentence_content[i], sentence_content[i + 1]) for i in range(len(sentence_content) - 1)]
        coherence = mean(pairwise, default=0.12)
        avg_content_len = mean([len(x) for x in sentence_content], default=0.0)
        idea_density = clamp01(avg_content_len / 14.0)
        tangentiality = clamp01(1.0 - coherence)

    s_coherence = scale_inverse(coherence, good=0.22, poor=0.05)
    s_idea_density = scale_inverse(idea_density, good=0.65, poor=0.25)
    s_tangentiality = scale_linear(tangentiality, low=0.35, high=0.85)

    overall = clamp01((0.45 * s_coherence) + (0.30 * s_idea_density) + (0.25 * s_tangentiality))

    details = {
        "coherence": round(s_coherence, 4),
        "idea_density": round(s_idea_density, 4),
        "tangentiality": round(s_tangentiality, 4),
    }
    raw = {
        "coherence_index": round(coherence, 4),
        "idea_density_index": round(idea_density, 4),
        "tangentiality_index": round(tangentiality, 4),
    }
    return DomainScore(round(overall, 4), details), raw


def prosody_domain(
    tokens: list[str], text: str, pause_map: Optional[list[float]], audio_duration: Optional[float]
) -> tuple[DomainScore, dict[str, float], bool]:
    word_count = max(len(tokens), 1)
    pauses = [float(p) for p in (pause_map or []) if p >= 0]
    has_audio_prosody = bool(pauses)

    if audio_duration and audio_duration > 5.0:
        duration_seconds = audio_duration
    else:
        estimated_speech_seconds = word_count / 2.5
        duration_seconds = estimated_speech_seconds + sum(pauses)

    duration_minutes = max(duration_seconds / 60.0, 0.1)
    speech_rate = word_count / duration_minutes

    if pauses:
        pause_freq = len(pauses) / duration_minutes
        hesitation_ratio = sum(1 for p in pauses if p >= 0.8) / len(pauses)
    else:
        punctuation_pauses = len(re.findall(r"[,;:\-]", text))
        pause_freq = (punctuation_pauses / max(word_count, 1)) * 100
        hesitation_ratio = sum(1 for t in tokens if t in FILLERS) / max(word_count, 1)

    s_rate = clamp01(abs(speech_rate - 140.0) / 95.0)
    s_pause = scale_linear(pause_freq, low=8.0, high=30.0)
    s_hes = scale_linear(hesitation_ratio, low=0.08, high=0.35)

    overall = clamp01((0.4 * s_rate) + (0.35 * s_pause) + (0.25 * s_hes))

    details = {
        "speech_rate": round(s_rate, 4),
        "pause_freq": round(s_pause, 4),
        "hesitation": round(s_hes, 4),
    }
    raw = {
        "speech_rate_wpm": round(speech_rate, 1),
        "pause_frequency_per_min": round(pause_freq, 2),
        "hesitation_ratio": round(hesitation_ratio, 4),
        "duration_seconds": round(duration_seconds, 2),
    }
    return DomainScore(round(overall, 4), details), raw, has_audio_prosody


def syntax_domain(tokens: list[str], sentences: list[str], text: str) -> tuple[DomainScore, dict[str, float]]:
    sentence_count = max(len(sentences), 1)
    mlu = len(tokens) / sentence_count

    per_sentence_depth = []
    for s in sentences:
        stoks = tokenize_words(s)
        sub_count = sum(1 for t in stoks if t in SUBORDINATORS)
        comma_count = s.count(",")
        per_sentence_depth.append(sub_count + (comma_count * 0.5))
    clause_depth = mean(per_sentence_depth, default=0.0)

    passive_matches = re.findall(r"\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en)\b", text.lower())
    passive_ratio = len(passive_matches) / max(sentence_count, 1)

    s_mlu = clamp01(abs(mlu - 17.0) / 12.0)
    s_depth = scale_linear(clause_depth, low=2.0, high=6.5)
    s_passive = scale_linear(passive_ratio, low=0.15, high=1.2)

    overall = clamp01((0.45 * s_mlu) + (0.35 * s_depth) + (0.20 * s_passive))

    details = {
        "mlu": round(s_mlu, 4),
        "clause_depth": round(s_depth, 4),
        "passive_ratio": round(s_passive, 4),
    }
    raw = {
        "mean_length_utterance": round(mlu, 2),
        "clause_depth_index": round(clause_depth, 2),
        "passive_ratio": round(passive_ratio, 3),
    }
    return DomainScore(round(overall, 4), details), raw


def affective_domain(tokens: list[str]) -> tuple[DomainScore, dict[str, float]]:
    total = max(len(tokens), 1)
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    arousal = sum(1 for t in tokens if t in AROUSAL_WORDS)
    hedge = sum(1 for t in tokens if t in HEDGE_WORDS)

    valence = (pos - neg) / (pos + neg + 1)
    valence_01 = (valence + 1.0) / 2.0
    arousal_rate = (arousal / total) * 100.0
    certainty = 1.0 - clamp01(hedge / max(total * 0.15, 1.0))

    s_valence = scale_inverse(valence_01, good=0.62, poor=0.20)
    s_arousal = scale_linear(arousal_rate, low=3.0, high=14.0)
    s_certainty = scale_inverse(certainty, good=0.72, poor=0.32)

    overall = clamp01((0.4 * s_valence) + (0.35 * s_arousal) + (0.25 * s_certainty))

    details = {
        "valence": round(s_valence, 4),
        "arousal": round(s_arousal, 4),
        "certainty": round(s_certainty, 4),
    }
    raw = {
        "valence_score": round(valence_01, 4),
        "arousal_rate_per_100w": round(arousal_rate, 2),
        "certainty_index": round(certainty, 4),
    }
    return DomainScore(round(overall, 4), details), raw


def compute_confidence(
    word_count: int, sentence_count: int, has_audio_prosody: bool, repeat_ratio: float
) -> tuple[float, list[str]]:
    notes: list[str] = []
    c_words = clamp01(word_count / 180.0)
    c_sents = clamp01(sentence_count / 8.0)
    c_repeat = clamp01(1.0 - (repeat_ratio * 1.4))
    c_audio = 1.0 if has_audio_prosody else 0.55

    confidence = clamp01((0.45 * c_words) + (0.2 * c_sents) + (0.2 * c_repeat) + (0.15 * c_audio))

    if word_count < 60:
        notes.append("Low sample length. Interpret results cautiously.")
    if not has_audio_prosody:
        notes.append("Prosody is inferred from text patterns because pause-map audio features were not provided.")
    if repeat_ratio > 0.45:
        notes.append("High repetition detected, which can reduce semantic reliability.")

    return round(confidence, 4), notes


def compute_analysis_state(
    text: str,
    pause_map: Optional[list[float]],
    audio_duration: Optional[float],
) -> AnalysisState:
    tokens = tokenize_words(text)
    sentences = split_sentences(text)
    cwords = content_words(tokens)

    repeat_ratio = 1.0 - (len(set(tokens)) / max(len(tokens), 1))

    lexical, lexical_raw = lexical_domain(tokens, cwords)
    semantic, semantic_raw = semantic_domain(sentences)
    prosody, prosody_raw, has_audio = prosody_domain(tokens, text, pause_map, audio_duration)
    syntax, syntax_raw = syntax_domain(tokens, sentences, text)
    affective, affective_raw = affective_domain(tokens)

    confidence, quality_notes = compute_confidence(
        word_count=len(tokens),
        sentence_count=len(sentences),
        has_audio_prosody=has_audio,
        repeat_ratio=repeat_ratio,
    )

    scores = {
        "lexical": lexical,
        "semantic": semantic,
        "prosody": prosody,
        "syntax": syntax,
        "affective": affective,
    }

    weighted = (
        (0.22 * lexical.overall)
        + (0.23 * semantic.overall)
        + (0.18 * prosody.overall)
        + (0.22 * syntax.overall)
        + (0.15 * affective.overall)
    )

    # Confidence-aware dampening prevents over-alerting when evidence quality is weak.
    confidence_factor = 0.75 + (0.25 * confidence)
    overall_load = clamp01(weighted * confidence_factor)

    metrics = {
        "word_count": len(tokens),
        "sentence_count": len(sentences),
        "repeat_ratio": round(repeat_ratio, 4),
        "lexical": lexical_raw,
        "semantic": semantic_raw,
        "prosody": prosody_raw,
        "syntax": syntax_raw,
        "affective": affective_raw,
    }

    return AnalysisState(
        scores=scores,
        overall_load=round(overall_load, 4),
        confidence=confidence,
        quality_notes=quality_notes,
        metrics=metrics,
    )


def severity_from_score(value: float) -> str:
    if value >= 0.72:
        return "high"
    if value >= 0.42:
        return "moderate"
    return "low"


def level_from_overall(overall_load: float, confidence: float) -> str:
    if overall_load >= 0.68:
        base = "high"
    elif overall_load >= 0.44:
        base = "moderate"
    else:
        base = "low"

    # Confidence guardrail: avoid "high" labels when signal quality is weak.
    if confidence < 0.45 and base == "high":
        return "moderate"
    return base


def summary_fallback(state: AnalysisState, risk_level: str) -> str:
    top_domain = max(state.scores.items(), key=lambda kv: kv[1].overall)[0]
    top_value = state.scores[top_domain].overall
    confidence_pct = round(state.confidence * 100)
    return (
        f"This analysis found a {risk_level} overall cognitive load signal based on linguistic and timing features. "
        f"The strongest deviation appeared in {top_domain} markers (score {top_value:.2f}). "
        f"Confidence is {confidence_pct}% and this output is screening support only, not a diagnosis."
    )


def make_highlights(state: AnalysisState) -> list[dict[str, Any]]:
    sorted_domains = sorted(state.scores.items(), key=lambda kv: kv[1].overall, reverse=True)
    highlights: list[dict[str, Any]] = []
    for domain, score in sorted_domains[:3]:
        if score.overall >= 0.66:
            finding = "Elevated deviation from expected baseline in this domain."
        elif score.overall >= 0.42:
            finding = "Mild-to-moderate deviation with mixed stability."
        else:
            finding = "Signals remain within expected variation for this domain."

        highlights.append(
            {
                "region": DOMAIN_REGION[domain],
                "activation": round(score.overall, 4),
                "finding": finding,
                "clinical_context": "Screening signal only. Interpret alongside clinical judgement and repeated assessments.",
            }
        )
    return highlights


def make_indicators(state: AnalysisState) -> list[dict[str, Any]]:
    indicators: list[dict[str, Any]] = []
    for domain, dscore in state.scores.items():
        for k, v in dscore.details.items():
            if v < 0.42:
                continue
            indicators.append(
                {
                    "indicator": f"{domain.title()} · {k.replace('_', ' ').title()}",
                    "severity": severity_from_score(v),
                    "explanation": f"Computed score {v:.2f} from measured input features; higher means greater deviation from baseline patterns.",
                }
            )

    indicators.sort(key=lambda x: {"high": 2, "moderate": 1, "low": 0}[x["severity"]], reverse=True)
    return indicators[:6]


def recommendation_for_level(level: str, confidence: float) -> str:
    if level == "high":
        return (
            "Repeat this assessment with a longer sample, then discuss the combined results with a qualified clinician. "
            "Do not treat this result as a diagnosis."
        )
    if level == "moderate":
        return (
            "Collect 1-2 additional samples across different times of day to confirm trend stability before drawing conclusions."
        )
    if confidence < 0.5:
        return "Provide a longer speech sample for stronger reliability before interpreting the result."
    return "Current signals are relatively stable. Continue periodic monitoring rather than one-off interpretation."


# -----------------------------------------------------------------------------
# Groq model discovery and controlled generation
# -----------------------------------------------------------------------------


async def fetch_available_models() -> list[str]:
    if not GROQ_API_KEY:
        return []

    async with _MODEL_CACHE_LOCK:
        now = time.time()
        if now - float(_MODEL_CACHE["updated"]) < MODEL_DISCOVERY_TTL_SECONDS:
            return list(_MODEL_CACHE["models"])

        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        try:
            async with httpx.AsyncClient(timeout=GROQ_TIMEOUT_SECONDS) as client:
                res = await client.get(f"{GROQ_API_BASE}/models", headers=headers)
                res.raise_for_status()
                data = res.json().get("data", [])
                models = sorted({item.get("id", "") for item in data if item.get("id")})
                _MODEL_CACHE["updated"] = now
                _MODEL_CACHE["models"] = models
                return models
        except Exception:
            return list(_MODEL_CACHE["models"])


def pick_model(available: list[str], override: str, candidates: list[str]) -> Optional[str]:
    if override and override in available:
        return override

    for m in candidates:
        if m in available:
            return m

    for m in available:
        lowered = m.lower()
        if "instruct" in lowered or "versatile" in lowered or "gpt-oss" in lowered:
            return m

    return available[0] if available else None


async def groq_chat(model: str, system: str, user: str, temperature: float = 0.2) -> Optional[str]:
    if not GROQ_API_KEY or not model:
        return None

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=GROQ_TIMEOUT_SECONDS) as client:
            res = await client.post(f"{GROQ_API_BASE}/chat/completions", headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


async def compose_safe_summary(state: AnalysisState, risk_level: str) -> tuple[str, dict[str, Optional[str]]]:
    available = await fetch_available_models()
    reasoning_model = pick_model(available, OVERRIDE_REASONING_MODEL, PREFERRED_REASONING_MODELS)
    safety_model = pick_model(available, OVERRIDE_SAFETY_MODEL, PREFERRED_SAFETY_MODELS)

    model_meta = {
        "reasoning_model": reasoning_model,
        "safety_model": safety_model,
    }

    baseline_summary = summary_fallback(state, risk_level)
    if not reasoning_model:
        return baseline_summary, model_meta

    features_for_prompt = {
        "risk_level": risk_level,
        "overall_cognitive_load": state.overall_load,
        "confidence": state.confidence,
        "scores": {k: v.overall for k, v in state.scores.items()},
        "quality_notes": state.quality_notes,
        "metrics": state.metrics,
    }

    system = (
        "You summarize computational language-screening outputs. "
        "Never diagnose disease, never use alarming wording, and always state uncertainty when confidence is limited. "
        "Output exactly 2-3 sentences in plain text."
    )
    user = "Write a careful summary for this analysis:\n" + json.dumps(features_for_prompt)

    summary = await groq_chat(reasoning_model, system, user, temperature=0.15)
    if not summary:
        return baseline_summary, model_meta

    if safety_model:
        safety_system = (
            "You are a safety editor for health-adjacent UX. "
            "Rewrite text to avoid panic, avoid diagnosis claims, and keep uncertainty explicit. "
            "Keep 2-3 sentences."
        )
        safety_user = (
            "Rewrite this summary to be non-alarmist and clinically careful while keeping factual content:\n"
            + summary
            + "\n\nConfidence: "
            + str(state.confidence)
        )
        safe = await groq_chat(safety_model, safety_system, safety_user, temperature=0.1)
        if safe:
            summary = safe

    return summary, model_meta


# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    available = await fetch_available_models()
    return {
        "ok": True,
        "service": "cortexflow-backend",
        "groq_configured": bool(GROQ_API_KEY),
        "model_count": len(available),
    }


@app.get("/models/recommended")
async def models_recommended() -> dict[str, Any]:
    available = await fetch_available_models()
    return {
        "available_models": available,
        "recommended": {
            "reasoning": pick_model(available, OVERRIDE_REASONING_MODEL, PREFERRED_REASONING_MODELS),
            "safety": pick_model(available, OVERRIDE_SAFETY_MODEL, PREFERRED_SAFETY_MODELS),
            "transcription": "whisper-large-v3-turbo",
        },
        "notes": {
            "production_primary": "openai/gpt-oss-120b",
            "production_fallback": "llama-3.3-70b-versatile",
            "fast_fallback": "openai/gpt-oss-20b",
        },
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    text = ensure_nonempty_text(req)
    session_id = req.session_id or str(uuid.uuid4())

    async def generate():
        # Initial state: first step running, rest pending
        for idx, step_name in enumerate(STEP_NAMES):
            yield safe_step_event(step_name, "running" if idx == 0 else "pending")

        try:
            # Step 1: preprocessing
            state = compute_analysis_state(text, req.pause_map, req.audio_duration)
            yield safe_step_event("STT preprocessor", "done", "Input normalized and validated")
            yield safe_step_event("Lexical agent", "running")

            # Step 2: lexical done
            await asyncio.sleep(0)
            yield safe_step_event("Lexical agent", "done")
            yield safe_step_event("Semantic agent", "running")

            # Step 3: semantic done
            await asyncio.sleep(0)
            yield safe_step_event("Semantic agent", "done")
            yield safe_step_event("Prosody agent", "running")

            # Step 4: prosody done
            await asyncio.sleep(0)
            yield safe_step_event("Prosody agent", "done")
            yield safe_step_event("Syntax agent", "running")

            # Step 5: syntax done
            await asyncio.sleep(0)
            yield safe_step_event("Syntax agent", "done")
            yield safe_step_event("Biomarker mapper", "running")

            scores_payload = {
                domain: {**score.details, "overall": score.overall}
                for domain, score in state.scores.items()
            }

            yield safe_step_event("Biomarker mapper", "done")
            yield safe_step_event("Report composer", "running")

            risk_level = level_from_overall(state.overall_load, state.confidence)
            summary, model_meta = await compose_safe_summary(state, risk_level)

            report = {
                "summary": summary,
                "risk_level": risk_level,
                "overall_cognitive_load": state.overall_load,
                "highlights": make_highlights(state),
                "risk_indicators": make_indicators(state),
                "recommendation": recommendation_for_level(risk_level, state.confidence),
                "disclaimer": (
                    "This tool is a non-diagnostic screening aid. It can be wrong and must not be used as a standalone "
                    "medical decision system. If you are concerned, consult a qualified clinician."
                ),
                "quality": {
                    "confidence": state.confidence,
                    "notes": state.quality_notes,
                },
                "model_info": model_meta,
            }

            yield safe_step_event("Report composer", "done")

            payload = {
                "type": "end",
                "message": summary,
                "scores": scores_payload,
                "report": report,
                "session_id": session_id,
            }
            yield (json.dumps(payload) + "\n").encode()

        except HTTPException as exc:
            yield (json.dumps({"type": "error", "message": exc.detail}) + "\n").encode()
        except Exception as exc:  # noqa: BLE001
            yield (json.dumps({"type": "error", "message": f"Analysis failed: {str(exc)}"}) + "\n").encode()

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
