# sarvam_client.py — complete fixed version
"""
Sarvam AI API client for STT, Translation, and TTS.

Models used:
    STT         : saaras:v3  
    Translation : mayura:v1
    TTS         : bulbul:v1

Docs: https://docs.sarvam.ai
"""

import os
import base64
import logging
import requests

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_BASE    = "https://api.sarvam.ai"
REQUEST_TIMEOUT = 30  # seconds

if not SARVAM_API_KEY:
    logger.warning("SARVAM_API_KEY not set — Sarvam features will fail")

SUPPORTED_LANGUAGES: dict[str, str] = {
    "Tamil":     "ta-IN",
    "Hindi":     "hi-IN",
    "Telugu":    "te-IN",
    "Kannada":   "kn-IN",
    "Malayalam": "ml-IN",
    "Bengali":   "bn-IN",
    "Gujarati":  "gu-IN",
    "Marathi":   "mr-IN",
    "English":   "en-IN",
}

def _sarvam_headers() -> dict:
    return {"api-subscription-key": SARVAM_API_KEY}


def _safe_json(response: requests.Response) -> dict:
    raw = response.text.strip()
    if not raw:
        raise RuntimeError(
            f"Sarvam API returned HTTP {response.status_code} "
            f"with EMPTY response body. "
            f"This usually means: invalid API key, quota exceeded, "
            f"or unsupported input. URL: {response.url}"
        )
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as e:
        raise RuntimeError(
            f"Sarvam API returned non-JSON body (HTTP {response.status_code}). "
            f"First 200 chars: {raw[:200]}"
        ) from e


def speech_to_text(audio_bytes: bytes, language_code: str) -> str:
    """
    Transcribes audio using Saaras v3 .
    """
    logger.info(f"STT request: language={language_code}, bytes={len(audio_bytes)}")

    response = requests.post(
        f"{SARVAM_BASE}/speech-to-text-translate",
        headers=_sarvam_headers(),
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        data={
            "model":         "saaras:v3",
            "language_code": language_code,
            "mode":          "transcribe",
        },
        timeout=REQUEST_TIMEOUT,
    )

    logger.info(f"STT response: HTTP {response.status_code}")
    response.raise_for_status()
    data = _safe_json(response)

    transcript = data.get("transcript", "").strip()
    logger.info(f"STT transcript: '{transcript[:80]}'")
    return transcript

def _translate(text: str, source: str, target: str) -> str:
    """
    Sarvam mayura has a ~1000 char per request limit.
    """
    if len(text) <= 900:
        chunks = [text]
    else:
        # Split at periods to respect sentence boundaries
        sentences = text.replace(". ", ".|").split("|")
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) <= 900:
                current += sentence + " "
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence + " "
        if current:
            chunks.append(current.strip())

    translated_chunks = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Translating chunk {i+1}/{len(chunks)}: {source}→{target}")

        response = requests.post(
            f"{SARVAM_BASE}/translate",
            headers={
                **_sarvam_headers(),
                "Content-Type": "application/json",
            },
            json={
                "input":                chunk,
                "source_language_code": source,
                "target_language_code": target,
                "model":                "mayura:v1",
            },
            timeout=REQUEST_TIMEOUT,
        )

        logger.info(f"Translate response: HTTP {response.status_code}")
        if not response.ok:
            logger.error(f"Translate error body: {response.text[:300]}")
        response.raise_for_status()
        data = _safe_json(response)

        translated = data.get("translated_text", chunk)
        translated_chunks.append(translated)

    return " ".join(translated_chunks)


def translate_to_english(text: str, language_code: str) -> str:
    """
    Translates text from Indian language to English.
    Returns original text unchanged if language is already English.
    """
    if language_code == "en-IN":
        return text
    return _translate(text, source=language_code, target="en-IN")


def translate_from_english(text: str, language_code: str) -> str:
    """
    Translates English text to target Indian language.
    Returns original text unchanged if target is English.
    """
    if language_code == "en-IN":
        return text
    return _translate(text, source="en-IN", target=language_code)



# Sarvam bulbul hard limit per request
_TTS_CHAR_LIMIT = 500

def text_to_speech(text: str, language_code: str) -> bytes:
    """
    Converts text to speech audio using Sarvam bulbul:v1.

    Truncates to 500 chars (bulbul hard limit).
    Returns WAV audio bytes.

    Raises:
        RuntimeError : on API failure (caller handles this)
    """
    # Truncate at word boundary, not mid-word
    if len(text) > _TTS_CHAR_LIMIT:
        truncated = text[:_TTS_CHAR_LIMIT].rsplit(" ", 1)[0] + "..."
        logger.warning(
            f"TTS input truncated from {len(text)} to {len(truncated)} chars"
        )
    else:
        truncated = text

    logger.info(f"TTS request: language={language_code}, chars={len(truncated)}")

    response = requests.post(
        f"{SARVAM_BASE}/text-to-speech",
        headers={
            **_sarvam_headers(),
            "Content-Type": "application/json",
        },
        json={
            "inputs":               [truncated],
            "target_language_code": language_code,  # NOTE: field name differs from translate
            "model":                "bulbul:v3",
            "speaker":              "shubh",
            "pace":                 0.9,
            "enable_preprocessing": True,
        },
        timeout=REQUEST_TIMEOUT,
    )

    logger.info(f"TTS response: HTTP {response.status_code}")

    # Surface the real error instead of JSONDecodeError
    if not response.ok:
        raw = response.text.strip()[:300]
        raise RuntimeError(
            f"TTS API failed: HTTP {response.status_code}. "
            f"Body: {raw}. "
            f"Common causes: invalid API key, quota exceeded, "
            f"unsupported language+speaker combination."
        )

    data = _safe_json(response)

    audios = data.get("audios")
    if not audios or not audios[0]:
        raise RuntimeError(
            f"TTS API returned success but 'audios' field is empty. "
            f"Full response: {data}"
        )

    audio_bytes = base64.b64decode(audios[0])
    logger.info(f"TTS audio decoded: {len(audio_bytes)} bytes")
    return audio_bytes