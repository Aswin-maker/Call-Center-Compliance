import os
from typing import Optional

import whisper


_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        model_name = os.getenv("WHISPER_MODEL", "base")
        _MODEL = whisper.load_model(model_name)
    return _MODEL


def _resolve_whisper_language(language: Optional[str]) -> Optional[str]:
    if not language:
        return None

    language_map = {
        "tamil": "ta",
        "english": "en",
        "hindi": "hi",
    }
    return language_map.get(language.strip().lower())


def transcribe_audio(audio_file_path: str, language: Optional[str] = None) -> str:
    """Transcribe an audio file using local Whisper and return plain text."""
    if not audio_file_path or not os.path.exists(audio_file_path):
        raise FileNotFoundError("Audio file not found for transcription")

    model = _get_model()
    whisper_language = _resolve_whisper_language(language)

    result = model.transcribe(audio_file_path, language=whisper_language)
    text = (result or {}).get("text", "").strip()

    if not text:
        raise RuntimeError("Transcription produced empty output")

    return text
