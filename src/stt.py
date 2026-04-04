import base64
import binascii
import os
import re
import subprocess
import tempfile

from openai import OpenAI


_DATA_URI_PREFIX_RE = r"^data:audio/[^;]+;base64,"


def _clean_base64_audio(base64_string: str) -> str:
    if not base64_string or not base64_string.strip():
        raise ValueError("Base64 invalid: audio payload is empty")

    cleaned = base64_string.strip().replace("\n", "").replace("\r", "")
    cleaned = re.sub(_DATA_URI_PREFIX_RE, "", cleaned, flags=re.IGNORECASE)

    if not cleaned:
        raise ValueError("Base64 invalid: no payload after cleanup")

    return cleaned


def _decode_base64_audio(cleaned_base64: str) -> bytes:
    try:
        audio_bytes = base64.b64decode(cleaned_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Base64 invalid: cannot decode audio") from exc

    if not audio_bytes:
        raise ValueError("Base64 invalid: decoded audio is empty")

    return audio_bytes


def process_audio(base64_string: str) -> str:
    """Process Base64 MP3 into normalized WAV and transcribe via OpenAI STT.

    Steps:
    1) Clean and validate Base64 input
    2) Save as input.mp3 and verify file size
    3) Convert to 16kHz mono WAV using ffmpeg
    4) Transcribe output.wav with gpt-4o-transcribe (language forced to ta)
    """
    cleaned_base64 = _clean_base64_audio(base64_string)
    audio_bytes = _decode_base64_audio(cleaned_base64)

    base_temp_dir = os.getenv("TEMP_AUDIO_DIR", "temp_audio")
    os.makedirs(base_temp_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="stt_", dir=base_temp_dir) as work_dir:
        input_mp3_path = os.path.join(work_dir, "input.mp3")
        output_wav_path = os.path.join(work_dir, "output.wav")

        with open(input_mp3_path, "wb") as mp3_file:
            mp3_file.write(audio_bytes)

        input_size = os.path.getsize(input_mp3_path)
        print(f"input.mp3 size bytes: {input_size}")
        if input_size <= 0:
            raise ValueError("Base64 invalid: input.mp3 is empty")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_mp3_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            output_wav_path,
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            print("ffmpeg success: converted input.mp3 to output.wav (16kHz mono)")
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg error: ffmpeg executable not found") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(f"ffmpeg error: {stderr or 'conversion failed'}") from exc

        if not os.path.isfile(output_wav_path) or os.path.getsize(output_wav_path) <= 0:
            raise RuntimeError("ffmpeg error: output.wav was not created correctly")

        try:
            client = OpenAI()
            with open(output_wav_path, "rb") as wav_file:
                result = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=wav_file,
                    language="ta",
                )
        except Exception as exc:
            raise RuntimeError(f"STT error: {exc}") from exc

        transcript = (getattr(result, "text", "") or "").strip()
        if not transcript:
            raise RuntimeError("STT error: empty transcript received")

        print(f"transcript output: {transcript}")
        return transcript


def transcribe_audio(audio_path: str, language_code: str = "en") -> str:
    """Legacy wrapper retained for compatibility with old call sites.

    New pipeline expects Base64 input and should call process_audio directly.
    """
    raise RuntimeError("Deprecated API: use process_audio(base64_string) instead")
