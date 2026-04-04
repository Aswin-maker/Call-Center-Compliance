import base64
import os
import uuid

from fastapi import HTTPException


def decode_base64_to_mp3(base64_string: str) -> str:
    """Decode a Base64-encoded MP3 payload and persist it to a unique file.

    The function validates input, applies a basic payload-size limit to reduce
    abuse risk, decodes Base64 safely, creates the target directory when
    missing, and writes the decoded bytes to a UUID-based `.mp3` file.

    Args:
        base64_string: The Base64 audio string from the request body.

    Returns:
        The filesystem path to the saved MP3 file.

    Raises:
        HTTPException: 400 if input is empty/invalid or exceeds size limits.
        HTTPException: 500 if file persistence fails.
    """
    if not base64_string or not base64_string.strip():
        raise HTTPException(status_code=400, detail="audioBase64 must not be empty")

    # Optional support for data URL format:
    # data:audio/mpeg;base64,<payload>
    sanitized = base64_string.strip()
    if "," in sanitized and sanitized.lower().startswith("data:"):
        sanitized = sanitized.split(",", 1)[1]

    # Basic abuse guard. Can be overridden via environment variable.
    max_base64_chars = int(os.getenv("MAX_AUDIO_BASE64_CHARS", "10000000"))
    if len(sanitized) > max_base64_chars:
        raise HTTPException(status_code=400, detail="audioBase64 payload is too large")

    # Approximate decoded size check before decoding to reduce memory pressure.
    approx_decoded_size = (len(sanitized) * 3) // 4
    max_audio_bytes = int(os.getenv("MAX_AUDIO_BYTES", str(8 * 1024 * 1024)))
    if approx_decoded_size > max_audio_bytes:
        raise HTTPException(status_code=413, detail="Decoded audio payload is too large")

    audio_dir = os.getenv("TEMP_AUDIO_DIR", "temp_audio")

    try:
        os.makedirs(audio_dir, exist_ok=True)
        filename = f"audio_{uuid.uuid4()}.mp3"
        file_path = os.path.join(audio_dir, filename)

        # Decode in chunks to avoid holding full decoded bytes in memory.
        # Must decode in 4-character aligned chunks for valid Base64 blocks.
        decoded_total = 0
        chunk_chars = 4 * 4096

        with open(file_path, "wb") as audio_file:
            for start in range(0, len(sanitized), chunk_chars):
                chunk = sanitized[start : start + chunk_chars]
                try:
                    decoded_chunk = base64.b64decode(chunk, validate=True)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail="Invalid Base64 audio payload") from exc

                if decoded_chunk:
                    decoded_total += len(decoded_chunk)
                    if decoded_total > max_audio_bytes:
                        raise HTTPException(status_code=413, detail="Decoded audio payload is too large")
                    audio_file.write(decoded_chunk)

        if decoded_total == 0:
            raise HTTPException(status_code=400, detail="Decoded audio payload is empty")
    except HTTPException:
        if "file_path" in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        raise
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to save decoded audio file") from exc

    return file_path
