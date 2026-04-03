import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from nlp import analyze_compliance, extract_keywords
from stt import transcribe_audio
from utils import decode_base64_to_mp3

load_dotenv()

API_KEY = os.getenv("API_KEY")

app = FastAPI(
    title="Call Center Compliance API",
    version="0.1.0",
    description="FastAPI scaffold for call center compliance workflows.",
)


class CallAnalyticsRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


def verify_api_key(x_api_key: str = Header(default="")) -> None:
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/api/call-analytics", dependencies=[Depends(verify_api_key)])
def call_analytics(payload: CallAnalyticsRequest) -> dict:
    if payload.audioFormat.strip().lower() != "mp3":
        raise HTTPException(status_code=400, detail="audioFormat must be mp3")

    try:
        audio_file_path = decode_base64_to_mp3(payload.audioBase64)
        transcript = transcribe_audio(audio_file_path, payload.language)
        compliance = analyze_compliance(transcript)
        keywords = extract_keywords(transcript)

        return {
            "status": "success",
            "language": payload.language,
            "transcript": transcript,
            "summary": compliance["summary"],
            "sop_validation": compliance["sop_validation"],
            "analytics": compliance["analytics"],
            "keywords": keywords,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal processing failure") from exc
