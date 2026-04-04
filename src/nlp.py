import re
import logging
from collections import Counter
from threading import Lock

from keybert import KeyBERT
from transformers import pipeline


_SUMMARIZER = None
_SENTIMENT = None
_KEYBERT = None
_MODEL_LOCK = Lock()
_LOGGER = logging.getLogger(__name__)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "i",
    "you",
    "we",
    "they",
    "this",
    "your",
    "our",
    "my",
    "me",
    "can",
    "could",
    "would",
    "should",
    "please",
    "sir",
    "madam",
    "hello",
    "hi",
}

_SUMMARY_MIN_WORDS = 20
_KEYWORD_MIN_COUNT = 5
_UNCLEAR_SUMMARY = "The conversation content is unclear or limited"

_INVALID_MESSAGE = "Invalid or low-quality transcription"
_PAYMENT_TERMS = {
    "payment",
    "pay",
    "emi",
    "installment",
    "partial",
    "full payment",
    "down payment",
    "due",
    "outstanding",
    "baki",
    "settlement",
}
_CALL_CONTEXT_TERMS = {
    "payment",
    "emi",
    "amount",
    "loan",
    "rupees",
    "rs",
    "due",
    "pending",
    "outstanding",
    "installment",
    "part payment",
    "full payment",
    "down payment",
    "interest",
    "budget",
    "settlement",
    "customer",
    "account",
    "plan",
    "today",
    "tomorrow",
    "panam",
    "kudunga",
    "paisa",
    "kist",
    "thavanai",
}
_FILLER_WORDS = {
    "um",
    "uh",
    "hmm",
    "mmm",
    "ah",
    "okay",
    "ok",
    "haan",
    "hmmmmm",
}

_BUSINESS_TERMS = {
    "payment",
    "emi",
    "installment",
    "partial payment",
    "full payment",
    "down payment",
    "due",
    "overdue",
    "outstanding",
    "settlement",
    "interest",
    "budget",
    "account",
    "plan",
    "loan",
    "amount",
    "customer",
    "agent",
    "collection",
    "pending",
    "panam",
    "kudunga",
    "paisa",
    "kist",
    "thavanai",
}

_EXPLICIT_OUTCOME_TERMS = {
    "will pay",
    "i will pay",
    "pay today",
    "pay tomorrow",
    "already paid",
    "payment done",
    "agreed",
    "agree",
    "accepted",
    "declined",
    "not interested",
}

_VALID_PAYMENT_ENUMS = {"EMI", "FULL_PAYMENT", "PARTIAL_PAYMENT", "DOWN_PAYMENT"}
_VALID_REJECTION_ENUMS = {
    "HIGH_INTEREST",
    "BUDGET_CONSTRAINTS",
    "ALREADY_PAID",
    "NOT_INTERESTED",
    "NONE",
}
_VALID_SENTIMENT_ENUMS = {"Positive", "Neutral", "Negative"}


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z']{2,}|[\u0B80-\u0BFF]{2,}|[\u0600-\u06FF]{2,}", text or "")


def _keyword_candidate_phrases(text: str, limit: int = 10) -> list[str]:
    chunks = re.split(r"[\n\r\.\!\?;:,]+", text or "")
    candidates: list[str] = []
    for chunk in chunks:
        words = [word for word in _tokenize_words(chunk) if word.lower() not in _STOPWORDS]
        if len(words) >= 2:
            phrase = " ".join(words[:3]).strip()
            if phrase and phrase not in candidates:
                candidates.append(phrase)
        for word in words:
            lowered = word.lower().strip()
            if lowered and lowered not in candidates and lowered not in _STOPWORDS:
                candidates.append(lowered)
            if len(candidates) >= limit:
                return candidates[:limit]
    return candidates[:limit]


def _fallback_keywords(text: str, existing: list[str]) -> list[str]:
    keywords = list(existing)
    lowered_existing = {keyword.lower() for keyword in keywords}

    tokens = [token.lower() for token in _tokenize_words(text) if token.lower() not in _STOPWORDS]
    for token, _count in Counter(tokens).most_common():
        if token not in lowered_existing:
            keywords.append(token)
            lowered_existing.add(token)
        if len(keywords) >= _KEYWORD_MIN_COUNT:
            return keywords[:10]

    for phrase in _keyword_candidate_phrases(text, limit=10):
        lowered = phrase.lower()
        if lowered not in lowered_existing:
            keywords.append(phrase)
            lowered_existing.add(lowered)
        if len(keywords) >= _KEYWORD_MIN_COUNT:
            break

    for chunk in re.split(r"\s+", text or ""):
        clean = re.sub(r"[^\w\u0B80-\u0BFF\u0600-\u06FF]+", "", chunk.lower())
        if len(clean) > 3 and clean not in lowered_existing and clean not in _STOPWORDS:
            keywords.append(clean)
            lowered_existing.add(clean)
        if len(keywords) >= _KEYWORD_MIN_COUNT:
            break

    return keywords[:10]


def _normalize(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "")
    return cleaned.strip().lower()


def _contains_any(text: str, phrases: set[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def clean_transcript(text: str) -> str:
    """Clean transcript by removing loops/noise while preserving meaning."""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return ""

    tokens = re.findall(r"[\w\u0B80-\u0BFF\u0600-\u06FF\u0900-\u097F']+|[\.,!?]", normalized, flags=re.UNICODE)
    if not tokens:
        return ""

    cleaned_tokens: list[str] = []
    last_word = ""
    run_length = 0

    for token in tokens:
        if re.fullmatch(r"[\.,!?]", token):
            if cleaned_tokens and cleaned_tokens[-1] != token:
                cleaned_tokens.append(token)
            last_word = ""
            run_length = 0
            continue

        lowered = token.lower()
        if lowered == last_word:
            run_length += 1
        else:
            last_word = lowered
            run_length = 1

        # Keep at most two consecutive occurrences for filler words, one for others.
        max_allowed = 2 if lowered in _FILLER_WORDS else 1
        if run_length <= max_allowed:
            cleaned_tokens.append(token)

    cleaned = " ".join(cleaned_tokens)
    cleaned = re.sub(r"\s+([\.,!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,!?")
    return cleaned


def is_invalid_transcript(text: str) -> bool:
    """Detect empty, repetitive, or meaningless transcripts."""
    cleaned = clean_transcript(text)
    if not cleaned:
        return True

    tokens = re.findall(r"[\w\u0B80-\u0BFF\u0600-\u06FF\u0900-\u097F']+", cleaned, flags=re.UNICODE)
    if not tokens:
        return True

    lowered = [token.lower() for token in tokens]
    lowered_text = " ".join(lowered)
    has_call_context = any(term in lowered_text for term in _CALL_CONTEXT_TERMS)

    # Any business context should be accepted even if short/informal.
    if has_call_context and len(tokens) >= 2:
        return False

    if len(tokens) < 4:
        return True

    # Keep repetitive transcripts valid if clear business/call context exists.
    if has_call_context:
        return False

    unique_ratio = len(set(lowered)) / max(len(lowered), 1)
    if len(lowered) >= 8 and unique_ratio < 0.35:
        return True

    counts = Counter(lowered)
    top_freq = counts.most_common(1)[0][1]
    if len(lowered) >= 10 and (top_freq / len(lowered)) > 0.45:
        return True

    # Repeating phrase loop detection, e.g., "kalum kalum kalum".
    if len(lowered) >= 6:
        bigrams = [f"{lowered[i]} {lowered[i + 1]}" for i in range(len(lowered) - 1)]
        bigram_freq = Counter(bigrams).most_common(1)[0][1]
        if (bigram_freq / max(len(bigrams), 1)) > 0.4:
            return True

    # No business context + weak lexical variety -> likely non-conversational noise.
    if len(set(lowered)) <= 3 and len(lowered) >= 6:
        return True

    return False


def detect_language(text: str) -> str:
    """Detect broad language bucket for response output."""
    sample = text or ""
    tamil_chars = sum(1 for ch in sample if "\u0B80" <= ch <= "\u0BFF")
    devanagari_chars = sum(1 for ch in sample if "\u0900" <= ch <= "\u097F")

    if tamil_chars >= 3:
        return "Tamil"
    if devanagari_chars >= 3:
        return "Hindi"

    lowered = sample.lower()
    latin_tokens = set(re.findall(r"[a-z']+", lowered))

    tanglish_tokens = {"enna", "seri", "vanakkam", "unga", "panam", "vendam"}
    hinglish_tokens = {
        "namaste",
        "haan",
        "nahi",
        "paisa",
        "kal",
        "aaj",
        "thik",
        "aap",
        "aapka",
        "kya",
        "hai",
    }

    if any(token in latin_tokens for token in hinglish_tokens) or "kar sakte" in lowered:
        return "Hinglish"
    if any(token in latin_tokens for token in tanglish_tokens):
        return "Tanglish"
    return "English"


def _get_summarizer():
    global _SUMMARIZER
    if _SUMMARIZER is None:
        with _MODEL_LOCK:
            if _SUMMARIZER is None:
                _SUMMARIZER = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",
                )
    return _SUMMARIZER


def _get_sentiment_model():
    global _SENTIMENT
    if _SENTIMENT is None:
        with _MODEL_LOCK:
            if _SENTIMENT is None:
                _SENTIMENT = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                )
    return _SENTIMENT


def _get_keybert_model():
    global _KEYBERT
    if _KEYBERT is None:
        with _MODEL_LOCK:
            if _KEYBERT is None:
                _KEYBERT = KeyBERT(model="all-MiniLM-L6-v2")
    return _KEYBERT


def _safe_extractive_summary(text: str) -> str:
    """Extract top sentences for summary when transformer model is unavailable."""
    normalized = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    
    # Filter empty and very short sentences
    valid_sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 3]
    
    # Return up to 3 sentences or full text if shorter
    if valid_sentences:
        summary = " ".join(valid_sentences[:3])
    else:
        summary = normalized[:240]

    summary = summary[:500].strip() if summary else normalized[:180]
    if len(summary.split()) < 12 and normalized:
        summary = normalized[:300].strip()
    return summary


def _ensure_useful_summary(text: str, summary: str) -> str:
    clean_text = re.sub(r"\s+", " ", text or "").strip()
    clean_summary = re.sub(r"\s+", " ", summary or "").strip()

    if not clean_summary or len(clean_summary.split()) < 12:
        return _safe_extractive_summary(clean_text)

    if len(clean_summary) < 60 and len(clean_text) > 120:
        fallback = _safe_extractive_summary(clean_text)
        if len(fallback.split()) >= len(clean_summary.split()):
            return fallback

    return clean_summary


def summarize_text(text: str) -> str:
    """Generate concise 2-3 line summary while preserving business context."""
    if not text or not text.strip():
        return ""

    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean.split()) < _SUMMARY_MIN_WORDS:
        return _ensure_useful_summary(clean, _safe_extractive_summary(clean))

    try:
        summarizer = _get_summarizer()
        # Use adaptive length based on text
        min_len = max(28, len(clean.split()) // 10)
        max_len = max(90, len(clean.split()) // 5)
        result = summarizer(
            clean,
            max_length=min(max_len, 130),
            min_length=min_len,
            do_sample=False,
            truncation=True,
        )
        summary = (result[0].get("summary_text", "") if result else "").strip()
        return _ensure_useful_summary(clean, summary)
    except Exception as e:
        _LOGGER.debug("Summarizer failed: %s, using extractive", e)
        return _ensure_useful_summary(clean, _safe_extractive_summary(clean))


def analyze_sop(text: str) -> dict:
    """Strict SOP detection using deterministic keyword/pattern rules.
    
    Detects presence of 5 key SOP components:
    1. Greeting - welcoming customer
    2. Identification - agent identifies self/company
    3. Problem Statement - explains issue/reason for call
    4. Solution Offering - proposes payment/solution options
    5. Closing - thanks customer and ends professionally
    """
    normalized = _normalize(text)

    # Greeting patterns
    greeting = _contains_any(
        normalized,
        {
            "hello",
            "hi",
            "good morning",
            "good afternoon",
            "good evening",
            "welcome",
            "vanakkam",
            "வணக்கம்",
            "namaste",
            "namaskar",
            "how are you",
        },
    )

    # Identification patterns
    identification = _contains_any(
        normalized,
        {
            "my name is",
            "this is",
            "i am calling from",
            "calling from",
            "from collection team",
            "from customer care",
            "pesurathu",
            "speaking from",
            "representative",
            "behalf of",
            "customer service",
        },
    )

    # Problem/Issue statement patterns
    problem_statement = _contains_any(
        normalized,
        {
            "due",
            "overdue",
            "outstanding",
            "pending payment",
            "pending amount",
            "issue",
            "problem",
            "delay in payment",
            "late payment",
            "baki",
            "outstanding balance",
            "dues",
            "unpaid",
            "amount due",
            "regarding your",
            "regarding the",
        },
    )

    # Solution/Offer patterns
    solution_offering = _contains_any(
        normalized,
        {
            "emi",
            "installment",
            "part payment",
            "partial payment",
            "full payment",
            "down payment",
            "payment option",
            "payment plan",
            "we can offer",
            "you can pay",
            "i can offer",
            "flexible",
            "arrangement",
            "payment arrangement",
            "settlement",
        },
    )

    # Closing/Polite end patterns
    closing = _contains_any(
        normalized,
        {
            "thank you",
            "thanks",
            "have a nice day",
            "have a good day",
            "goodbye",
            "bye",
            "nandri",
            "நன்றி",
            "appreciate",
            "kind regards",
            "take care",
            "talk to you",
        },
    )

    checks = [greeting, identification, problem_statement, solution_offering, closing]
    steps_present = sum(checks)
    score = round(steps_present / 5.0, 2)
    adherence_status = "FOLLOWED" if steps_present >= 4 else "NOT_FOLLOWED"

    missing = []
    if not greeting:
        missing.append("greeting")
    if not identification:
        missing.append("identification")
    if not problem_statement:
        missing.append("problemStatement")
    if not solution_offering:
        missing.append("solutionOffering")
    if not closing:
        missing.append("closing")

    if missing:
        explanation = f"{steps_present}/5 SOP steps present; missing: {', '.join(missing)}"
    else:
        explanation = "5/5 SOP steps present"

    return {
        "greeting": greeting,
        "identification": identification,
        "problemStatement": problem_statement,
        "solutionOffering": solution_offering,
        "closing": closing,
        "complianceScore": score,
        "adherenceStatus": adherence_status,
        "explanation": explanation,
    }


def classify_payment(text: str) -> str:
    """Classify payment intent into fixed enums using deterministic rules."""
    normalized = _normalize(text)

    if not _contains_any(normalized, _PAYMENT_TERMS):
        # Safest valid fallback when preference is unclear.
        return "FULL_PAYMENT"

    if _contains_any(normalized, {"emi", "monthly", "installment", "every month"}):
        return "EMI"
    if _contains_any(
        normalized,
        {"partial", "part payment", "pay some", "half payment", "konjam pay"},
    ):
        return "PARTIAL_PAYMENT"
    if (
        re.search(r"pay\s+\d+", normalized)
        and _contains_any(normalized, {"later", "remaining", "rest", "next month"})
    ):
        return "PARTIAL_PAYMENT"
    if _contains_any(
        normalized,
        {"down payment", "advance payment", "initial payment", "booking amount"},
    ):
        return "DOWN_PAYMENT"
    if _contains_any(
        normalized,
        {"full payment", "pay full", "one shot", "lump sum", "settle full"},
    ):
        return "FULL_PAYMENT"

    return "FULL_PAYMENT"


def detect_rejection_reason(text: str) -> str:
    """Detect rejection reason using strict enum-based keyword logic."""
    normalized = _normalize(text)

    if _contains_any(
        normalized,
        {
            "high interest",
            "too much interest",
            "interest is high",
            "interest rate high",
            "vatti jasthi",
        },
    ):
        return "HIGH_INTEREST"

    if _contains_any(
        normalized,
        {
            "budget issue",
            "budget problem",
            "no money",
            "financial problem",
            "cannot afford",
            "cash issue",
            "panam illa",
        },
    ):
        return "BUDGET_CONSTRAINTS"

    if _contains_any(
        normalized,
        {
            "already paid",
            "payment done",
            "already settled",
            "already closed",
            "naan already paid",
            "pehle hi pay kiya",
        },
    ):
        return "ALREADY_PAID"

    if _contains_any(
        normalized,
        {
            "not interested",
            "don't want",
            "do not want",
            "not willing",
            "vendam",
            "வேண்டாம்",
        },
    ):
        return "NOT_INTERESTED"

    if not _contains_any(normalized, _PAYMENT_TERMS):
        return "NONE"

    return "NONE"


def detect_sentiment(text: str) -> str:
    """Detect sentiment using HF model and normalize to Positive/Negative/Neutral."""
    normalized = _normalize(text)
    if not normalized:
        return "Neutral"

    try:
        sentiment_model = _get_sentiment_model()
        result = sentiment_model(normalized[:1000], truncation=True)
        label = (result[0].get("label", "") if result else "").strip().upper()

        label_map = {
            "POSITIVE": "Positive",
            "NEGATIVE": "Negative",
            "NEUTRAL": "Neutral",
            "LABEL_2": "Positive",
            "LABEL_1": "Neutral",
            "LABEL_0": "Negative",
        }
        return label_map.get(label, "Neutral")
    except Exception:
        return "Neutral"


def extract_keywords(text: str) -> list[str]:
    """Extract top 8-10 meaningful keywords present in transcript text.
    
    Uses multiple strategies:
    1. KeyBERT semantic extraction with fallback
    2. Frequency-based extraction from non-stopwords
    3. Ensures keywords actually exist in original text
    """
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []

    keywords = []
    
    # Strategy 1: Try KeyBERT extraction
    try:
        kw_model = _get_keybert_model()
        candidates = kw_model.extract_keywords(
            normalized,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            use_mmr=True,
            diversity=0.5,
            top_n=15,
        )
        lowered = normalized.lower()
        for phrase, score in candidates:
            phrase_clean = phrase.strip()
            if not phrase_clean:
                continue
            if phrase_clean.lower() in lowered and phrase_clean not in keywords:
                keywords.append(phrase_clean)
            if len(keywords) >= 10:
                break
    except Exception as e:
        _LOGGER.debug("KeyBERT extraction failed: %s", e)

    keywords = _fallback_keywords(normalized, keywords)

    # Prioritize business-relevant terms that appear in transcript/summary.
    lowered_text = normalized.lower()
    prioritized = [term for term in _BUSINESS_TERMS if term in lowered_text]
    ordered: list[str] = []
    for term in prioritized + keywords:
        if term not in ordered:
            ordered.append(term)
    keywords = ordered

    # Keep only meaningful business-relevant phrases.
    filtered: list[str] = []
    for kw in keywords:
        lowered_kw = kw.lower().strip()
        if not lowered_kw:
            continue
        if lowered_kw in _STOPWORDS:
            continue
        if len(lowered_kw) < 3:
            continue
        if lowered_kw in _BUSINESS_TERMS or any(term in lowered_kw for term in _BUSINESS_TERMS):
            if lowered_kw not in {k.lower() for k in filtered}:
                filtered.append(kw)

    keywords = filtered

    # Ensure a minimum of 5 useful keywords whenever possible.
    if len(keywords) < _KEYWORD_MIN_COUNT:
        lowered_existing = {k.lower() for k in keywords}
        source_tokens = [
            token.lower()
            for token in re.findall(r"[\w\u0B80-\u0BFF\u0600-\u06FF\u0900-\u097F']+", normalized, flags=re.UNICODE)
            if len(token) >= 3 and token.lower() not in _STOPWORDS
        ]
        for token in source_tokens:
            if token in lowered_existing:
                continue
            if token in _BUSINESS_TERMS or any(term in token for term in _BUSINESS_TERMS):
                keywords.append(token)
                lowered_existing.add(token)
            if len(keywords) >= _KEYWORD_MIN_COUNT:
                break

    if len(keywords) < _KEYWORD_MIN_COUNT:
        fallback = _fallback_keywords(normalized, keywords)
        lowered_existing = {k.lower() for k in keywords}
        for kw in fallback:
            lowered_kw = kw.lower().strip()
            if lowered_kw in lowered_existing or lowered_kw in _STOPWORDS or len(lowered_kw) < 3:
                continue
            if lowered_kw in _BUSINESS_TERMS or any(term in lowered_kw for term in _BUSINESS_TERMS):
                keywords.append(kw)
                lowered_existing.add(lowered_kw)
            if len(keywords) >= _KEYWORD_MIN_COUNT:
                break

    # Last fallback: add meaningful transcript tokens (still from transcript only).
    if len(keywords) < _KEYWORD_MIN_COUNT:
        lowered_existing = {k.lower() for k in keywords}
        all_tokens = [
            token.lower()
            for token in re.findall(r"[\w\u0B80-\u0BFF\u0600-\u06FF\u0900-\u097F']+", normalized, flags=re.UNICODE)
            if len(token) >= 3 and token.lower() not in _STOPWORDS
        ]
        for token, _count in Counter(all_tokens).most_common():
            if token in lowered_existing:
                continue
            keywords.append(token)
            lowered_existing.add(token)
            if len(keywords) >= _KEYWORD_MIN_COUNT:
                break

    return keywords[:10]


def _build_summary(text: str, payment_preference: str, rejection_reason: str) -> str:
    """Build concise factual summary from transcript evidence only."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned or is_invalid_transcript(cleaned):
        return _UNCLEAR_SUMMARY

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    key_points = " ".join(sentences[:2]) if sentences else cleaned[:220]

    lowered = cleaned.lower()
    purpose = "Purpose: not explicitly stated"
    for marker, statement in [
        ("outstanding", "Purpose: discussion on outstanding payment"),
        ("due", "Purpose: discussion on due amount"),
        ("pending", "Purpose: discussion on pending payment"),
        ("payment", "Purpose: payment-related discussion"),
    ]:
        if marker in lowered:
            purpose = statement
            break

    rejection_phrase_map = {
        "HIGH_INTEREST": "high interest",
        "BUDGET_CONSTRAINTS": "budget constraints",
        "ALREADY_PAID": "already paid",
        "NOT_INTERESTED": "not interested",
        "NONE": "no explicit rejection",
    }

    explicit_outcome = any(term in lowered for term in _EXPLICIT_OUTCOME_TERMS)
    if rejection_reason != "NONE" and explicit_outcome:
        outcome = f"Outcome: customer indicated {rejection_phrase_map[rejection_reason]}."
    elif explicit_outcome:
        if payment_preference == "EMI":
            outcome = "Outcome: customer explicitly discussed EMI plan."
        elif payment_preference == "PARTIAL_PAYMENT":
            outcome = "Outcome: customer explicitly discussed partial payment."
        elif payment_preference == "DOWN_PAYMENT":
            outcome = "Outcome: customer explicitly discussed down payment."
        else:
            outcome = "Outcome: explicit payment intent was mentioned."
    else:
        outcome = "Outcome: not explicitly stated in transcript."

    key_points = key_points.rstrip(" .")
    return f"{purpose}. Key points: {key_points}. {outcome}".strip()


def _final_validate_success_payload(payload: dict) -> dict:
    """Enforce strict output contract for success responses."""
    normalized_payload = dict(payload or {})
    normalized_payload["status"] = "success"

    normalized_payload.setdefault("language", "English")
    normalized_payload["language"] = str(normalized_payload.get("language") or "English")

    normalized_payload.setdefault("transcript", "")
    normalized_payload["transcript"] = str(normalized_payload.get("transcript") or "")

    normalized_payload.setdefault("summary", _UNCLEAR_SUMMARY)
    normalized_payload["summary"] = str(normalized_payload.get("summary") or _UNCLEAR_SUMMARY)

    transcript = normalized_payload["transcript"]
    lowered_transcript = transcript.lower()
    keywords = normalized_payload.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []

    # Keep keywords meaningful and sourced from transcript text.
    clean_keywords: list[str] = []
    seen = set()
    for kw in keywords:
        if not isinstance(kw, str):
            continue
        norm = kw.strip().lower()
        if not norm or norm in seen or len(norm) < 3:
            continue
        if norm in _STOPWORDS:
            continue
        if norm not in lowered_transcript:
            continue
        seen.add(norm)
        clean_keywords.append(kw.strip())
        if len(clean_keywords) >= 10:
            break

    normalized_payload["keywords"] = clean_keywords[:10]

    sop_validation = normalized_payload.get("sop_validation") or {}
    if not isinstance(sop_validation, dict):
        sop_validation = {}
    normalized_payload["sop_validation"] = {
        "greeting": bool(sop_validation.get("greeting", False)),
        "identification": bool(sop_validation.get("identification", False)),
        "problemStatement": bool(sop_validation.get("problemStatement", False)),
        "solutionOffering": bool(sop_validation.get("solutionOffering", False)),
        "closing": bool(sop_validation.get("closing", False)),
        "complianceScore": float(sop_validation.get("complianceScore", 0.0) or 0.0),
        "adherenceStatus": sop_validation.get("adherenceStatus", "NOT_FOLLOWED")
        if sop_validation.get("adherenceStatus") in {"FOLLOWED", "NOT_FOLLOWED"}
        else "NOT_FOLLOWED",
        "explanation": str(sop_validation.get("explanation") or _UNCLEAR_SUMMARY),
    }

    analytics = normalized_payload.get("analytics") or {}
    if not isinstance(analytics, dict):
        analytics = {}
    payment_preference = analytics.get("paymentPreference")
    if payment_preference not in _VALID_PAYMENT_ENUMS:
        payment_preference = "FULL_PAYMENT"
    rejection_reason = analytics.get("rejectionReason")
    if rejection_reason not in _VALID_REJECTION_ENUMS:
        rejection_reason = "NONE"
    sentiment = analytics.get("sentiment")
    if sentiment not in _VALID_SENTIMENT_ENUMS:
        sentiment = "Neutral"

    normalized_payload["analytics"] = {
        "paymentPreference": payment_preference,
        "rejectionReason": rejection_reason,
        "sentiment": sentiment,
    }

    return normalized_payload


def build_compliance_json(text: str) -> dict:
    """Return strict success JSON payload for compliance engine output."""
    cleaned = clean_transcript(text)
    # Deterministic priority order: validate -> clean -> SOP -> analytics -> summary -> keywords.
    sop = analyze_sop(cleaned)
    payment_preference = classify_payment(cleaned)
    rejection_reason = detect_rejection_reason(cleaned)
    sentiment = detect_sentiment(cleaned)
    summary = _build_summary(cleaned, payment_preference, rejection_reason)
    keywords = extract_keywords(cleaned)

    result = {
        "status": "success",
        "language": detect_language(cleaned or text),
        "transcript": cleaned,
        "summary": summary,
        "sop_validation": sop,
        "analytics": {
            "paymentPreference": payment_preference,
            "rejectionReason": rejection_reason,
            "sentiment": sentiment,
        },
        "keywords": keywords,
    }
    return _final_validate_success_payload(result)


def analyze_compliance(text: str) -> dict:
    """Backward-compatible wrapper for strict compliance JSON builder."""
    return build_compliance_json(text)
