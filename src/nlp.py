import re
from collections import Counter


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


def _contains_any(text: str, phrases: set[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _normalize(transcript: str) -> str:
    return re.sub(r"\s+", " ", transcript.strip().lower())


def _build_summary(transcript: str, max_chars: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", transcript).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3].rstrip()}..."


def _detect_sop(normalized: str) -> dict:
    greeting = _contains_any(
        normalized,
        {"good morning", "good afternoon", "good evening", "hello", "hi"},
    )
    identification = _contains_any(
        normalized,
        {
            "my name is",
            "this is",
            "calling from",
            "speaking from",
            "i am from",
        },
    )
    problem_statement = _contains_any(
        normalized,
        {
            "issue",
            "problem",
            "concern",
            "outstanding",
            "due amount",
            "payment pending",
            "delay",
        },
    )
    solution_offering = _contains_any(
        normalized,
        {
            "emi",
            "installment",
            "part payment",
            "partial payment",
            "full payment",
            "down payment",
            "option",
            "plan",
            "we can",
            "you can",
        },
    )
    closing = _contains_any(
        normalized,
        {
            "thank you",
            "have a nice day",
            "have a good day",
            "goodbye",
            "bye",
            "thanks for your time",
        },
    )

    checks = [greeting, identification, problem_statement, solution_offering, closing]
    score = sum(checks) / 5.0
    adherence_status = "FOLLOWED" if score >= 0.8 else "NOT_FOLLOWED"

    explanation = (
        "SOP adherence is based on detection of greeting, identification, "
        "problem statement, solution offering, and closing in the transcript."
    )

    return {
        "greeting": greeting,
        "identification": identification,
        "problemStatement": problem_statement,
        "solutionOffering": solution_offering,
        "closing": closing,
        "complianceScore": round(score, 2),
        "adherenceStatus": adherence_status,
        "explanation": explanation,
    }


def _classify_payment_preference(normalized: str) -> str:
    if _contains_any(normalized, {"emi", "installment", "monthly payment"}):
        return "EMI"
    if _contains_any(normalized, {"full payment", "pay in full", "one shot", "lump sum"}):
        return "FULL_PAYMENT"
    if _contains_any(normalized, {"partial payment", "part payment", "pay some", "half now"}):
        return "PARTIAL_PAYMENT"
    if _contains_any(normalized, {"down payment", "advance payment", "pay advance"}):
        return "DOWN_PAYMENT"
    return "FULL_PAYMENT"


def _classify_rejection_reason(normalized: str) -> str:
    if _contains_any(
        normalized,
        {"high interest", "interest is high", "too much interest", "interest rate"},
    ):
        return "HIGH_INTEREST"
    if _contains_any(
        normalized,
        {"budget", "no money", "financial issue", "can't afford", "cash crunch"},
    ):
        return "BUDGET_CONSTRAINTS"
    if _contains_any(normalized, {"already paid", "payment done", "already settled"}):
        return "ALREADY_PAID"
    if _contains_any(normalized, {"not interested", "don't want", "do not want", "no thanks"}):
        return "NOT_INTERESTED"
    return "NONE"


def _classify_sentiment(normalized: str) -> str:
    positive_markers = {
        "okay",
        "sure",
        "agree",
        "can pay",
        "will pay",
        "thank you",
        "fine",
    }
    negative_markers = {
        "cannot",
        "can't",
        "won't",
        "angry",
        "bad",
        "frustrated",
        "not possible",
        "no",
    }

    pos = sum(1 for marker in positive_markers if marker in normalized)
    neg = sum(1 for marker in negative_markers if marker in normalized)

    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"


def extract_keywords(transcript: str, top_k: int = 10) -> list[str]:
    """Extract top keywords from transcript using frequency-based filtering."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z']{2,}", transcript.lower())
    filtered = [token for token in tokens if token not in _STOPWORDS]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_k)]


def analyze_compliance(transcript: str) -> dict:
    """Analyze transcript and return SOP, analytics, and summary outputs."""
    normalized = _normalize(transcript)
    sop_validation = _detect_sop(normalized)

    analytics = {
        "paymentPreference": _classify_payment_preference(normalized),
        "rejectionReason": _classify_rejection_reason(normalized),
        "sentiment": _classify_sentiment(normalized),
    }

    return {
        "summary": _build_summary(transcript),
        "sop_validation": sop_validation,
        "analytics": analytics,
    }
