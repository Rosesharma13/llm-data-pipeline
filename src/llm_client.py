"""
llm_client.py — LLM API integration module
Uses Groq API (gemma2-9b-it) with retry logic and exponential backoff
Handles malformed JSON responses gracefully
Author: Rose Sharma

Why Groq?
- Free tier with generous rate limits
- Fast inference (fastest LLM API available)
- Supports gemma2-9b-it — good balance of speed and quality
- No credit card required for free tier
"""

import os
import json
import re
import logging
import time
from groq import Groq, APIError, RateLimitError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)

logger = logging.getLogger(__name__)

# ─── Groq client setup ───────────────────────────────────────────────────────
def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable not set. "
            "Get your free key at console.groq.com"
        )
    return Groq(api_key=api_key)


# ─── System prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise text analysis assistant. 
For every text provided, extract structured information and return ONLY valid JSON.
Do not include any explanation, markdown, or text outside the JSON object.
Always return exactly this structure:
{
  "summary": "2-3 sentence summary of the text",
  "entities": {
    "people": ["name1", "name2"],
    "places": ["place1", "place2"],
    "organizations": ["org1", "org2"]
  },
  "sentiment": {
    "label": "positive|neutral|negative",
    "confidence": 0.85
  },
  "questions": [
    "Question 1 the text raises?",
    "Question 2 the text raises?",
    "Question 3 the text raises?"
  ]
}"""


# ─── JSON extraction ─────────────────────────────────────────────────────────
def extract_json(text: str) -> dict:
    """
    Robustly extract JSON from LLM response.
    Handles: pure JSON, JSON in markdown blocks, JSON buried in text.
    """
    if not text or not text.strip():
        raise ValueError("Empty response from LLM")

    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try fixing common JSON issues (trailing commas)
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}...")


def validate_result(result: dict) -> dict:
    """Ensure all required fields are present with correct types."""
    # Ensure summary
    if not isinstance(result.get("summary"), str):
        result["summary"] = "Summary not available"

    # Ensure entities
    if not isinstance(result.get("entities"), dict):
        result["entities"] = {"people": [], "places": [], "organizations": []}
    for key in ["people", "places", "organizations"]:
        if not isinstance(result["entities"].get(key), list):
            result["entities"][key] = []

    # Ensure sentiment
    if not isinstance(result.get("sentiment"), dict):
        result["sentiment"] = {"label": "neutral", "confidence": 0.5}
    if result["sentiment"].get("label") not in ["positive", "neutral", "negative"]:
        result["sentiment"]["label"] = "neutral"
    if not isinstance(result["sentiment"].get("confidence"), (int, float)):
        result["sentiment"]["confidence"] = 0.5

    # Ensure questions
    if not isinstance(result.get("questions"), list):
        result["questions"] = ["No questions extracted"]
    result["questions"] = result["questions"][:3]  # Max 3

    return result


# ─── LLM call with retry ─────────────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=False
)
def _call_groq(client: Groq, text: str) -> str:
    """Make a single Groq API call with retry on failure."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this text:\n\n{text}"}
        ],
        temperature=0.1,
        max_tokens=800,
        timeout=30
    )
    return response.choices[0].message.content


# ─── Main analysis function ───────────────────────────────────────────────────
def analyze_chunk(client: Groq, chunk: dict) -> dict:
    """
    Analyze a single chunk using Groq LLM.
    Returns enriched chunk dict with LLM results.
    Never crashes — logs errors and returns error state.
    """
    source = chunk["source"]
    idx = chunk["chunk_index"]

    try:
        logger.info(f"Analyzing chunk {idx}/{chunk['total_chunks']} from: {source}")

        raw_response = _call_groq(client, chunk["text"])
        result = extract_json(raw_response)
        result = validate_result(result)

        logger.info(f"✅ Chunk {idx} analyzed — sentiment: {result['sentiment']['label']}")

        return {
            **chunk,
            "summary": result["summary"],
            "entities_people": ", ".join(result["entities"]["people"]),
            "entities_places": ", ".join(result["entities"]["places"]),
            "entities_orgs": ", ".join(result["entities"]["organizations"]),
            "sentiment_label": result["sentiment"]["label"],
            "sentiment_confidence": result["sentiment"]["confidence"],
            "questions": result["questions"],
            "status": "success",
            "error": None
        }

    except RetryError as e:
        logger.error(f"❌ All retries exhausted for chunk {idx} from {source}: {e}")
        return {**chunk, "status": "failed", "error": "Max retries exceeded", "summary": "", "entities_people": "", "entities_places": "", "entities_orgs": "", "sentiment_label": "unknown", "sentiment_confidence": 0.0, "questions": []}

    except ValueError as e:
        logger.error(f"❌ JSON parsing failed for chunk {idx} from {source}: {e}")
        return {**chunk, "status": "failed", "error": f"JSON parse error: {str(e)}", "summary": "", "entities_people": "", "entities_places": "", "entities_orgs": "", "sentiment_label": "unknown", "sentiment_confidence": 0.0, "questions": []}

    except Exception as e:
        logger.error(f"❌ Unexpected error for chunk {idx} from {source}: {e}")
        return {**chunk, "status": "failed", "error": str(e), "summary": "", "entities_people": "", "entities_places": "", "entities_orgs": "", "sentiment_label": "unknown", "sentiment_confidence": 0.0, "questions": []}


def analyze_all(chunks: list) -> list:
    """Analyze all chunks. Continues even if individual chunks fail."""
    client = get_client()
    results = []

    for i, chunk in enumerate(chunks):
        result = analyze_chunk(client, chunk)
        results.append(result)
        # Small delay to respect rate limits
        if i < len(chunks) - 1:
            time.sleep(0.5)

    success = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - success
    logger.info(f"LLM analysis complete: {success} success, {failed} failed")
    return results
