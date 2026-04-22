"""
preprocessor.py — Text preprocessing and chunking module
Cleans text and splits into LLM-safe chunks using token counting
Author: Rose Sharma
"""

import re
import logging

logger = logging.getLogger(__name__)

# Max tokens per chunk — safe for gemma2-9b-it context window
MAX_TOKENS = 1500
AVG_CHARS_PER_TOKEN = 4  # Conservative estimate


def estimate_tokens(text: str) -> int:
    """Estimate token count using character-based heuristic."""
    return len(text) // AVG_CHARS_PER_TOKEN


def clean_chunk(text: str) -> str:
    """Final cleaning pass on a chunk before sending to LLM."""
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove URLs from text (they add noise without value)
    text = re.sub(r"http[s]?://\S+", "[URL]", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)

    return text.strip()


def chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> list:
    """
    Split text into chunks that fit within LLM context limits.
    Strategy: Split by paragraphs first, then sentences if needed.
    Returns list of chunk strings.
    """
    max_chars = max_tokens * AVG_CHARS_PER_TOKEN

    # If text fits in one chunk, return as-is
    if len(text) <= max_chars:
        return [clean_chunk(text)]

    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If single paragraph is too long, split by sentences
        if len(para) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if current_len + len(sent) > max_chars and current_chunk:
                    chunks.append(clean_chunk("\n\n".join(current_chunk)))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(sent)
                current_len += len(sent)
        else:
            if current_len + len(para) > max_chars and current_chunk:
                chunks.append(clean_chunk("\n\n".join(current_chunk)))
                current_chunk = []
                current_len = 0
            current_chunk.append(para)
            current_len += len(para)

    # Add remaining
    if current_chunk:
        chunks.append(clean_chunk("\n\n".join(current_chunk)))

    # Filter empty chunks
    chunks = [c for c in chunks if len(c.strip()) > 50]

    logger.info(f"Text chunked into {len(chunks)} chunks (max {max_tokens} tokens each)")
    return chunks


def preprocess(sources: list) -> list:
    """
    Preprocess all ingested sources.
    Returns flat list of chunk dicts ready for LLM processing.
    Each dict: {source, source_type, chunk_index, total_chunks, text}
    """
    all_chunks = []

    for source_data in sources:
        source = source_data["source"]
        source_type = source_data["source_type"]
        text = source_data["text"]

        if not text or not text.strip():
            logger.warning(f"Empty text for source: {source}. Skipping.")
            continue

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": source,
                "source_type": source_type,
                "chunk_index": i + 1,
                "total_chunks": len(chunks),
                "text": chunk,
                "token_estimate": estimate_tokens(chunk)
            })

    logger.info(f"Preprocessing complete: {len(all_chunks)} chunks ready for LLM")
    return all_chunks
