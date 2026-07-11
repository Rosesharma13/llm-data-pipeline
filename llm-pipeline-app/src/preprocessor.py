"""
preprocessor.py — Text preprocessing and chunking module
Author: Rose Sharma
"""
import re
import logging

logger = logging.getLogger(__name__)

MAX_TOKENS = 1500
AVG_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return len(text) // AVG_CHARS_PER_TOKEN


def clean_chunk(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"http[s]?://\S+", "[URL]", text)
    text = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)
    return text.strip()


def chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> list:
    max_chars = max_tokens * AVG_CHARS_PER_TOKEN
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
    if current_chunk:
        chunks.append(clean_chunk("\n\n".join(current_chunk)))
    return [c for c in chunks if len(c.strip()) > 50]


def preprocess(sources: list) -> list:
    all_chunks = []
    for source_data in sources:
        source = source_data["source"]
        source_type = source_data["source_type"]
        text = source_data["text"]
        if not text or not text.strip():
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
    return all_chunks
