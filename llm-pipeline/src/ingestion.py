"""
ingestion.py — Input ingestion module
Handles: .txt files, .pdf files, and URLs
Author: Rose Sharma
"""

import logging
import re
import requests
import httpx
from pathlib import Path
from bs4 import BeautifulSoup

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


def read_txt_file(filepath: str) -> str:
    """Read a .txt file, handling encoding issues gracefully."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            text = path.read_text(encoding=encoding)
            logger.info(f"Read txt file: {filepath} ({len(text)} chars)")
            return text
        except UnicodeDecodeError:
            continue

    # Last resort: ignore bad bytes
    text = path.read_text(encoding="utf-8", errors="ignore")
    logger.warning(f"Read {filepath} with utf-8 ignore mode — some chars may be lost")
    return text


def read_pdf_file(filepath: str) -> str:
    """Extract text from a PDF file page by page."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    try:
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
            except Exception as e:
                logger.warning(f"Could not extract page {i} from {filepath}: {e}")
        full_text = "\n\n".join(pages)
        logger.info(f"Read PDF: {filepath} ({len(reader.pages)} pages, {len(full_text)} chars)")
        return full_text
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF {filepath}: {e}")


def fetch_url(url: str, timeout: int = 15) -> str:
    """Fetch and extract clean text from a URL using BeautifulSoup."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LLMPipeline/1.0)"
    }
    try:
        response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove boilerplate tags
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "advertisement", "iframe", "form"]):
            tag.decompose()

        # Extract main text
        text = soup.get_text(separator="\n")
        text = clean_text(text)

        logger.info(f"Fetched URL: {url} ({len(text)} chars)")
        return text

    except httpx.TimeoutException:
        raise RuntimeError(f"Timeout fetching URL: {url}")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP {e.response.status_code} for URL: {url}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")


def clean_text(text: str) -> str:
    """Clean encoding issues, excessive whitespace, and boilerplate noise."""
    # Fix common encoding artifacts
    text = text.replace("\u00e2\u0080\u0099", "'")
    text = text.replace("\u00e2\u0080\u009c", '"')
    text = text.replace("\u00e2\u0080\u009d", '"')
    text = text.replace("\u00c2\u00a0", " ")

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are only whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l]

    # Remove very short boilerplate lines
    lines = [l for l in lines if len(l) > 20 or l.endswith((".", "!", "?", ":"))]

    return "\n".join(lines).strip()


def ingest(file_path: str = None, urls: list = None) -> list:
    """
    Main ingestion function.
    Returns list of dicts: {source, source_type, text}
    Skips failed inputs with logging — does not crash.
    """
    results = []

    # Handle file input
    if file_path:
        try:
            ext = Path(file_path).suffix.lower()
            if ext == ".txt":
                text = read_txt_file(file_path)
                source_type = "txt"
            elif ext == ".pdf":
                text = read_pdf_file(file_path)
                source_type = "pdf"
            else:
                logger.error(f"Unsupported file type: {ext}. Skipping.")
                text = None

            if text and text.strip():
                results.append({
                    "source": file_path,
                    "source_type": source_type,
                    "text": clean_text(text)
                })
            else:
                logger.warning(f"Empty content from file: {file_path}. Skipping.")

        except Exception as e:
            logger.error(f"Failed to ingest file {file_path}: {e}")

    # Handle URL inputs
    if urls:
        for url in urls:
            try:
                text = fetch_url(url)
                if text.strip():
                    results.append({
                        "source": url,
                        "source_type": "url",
                        "text": text
                    })
                else:
                    logger.warning(f"Empty content from URL: {url}. Skipping.")
            except Exception as e:
                logger.error(f"Failed to ingest URL {url}: {e}")

    logger.info(f"Ingestion complete: {len(results)} sources loaded")
    return results
