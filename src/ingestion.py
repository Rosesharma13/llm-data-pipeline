"""
ingestion.py — Input ingestion module
Handles: .txt files, .pdf files, and URLs
Author: Rose Sharma
"""

import logging
import re
import httpx
from pathlib import Path
from bs4 import BeautifulSoup

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


def read_txt_file(filepath: str) -> str:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            text = path.read_text(encoding=encoding)
            return text
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(filepath: str) -> str:
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
                logger.warning(f"Could not extract page {i}: {e}")
        return "\n\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF {filepath}: {e}")


def fetch_url(url: str, timeout: int = 15) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LLMPipeline/1.0)"}
    try:
        response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "form"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return clean_text(text)
    except httpx.TimeoutException:
        raise RuntimeError(f"Timeout fetching URL: {url}")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP {e.response.status_code} for URL: {url}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")


def clean_text(text: str) -> str:
    text = text.replace("\u00e2\u0080\u0099", "'")
    text = text.replace("\u00e2\u0080\u009c", '"')
    text = text.replace("\u00e2\u0080\u009d", '"')
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l]
    lines = [l for l in lines if len(l) > 20 or l.endswith((".", "!", "?", ":"))]
    return "\n".join(lines).strip()


def ingest(file_path: str = None, urls: list = None) -> list:
    results = []
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
                logger.error(f"Unsupported file type: {ext}")
                text = None
            if text and text.strip():
                results.append({
                    "source": file_path,
                    "source_type": source_type,
                    "text": clean_text(text)
                })
        except Exception as e:
            logger.error(f"Failed to ingest file {file_path}: {e}")
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
            except Exception as e:
                logger.error(f"Failed to ingest URL {url}: {e}")
    return results
