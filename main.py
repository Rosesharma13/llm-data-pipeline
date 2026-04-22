"""
main.py — LLM Data Pipeline Entry Point
Usage:
  python main.py --file inputs/sample.txt --urls https://example.com https://bbc.com/news
  python main.py --file inputs/sample.pdf
  python main.py --urls https://example.com

Author: Rose Sharma
LLM Used: Groq — llama-3.1-8b-instant
Why Groq: Free tier, fastest inference speed, no CC required, gemma2-9b-it handles structured JSON well
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestion import ingest
from preprocessor import preprocess
from llm_client import analyze_all
from storage import save_json, save_excel, save_report


# ─── Logging setup ───────────────────────────────────────────────────────────
def setup_logging(output_dir: str = "logs"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"pipeline_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


# ─── Argument parser ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Data Pipeline — Analyze text from files and URLs using Groq AI"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a .txt or .pdf file to analyze"
    )
    parser.add_argument(
        "--urls", nargs="+", default=None,
        help="One or more URLs to fetch and analyze"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory to save output files (default: outputs)"
    )
    return parser.parse_args()


# ─── Main pipeline ───────────────────────────────────────────────────────────
def run_pipeline(file_path=None, urls=None, output_dir="outputs"):
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("LLM DATA PIPELINE STARTED")
    logger.info(f"File: {file_path or 'None'}")
    logger.info(f"URLs: {urls or 'None'}")
    logger.info("=" * 60)

    # Validate at least one input
    if not file_path and not urls:
        logger.error("No inputs provided. Use --file and/or --urls")
        sys.exit(1)

    # Check API key
    if not os.environ.get("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY environment variable not set!")
        logger.error("Set it with: export GROQ_API_KEY=your_key_here")
        sys.exit(1)

    # Step 1: Ingest
    logger.info("STEP 1: Ingesting sources...")
    sources = ingest(file_path=file_path, urls=urls)

    if not sources:
        logger.error("No valid sources could be ingested. Check your inputs.")
        sys.exit(1)

    logger.info(f"Ingested {len(sources)} source(s)")

    # Step 2: Preprocess
    logger.info("STEP 2: Preprocessing and chunking...")
    chunks = preprocess(sources)

    if not chunks:
        logger.error("No chunks produced from preprocessing. Exiting.")
        sys.exit(1)

    logger.info(f"Produced {len(chunks)} chunk(s) for LLM analysis")

    # Step 3: LLM Analysis
    logger.info("STEP 3: Running LLM analysis via Groq...")
    results = analyze_all(chunks)

    # Step 4: Save outputs
    logger.info("STEP 4: Saving outputs...")
    json_path  = save_json(results, output_dir)
    excel_path = save_excel(results, output_dir)
    report_path = save_report(results, output_dir)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  JSON output   : {json_path}")
    logger.info(f"  Excel output  : {excel_path}")
    logger.info(f"  Summary report: {report_path}")
    logger.info("=" * 60)

    return {
        "results": results,
        "json": json_path,
        "excel": excel_path,
        "report": report_path
    }


if __name__ == "__main__":
    args = parse_args()
    log_file = setup_logging()
    run_pipeline(
        file_path=args.file,
        urls=args.urls,
        output_dir=args.output_dir
    )
