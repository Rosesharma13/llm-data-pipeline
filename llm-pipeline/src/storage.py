"""
storage.py — Output storage module
Saves results to: JSON file, Excel/CSV file, plain text summary report
Author: Rose Sharma
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(results: list, output_dir: str = "outputs") -> str:
    """Save all results to a structured JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    filepath = os.path.join(output_dir, f"results_{timestamp}.json")

    output = {
        "pipeline_run": timestamp,
        "total_chunks": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] != "success"),
        "results": []
    }

    for r in results:
        output["results"].append({
            "source": r["source"],
            "source_type": r["source_type"],
            "chunk_index": r["chunk_index"],
            "total_chunks": r["total_chunks"],
            "token_estimate": r.get("token_estimate", 0),
            "status": r["status"],
            "error": r.get("error"),
            "summary": r.get("summary", ""),
            "entities": {
                "people": r.get("entities_people", "").split(", ") if r.get("entities_people") else [],
                "places": r.get("entities_places", "").split(", ") if r.get("entities_places") else [],
                "organizations": r.get("entities_orgs", "").split(", ") if r.get("entities_orgs") else []
            },
            "sentiment": {
                "label": r.get("sentiment_label", "unknown"),
                "confidence": r.get("sentiment_confidence", 0.0)
            },
            "questions": r.get("questions", [])
        })

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ JSON saved: {filepath}")
    return filepath


def save_excel(results: list, output_dir: str = "outputs") -> str:
    """Save results to Excel file — one row per chunk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    filepath = os.path.join(output_dir, f"results_{timestamp}.xlsx")

    rows = []
    for r in results:
        questions = r.get("questions", [])
        rows.append({
            "Source": r["source"],
            "Source Type": r["source_type"],
            "Chunk": f"{r['chunk_index']}/{r['total_chunks']}",
            "Token Estimate": r.get("token_estimate", 0),
            "Status": r["status"],
            "Summary": r.get("summary", ""),
            "People": r.get("entities_people", ""),
            "Places": r.get("entities_places", ""),
            "Organizations": r.get("entities_orgs", ""),
            "Sentiment": r.get("sentiment_label", ""),
            "Confidence": r.get("sentiment_confidence", 0.0),
            "Question 1": questions[0] if len(questions) > 0 else "",
            "Question 2": questions[1] if len(questions) > 1 else "",
            "Question 3": questions[2] if len(questions) > 2 else "",
            "Error": r.get("error", ""),
        })

    df = pd.DataFrame(rows)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
        # Auto-adjust column widths
        ws = writer.sheets["Results"]
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)

    logger.info(f"✅ Excel saved: {filepath}")
    return filepath


def save_report(results: list, output_dir: str = "outputs") -> str:
    """Generate a plain-text summary report aggregating all findings."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    filepath = os.path.join(output_dir, f"summary_report_{timestamp}.txt")

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    # Aggregate sentiments
    sentiments = [r.get("sentiment_label", "unknown") for r in successful]
    sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}

    # Aggregate entities
    all_people = []
    all_places = []
    all_orgs = []
    for r in successful:
        if r.get("entities_people"):
            all_people.extend([p.strip() for p in r["entities_people"].split(",") if p.strip()])
        if r.get("entities_places"):
            all_places.extend([p.strip() for p in r["entities_places"].split(",") if p.strip()])
        if r.get("entities_orgs"):
            all_orgs.extend([p.strip() for p in r["entities_orgs"].split(",") if p.strip()])

    # Top entities
    def top_n(lst, n=5):
        from collections import Counter
        return [item for item, _ in Counter(lst).most_common(n) if item]

    # Sources breakdown
    sources = {}
    for r in results:
        src = r["source"]
        if src not in sources:
            sources[src] = {"total": 0, "success": 0, "type": r["source_type"]}
        sources[src]["total"] += 1
        if r["status"] == "success":
            sources[src]["success"] += 1

    lines = [
        "=" * 70,
        "LLM DATA PIPELINE — SUMMARY REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "PIPELINE OVERVIEW",
        "-" * 40,
        f"Total chunks processed : {len(results)}",
        f"Successfully analyzed  : {len(successful)}",
        f"Failed / skipped       : {len(failed)}",
        f"Success rate           : {len(successful)/max(len(results),1)*100:.1f}%",
        "",
        "SOURCES PROCESSED",
        "-" * 40,
    ]

    for src, info in sources.items():
        lines.append(f"  [{info['type'].upper()}] {src[:80]}")
        lines.append(f"         Chunks: {info['total']} | Successful: {info['success']}")

    lines += [
        "",
        "SENTIMENT ANALYSIS",
        "-" * 40,
    ]
    for label, count in sorted(sentiment_counts.items()):
        pct = count / max(len(successful), 1) * 100
        bar = "█" * int(pct / 5)
        lines.append(f"  {label:<10} {bar:<20} {count} chunks ({pct:.1f}%)")

    lines += [
        "",
        "KEY ENTITIES FOUND",
        "-" * 40,
        f"  People        : {', '.join(top_n(all_people)) or 'None identified'}",
        f"  Places        : {', '.join(top_n(all_places)) or 'None identified'}",
        f"  Organizations : {', '.join(top_n(all_orgs)) or 'None identified'}",
        "",
        "CHUNK SUMMARIES",
        "-" * 40,
    ]

    for r in successful[:10]:  # Show first 10 summaries
        lines.append(f"\n  Source: {r['source'][:60]} [Chunk {r['chunk_index']}]")
        lines.append(f"  Sentiment: {r.get('sentiment_label','?')} ({r.get('sentiment_confidence',0):.0%})")
        lines.append(f"  Summary: {r.get('summary','')[:200]}")

    if failed:
        lines += [
            "",
            "FAILED / SKIPPED INPUTS",
            "-" * 40,
        ]
        for r in failed:
            lines.append(f"  ❌ {r['source']} [Chunk {r['chunk_index']}]: {r.get('error','Unknown error')}")

    lines += [
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70
    ]

    report_text = "\n".join(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"✅ Report saved: {filepath}")
    print(report_text)
    return filepath
