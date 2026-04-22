# 🔬 LLM Data Pipeline

## AI Engineer Intern Assignment 2 — LLM Integration & Data Pipeline

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-FF6B00?style=flat)](https://groq.com)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)

---

## 📌 Objective

A production-style Python data pipeline that ingests unstructured text from multiple sources (files + URLs), preprocesses it, sends it through a Groq LLM for structured extraction, handles failures gracefully, and stores clean results in multiple formats.

**Built without LangChain or any orchestration framework — direct API calls only.**

---

## 🤖 LLM Used — Why Groq?

**Model:** `llama-3.1-8b-instant` via Groq API

**Why Groq:**
- Fastest LLM inference API available (sub-second latency)
- Free tier with generous rate limits — no credit card required
- `llama-3.1-8b-instant` produces clean, structured JSON reliably
- Simple Python SDK with no hidden abstractions

---

## 📁 Project Structure

```
llm-pipeline/
├── inputs/
│   └── sample.txt        ← Sample input file for testing
├── src/
│   ├── ingestion.py      ← Reads .txt, .pdf files and fetches URLs
│   ├── preprocessor.py   ← Cleans text and chunks into LLM-safe sizes
│   ├── llm_client.py     ← Groq API calls with retry + JSON parsing
│   └── storage.py        ← Saves JSON, Excel, and text report
├── outputs/
│   ├── sample_results.json
│   ├── sample_results.xlsx
│   └── sample_summary_report.txt
├── logs/                 ← Auto-created pipeline run logs
├── main.py               ← Entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### 1. Clone and install
```bash
git clone https://github.com/Rosesharma13/llm-pipeline.git
cd llm-pipeline
pip install -r requirements.txt
```

### 2. Set your Groq API key
```bash
# Linux / Mac
export GROQ_API_KEY=your_groq_api_key_here

# Windows PowerShell
$env:GROQ_API_KEY="your_groq_api_key_here"
```
Get a free key at: [console.groq.com](https://console.groq.com)

### 3. Run the pipeline

**With a text file:**
```bash
python main.py --file inputs/sample.txt
```

**With URLs:**
```bash
python main.py --urls https://en.wikipedia.org/wiki/Artificial_intelligence https://bbc.com/news
```

**Both at once (required by assignment):**
```bash
python main.py --file inputs/sample.txt --urls https://en.wikipedia.org/wiki/Machine_learning
```

**Custom output directory:**
```bash
python main.py --file inputs/sample.txt --output-dir my_results
```

---

## 📤 Output Files

Every pipeline run produces 3 output files in the `outputs/` folder:

| File | Format | Description |
|---|---|---|
| `results_TIMESTAMP.json` | JSON | Full structured results per chunk |
| `results_TIMESTAMP.xlsx` | Excel | One row per chunk — easy to filter |
| `summary_report_TIMESTAMP.txt` | Plain text | Aggregated findings report |

---

## 🏗️ Design Decisions

### Modular Architecture
Code is split into 4 focused modules — no single file exceeds 200 lines:
- `ingestion.py` — reads files and fetches URLs
- `preprocessor.py` — cleans and chunks text
- `llm_client.py` — handles all LLM communication
- `storage.py` — writes all output formats

### Chunking Strategy
Text is split by paragraphs first, then by sentences if needed. Target: ~1,500 tokens per chunk (conservative for gemma2-9b-it context window). Token count estimated via character heuristic (4 chars ≈ 1 token).

### Retry Logic
Uses `tenacity` library for exponential backoff:
- Retries on: `RateLimitError`, `APITimeoutError`, `APIError`
- 4 attempts max
- Wait: 2s → 4s → 8s → 16s between retries
- All retries logged — no silent failures

### JSON Parsing
LLM output is parsed with 4 fallback strategies:
1. Direct `json.loads()`
2. Extract from markdown code blocks
3. Find JSON object anywhere in response
4. Fix common issues (trailing commas) and retry

### Failure Handling
- Bad file → logged and skipped, pipeline continues
- Bad URL → logged and skipped, pipeline continues
- LLM API failure after retries → chunk marked as failed, pipeline continues
- All failures logged to file in `logs/` directory

---

## 🧪 Inputs Tested

| Input | Type | Result |
|---|---|---|
| `inputs/sample.txt` (AI in Healthcare article) | TXT file | ✅ Success |
| `https://en.wikipedia.org/wiki/Artificial_intelligence` | URL | ✅ Success |
| Non-existent file path | TXT file | ✅ Skipped with log |
| Dead URL | URL | ✅ Skipped with log |

---

## ⚠️ Known Limitations

- PDF extraction may lose formatting from complex/scanned PDFs
- Token estimation is approximate (character-based, not exact tokenizer)
- Rate limits on Groq free tier may slow processing of large inputs
- URLs requiring JavaScript rendering are not supported (static HTML only)
- Entity extraction quality depends on chunk size and text clarity

---

## 📊 Sample Output

### JSON (truncated)
```json
{
  "pipeline_run": "20260422_143022",
  "total_chunks": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "source": "inputs/sample.txt",
      "summary": "AI is transforming healthcare through diagnostics and cost reduction...",
      "entities": {
        "people": ["Eric Topol", "Elizabeth Warren"],
        "organizations": ["Google DeepMind", "FDA", "Microsoft"]
      },
      "sentiment": {"label": "neutral", "confidence": 0.78},
      "questions": ["How can algorithmic bias be addressed?", ...]
    }
  ]
}
```

---

## 👩‍💻 Author

**Rose Sharma** | AI Engineer Intern Candidate

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rose-sharma13)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Rosesharma13)
