"""
app.py — Streamlit UI for LLM Data Pipeline
Wraps ingestion → preprocessing → LLM analysis → display
Author: Rose Sharma
"""

import streamlit as st
import os
import json
import tempfile
import time
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Data Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0F1117; }
    .stApp { background-color: #0F1117; }
    h1 { color: #F0F2F6; font-family: 'Segoe UI', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1E2530, #252D3A);
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-val { font-size: 32px; font-weight: 700; color: #68D391; }
    .metric-label { font-size: 12px; color: #A0AEC0; text-transform: uppercase; letter-spacing: 0.08em; }
    .chunk-card {
        background: #1A202C;
        border: 1px solid #2D3748;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .tag-positive { background: #1C4532; color: #68D391; border: 1px solid #276749; }
    .tag-neutral  { background: #1A2942; color: #63B3ED; border: 1px solid #2B6CB0; }
    .tag-negative { background: #3D1A1A; color: #FC8181; border: 1px solid #9B2C2C; }
    .tag-entity   { background: #2D2042; color: #B794F4; border: 1px solid #553C9A; }
    .badge-success { background: #1C4532; color: #68D391; padding: 2px 10px; border-radius: 50px; font-size: 12px; }
    .badge-failed  { background: #3D1A1A; color: #FC8181; padding: 2px 10px; border-radius: 50px; font-size: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Import pipeline modules ───────────────────────────────────────────────────
@st.cache_resource
def load_modules():
    from src.ingestion import ingest
    from src.preprocessor import preprocess
    from src.llm_client import analyze_all
    return ingest, preprocess, analyze_all


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 LLM Data Pipeline")
    st.markdown("*Production-grade text analysis powered by Groq*")
    st.divider()

    st.markdown("### ⚙️ Configuration")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com"
    )

    st.divider()
    st.markdown("### 📥 Input Sources")

    input_mode = st.radio(
        "Choose input type",
        ["Paste Text", "Upload File", "URL(s)"],
        index=0
    )

    st.divider()
    st.markdown("### 📖 About")
    st.markdown("""
**What this pipeline does:**
1. Ingests text from files or URLs
2. Chunks text into LLM-safe segments
3. Sends each chunk to Groq LLaMA 3.1 8B
4. Extracts: summary, entities, sentiment, questions
5. Displays structured results

**Built without LangChain** — direct Groq API calls with retry logic and exponential backoff.

**94.4% chunk success rate** in production testing.
    """)
    st.markdown("[GitHub](https://github.com/Rosesharma13/llm-data-pipeline) · [LinkedIn](https://linkedin.com/in/rose-sharma13)")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("# 🔬 LLM Data Pipeline")
st.markdown("Extract structured insights from any text — summaries, entities, sentiment, and key questions — using Groq's ultra-fast LLM inference.")
st.divider()

# Input collection
text_input = None
url_input = None
uploaded_file = None

if input_mode == "Paste Text":
    text_input = st.text_area(
        "Paste your text here",
        height=220,
        placeholder="Paste any article, report, document, or unstructured text...",
    )

elif input_mode == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload a .txt or .pdf file",
        type=["txt", "pdf"],
        help="Max ~50KB recommended for fast processing"
    )

elif input_mode == "URL(s)":
    url_raw = st.text_area(
        "Enter URLs (one per line)",
        height=120,
        placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence\nhttps://example.com/article",
    )
    url_input = [u.strip() for u in url_raw.strip().splitlines() if u.strip()] if url_raw else []

# Run button
col1, col2 = st.columns([1, 4])
with col1:
    run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

# ── Pipeline execution ────────────────────────────────────────────────────────
if run_btn:
    # Validate inputs
    if not groq_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
        st.stop()

    has_input = (
        (input_mode == "Paste Text" and text_input and text_input.strip()) or
        (input_mode == "Upload File" and uploaded_file is not None) or
        (input_mode == "URL(s)" and url_input)
    )

    if not has_input:
        st.error("⚠️ Please provide at least one input source.")
        st.stop()

    os.environ["GROQ_API_KEY"] = groq_key

    try:
        ingest, preprocess, analyze_all = load_modules()
    except Exception as e:
        st.error(f"Failed to load pipeline modules: {e}")
        st.stop()

    # Progress display
    progress_bar = st.progress(0)
    status = st.empty()

    try:
        # Step 1 — Ingestion
        status.info("📥 Step 1/3 — Ingesting sources...")
        progress_bar.progress(10)

        sources = []
        file_path_temp = None

        if input_mode == "Paste Text":
            sources = [{
                "source": "pasted_text",
                "source_type": "txt",
                "text": text_input.strip()
            }]

        elif input_mode == "Upload File":
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                file_path_temp = tmp.name
            sources = ingest(file_path=file_path_temp)

        elif input_mode == "URL(s)":
            sources = ingest(urls=url_input)

        if not sources:
            st.error("⚠️ Could not extract any content from the provided input. Check URLs are accessible or file is not empty.")
            progress_bar.empty()
            status.empty()
            st.stop()

        progress_bar.progress(30)

        # Step 2 — Preprocessing
        status.info("✂️ Step 2/3 — Chunking and preprocessing...")
        chunks = preprocess(sources)
        progress_bar.progress(50)

        if not chunks:
            st.error("⚠️ No chunks produced after preprocessing.")
            progress_bar.empty()
            status.empty()
            st.stop()

        # Step 3 — LLM Analysis
        status.info(f"🤖 Step 3/3 — Analyzing {len(chunks)} chunk(s) with Groq LLaMA...")
        results = analyze_all(chunks)
        progress_bar.progress(100)
        status.success(f"✅ Pipeline complete — {len(results)} chunk(s) processed.")
        time.sleep(0.8)
        progress_bar.empty()
        status.empty()

        # Clean up temp file
        if file_path_temp:
            try:
                os.unlink(file_path_temp)
            except:
                pass

        # ── Results display ───────────────────────────────────────────────────
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] != "success"]

        # Metrics row
        st.markdown("## 📊 Pipeline Results")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{len(results)}</div>
                <div class="metric-label">Total Chunks</div>
            </div>""", unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:#68D391">{len(successful)}</div>
                <div class="metric-label">Successful</div>
            </div>""", unsafe_allow_html=True)

        with m3:
            rate = f"{len(successful)/max(len(results),1)*100:.0f}%"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:#63B3ED">{rate}</div>
                <div class="metric-label">Success Rate</div>
            </div>""", unsafe_allow_html=True)

        with m4:
            sentiments = [r.get("sentiment_label","") for r in successful]
            dominant = max(set(sentiments), key=sentiments.count) if sentiments else "N/A"
            color = {"positive":"#68D391","neutral":"#63B3ED","negative":"#FC8181"}.get(dominant,"#A0AEC0")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:{color}">{dominant.title()}</div>
                <div class="metric-label">Dominant Sentiment</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Aggregate entities
        if successful:
            all_people = list(set([
                p.strip() for r in successful
                for p in (r.get("entities_people") or "").split(",")
                if p.strip()
            ]))
            all_places = list(set([
                p.strip() for r in successful
                for p in (r.get("entities_places") or "").split(",")
                if p.strip()
            ]))
            all_orgs = list(set([
                p.strip() for r in successful
                for p in (r.get("entities_orgs") or "").split(",")
                if p.strip()
            ]))

            if any([all_people, all_places, all_orgs]):
                st.markdown("### 🏷️ All Entities Found")
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    st.markdown("**👤 People**")
                    if all_people:
                        tags = "".join([f'<span class="tag tag-entity">{p}</span>' for p in all_people[:10]])
                        st.markdown(tags, unsafe_allow_html=True)
                    else:
                        st.caption("None identified")
                with ec2:
                    st.markdown("**📍 Places**")
                    if all_places:
                        tags = "".join([f'<span class="tag tag-entity">{p}</span>' for p in all_places[:10]])
                        st.markdown(tags, unsafe_allow_html=True)
                    else:
                        st.caption("None identified")
                with ec3:
                    st.markdown("**🏢 Organizations**")
                    if all_orgs:
                        tags = "".join([f'<span class="tag tag-entity">{p}</span>' for p in all_orgs[:10]])
                        st.markdown(tags, unsafe_allow_html=True)
                    else:
                        st.caption("None identified")
                st.divider()

        # Chunk-by-chunk results
        st.markdown("### 📄 Chunk Analysis")

        for r in results:
            sentiment = r.get("sentiment_label", "unknown")
            conf = r.get("sentiment_confidence", 0)
            tag_class = {"positive":"tag-positive","neutral":"tag-neutral","negative":"tag-negative"}.get(sentiment,"tag-neutral")
            status_badge = '<span class="badge-success">✅ success</span>' if r["status"]=="success" else '<span class="badge-failed">❌ failed</span>'

            with st.expander(f"Chunk {r['chunk_index']}/{r['total_chunks']} — {r['source'][:60]}", expanded=len(results)==1):
                col_a, col_b = st.columns([3,1])
                with col_a:
                    st.markdown(f"**Source:** `{r['source']}`")
                    st.markdown(f"**Type:** {r['source_type'].upper()} &nbsp;|&nbsp; **Tokens (est.):** {r.get('token_estimate',0)} &nbsp;|&nbsp; {status_badge}", unsafe_allow_html=True)
                with col_b:
                    st.markdown(f'<span class="tag {tag_class}">{sentiment.title()} ({conf:.0%})</span>', unsafe_allow_html=True)

                if r["status"] == "success":
                    st.markdown("**📝 Summary**")
                    st.info(r.get("summary",""))

                    q1, q2 = st.columns(2)
                    with q1:
                        questions = r.get("questions", [])
                        if questions:
                            st.markdown("**❓ Key Questions**")
                            for q in questions:
                                st.markdown(f"- {q}")
                    with q2:
                        people = r.get("entities_people","")
                        places = r.get("entities_places","")
                        orgs = r.get("entities_orgs","")
                        if any([people, places, orgs]):
                            st.markdown("**🏷️ Entities**")
                            if people: st.markdown(f"👤 {people}")
                            if places: st.markdown(f"📍 {places}")
                            if orgs:   st.markdown(f"🏢 {orgs}")
                else:
                    st.error(f"Error: {r.get('error','Unknown error')}")

        # JSON download
        st.divider()
        st.markdown("### ⬇️ Export Results")

        export_data = {
            "pipeline_run": time.strftime("%Y%m%d_%H%M%S"),
            "total_chunks": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "results": [
                {
                    "source": r["source"],
                    "chunk_index": r["chunk_index"],
                    "status": r["status"],
                    "summary": r.get("summary",""),
                    "entities": {
                        "people": [p.strip() for p in (r.get("entities_people") or "").split(",") if p.strip()],
                        "places": [p.strip() for p in (r.get("entities_places") or "").split(",") if p.strip()],
                        "organizations": [p.strip() for p in (r.get("entities_orgs") or "").split(",") if p.strip()],
                    },
                    "sentiment": {"label": r.get("sentiment_label",""), "confidence": r.get("sentiment_confidence",0)},
                    "questions": r.get("questions",[]),
                }
                for r in results
            ]
        }

        st.download_button(
            label="⬇️ Download JSON Results",
            data=json.dumps(export_data, indent=2),
            file_name=f"pipeline_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    except Exception as e:
        progress_bar.empty()
        status.empty()
        st.error(f"Pipeline error: {str(e)}")
        st.exception(e)

else:
    # Landing state
    st.markdown("### How it works")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**1. Input**\n\nPaste text, upload a PDF/TXT file, or provide URLs")
    with c2:
        st.markdown("**2. Chunk**\n\nText is split into LLM-safe segments (~1500 tokens each)")
    with c3:
        st.markdown("**3. Analyze**\n\nEach chunk is sent to Groq LLaMA 3.1 8B with retry logic")
    with c4:
        st.markdown("**4. Extract**\n\nGet summaries, named entities, sentiment scores, and key questions")

    st.divider()
    st.info("👈 Enter your Groq API key in the sidebar and choose an input type to get started. Get a free key at [console.groq.com](https://console.groq.com)")
