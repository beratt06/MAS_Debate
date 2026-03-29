"""Streamlit UI for the unified multi-agent debate system."""

from __future__ import annotations

import json

import streamlit as st

from config import OLLAMA_MODEL_NAME, RETRIEVAL_TOP_K
from multiagent_system import MultiAgentDebateSystem


st.set_page_config(
    page_title="Mevzuu AI Debate Studio",
    page_icon="DA",
    layout="wide",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --paper: #f6f1e8;
  --ink: #1f2a37;
  --accent: #cc5f2a;
  --accent-soft: #f1c9ad;
  --card: #fffaf3;
}

.stApp {
  background:
    radial-gradient(circle at 15% 20%, #f3dfcc 0, transparent 35%),
    radial-gradient(circle at 85% 10%, #e8e1d2 0, transparent 40%),
    linear-gradient(180deg, #f7f0e4 0%, #f0e7d8 100%);
  color: var(--ink);
}

.block-container {
  max-width: 1200px;
  padding-top: 1.4rem;
}

h1, h2, h3 {
  font-family: 'Space Grotesk', sans-serif;
  letter-spacing: 0.2px;
}

p, li, .stTextInput label, .stTextArea label, .stSlider label {
  font-family: 'IBM Plex Sans', sans-serif;
}

.hero {
  background: linear-gradient(135deg, #232f3f 0%, #18212d 100%);
  color: #f4f1ea;
  border-radius: 16px;
  padding: 22px 24px;
  border: 1px solid #39495d;
  box-shadow: 0 10px 30px rgba(20, 33, 49, 0.22);
}

.hero-title {
  font-size: 1.8rem;
  margin: 0;
}

.hero-sub {
  margin-top: 0.4rem;
  opacity: 0.9;
}

.card {
  background: var(--card);
  border: 1px solid #ecdcc8;
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 6px 18px rgba(90, 67, 45, 0.07);
}

.metric {
  border-left: 5px solid var(--accent);
}

.stButton>button {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  background: linear-gradient(90deg, #cc5f2a 0%, #b04a1b 100%);
  color: #fff;
  border: 0;
  border-radius: 10px;
  padding: 0.55rem 1rem;
}

.stButton>button:hover {
  filter: brightness(1.07);
}

.stDownloadButton>button {
  border-radius: 10px;
}

.badge {
  display: inline-block;
  background: var(--accent-soft);
  color: #6b2f12;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.84rem;
  margin-right: 8px;
}

@media (max-width: 840px) {
  .hero-title { font-size: 1.4rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1 class="hero-title">Mevzuu AI Debate Studio</h1>
  <p class="hero-sub">Router + Research + Pro + Contra + Evidence Verifier + Judge butunlesik tek multi-agent akisi.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Sistem Ayarlari")
    model_name = st.text_input("Ollama Modeli", value=OLLAMA_MODEL_NAME)
    max_rounds = st.slider("Debate Round", min_value=1, max_value=5, value=3)
    top_k = st.slider("Retrieval Top-K", min_value=1, max_value=10, value=RETRIEVAL_TOP_K)

    st.markdown("---")
    if st.button("PDF Indexi Olustur / Guncelle", use_container_width=True):
        with st.spinner("PDF'ler taraniyor ve index olusturuluyor..."):
            try:
                index_system = MultiAgentDebateSystem(
                    model_name=model_name,
                    max_rounds=max_rounds,
                    retrieval_top_k=top_k,
                )
                report = index_system.build_index()
                st.success(
                    f"Index hazir: {report.pdf_count} PDF, {report.page_count} sayfa, "
                    f"{report.chunk_count} chunk, {report.indexed_chunk_count} yeni chunk."
                )
            except Exception as exc:
                st.error(f"Index islemi basarisiz: {exc}")

question = st.text_area(
    "Tartisma Sorusu",
    height=140,
    placeholder="Ornek: Yapay zeka kamu sagligi kararlarinda ana karar verici olmali mi?",
)

run_clicked = st.button("Multi-Agent Debate Baslat", type="primary", use_container_width=True)

if run_clicked:
    if not question.strip():
        st.warning("Lutfen once bir soru girin.")
    else:
        with st.spinner("Ajanlar calisiyor, tartisma uretiliyor..."):
            try:
                system = MultiAgentDebateSystem(
                    model_name=model_name,
                    max_rounds=max_rounds,
                    retrieval_top_k=top_k,
                )
                result = system.run(question)
            except Exception as exc:
                st.error(f"Calistirma hatasi: {exc}")
                st.stop()

        research = result["research"]
        debate = result["debate"]
        judge = result["judge"]
        verifier_logs = result["verifier_logs"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"<div class='card metric'><span class='badge'>Rounds</span><h3>{debate['total_rounds']}</h3></div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div class='card metric'><span class='badge'>Facts</span><h3>{len(research['facts'])}</h3></div>",
                unsafe_allow_html=True,
            )
        with col3:
            confidence = judge.get("Guven_Skoru") if isinstance(judge, dict) else "N/A"
            st.markdown(
                f"<div class='card metric'><span class='badge'>Guven Skoru</span><h3>{confidence}</h3></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### Arastirma Ozeti")
        st.markdown(f"<div class='card'>{research['topic_summary']}</div>", unsafe_allow_html=True)

        st.markdown("### Toplanan Bulgular")
        for fact in research["facts"]:
            st.write(f"- {fact}")

        st.markdown("### Debate Turlari")
        tabs = st.tabs([f"Round {round_data['round_number']}" for round_data in debate["rounds"]])

        for idx, round_data in enumerate(debate["rounds"]):
            with tabs[idx]:
                pro = round_data["pro_entry"]["content"]
                contra = round_data["contra_entry"]["content"]

                left, right = st.columns(2)
                with left:
                    st.subheader("Pro")
                    for arg in pro.get("arguments", []):
                        st.markdown(f"**{arg['title']}**")
                        st.write(arg["explanation"])
                with right:
                    st.subheader("Contra")
                    for counter in contra.get("counter_arguments", []):
                        st.markdown(f"**Hedef:** {counter['target_argument']}")
                        st.write(counter["criticism"])

                st.markdown("#### Riskler")
                for risk in contra.get("risks", []):
                    st.write(f"- [{risk['severity']}] {risk['title']}: {risk['description']}")

        st.markdown("### Evidence Verifier + Scoring")
        st.json(verifier_logs)

        st.markdown("### Judge Karari")
        st.json(judge)

        pretty_json = json.dumps(result, ensure_ascii=False, indent=2)
        st.download_button(
            "Tum Sonucu JSON Olarak Indir",
            data=pretty_json,
            file_name="multiagent_debate_result.json",
            mime="application/json",
            use_container_width=True,
        )
