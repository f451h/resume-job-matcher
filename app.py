"""Resume vs job description matcher.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import os
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from matcher import SemanticScorer, match
from matcher.feedback import get_feedback
from matcher.parsing import read_file

load_dotenv()
SAMPLES = Path(__file__).parent / "samples"

st.set_page_config(page_title="Resume Match", page_icon="🎯", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  h1 {font-family: Georgia, serif; letter-spacing: -0.5px;}
  .sub {color: #6b5e4f; margin-top: -0.6rem; margin-bottom: 1.2rem;}
  .chip {display: inline-block; padding: 2px 10px; margin: 3px 4px 3px 0; border-radius: 12px; font-size: 0.82rem;}
  .ok {background: #e3f1e4; color: #24572b; border: 1px solid #b9dcbc;}
  .miss {background: #fbe4dc; color: #8a2c12; border: 1px solid #f0bfae;}
  .soft {background: #f1e8da; color: #5a4a36; border: 1px solid #e3d9c8;}
  .label {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; color: #6b5e4f; margin: 0.6rem 0 0.2rem;}
  .verdict {font-size: 1.15rem; font-weight: 600; text-align: center; margin-top: -1rem;}
  .rewrite {border-left: 3px solid #b5562b; padding: 0.4rem 0.8rem; margin-bottom: 0.7rem; background: #fffdf8;}
  .rewrite .old {color: #8b7d6b; text-decoration: line-through; font-size: 0.88rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_scorer(use_embeddings: bool) -> SemanticScorer:
    return SemanticScorer(use_embeddings=use_embeddings)


def chips(items: list[str], cls: str) -> str:
    if not items:
        return '<span style="color:#8b7d6b;font-size:0.85rem">none</span>'
    return "".join(f'<span class="chip {cls}">{html.escape(i)}</span>' for i in items)


def gauge(score: float) -> go.Figure:
    color = "#2e7d32" if score >= 75 else "#c98a1b" if score >= 55 else "#b5562b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "/100", "font": {"size": 38}},
        gauge={"axis": {"range": [0, 100], "visible": False}, "bar": {"color": color, "thickness": 0.35},
               "bgcolor": "#f1e8da", "borderwidth": 0,
               "steps": [{"range": [0, 40], "color": "#f7eee3"}, {"range": [40, 75], "color": "#f3e7d6"},
                         {"range": [75, 100], "color": "#eadfcd"}]}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def breakdown_chart(b: dict[str, float]) -> go.Figure:
    names = {"required_skills": "Required skills", "semantic": "Overall relevance",
             "experience": "Experience", "preferred_skills": "Nice-to-have skills"}
    keys = ["required_skills", "semantic", "experience", "preferred_skills"]
    fig = go.Figure(go.Bar(x=[b[k] for k in keys], y=[names[k] for k in keys], orientation="h",
                           marker_color="#b5562b", text=[f"{b[k]:.0f}%" for k in keys], textposition="outside"))
    fig.update_layout(height=210, margin=dict(l=0, r=30, t=10, b=10), xaxis=dict(range=[0, 110], visible=False),
                      yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


with st.sidebar:
    st.subheader("Settings")
    api_key = st.text_input("Groq API key", type="password", value=os.getenv("GROQ_API_KEY", ""),
                            help="Needed only for the written feedback. The score works without it.")
    use_embeddings = st.toggle("Semantic similarity (embeddings)", value=True,
                               help="Off = TF-IDF similarity, which needs no model download.")
    st.caption("Score = 45% required skills · 30% overall relevance · 15% experience · 10% nice-to-have skills")

st.title("How well does your resume fit the job?")
st.markdown('<div class="sub">Paste a job description, add your resume, and see what matches, what\'s missing '
            'and how to improve it.</div>', unsafe_allow_html=True)

left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Resume**")
    up = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"], label_visibility="collapsed")
    if up:
        resume_text = read_file(up.getvalue(), up.name)
    else:
        resume_text = st.text_area("or paste it here", height=260,
                                   value=(SAMPLES / "resume_ayesha.txt").read_text(), label_visibility="collapsed")
with right:
    st.markdown("**Job description**")
    sample_job = st.selectbox("Load a sample", ["ML Engineer", "Senior Data Analyst"], label_visibility="collapsed")
    default_jd = (SAMPLES / ("job_ml_engineer.txt" if sample_job == "ML Engineer" else "job_data_analyst.txt")).read_text()
    job_text = st.text_area("Job description", height=260, value=default_jd, label_visibility="collapsed")

if st.button("Analyse match", type="primary", use_container_width=True) or st.session_state.get("ran"):
    st.session_state.ran = True
    if not resume_text.strip() or not job_text.strip():
        st.warning("Add both a resume and a job description.")
        st.stop()
    scorer = get_scorer(use_embeddings)
    result = match(resume_text, job_text, scorer)

    st.divider()
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        st.plotly_chart(gauge(result.score), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="verdict">{result.verdict}</div>', unsafe_allow_html=True)
        need = f"{result.required_years:g}+ yrs asked" if result.required_years else "no minimum stated"
        st.caption(f"Experience on resume: ~{result.resume_years:g} yrs ({need}) · similarity: {result.similarity_method}")
    with c2:
        st.plotly_chart(breakdown_chart(result.breakdown), use_container_width=True, config={"displayModeBar": False})

    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown('<div class="label">Required skills you have</div>' + chips(result.matched_required, "ok"),
                    unsafe_allow_html=True)
        st.markdown('<div class="label">Required skills missing</div>' + chips(result.missing_required, "miss"),
                    unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="label">Nice-to-have: have / missing</div>' + chips(result.matched_preferred, "ok")
                    + chips(result.missing_preferred, "miss"), unsafe_allow_html=True)
        st.markdown('<div class="label">Soft skills the job mentions</div>' + chips(result.soft_skills_in_job, "soft"),
                    unsafe_allow_html=True)
        st.markdown('<div class="label">Other skills on your resume</div>' + chips(result.extra_skills, "soft"),
                    unsafe_allow_html=True)

    st.divider()
    st.subheader("Feedback")
    if not api_key:
        st.info("Add a Groq API key in the sidebar for written feedback and bullet-point rewrites.")
    else:
        with st.spinner("Reading your resume..."):
            try:
                fb = get_feedback(resume_text, job_text, result, api_key=api_key)
            except Exception as e:
                st.error(f"The model call failed: {e}")
                st.stop()
        st.write(fb["summary"])
        f1, f2 = st.columns(2, gap="large")
        with f1:
            st.markdown("**Strengths**")
            for s in fb["strengths"]:
                st.markdown(f"- {s}")
        with f2:
            st.markdown("**Gaps to close**")
            for g in fb["gaps"]:
                st.markdown(f"- {g}")
        if fb["bullet_rewrites"]:
            st.markdown("**Stronger bullet points**")
            for b in fb["bullet_rewrites"]:
                st.markdown(f'<div class="rewrite"><div class="old">{html.escape(b["original"])}</div>'
                            f'<div>{html.escape(b["improved"])}</div></div>', unsafe_allow_html=True)
        if fb["keywords_to_add"]:
            st.markdown('<div class="label">Keywords to add (if true for you)</div>'
                        + chips(fb["keywords_to_add"], "soft"), unsafe_allow_html=True)
