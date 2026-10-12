"""LLM feedback on top of the numeric match, using Groq with JSON output."""
from __future__ import annotations

import json
import os

from .scoring import MatchResult

DEFAULT_MODEL = "llama-3.3-70b-versatile"

PROMPT = """You are an experienced technical recruiter reviewing a resume against a job description.
A scoring system has already measured the overlap:

- Overall match: {score}/100 ({verdict})
- Required skills found: {matched}
- Required skills missing: {missing}
- Preferred skills missing: {missing_pref}
- Experience on resume: ~{years} years (job asks for {need})

Return JSON with exactly these keys:
- "summary": 2-3 sentences on how well the candidate fits, honest and specific
- "strengths": list of 3 concrete strengths relevant to this job, each tied to something on the resume
- "gaps": list of up to 4 gaps that matter most for this job, each with a practical way to close it
- "bullet_rewrites": list of 3 objects {{"original": "...", "improved": "..."}}. Take real bullets from the resume
  and rewrite them to be more specific and relevant to this job. Never invent numbers or experience the resume
  doesn't support; use [X] as a placeholder where a metric would help.
- "keywords_to_add": list of up to 6 job keywords missing from the resume that the candidate could truthfully add
  if they have that experience

RESUME:
{resume}

JOB DESCRIPTION:
{job}"""


def get_feedback(resume: str, job: str, result: MatchResult, api_key: str | None = None,
                 model: str | None = None) -> dict:
    from groq import Groq
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    client = Groq(api_key=key)
    prompt = PROMPT.format(
        score=result.score, verdict=result.verdict,
        matched=", ".join(result.matched_required) or "none",
        missing=", ".join(result.missing_required) or "none",
        missing_pref=", ".join(result.missing_preferred) or "none",
        years=result.resume_years, need=f"{result.required_years:g}+ years" if result.required_years else "not specified",
        resume=resume[:8000], job=job[:5000])
    resp = client.chat.completions.create(
        model=model or os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, response_format={"type": "json_object"})
    return parse_feedback(resp.choices[0].message.content)


def parse_feedback(raw: str) -> dict:
    data = json.loads(raw)
    return {
        "summary": str(data.get("summary", "")),
        "strengths": [str(s) for s in data.get("strengths", [])][:5],
        "gaps": [str(g) for g in data.get("gaps", [])][:5],
        "bullet_rewrites": [b for b in data.get("bullet_rewrites", [])
                            if isinstance(b, dict) and "original" in b and "improved" in b][:5],
        "keywords_to_add": [str(k) for k in data.get("keywords_to_add", [])][:8],
    }
