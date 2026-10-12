"""Combine skill coverage, semantic similarity and experience into one match score."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .parsing import experience_months, required_years, split_job_description
from .skills import CATEGORY_OF, SOFT_SKILLS, extract_skills, requirement_groups, with_implied

WEIGHTS = {"required_skills": 0.45, "semantic": 0.30, "preferred_skills": 0.10, "experience": 0.15}


@dataclass
class MatchResult:
    score: float                       # 0-100
    breakdown: dict[str, float]        # each component 0-100
    matched_required: list[str]
    missing_required: list[str]
    matched_preferred: list[str]
    missing_preferred: list[str]
    extra_skills: list[str]            # on the resume, not asked for
    resume_years: float
    required_years: float | None
    similarity_method: str
    categories: dict[str, str] = field(default_factory=dict)
    soft_skills_in_job: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.score >= 75:
            return "Strong match"
        if self.score >= 55:
            return "Good match with some gaps"
        if self.score >= 40:
            return "Partial match"
        return "Weak match"


class SemanticScorer:
    """Embedding similarity with a TF-IDF fallback when the embedding model isn't available."""

    def __init__(self, use_embeddings: bool = True):
        self.method = "tfidf"
        self._model = None
        if use_embeddings:
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                self.method = "embeddings"
            except Exception:
                self._model = None

    def similarity(self, a: str, b: str) -> float:
        if self._model is not None:
            va, vb = np.array(list(self._model.embed([_window(a), _window(b)])))
            cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
            # bge cosine for related documents sits roughly in 0.5-0.9; stretch that to 0-1
            return float(np.clip((cos - 0.5) / 0.4, 0, 1))
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit([a, b])
        m = vec.transform([a, b])
        cos = float((m[0] @ m[1].T).toarray()[0, 0])
        return float(np.clip(cos / 0.35, 0, 1))      # tf-idf cosine between related docs rarely passes ~0.35


def _window(text: str, max_words: int = 380) -> str:
    return " ".join(text.split()[:max_words])


def _coverage(have: set[str], groups: list[set[str]]) -> tuple[float, list[str], list[str]]:
    """Share of requirement groups the resume satisfies, plus readable matched/missing labels."""
    if not groups:
        return 1.0, [], []
    matched, missing = [], []
    for g in groups:
        label = " or ".join(sorted(g))
        (matched if have & g else missing).append(label if not have & g else " / ".join(sorted(have & g)))
    return len(matched) / len(groups), sorted(matched), sorted(missing)


def match(resume: str, job: str, scorer: SemanticScorer | None = None) -> MatchResult:
    scorer = scorer or SemanticScorer(use_embeddings=False)
    sections = split_job_description(job)
    resume_skills = with_implied(extract_skills(resume))
    have = set(resume_skills)

    req_groups = [g for g in requirement_groups(sections.required) if not g <= SOFT_SKILLS]
    req_skills = set().union(*req_groups) if req_groups else set()
    pref_groups = [g for g in requirement_groups(sections.preferred) if not g <= SOFT_SKILLS and not g <= req_skills]
    pref_skills = set().union(*pref_groups) if pref_groups else set()

    req_score, matched_req, missing_req = _coverage(have, req_groups)
    pref_score, matched_pref, missing_pref = _coverage(have, pref_groups)

    years = experience_months(resume) / 12
    need = required_years(job)
    exp_score = 1.0 if not need else min(1.0, years / need)

    breakdown = {
        "required_skills": req_score,
        "preferred_skills": pref_score,
        "semantic": scorer.similarity(resume, job),
        "experience": exp_score,
    }
    total = sum(WEIGHTS[k] * v for k, v in breakdown.items()) * 100
    job_soft = sorted(set(extract_skills(job)) & SOFT_SKILLS)
    return MatchResult(
        score=round(total, 1),
        breakdown={k: round(v * 100, 1) for k, v in breakdown.items()},
        matched_required=matched_req, missing_required=missing_req,
        matched_preferred=matched_pref, missing_preferred=missing_pref,
        extra_skills=sorted(set(extract_skills(resume)) - req_skills - pref_skills - SOFT_SKILLS),
        resume_years=round(years, 1), required_years=need, similarity_method=scorer.method,
        categories={**CATEGORY_OF}, soft_skills_in_job=job_soft,
    )
