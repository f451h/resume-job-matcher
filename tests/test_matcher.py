from datetime import date
from pathlib import Path

import pytest

from matcher import SemanticScorer, match
from matcher.feedback import parse_feedback
from matcher.parsing import experience_months, required_years, split_job_description
from matcher.skills import extract_skills

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
RESUME = (SAMPLES / "resume_ayesha.txt").read_text()
ML_JOB = (SAMPLES / "job_ml_engineer.txt").read_text()
DA_JOB = (SAMPLES / "job_data_analyst.txt").read_text()


@pytest.mark.parametrize("text,skill", [
    ("Built models with sklearn", "Scikit-learn"),
    ("Experience in C++ and C#", "C++"),
    ("Experience in C++ and C#", "C#"),
    ("Frontend in React.js and Next.js", "Next.js"),
    ("Deployed with GitHub Actions", "CI/CD"),
    ("Worked on retrieval-augmented generation", "RAG"),
])
def test_aliases_are_recognised(text, skill):
    assert skill in extract_skills(text)


def test_no_false_positives_on_common_words():
    skills = extract_skills("Javascript developer. Reading, writing and arithmetic. Go to the market.")
    assert "Java" not in skills and "R" not in skills and "Go" not in skills


def test_r_language_needs_capital_r():
    assert "R" in extract_skills("Statistical modelling in R and Python")
    assert "R" not in extract_skills("r and d team")


def test_required_years():
    assert required_years("2+ years of experience") == 2
    assert required_years("at least 5 years in data") == 5
    assert required_years("3-5 years") == 3
    assert required_years("No experience needed") is None


def test_experience_merges_overlapping_jobs():
    text = "Job A Jan 2020 - Dec 2020\nJob B Jul 2020 - Jun 2021\nJob C 2023 - Present"
    months = experience_months(text, today=date(2024, 12, 1))
    assert months == 18 + 24          # 2020-01..2021-06, plus 2023-01..2024-12


def test_preferred_section_is_separated():
    s = split_job_description(ML_JOB)
    assert "Kubernetes" in s.preferred and "Kubernetes" not in s.required
    assert "FastAPI" in s.required


def test_match_on_sample_resume():
    r = match(RESUME, ML_JOB, SemanticScorer(use_embeddings=False))
    assert 0 <= r.score <= 100
    assert "Python" in r.matched_required and "Docker" in r.missing_required
    assert "TensorFlow" in r.matched_required            # satisfies "PyTorch or TensorFlow"
    assert not any("PyTorch" in m for m in r.missing_required)
    assert "AWS or Kubernetes" in r.missing_preferred
    assert 5 <= r.resume_years <= 6.5                    # education dates are ignored
    assert "Communication" in r.soft_skills_in_job


def test_analyst_job_fits_better_than_ml_job():
    scorer = SemanticScorer(use_embeddings=False)
    assert match(RESUME, DA_JOB, scorer).score > match(RESUME, ML_JOB, scorer).score


def test_parse_feedback_drops_bad_items():
    raw = '{"summary": "ok", "strengths": ["a"], "gaps": [], "bullet_rewrites": [{"original": "x", "improved": "y"}, "bad"], "keywords_to_add": ["Docker"]}'
    fb = parse_feedback(raw)
    assert fb["bullet_rewrites"] == [{"original": "x", "improved": "y"}]
    assert fb["keywords_to_add"] == ["Docker"]


def test_or_requirements_and_implied_skills():
    from matcher.skills import requirement_groups, with_implied, extract_skills
    groups = requirement_groups("Experience with PyTorch or TensorFlow. Also Docker.")
    assert {"PyTorch", "TensorFlow"} in groups and {"Docker"} in groups
    assert "Deep Learning" in with_implied(extract_skills("Built CNNs in Keras"))


def test_education_dates_are_not_experience():
    text = "EXPERIENCE\nAnalyst Jan 2022 - Dec 2022\n\nEDUCATION\nBS CS 2017 - 2021\n"
    assert experience_months(text) == 12
