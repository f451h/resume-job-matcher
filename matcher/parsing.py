"""Reading resumes / job descriptions and pulling out structure."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

_PREFERRED_HEADINGS = re.compile(
    r"^\s*(nice to have|nice-to-have|preferred|bonus|good to have|plus(es)?|desirable)\b.*$", re.I | re.M)
_REQUIRED_HEADINGS = re.compile(
    r"^\s*(requirements?|required|must have|must-have|qualifications|what you.ll need|you have|skills)\b.*$", re.I | re.M)


def read_file(data: bytes, name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    if suffix == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")


@dataclass
class JobSections:
    required: str
    preferred: str


def split_job_description(jd: str) -> JobSections:
    """Separate 'nice to have' text from the rest so missing bonus skills hurt the score less."""
    m = _PREFERRED_HEADINGS.search(jd)
    if not m:
        return JobSections(required=jd, preferred="")
    preferred = jd[m.start():]
    # the preferred block ends at the next heading that isn't another "preferred" heading
    nxt = _REQUIRED_HEADINGS.search(preferred, 1)
    if nxt:
        preferred, rest = preferred[:nxt.start()], preferred[nxt.start():]
        return JobSections(required=jd[:m.start()] + rest, preferred=preferred)
    return JobSections(required=jd[:m.start()], preferred=preferred)


def required_years(jd: str) -> float | None:
    """'3+ years', '2-4 years', 'at least 5 years', 'minimum of 2 years' -> lower bound."""
    pats = [r"(\d+(?:\.\d+)?)\s*\+?\s*(?:-|to|–)?\s*\d*\s*\+?\s*years?",
            r"(?:at least|minimum(?: of)?)\s+(\d+)\s+years?"]
    vals = [float(x) for p in pats for x in re.findall(p, jd, re.I)]
    vals = [v for v in vals if 0 < v <= 20]
    return min(vals) if vals else None


_DATE = r"(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+)?((?:19|20)\d{2})"
_RANGE = re.compile(rf"{_DATE}\s*(?:-|–|—|to)\s*(?:{_DATE}|(present|current|now))", re.I)


_SECTION = re.compile(r"^\s*([A-Z][A-Za-z &/]{2,40}):?\s*$", re.M)
_SKIP_SECTIONS = {"education", "academic background", "qualifications", "certifications", "courses"}


def strip_sections(resume: str, names: set[str] = _SKIP_SECTIONS) -> str:
    """Remove sections like EDUCATION so degree dates don't count as work experience."""
    heads = list(_SECTION.finditer(resume))
    out, last = [], 0
    for i, h in enumerate(heads):
        if h.group(1).strip().lower() in names:
            end = heads[i + 1].start() if i + 1 < len(heads) else len(resume)
            out.append(resume[last:h.start()])
            last = end
    out.append(resume[last:])
    return "".join(out)


def experience_months(resume: str, today: date | None = None) -> int:
    """Total months covered by date ranges outside the education section, with overlapping jobs merged."""
    today = today or date.today()
    spans = []
    for m in _RANGE.finditer(strip_sections(resume)):
        sm, sy, em, ey, present = m.groups()
        start = int(sy) * 12 + (MONTHS[sm[:3].lower()] if sm else 1) - 1
        if present:
            end = today.year * 12 + today.month - 1
        else:
            end = int(ey) * 12 + (MONTHS[em[:3].lower()] if em else 12) - 1
        if end >= start:
            spans.append((start, end))
    spans.sort()
    total, cur_s, cur_e = 0, None, None
    for s, e in spans:
        if cur_e is None or s > cur_e + 1:
            if cur_e is not None:
                total += cur_e - cur_s + 1
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s + 1
    return total
