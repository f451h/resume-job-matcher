"""Skill taxonomy and extraction.

Each canonical skill maps to the different ways people write it ("scikit-learn",
"sklearn", "scikit learn"). Matching on aliases means a resume that says "sklearn"
still counts for a job that asks for "scikit-learn".
"""
from __future__ import annotations

import re

SKILLS: dict[str, dict[str, list[str]]] = {
    "Programming": {
        "Python": ["python"], "R": [r"\bR\b(?!&)"], "SQL": ["sql"], "Java": ["java(?!script)"],
        "C++": [r"c\+\+"], "C#": ["c#"], "JavaScript": ["javascript", r"\bjs\b"], "TypeScript": ["typescript"],
        "Go": ["golang"], "Scala": ["scala"], "Bash": ["bash", "shell scripting"],
    },
    "Machine Learning": {
        "Machine Learning": ["machine learning", r"\bml\b"], "Deep Learning": ["deep learning"],
        "Scikit-learn": ["scikit-learn", "sklearn", "scikit learn"], "TensorFlow": ["tensorflow"],
        "PyTorch": ["pytorch", "torch"], "Keras": ["keras"], "XGBoost": ["xgboost"], "LightGBM": ["lightgbm"],
        "Computer Vision": ["computer vision", "image classification", "object detection"],
        "NLP": ["nlp", "natural language processing"], "Time Series": ["time series", "forecasting"],
        "Feature Engineering": ["feature engineering"], "Model Deployment": ["model deployment", "mlops", "model serving"],
        "Statistics": ["statistics", "statistical analysis", "hypothesis testing", "a/b testing"],
        "Recommender Systems": ["recommender system", "recommendation system"],
    },
    "LLMs & GenAI": {
        "LLMs": ["llm", "llms", "large language model"], "RAG": ["rag", "retrieval-augmented", "retrieval augmented"],
        "LangChain": ["langchain"], "LangGraph": ["langgraph"], "LlamaIndex": ["llamaindex", "llama index"],
        "Hugging Face": ["hugging face", "huggingface", "transformers library"], "Prompt Engineering": ["prompt engineering"],
        "Vector Databases": ["vector database", "vector db", "pinecone", "chromadb", "chroma", "faiss", "qdrant", "weaviate"],
        "Fine-tuning": ["fine-tuning", "fine tuning", "finetuning", "lora", "peft"], "AI Agents": ["ai agent", "agentic", "agents"],
        "OpenAI API": ["openai api", "gpt-4", "chatgpt api"],
    },
    "Data": {
        "Pandas": ["pandas"], "NumPy": ["numpy"], "Spark": ["spark", "pyspark"], "Airflow": ["airflow"],
        "dbt": [r"\bdbt\b"], "ETL": ["etl", "data pipeline"], "Data Visualization": ["data visualization", "matplotlib", "seaborn", "plotly"],
        "Power BI": ["power bi", "powerbi"], "Tableau": ["tableau"], "Excel": ["excel"],
        "PostgreSQL": ["postgresql", "postgres"], "MySQL": ["mysql"], "MongoDB": ["mongodb"], "BigQuery": ["bigquery"],
    },
    "Engineering & Cloud": {
        "Docker": ["docker"], "Kubernetes": ["kubernetes", "k8s"], "AWS": ["aws", "amazon web services", "sagemaker"],
        "GCP": ["gcp", "google cloud"], "Azure": ["azure"], "Git": ["git", "github", "gitlab"], "CI/CD": ["ci/cd", "github actions", "jenkins"],
        "FastAPI": ["fastapi"], "Flask": ["flask"], "Django": ["django"], "REST APIs": ["rest api", "restful", "rest apis"],
        "Linux": ["linux"], "Streamlit": ["streamlit"],
    },
    "Web": {
        "React": ["react", "reactjs", "react.js"], "Next.js": ["next.js", "nextjs"], "Node.js": ["node.js", "nodejs", "node js"],
        "HTML/CSS": ["html", "css"], "Tailwind CSS": ["tailwind"],
    },
    "Soft skills": {
        "Communication": ["communication", "communicate"], "Teamwork": ["teamwork", "collaborat"],
        "Leadership": ["leadership", "led a team", "mentor"], "Problem Solving": ["problem solving", "problem-solving"],
        "Stakeholder Management": ["stakeholder"],
    },
}

_PATTERNS: list[tuple[str, str, re.Pattern]] = []
for category, skills in SKILLS.items():
    for name, aliases in skills.items():
        for alias in aliases:
            # an alias that already contains regex syntax is used as-is, otherwise match on word boundaries
            body = alias if alias.startswith(r"\b") or "(?" in alias else rf"(?<![\w+#.]){alias}(?![\w+#])"
            flags = 0 if name == "R" else re.IGNORECASE
            _PATTERNS.append((category, name, re.compile(body, flags)))


def extract_skills(text: str) -> dict[str, str]:
    """Return {skill: category} for every skill mentioned in the text."""
    found: dict[str, str] = {}
    for category, name, pattern in _PATTERNS:
        if name not in found and pattern.search(text):
            found[name] = category
    return found


# Knowing a specific tool implies the broader skill, e.g. TensorFlow -> Deep Learning.
IMPLIES = {
    "TensorFlow": ["Deep Learning", "Machine Learning"], "PyTorch": ["Deep Learning", "Machine Learning"],
    "Keras": ["Deep Learning", "Machine Learning"], "Scikit-learn": ["Machine Learning"],
    "XGBoost": ["Machine Learning"], "LightGBM": ["Machine Learning"], "Deep Learning": ["Machine Learning"],
    "LangChain": ["LLMs"], "LlamaIndex": ["LLMs"], "OpenAI API": ["LLMs"], "RAG": ["LLMs"], "Fine-tuning": ["LLMs"],
    "PostgreSQL": ["SQL"], "MySQL": ["SQL"], "BigQuery": ["SQL"], "FastAPI": ["REST APIs"], "Flask": ["REST APIs"],
    "Next.js": ["React"], "Airflow": ["ETL"], "dbt": ["ETL"], "Kubernetes": ["Docker"],
}

SOFT_SKILLS = set(SKILLS["Soft skills"])


def with_implied(skills: dict[str, str]) -> dict[str, str]:
    out = dict(skills)
    for s in skills:
        for parent in IMPLIES.get(s, []):
            out.setdefault(parent, CATEGORY_OF[parent])
    return out


CATEGORY_OF = {name: cat for cat, group in SKILLS.items() for name in group}

_OR = re.compile(r"([\w.+#/-]+(?:\s[\w.+#/-]+)?)\s+or\s+([\w.+#/-]+(?:\s[\w.+#/-]+)?)", re.I)


def requirement_groups(text: str) -> list[set[str]]:
    """Turn job text into requirements. 'PyTorch or TensorFlow' becomes one requirement either skill satisfies."""
    found = extract_skills(text)
    groups: list[set[str]] = []
    grouped: set[str] = set()
    for m in _OR.finditer(text):
        left, right = set(extract_skills(m.group(1))), set(extract_skills(m.group(2)))
        if left and right:
            g = left | right
            groups.append(g)
            grouped |= g
    groups += [{s} for s in found if s not in grouped]
    # merge duplicate groups
    unique = []
    for g in groups:
        if g not in unique:
            unique.append(g)
    return unique
