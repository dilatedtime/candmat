"""The JD distilled into a structured, weighted rubric.

This is the hand-authored distillation of ``job_description.docx`` (Senior AI
Engineer, Founding Team @ Redrob AI). The JD is unusually explicit about what it
rewards and what it considers a trap, so we encode that directly rather than
inferring it. The philosophy — "credit transferable evidence, don't reward
keyword matching, penalise contradiction" — is ported from verdix's
``MATCH_FIT_PROMPT`` / ``_jd_required_skills`` (resume-matcher) and
``analyze_job_description`` (agent-python engine).

Every vocabulary entry is lower-cased; matching is alias-aware and evidence-based
(see ``ranker/text.py``). Nothing here calls a network or an LLM — it is a static
rubric consumed by the deterministic scorer.
"""

from __future__ import annotations

# ── Core role identity ───────────────────────────────────────────────────────
# Titles that signal the candidate actually does engineering of the right kind.
# Grounded in the 47 observed titles; only these read as real SWE/ML/data work.
STRONG_TITLES = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "data engineer", "analytics engineer", "data scientist",
    "backend engineer", "software engineer", "full stack developer",
    "senior software engineer", "senior software engineer (ml)",
    "cloud engineer", "devops engineer",
}
# Engineering-adjacent: real code, but further from the ML/IR core.
ADJACENT_TITLES = {
    "frontend engineer", "mobile developer", "qa engineer",
    "java developer", ".net developer", "data analyst",
}
# Titles that, when paired with a stuffed AI skill list, mark a keyword stuffer.
# (JD: "all the AI keywords ... but whose title is 'Marketing Manager' is not a fit")
NON_ENGINEERING_TITLES = {
    "marketing manager", "hr manager", "sales executive", "accountant",
    "content writer", "graphic designer", "civil engineer", "mechanical engineer",
    "operations manager", "customer support", "project manager", "business analyst",
}

# ── Must-have competencies (the JD's "things you absolutely need") ────────────
# Weighted; the retrieval/ranking/vector core is what the role is actually about.
MUST_HAVE_SKILLS: dict[str, float] = {
    "embeddings": 1.0,
    "retrieval": 1.0,
    "information retrieval": 1.0,
    "ranking": 1.0,
    "learning to rank": 1.0,
    "recommendation systems": 0.9,
    "vector search": 1.0,
    "semantic search": 1.0,
    "hybrid search": 1.0,
    # vector DBs / search infra — JD lists these as interchangeable evidence
    "faiss": 0.85, "pinecone": 0.85, "weaviate": 0.85, "qdrant": 0.85,
    "milvus": 0.85, "opensearch": 0.8, "elasticsearch": 0.8, "bm25": 0.8,
    # eval literacy
    "ndcg": 0.7, "mrr": 0.6, "map": 0.5, "a/b testing": 0.6,
    # strong python / ML foundation
    "python": 0.5, "nlp": 0.8, "transformers": 0.6,
    "sentence-transformers": 0.7, "bge": 0.5, "e5": 0.5,
}
# Nice-to-have ("we'd like but won't reject"): smaller positive weight.
NICE_TO_HAVE_SKILLS: dict[str, float] = {
    "fine-tuning llms": 0.4, "lora": 0.3, "qlora": 0.3, "peft": 0.3,
    "xgboost": 0.3, "learning-to-rank": 0.4,
    "distributed systems": 0.3, "spark": 0.2, "kafka": 0.2, "airflow": 0.2,
    "feature engineering": 0.25, "mlflow": 0.2, "weights & biases": 0.2,
}

# Skills the JD warns are NOT the core (CV/speech/robotics without NLP/IR).
# Presence isn't fatal, but a profile that is ONLY these is down-weighted.
OFF_DOMAIN_SKILLS = {
    "image classification", "object detection", "gans", "diffusion models",
    "speech recognition", "tts", "image segmentation", "robotics", "slam",
    "photoshop", "content writing", "figma", "illustrator",
}

# ── Career-history evidence phrases (the Tier-5 detector) ─────────────────────
# JD: a candidate who never says "RAG"/"Pinecone" but whose CAREER HISTORY shows
# they built a recommender/search/ranking system at a product company IS a fit.
# These are matched against career_history[].description, not the skills array.
CAREER_FIT_PHRASES = (
    "recommend", "ranking", "ranker", "retrieval", "search system",
    "search engine", "embedding", "vector", "personaliz", "relevance",
    "matching", "semantic", "information retrieval", "recsys",
    "collaborative filtering", "learning to rank", "nearest neighbor",
)
EVAL_PHRASES = ("ndcg", "mrr", "map", "a/b test", "offline eval", "online eval",
                "precision@", "recall@", "relevance metric")

# ── Negative signals / disqualifiers (the JD's "explicitly do NOT want") ──────
CONSULTING_FIRMS = (
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "mphasis", "ltimindtree",
)
RESEARCH_ONLY_MARKERS = (
    "phd researcher", "research scientist", "postdoc", "academic", "research lab",
    "published", "thesis",
)
LANGCHAIN_ONLY_MARKERS = ("langchain", "prompt engineering")

# Geography the JD prefers (Noida/Pune; nearby metros welcome).
PREFERRED_LOCATIONS = ("noida", "pune", "delhi", "ncr", "gurgaon", "gurugram",
                       "hyderabad", "mumbai", "bangalore", "bengaluru")
PRODUCT_INDUSTRIES = (
    "software", "fintech", "saas", "ai/ml", "ai services", "conversational ai",
    "e-commerce", "edtech", "adtech", "gaming", "healthtech", "food delivery",
    "insurance tech", "transportation",
)
SERVICES_INDUSTRIES = ("it services", "consulting", "conglomerate", "paper products")

# Experience band the JD names (5–9 yrs, soft).
TARGET_YOE_MIN, TARGET_YOE_LOW, TARGET_YOE_HIGH, TARGET_YOE_MAX = 4.0, 6.0, 8.0, 11.0

# ── Fusion weights (how the component scores combine) ─────────────────────────
# Top-10/50 ordering dominates the metric, so the role-fit components carry most
# weight; semantic recall surfaces Tier-5s; signals modulate availability.
WEIGHTS = {
    "semantic": 0.30,      # cosine(JD, candidate composed text)
    "skill_fit": 0.28,     # trusted must-have skill coverage
    "career_fit": 0.24,    # evidence in career history (catches Tier-5s)
    "experience": 0.10,    # YoE band + applied-ML share
    "context": 0.08,       # product-company + geo + eval literacy
}

# JD text kept for embedding the query side (kept short + focused on the core).
JD_QUERY_TEXT = (
    "Senior AI Engineer for a talent-intelligence product. Owns the ranking, "
    "retrieval and matching systems that decide what recruiters see. Needs "
    "production experience with embeddings-based retrieval (sentence-transformers, "
    "BGE, E5), vector databases and hybrid search (FAISS, Pinecone, Weaviate, "
    "Qdrant, Milvus, OpenSearch, Elasticsearch), strong Python, and rigorous "
    "evaluation of ranking systems (NDCG, MRR, MAP, offline-to-online, A/B tests). "
    "Has shipped an end-to-end ranking, search or recommendation system to real "
    "users at a product company. Not a researcher-only profile, not a "
    "LangChain-only profile, not a title-chaser, not pure consulting-services."
)
