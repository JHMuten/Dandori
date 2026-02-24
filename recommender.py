from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai

PERSIST_DIR = "data/courses_db"
COLLECTION_NAME = "course_catalog"

_BUDGET_RE = re.compile(r"(?:under|below|less than|<=?)\s*£?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_budget_gbp(text: str) -> Optional[float]:
    m = _BUDGET_RE.search(text or "")
    return float(m.group(1)) if m else None


@dataclass
class Recommendation:
    class_id: str
    title: str
    location: str
    instructor: str
    course_type: str
    cost_gbp: Any
    distance: Optional[float] = None


class CourseRecommender:
    def __init__(
        self,
        persist_dir: str = PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        local_model_name: str = "all-MiniLM-L6-v2",
        gemini_model: str = "gemini-2.5-flash",
        dataset_path: str = "data/courses.pkl",
    ):
        # -----------------------
        # Load dataset (for deterministic counts)
        # -----------------------
        self.df = pd.read_pickle(dataset_path)
        self.df["class_id"] = self.df["class_id"].astype(str)
        self.df = self.df.drop_duplicates(subset=["class_id"], keep="last").reset_index(drop=True)

        # Known locations for simple extraction
        self.known_locations = sorted(
            {str(x).strip() for x in self.df.get("location", pd.Series([])).dropna().unique() if str(x).strip()}
        )

        # -----------------------
        # Local embeddings (must match grounding.py)
        # -----------------------
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=local_model_name
        )
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

        # -----------------------
        # Gemini (dotenv)
        # -----------------------
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Put it in your .env file.")
        self.gemini = genai.Client(api_key=api_key)
        self.gemini_model = gemini_model

    # -----------------------
    # Deterministic helpers
    # -----------------------
    def total_courses(self) -> int:
        # True total (unique class_id)
        return int(len(self.df))

    def count_in_location(self, location: str) -> int:
        loc = (location or "").strip().lower()
        if "location" not in self.df.columns:
            return 0
        return int((self.df["location"].fillna("").str.strip().str.lower() == loc).sum())

    def extract_location_from_text(self, text: str) -> Optional[str]:
        t = (text or "").lower()
        for loc in self.known_locations:
            if loc.lower() in t:
                return loc
        return None

    def _ensure_cost_num(self) -> None:
        """
        Ensure a numeric cost column exists for price-based counting.
        This also protects you from Streamlit holding an older recommender instance.
        """
        if "cost_num" in self.df.columns:
            return

        # Prefer cost_gbp, but tolerate missing column
        if "cost_gbp" not in self.df.columns:
            self.df["cost_num"] = pd.Series([pd.NA] * len(self.df))
            return

        self.df["cost_num"] = (
            self.df["cost_gbp"]
            .astype(str)
            .str.replace("£", "", regex=False)
            .str.strip()
        )
        self.df["cost_num"] = pd.to_numeric(self.df["cost_num"], errors="coerce")

    def count_above_price(self, price: float) -> int:
        self._ensure_cost_num()
        return int((self.df["cost_num"] > float(price)).sum())

    def count_below_price(self, price: float) -> int:
        self._ensure_cost_num()
        return int((self.df["cost_num"] < float(price)).sum())

    def count_between_prices(self, lo: float, hi: float) -> int:
        self._ensure_cost_num()
        lo, hi = float(lo), float(hi)
        return int(((self.df["cost_num"] >= lo) & (self.df["cost_num"] <= hi)).sum())

    def count_exact_price(self, price: float) -> int:
        self._ensure_cost_num()
        return int((self.df["cost_num"] == float(price)).sum())

    # -----------------------
    # Retrieval (local embeddings)
    # -----------------------
    def retrieve(self, user_query: str, n_results: int = 8) -> List[Recommendation]:
        if not user_query or not user_query.strip():
            return []

        # Optional: budget filter from natural language ("under £80")
        max_budget = parse_budget_gbp(user_query)

        res = self.collection.query(
            query_texts=[user_query],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        metadatas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]

        recs: List[Recommendation] = []
        for cid, md, dist in zip(ids, metadatas, dists):
            if not md:
                continue

            if max_budget is not None:
                try:
                    if float(md.get("cost")) > float(max_budget):
                        continue
                except Exception:
                    pass

            recs.append(
                Recommendation(
                    class_id=str(cid),
                    title=md.get("title", ""),
                    location=md.get("location", ""),
                    instructor=md.get("instructor", ""),
                    course_type=md.get("type", ""),
                    cost_gbp=md.get("cost", ""),
                    distance=float(dist) if dist is not None else None,
                )
            )

        return recs[:5]

    # -----------------------
    # Generation (Gemini)
    # -----------------------
    def respond(self, user_query: str, recs: List[Recommendation]) -> str:
        if not recs:
            prompt = (
                "You are a friendly course concierge for the School of Dandori.\n"
                f"User request: {user_query}\n\n"
                "No matching courses were found from the catalogue.\n"
                "Ask ONE short follow-up question to clarify (location, budget, or course type)."
            )
        else:
            courses_text = "\n".join(
                [
                    f"- {r.title} (Class ID: {r.class_id}) | {r.location} | £{r.cost_gbp} | "
                    f"{r.course_type} | Instructor: {r.instructor}"
                    for r in recs
                ]
            )

            prompt = (
                "You are a friendly course concierge for the School of Dandori.\n"
                "Only recommend courses from the list provided.\n\n"
                f"User request: {user_query}\n\n"
                "Retrieved courses:\n"
                f"{courses_text}\n\n"
                "Instructions:\n"
                "- Recommend up to 3 courses.\n"
                "- Give a 1–2 sentence reason for each.\n"
                "- If a key detail is missing, ask ONE follow-up question.\n"
                "- Keep it playful but not cringe.\n"
            )

        resp = self.gemini.models.generate_content(
            model=self.gemini_model,
            contents=prompt,
        )
        return getattr(resp, "text", str(resp))