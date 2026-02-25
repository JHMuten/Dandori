import os
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Union

import pandas as pd
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai


# ---------------------------
# Helpers / Models
# ---------------------------

@dataclass
class CourseResult:
    class_id: str
    title: str
    location: str
    course_type: str
    instructor: str
    cost_gbp: str  # keep as display string
    distance: float = 0.0


# ---------------------------
# Course Recommender
# ---------------------------

class CourseRecommender:
    """
    Loads course data, provides:
      - deterministic counting/filtering (for "how many..." questions)
      - semantic retrieval via Chroma
      - deterministic fallback retrieval if semantic retrieval returns nothing
      - LLM response generation
    """

    def __init__(
        self,
        dataset_path: str = "data/courses.pkl",
        persist_dir: str = PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        local_model_name: str = "all-MiniLM-L6-v2",
        gemini_model: str = "gemini-1.5-flash",
    ):
        # -----------------------
        # Configure Gemini
        # -----------------------
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in your environment or .env file.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(gemini_model)

        # -----------------------
        # Load and normalise dataset
        # -----------------------
        self.dataset_path = dataset_path
        self.df = pd.read_pickle(self.dataset_path).copy()

        # Normalize class_id
        if "class_id" in self.df.columns:
            self.df["class_id"] = self.df["class_id"].astype(str)
            self.df = self.df.drop_duplicates(subset=["class_id"], keep="last").reset_index(drop=True)

        # Text columns safety
        for c in ["title", "description", "location", "course_type", "instructor"]:
            if c in self.df.columns:
                self.df[c] = self.df[c].fillna("")

        # Ensure a stable numeric cost field for filtering
        self._ensure_cost_num()

        # --- Locations for quick extraction ---
        self.locations = sorted([x for x in self.df.get("location", pd.Series([], dtype=str)).unique().tolist() if x])

        # --- Chroma persistent client ---
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(name=chroma_collection)

    # ---------------------------
    # Deterministic utilities
    # ---------------------------

    def _ensure_cost_num(self) -> None:
        """
        Ensures self.df has a numeric 'cost_num' column for robust filtering.
        Handles strings like '£60', '60', 'Free', '£50–£80', etc.
        For ranges, uses the lower bound as a pragmatic numeric proxy for filtering.
        """
        if "cost_num" in self.df.columns:
            return

        if "cost_gbp" not in self.df.columns:
            self.df["cost_num"] = pd.NA
            return

        s = self.df["cost_gbp"].astype(str).fillna("").str.strip()

        # Common free patterns
        free_mask = s.str.match(r"^(free|£?0(\.0+)?|no charge)$", case=False, na=False)

        # Extract any numbers (including decimals)
        nums = s.str.findall(r"\d+(?:\.\d+)?")

        def to_num(x: List[str]) -> Optional[float]:
            if not x:
                return None
            try:
                return float(x[0])  # lower bound proxy if multiple numbers exist
            except Exception:
                return None

        cost_num = nums.apply(to_num)
        cost_num = pd.to_numeric(cost_num, errors="coerce")
        cost_num[free_mask] = 0.0

        self.df["cost_num"] = cost_num

    def total_courses(self) -> int:
        return int(len(self.df))

    # ---------------------------
    # Location extraction (single + multi)
    # ---------------------------

    def extract_locations_from_text(self, text: str) -> List[str]:
        """
        Returns ALL locations mentioned in the text.
        Works for:
            - "Brighton and York"
            - "Brighton, York, Bath"
            - "near Brighton / York"
        Handles any number of locations.
        """
        if not text:
            return []

        t = text.lower()
        matches: List[str] = []

        # Prefer longest first to avoid partial overlaps
        for loc in sorted(self.locations, key=len, reverse=True):
            loc_l = loc.lower().strip()
            if not loc_l:
                continue
            if re.search(rf"(?<!\w){re.escape(loc_l)}(?!\w)", t):
                matches.append(loc)

        # Deduplicate while preserving order
        seen = set()
        out: List[str] = []
        for m in matches:
            if m not in seen:
                out.append(m)
                seen.add(m)

        return out

    def extract_location_from_text(self, text: str) -> Optional[str]:
        """
        Backwards-compatible single-location helper.
        Returns the first detected location if multiple exist.
        """
        locs = self.extract_locations_from_text(text)
        return locs[0] if locs else None

    def locations_label(self, locs: List[str]) -> str:
        """
        For nicer phrasing in replies:
          ["Brighton"] -> "Brighton"
          ["Brighton","York"] -> "Brighton and York"
          ["Brighton","York","Bath"] -> "Brighton, York and Bath"
        """
        locs = [l for l in locs if l]
        if not locs:
            return ""
        if len(locs) == 1:
            return locs[0]
        if len(locs) == 2:
            return f"{locs[0]} and {locs[1]}"
        return f"{', '.join(locs[:-1])} and {locs[-1]}"

    # ---------------------------
    # Deterministic counting
    # ---------------------------

    def count_filtered(
        self,
        location: Optional[Union[str, List[str]]] = None,
        price_mode: Optional[str] = None,
        a: Optional[float] = None,
        b: Optional[float] = None,
    ) -> int:
        """
        Generic deterministic counter for intersections:
          - location: None | str | list[str]
          - price_mode: {"between", "above", "at_least", "below", "at_most", "exact"}
          - a/b thresholds depending on mode
        """
        df = self.df

        # Location filter (supports multi)
        if location and "location" in df.columns:
            if isinstance(location, list):
                locs = [x.strip().lower() for x in location if x and str(x).strip()]
                if locs:
                    df = df[df["location"].fillna("").str.strip().str.lower().isin(locs)]
            else:
                loc = str(location).strip().lower()
                df = df[df["location"].fillna("").str.strip().str.lower() == loc]

        # Price filter
        if price_mode:
            self._ensure_cost_num()
            c = pd.to_numeric(df["cost_num"], errors="coerce")

            if price_mode == "above":          # strictly >
                df = df[c > float(a)]
            elif price_mode == "at_least":     # >=
                df = df[c >= float(a)]
            elif price_mode == "below":        # strictly <
                df = df[c < float(a)]
            elif price_mode == "at_most":      # <=
                df = df[c <= float(a)]
            elif price_mode == "between":
                lo, hi = float(a), float(b)
                df = df[(c >= lo) & (c <= hi)]
            elif price_mode == "exact":
                df = df[c == float(a)]

        return int(len(df))

    # ---------------------------
    # Retrieval
    # ---------------------------

    def _coerce_cost_display(self, md: Dict[str, Any]) -> str:
        """
        Normalise cost for display (string).
        """
        v = md.get("cost", md.get("cost_gbp", "Unknown"))
        if v is None:
            return "Unknown"
        s = str(v).strip()
        return s if s else "Unknown"

    def retrieve(self, query: str, n_results: int = 8) -> List[CourseResult]:
        """
        Retrieve top N semantically relevant courses via Chroma.
        Returns CourseResult list populated from metadata.
        """
        res = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]

        results: List[CourseResult] = []
        for md, dist in zip(metadatas, distances):
            if not md:
                continue
            results.append(
                CourseResult(
                    class_id=str(md.get("class_id", "")),
                    title=str(md.get("title", "")),
                    location=str(md.get("location", "")),
                    course_type=str(md.get("course_type", "")),
                    instructor=str(md.get("instructor", "")),
                    cost_gbp=self._coerce_cost_display(md),
                    distance=float(dist) if dist is not None else 0.0,
                )
            )
        return results

    # ---------------------------
    # E) Deterministic fallback retrieval
    # ---------------------------

    def fallback_search(
        self,
        query: str,
        locations: Optional[List[str]] = None,
        price_filter: Optional[Tuple[str, float, Optional[float]]] = None,
        limit: int = 8,
    ) -> List[CourseResult]:
        """
        Deterministic fallback when vector retrieval returns nothing.
        Filters by location(s) + price if provided, then does simple keyword scoring.
        """
        df = self.df.copy()

        # Location filter
        if locations:
            locs = [x.strip().lower() for x in locations if x and str(x).strip()]
            if locs:
                df = df[df["location"].fillna("").str.strip().str.lower().isin(locs)]

        # Price filter
        if price_filter:
            mode, a, b = price_filter
            self._ensure_cost_num()
            c = pd.to_numeric(df["cost_num"], errors="coerce")

            if mode == "above":
                df = df[c > float(a)]
            elif mode == "at_least":
                df = df[c >= float(a)]
            elif mode == "below":
                df = df[c < float(a)]
            elif mode == "at_most":
                df = df[c <= float(a)]
            elif mode == "between":
                lo, hi = float(a), float(b)
                df = df[(c >= lo) & (c <= hi)]
            elif mode == "exact":
                df = df[c == float(a)]

        if df.empty:
            return []

        # Simple keyword scoring (generic + fast)
        q = (query or "").strip().lower()
        tokens = [t for t in re.findall(r"[a-zA-Z']+", q) if len(t) >= 3]

        def score_row(row) -> int:
            hay = " ".join([
                str(row.get("title", "")),
                str(row.get("description", "")),
                str(row.get("course_type", "")),
                str(row.get("instructor", "")),
            ]).lower()
            return sum(1 for tok in tokens if tok in hay)

        df["_score"] = df.apply(score_row, axis=1)
        df = df.sort_values(["_score"], ascending=False).head(limit)

        out: List[CourseResult] = []
        for _, r in df.iterrows():
            out.append(
                CourseResult(
                    class_id=str(r.get("class_id", "")),
                    title=str(r.get("title", "")),
                    location=str(r.get("location", "")),
                    course_type=str(r.get("course_type", "")),
                    instructor=str(r.get("instructor", "")),
                    cost_gbp=str(r.get("cost_gbp", "Unknown")),
                    distance=0.0,
                )
            )
        return out

    # ---------------------------
    # LLM Response
    # ---------------------------

    def respond(self, user_query: str, recs: List[CourseResult]) -> str:
        """
        Uses Gemini to generate a friendly response based on retrieved matches.
        """
        if not recs:
            return (
                "I couldn’t find anything that matches that right now. "
                "Could you tell me a location (or if you’re open to anywhere) and a rough budget?"
            )

        context_lines = []
        for r in recs:
            context_lines.append(
                f"- {r.title} | id={r.class_id} | location={r.location} | type={r.course_type} | "
                f"instructor={r.instructor} | cost={r.cost_gbp}"
            )

        prompt = f"""
You are the School of Dandori Course Chatbot.

Rules:
- Only recommend courses that appear in the provided matches.
- Do not invent details (dates, prerequisites, materials) unless explicitly in the match line.
- If the user request is missing key info (like location or budget), ask ONE follow-up question.
- Provide 3-5 suggestions maximum.
- Keep the tone warm and playful but concise.

User request:
{user_query}

Matches:
{chr(10).join(context_lines)}

Write the reply:
"""

        try:
            out = self.model.generate_content(prompt)
            return (out.text or "").strip() or "Sorry — I ran into an issue generating a reply."
        except Exception as e:
            return f"Sorry — I ran into an error generating a reply: {e}"

    def respond_smart(self, user_query: str, recs: List[CourseResult], has_location: bool, has_budget: bool) -> str:
        """
        Smarter empty-results fallback:
        - If user already provided location/budget, don't ask again.
        - Ask for what's missing or ask for "vibe/type" instead.
        """
        if recs:
            return self.respond(user_query, recs)

        if not has_location and not has_budget:
            return "I can help you find a class — what location are you looking in, and what’s your rough budget?"
        if not has_location:
            return "Which location should I search in? (Or tell me if you’re open to anywhere.)"
        if not has_budget:
            return "What’s your rough budget for the class?"
        return "Got it. What vibe are you after — art, movement, storytelling, or something else?"