import os
import re
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Union
from math import radians, cos, sin, asin, sqrt

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from grounding import PERSIST_DIR, COLLECTION_NAME


# ---------------------------
# Helpers / Models
# ---------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth (in miles).
    
    Args:
        lat1, lon1: Latitude and longitude of point 1
        lat2, lon2: Latitude and longitude of point 2
    
    Returns:
        Distance in miles
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of Earth in miles
    r = 3956
    
    return c * r

@dataclass
class CourseResult:
    class_id: str
    title: str
    location: str
    course_type: str
    instructor: str
    cost_gbp: str  # keep as display string
    distance: float = 0.0
    distance_miles: Optional[float] = None  # Distance from user's query location


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
        gemini_model: str = "gemini-2.5-flash",
    ):
        load_dotenv()

        # -----------------------
        # Configure Gemini
        # -----------------------
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in your environment or .env file.")
        self.client = genai.Client(api_key=api_key)
        self.gemini_model = gemini_model

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
        
        # --- Build location coordinate cache ---
        self.location_coords = {}
        for loc in self.locations:
            loc_rows = self.df[self.df["location"] == loc]
            if not loc_rows.empty:
                lat = loc_rows.iloc[0].get("latitude")
                lon = loc_rows.iloc[0].get("longitude")
                if pd.notna(lat) and pd.notna(lon):
                    self.location_coords[loc] = (float(lat), float(lon))
        
        # --- Initialize geocoder for on-the-fly location queries ---
        self.geolocator = Nominatim(user_agent="dandori_course_app_v1.0")
        self.geocode_cache = {}  # Cache for user-provided locations not in dataset
        self.last_geocode_time = 0  # Track last API call for rate limiting

        # --- Chroma persistent client ---
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Get HF token from environment for embeddings
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=local_model_name,
            device="cpu"
        )
        
        self.collection = self.chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )

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
    
    def extract_any_location_from_text(self, text: str) -> Optional[str]:
        """
        Extract a location from text, even if it's not in the dataset.
        First tries to match known locations, then looks for common location patterns.
        
        Args:
            text: User query text
        
        Returns:
            Location name or None
        """
        # First try known locations
        known_locs = self.extract_locations_from_text(text)
        if known_locs:
            return known_locs[0]
        
        # Try to extract unknown location using patterns
        # Patterns like "in London", "near Manchester", "around Leeds"
        location_patterns = [
            r'\b(?:in|near|around|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:area|courses?)\b',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None

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
    
    def get_location_coords(self, location: str) -> Optional[Tuple[float, float]]:
        """
        Get lat/lon coordinates for a location name.
        First checks dataset locations, then tries on-the-fly geocoding for unknown locations.
        
        Args:
            location: Location name (e.g., "Brighton", "York", "London")
        
        Returns:
            (latitude, longitude) tuple or None if not found
        """
        # First check if it's a known location from the dataset
        if location in self.location_coords:
            return self.location_coords[location]
        
        # Check geocode cache for previously queried locations
        if location in self.geocode_cache:
            return self.geocode_cache[location]
        
        # Try to geocode the location on-the-fly
        coords = self._geocode_location(location)
        
        # Cache the result (even if None, to avoid repeated failed lookups)
        self.geocode_cache[location] = coords
        
        return coords
    
    def _geocode_location(self, location_name: str, retry: int = 2) -> Optional[Tuple[float, float]]:
        """
        Geocode a location name using Nominatim API.
        Respects rate limiting (1 request per second).
        
        Args:
            location_name: Name of the location
            retry: Number of retry attempts
        
        Returns:
            (latitude, longitude) tuple or None if failed
        """
        if not location_name or location_name == 'TBC':
            return None
        
        # Add UK context for better results
        query = f"{location_name}, UK"
        
        # Respect Nominatim rate limit (1 request per second)
        current_time = time.time()
        time_since_last = current_time - self.last_geocode_time
        if time_since_last < 1.0:
            time.sleep(1.0 - time_since_last)
        
        for attempt in range(retry):
            try:
                location = self.geolocator.geocode(query, timeout=5)
                self.last_geocode_time = time.time()
                
                if location:
                    return (location.latitude, location.longitude)
                else:
                    return None
                    
            except GeocoderTimedOut:
                if attempt < retry - 1:
                    time.sleep(1)
                continue
            except (GeocoderServiceError, Exception):
                return None
        
        return None

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

    def retrieve(self, query: str, n_results: int = 8, reference_location: Optional[str] = None) -> List[CourseResult]:
        """
        Retrieve top N semantically relevant courses via Chroma.
        Returns CourseResult list populated from metadata.
        
        Args:
            query: User's search query
            n_results: Number of results to return
            reference_location: Optional location name to calculate distances from
        
        Returns:
            List of CourseResult objects, sorted by distance if reference_location provided
        """
        res = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        ids = (res.get("ids") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        
        # Get reference coordinates if location provided
        ref_coords = None
        if reference_location:
            ref_coords = self.get_location_coords(reference_location)

        results: List[CourseResult] = []
        for doc_id, md, dist in zip(ids, metadatas, distances):
            if not md:
                continue
            
            # Calculate distance from reference location if available
            distance_miles = None
            if ref_coords:
                course_lat = md.get("latitude")
                course_lon = md.get("longitude")
                if course_lat is not None and course_lon is not None:
                    distance_miles = haversine_distance(
                        ref_coords[0], ref_coords[1],
                        float(course_lat), float(course_lon)
                    )
                    # Round very small distances (same location) to 0
                    if distance_miles < 0.1:
                        distance_miles = 0.0
            
            results.append(
                CourseResult(
                    class_id=str(doc_id),  # ID is stored as document ID, not in metadata
                    title=str(md.get("title", "")),
                    location=str(md.get("location", "")),
                    course_type=str(md.get("type", "")),  # Stored as "type" in grounding.py
                    instructor=str(md.get("instructor", "")),
                    cost_gbp=self._coerce_cost_display(md),
                    distance=float(dist) if dist is not None else 0.0,
                    distance_miles=distance_miles,
                )
            )
        
        # Sort by distance if reference location was provided and distances calculated
        if ref_coords and any(r.distance_miles is not None for r in results):
            # Put courses with distances first, sorted by distance
            # Then courses without distances (TBC locations)
            results.sort(key=lambda r: (r.distance_miles is None, r.distance_miles if r.distance_miles else 0))
        
        return results

    # ---------------------------
    # E) Deterministic fallback retrieval
    # ---------------------------

    def format_recommendations(self, recs: List[CourseResult], limit: int = 5) -> str:
        if not recs:
            return "I couldn’t find any matching courses."

        lines = ["Here are a few options:"]
        for r in recs[:limit]:
            lines.append(
                f"- **{r.title}** (📍 {r.location}, 💷 {r.cost_gbp}, 🆔 {r.class_id})"
            )
        return "\n".join(lines)


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
        If Gemini fails (model not available / API error), fall back to deterministic formatting.
        """
        if not recs:
            return (
                "I couldn’t find anything that matches that right now. "
                "Could you tell me a location (or if you’re open to anywhere) and a rough budget?"
            )

        context_lines = []
        for r in recs:
            # Include distance if available
            distance_info = ""
            if r.distance_miles is not None:
                distance_info = f" | distance={r.distance_miles:.1f} miles"
            
            context_lines.append(
                f"- {r.title} | id={r.class_id} | location={r.location} | type={r.course_type} | "
                f"instructor={r.instructor} | cost={r.cost_gbp}{distance_info}"
            )

        prompt = f"""
    You are the School of Dandori Course Chatbot.

    Rules:
    - Only recommend courses that appear in the provided matches.
    - Do NOT list course details (title, location, price, ID) in your response - they will be shown separately in cards below your message.
    - Instead, provide a brief, friendly introduction (1-2 sentences) about the courses found.
    - If distance information is provided, you can mention proximity (e.g., "I found some lovely options nearby").
    - If the user request is missing key info (like location or budget), ask ONE follow-up question.
    - Keep the tone warm and playful but concise.
    - Focus on the vibe/theme of the courses rather than listing them.

    User request:
    {user_query}

    Matches:
    {chr(10).join(context_lines)}

    Write a brief, friendly response (1-2 sentences max) introducing the courses without listing their details:
    """

        try:
            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=prompt
            )
            return response.text.strip() if response.text else self.format_recommendations(recs)
        except Exception:
            # Do NOT leak Gemini errors to the user
            return self.format_recommendations(recs)

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