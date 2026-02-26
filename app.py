import re
import math
import streamlit as st
import pandas as pd
import chromadb
from recommender import CourseRecommender
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------
# Page config MUST be the first Streamlit call
# ---------------------------
st.set_page_config(page_title="Course Search", page_icon="📚", layout="wide")

# ---------------------------
# Deterministic count + price parsing for chatbot
# ---------------------------
COUNT_Q_RE = re.compile(r"\b(how many|number of courses|total courses|total number)\b", re.I)

PRICE_BETWEEN_RE = re.compile(
    r"\b(?:between|from)\s*£?\s*(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*£?\s*(\d+(?:\.\d+)?)\b",
    re.I,
)
PRICE_RANGE_DASH_RE = re.compile(
    r"\b£?\s*(\d+(?:\.\d+)?)\s*[-–]\s*£?\s*(\d+(?:\.\d+)?)\b",
    re.I,
)

PRICE_PLUS_RE = re.compile(r"(?:^|\s)£?\s*(\d+(?:\.\d+)?)\s*\+\s*(?=$|[^\w])", re.I)

PRICE_NUM_AND_ABOVE_RE = re.compile(
    r"\b£?\s*(\d+(?:\.\d+)?)\s*(?:\+|and\s+(?:above|over)|or\s+more)\s*(?=$|[^\w])",
    re.I,
)

PRICE_NUM_AND_BELOW_RE = re.compile(
    r"\b£?\s*(\d+(?:\.\d+)?)\s*(?:and\s+(?:below|under)|or\s+less)\b",
    re.I,
)


# Strict comparisons
PRICE_AND_ABOVE_RE = re.compile(r"£?\s*(\d+(?:\.\d+)?)\s*(?:and\s+above|and\s+over)\b", re.I)

PRICE_ABOVE_RE = re.compile(
    r"\b(?:above|over|more than|greater than)\s*£?\s*(\d+(?:\.\d+)?)\b",
    re.I,
)
PRICE_BELOW_RE = re.compile(
    r"\b(?:under|below|less than)\s*£?\s*(\d+(?:\.\d+)?)\b",
    re.I,
)

# Inclusive comparisons
PRICE_AT_LEAST_RE = re.compile(
    r"\b(?:at least|min(?:imum)?)\s*£?\s*(\d+(?:\.\d+)?)\b",
    re.I,
)
PRICE_AT_MOST_RE = re.compile(
    r"\b(?:at most|max(?:imum)?|up to)\s*£?\s*(\d+(?:\.\d+)?)\b",
    re.I,
)

PRICE_EXACT_RE = re.compile(
    r"\b(?:exactly|at)\s*£?\s*(\d+(?:\.\d+)?)\b",
    re.I,
)

UNKNOWN_LOC_RE = re.compile(r"\b(?:in|located in|near)\s+([A-Za-z][A-Za-z\s\-']{2,})\b", re.I)

SMALLTALK_RE = re.compile(
    r"\b(hi|hello|hey|thanks|thank you|what do you suggest)\b",
    re.I,
)

COURSE_INTENT_RE = re.compile(
    r"\b(course|courses|class|classes|workshop|workshops|learn|session|dandori)\b",
    re.I,
)

def is_out_of_scope(text: str) -> bool:
    """Check if user query is out of scope for course recommendations."""
    t = text or ""
    
    # Allow clear course keywords
    if COURSE_INTENT_RE.search(t):
        return False
    
    # Allow small talk
    if SMALLTALK_RE.search(t):
        return False
    
    # Allow counting queries
    if COUNT_Q_RE.search(t):
        return False
    
    # Allow if it contains a recognised location
    if st.session_state.recommender:
        locs = st.session_state.recommender.extract_locations_from_text(t)
        if locs:
            return False
    
    # Allow if it contains a price constraint
    if parse_price_filter(t):
        return False
    
    # Otherwise block
    return True


def parse_price_filter(text: str):
    """
    Returns (mode, a, b)
    mode in {"between", "above", "at_least", "below", "at_most", "exact"}.
    """
    t = text or ""

    # Free
    if re.search(r"\bfree\b", t, re.I):
        return ("exact", 0.0, None)

    # Between / ranges
    m_between = PRICE_BETWEEN_RE.search(t) or PRICE_RANGE_DASH_RE.search(t)
    if m_between:
        a = float(m_between.group(1))
        b = float(m_between.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return ("between", lo, hi)

    # "£60 and above" (number + and above)
    m_and_above = PRICE_AND_ABOVE_RE.search(t)
    if m_and_above:
        return ("at_least", float(m_and_above.group(1)), None)

    # "£60+" or "60+"
    m_plus = PRICE_PLUS_RE.search(t)
    if m_plus:
        return ("at_least", float(m_plus.group(1)), None)

    # Other number-first forms
    m_num_above = PRICE_NUM_AND_ABOVE_RE.search(t)
    if m_num_above:
        return ("at_least", float(m_num_above.group(1)), None)

    m_num_below = PRICE_NUM_AND_BELOW_RE.search(t)
    if m_num_below:
        return ("at_most", float(m_num_below.group(1)), None)

    # Phrase-first forms
    m_at_least = PRICE_AT_LEAST_RE.search(t)
    if m_at_least:
        return ("at_least", float(m_at_least.group(1)), None)

    m_above = PRICE_ABOVE_RE.search(t)
    if m_above:
        return ("above", float(m_above.group(1)), None)

    m_at_most = PRICE_AT_MOST_RE.search(t)
    if m_at_most:
        return ("at_most", float(m_at_most.group(1)), None)

    m_below = PRICE_BELOW_RE.search(t)
    if m_below:
        return ("below", float(m_below.group(1)), None)

    m_exact = PRICE_EXACT_RE.search(t)
    if m_exact:
        return ("exact", float(m_exact.group(1)), None)

    return None


def _format_price_phrase(mode: str, a: float, b: float | None):
    if mode == "above":
        return f"above **£{a:g}**"
    if mode == "at_least":
        return f"**£{a:g} and above**"
    if mode == "below":
        return f"below **£{a:g}**"
    if mode == "at_most":
        return f"**£{a:g} and below**"
    if mode == "between":
        return f"between **£{a:g} and £{b:g}**"
    if mode == "exact":
        return f"exactly **£{a:g}**"
    return "that price"


def handle_count_question(user_text: str, recommender: CourseRecommender):
    """
    Deterministic handler for:
    - total courses
    - courses in a location
    - courses above/under/between/exact £ threshold
    - combined constraints (e.g., location AND price)
    """
    text = user_text or ""
    if not COUNT_Q_RE.search(text):
        return None

    locs = recommender.extract_locations_from_text(text)
    
    # If no known locations found, try to extract any location
    if not locs:
        any_loc = recommender.extract_any_location_from_text(text)
        if any_loc:
            # Verify it can be geocoded
            coords = recommender.get_location_coords(any_loc)
            if coords:
                locs = [any_loc]
    
    parsed = parse_price_filter(text)


# If it looks like the user specified a location, but we don't recognise it, don't return total.
    m = UNKNOWN_LOC_RE.search(text)
    if m and not locs:
        unknown = m.group(1).strip()
        # Optional: show a few valid locations as hints
        examples = ", ".join(recommender.locations[:8]) if hasattr(recommender, "locations") else ""
        hint = f" Here are some locations I *do* have: {examples}." if examples else ""
        return f"I couldn’t find any courses in **{unknown}** in our dataset.{hint}"


    loc_label = recommender.locations_label(locs) if locs else None
    parsed = parse_price_filter(text)

    # Combined (locations + price)
    if locs and parsed:
        mode, a, b = parsed
        n = recommender.count_filtered(location=locs, price_mode=mode, a=a, b=b)
        return f"There are **{n}** course(s) in **{loc_label}** priced {_format_price_phrase(mode, a, b)}."

    # Price only
    if parsed:
        mode, a, b = parsed
        n = recommender.count_filtered(price_mode=mode, a=a, b=b)
        return f"There are **{n}** course(s) priced {_format_price_phrase(mode, a, b)}."

    # Locations only
    if locs:
        n = recommender.count_filtered(location=locs)
        return f"We currently have **{n}** course(s) listed in **{loc_label}**."

    # Total
    n = recommender.total_courses()
    return f"We currently have **{n}** courses in our catalogue."


# ---------------------------
# Tabs
# ---------------------------
tab_search, tab_chat = st.tabs(["🔎 Search", "🤖 Chatbot"])


# =========================================================
# TAB 1: SEARCH
# =========================================================
with tab_search:
    @st.cache_data
    def load_data():
        return pd.read_pickle("data/courses.pkl")

    @st.cache_resource
    def load_chroma_collection():
        """Load ChromaDB collection for semantic search - lazy loaded"""
        try:
            from chromadb.utils import embedding_functions
            import os
            import requests
            
            db_path = "data/courses_db"
            
            # Check if database exists
            if not os.path.exists(db_path):
                return None
            
            # Get HF token from environment
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                # Set token for huggingface_hub
                os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            
            try:
                # Use device='cpu' and optimize for lower memory usage
                embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2",
                    device="cpu"  # Explicitly use CPU to avoid GPU memory allocation attempts
                )
            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    st.error("⚠️ Hugging Face rate limit exceeded. Please try again in a few minutes.")
                    if not hf_token:
                        st.warning("💡 HF_TOKEN not found in environment. Set it in Cloud Run secrets for higher rate limits.")
                    else:
                        st.info("HF_TOKEN is set but rate limit still exceeded. You may need to wait a few minutes.")
                else:
                    st.error(f"Failed to download model from Hugging Face: {str(e)}")
                return None
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    st.error("⚠️ Hugging Face rate limit exceeded. Please try again in a few minutes.")
                    if not hf_token:
                        st.warning("💡 HF_TOKEN not found in environment. Set it in Cloud Run secrets for higher rate limits.")
                    else:
                        st.info("HF_TOKEN is set but rate limit still exceeded. You may need to wait a few minutes.")
                else:
                    st.error(f"Failed to load embedding model: {str(e)}")
                return None
            
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_collection(
                name="course_catalog",
                embedding_function=embedding_fn
            )
            
            return collection
        except Exception as e:
            st.error(f"ChromaDB error: {str(e)}")
            return None

    df = load_data().copy()
    
    # Don't load ChromaDB on startup - only when needed
    chroma_collection = None

    # Ensure consistency with Chroma index (unique class_id)
    df["class_id"] = df["class_id"].astype(str)
    df = df.drop_duplicates(subset=["class_id"], keep="last").reset_index(drop=True)

    # Data hygiene for text columns
    text_cols = ["title", "description", "location", "course_type", "instructor"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].fillna("")

    # Ensure cost is numeric
    if "cost_gbp" in df.columns:
        df["cost_gbp"] = (
            df["cost_gbp"]
            .astype(str)
            .str.replace("£", "", regex=False)
            .str.strip()
        )
        df["cost_gbp"] = pd.to_numeric(df["cost_gbp"], errors="coerce")

    # Ensure list-ish fields are safe to render
    listish_cols = ["learning_objectives", "skills_developed", "provided_materials"]

    def to_list_safe(x):
        if x is None:
            return []
        try:
            if isinstance(x, float) and pd.isna(x):
                return []
        except Exception:
            pass
        if isinstance(x, list):
            return x
        return x

    for c in listish_cols:
        if c in df.columns:
            df[c] = df[c].apply(to_list_safe)

    if "expanded_id" not in st.session_state:
        st.session_state["expanded_id"] = None

    def clear_filters():
        st.session_state["loc"] = []
        st.session_state["ctype"] = []
        st.session_state["title"] = ""
        st.session_state["kw"] = ""
        st.session_state["semantic"] = False
        st.session_state["price"] = (
            float(df["cost_gbp"].min()) if df["cost_gbp"].notna().any() else 0.0,
            float(df["cost_gbp"].max()) if df["cost_gbp"].notna().any() else 0.0,
        )
        st.session_state["sort"] = "Relevance (default)"
        st.session_state["page"] = 1
        st.session_state["expanded_id"] = None

    st.title("📚 Course Search")
    st.markdown("---")

    # Filters
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

    with col1:
        locations = sorted([x for x in df["location"].unique().tolist() if x])
        location_search = st.multiselect("🌍 Filter by Location", locations, key="loc")

    with col2:
        course_types = sorted([x for x in df["course_type"].unique().tolist() if x])
        course_type_search = st.multiselect("📚 Filter by Course Type", course_types, key="ctype")

    with col3:
        title_search = st.text_input("📖 Search by Title", placeholder="e.g., Waffle", key="title")

    with col4:
        keyword_search = st.text_input("🔍 Search by Keyword", placeholder="e.g., culinary", key="kw")

    # Semantic search toggle
    col_semantic, col_empty = st.columns([2, 6])
    with col_semantic:
        # Check if ChromaDB is available
        import os
        chroma_available = os.path.exists("data/courses_db")
        
        use_semantic = st.checkbox(
            "🧠 Use Semantic Search",
            value=False,
            key="semantic",
            disabled=(not chroma_available),
            help="Find courses by meaning, not just exact words (e.g., 'baking' finds 'waffle' courses)"
        )
        
        # Load ChromaDB only when user enables semantic search
        if use_semantic and chroma_collection is None:
            with st.spinner("Loading semantic search model (first time may take 30-60 seconds)..."):
                chroma_collection = load_chroma_collection()
            if chroma_collection is None:
                st.error("Failed to load semantic search. Using regular search instead.")
                use_semantic = False

    colA, colB, colC = st.columns([3, 2, 1])

    with colA:
        if df["cost_gbp"].notna().any():
            min_cost = float(df["cost_gbp"].min())
            max_cost = float(df["cost_gbp"].max())
        else:
            min_cost, max_cost = 0.0, 0.0

        # Initialize price in session state if not present
        if "price" not in st.session_state:
            st.session_state["price"] = (min_cost, max_cost)

        price_range = st.slider(
            "💷 Price range (£)",
            min_value=min_cost,
            max_value=max_cost,
            step=1.0,
            key="price",
        )

    with colB:
        sort_by = st.selectbox(
            "↕️ Sort by",
            [
                "Relevance (default)",
                "Title (A→Z)",
                "Location (A→Z)",
                "Cost (low→high)",
                "Cost (high→low)",
            ],
            key="sort",
        )

    with colC:
        st.write("")
        st.write("")
        st.button("🧹 Clear filters", on_click=clear_filters)

    current_filters = (tuple(location_search), tuple(course_type_search), title_search, keyword_search, use_semantic, price_range, sort_by)
    if "prev_filters" not in st.session_state:
        st.session_state.prev_filters = current_filters
    else:
        if st.session_state.prev_filters != current_filters:
            st.session_state.expanded_id = None
            st.session_state.page = 1
            st.session_state.prev_filters = current_filters

    # Apply filters
    filtered_df = df.copy()

    # Semantic search takes priority if enabled and keyword is provided
    if use_semantic and keyword_search and chroma_collection is not None:
        try:
            # Query ChromaDB for semantically similar courses
            results = chroma_collection.query(
                query_texts=[keyword_search],
                n_results=min(50, len(df))  # Get top 50 semantic matches
            )

            # Get the class IDs from semantic search results
            if results['ids'] and len(results['ids'][0]) > 0:
                semantic_ids = results['ids'][0]
                filtered_df = filtered_df[filtered_df['class_id'].isin(semantic_ids)]

                # Remove duplicates and preserve semantic ranking order
                id_to_rank = {cid: idx for idx, cid in enumerate(semantic_ids)}
                filtered_df['_semantic_rank'] = filtered_df['class_id'].map(id_to_rank)
                filtered_df = filtered_df.drop_duplicates(subset=['class_id'], keep='first')
                filtered_df = filtered_df.sort_values('_semantic_rank').drop(columns=['_semantic_rank'])
            else:
                filtered_df = filtered_df.iloc[0:0]  # Empty result

        except Exception as e:
            st.error(f"Semantic search error: {e}")
            filtered_df = df.copy()

    if location_search:
        filtered_df = filtered_df[filtered_df["location"].isin(location_search)]

    if course_type_search:
        filtered_df = filtered_df[filtered_df["course_type"].isin(course_type_search)]

    if title_search:
        filtered_df = filtered_df[filtered_df["title"].str.contains(title_search, case=False, na=False)]

    # Only apply keyword filter if NOT using semantic search
    if keyword_search and not use_semantic:
        mask = (
            filtered_df["title"].str.contains(keyword_search, case=False, na=False)
            | filtered_df["description"].str.contains(keyword_search, case=False, na=False)
            | filtered_df["skills_developed"].astype(str).str.contains(keyword_search, case=False, na=False)
            | filtered_df["learning_objectives"].astype(str).str.contains(keyword_search, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    if "cost_gbp" in filtered_df.columns and filtered_df["cost_gbp"].notna().any():
        lo, hi = price_range
        filtered_df = filtered_df[(filtered_df["cost_gbp"] >= lo) & (filtered_df["cost_gbp"] <= hi)]

    # Sorting
    if sort_by == "Title (A→Z)":
        filtered_df = filtered_df.sort_values("title", ascending=True)
    elif sort_by == "Location (A→Z)":
        filtered_df = filtered_df.sort_values("location", ascending=True)
    elif sort_by == "Cost (low→high)":
        filtered_df = filtered_df.sort_values("cost_gbp", ascending=True, na_position="last")
    elif sort_by == "Cost (high→low)":
        filtered_df = filtered_df.sort_values("cost_gbp", ascending=False, na_position="last")

    # Results header + pagination
    page_size = 12
    total = len(filtered_df)
    total_pages = max(1, math.ceil(total / page_size))

    # Initialize page in session state if not present
    if "page" not in st.session_state:
        st.session_state["page"] = 1

    # Ensure page is within valid range
    if st.session_state["page"] > total_pages:
        st.session_state["page"] = 1

    page = st.number_input("Page", min_value=1, max_value=total_pages, step=1, key="page")
    start = (page - 1) * page_size
    end = start + page_size

    # Display results header
    if total > 0:
        start_num = start + 1
        end_num = min(end, total)
        st.markdown(f"### Showing courses {start_num}-{end_num} of {total}")
    else:
        st.markdown(f"### Found 0 courses")

    st.markdown("---")

    page_df = filtered_df.iloc[start:end]
    courses_list = page_df.to_dict("records")

    # Render results
    if total == 0:
        st.info("No courses found matching your search criteria. Try adjusting your filters.")
    else:
        if st.session_state.expanded_id is not None:
            match = filtered_df[filtered_df["class_id"] == st.session_state.expanded_id]
            if match.empty:
                st.session_state.expanded_id = None
                st.rerun()

            course = match.iloc[0].to_dict()

            with st.container(border=True):
                col_header, col_button = st.columns([5, 1])
                with col_header:
                    st.markdown(f"### {course['title']}")
                with col_button:
                    if st.button("Hide Details"):
                        st.session_state.expanded_id = None
                        st.rerun()

                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown(f"**Instructor:** {course.get('instructor','') or 'Unknown'}")
                    st.markdown(f"**Location:** 📍 {course.get('location','') or 'Unknown'}")
                    st.markdown(f"**Course Type:** 📚 {course.get('course_type','') or 'Unknown'}")
                    cost = course.get("cost_gbp", None)
                    cost_display = f"£{cost:.2f}" if pd.notna(cost) else "Unknown"
                    st.markdown(f"**Cost:** 💷 {cost_display}")
                with col_b:
                    st.markdown(f"**Class ID:** {course['class_id']}")

                st.markdown("---")
                st.markdown("**Description:**")
                st.write(course.get("description", ""))

                st.markdown("---")
                c1, c2, c3 = st.columns(3)

                def render_list_or_text(val):
                    if isinstance(val, list):
                        if not val:
                            st.write("—")
                        else:
                            for item in val:
                                st.markdown(f"- {item}")
                    else:
                        st.write(val if val else "—")

                with c1:
                    st.markdown("**Learning Objectives:**")
                    render_list_or_text(course.get("learning_objectives", []))

                with c2:
                    st.markdown("**Skills Developed:**")
                    render_list_or_text(course.get("skills_developed", []))

                with c3:
                    st.markdown("**Provided Materials:**")
                    render_list_or_text(course.get("provided_materials", []))

            st.caption("Tip: change filters to go back to the grid automatically, or click Hide Details.")
        else:
            for i in range(0, len(courses_list), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j >= len(courses_list):
                        continue

                    course = courses_list[i + j]
                    with col:
                        with st.container(border=True):
                            st.markdown(f"**{course['title']}**")
                            st.markdown(f"📍 {course.get('location','') or 'Unknown'}")
                            st.markdown(f"📚 {course.get('course_type','') or 'Unknown'}")
                            cost = course.get("cost_gbp", None)
                            cost_display = f"£{cost:.2f}" if pd.notna(cost) else "Unknown"
                            st.markdown(f"💷 {cost_display}")

                            if st.button("View Details", key=f"view_{course['class_id']}"):
                                st.session_state.expanded_id = course["class_id"]
                                st.rerun()

    # Sidebar
    with st.sidebar:
        st.image("dandori-logo.png", width='stretch')
        st.metric("Courses Available", len(df))
        st.metric("Courses Found", len(filtered_df))

        st.markdown("---")
        st.markdown("### Available Locations (with counts)")
        loc_counts = df["location"].value_counts(dropna=True)
        for loc, cnt in loc_counts.items():
            if loc:
                st.markdown(f"- {loc} ({cnt})")


# =========================================================
# TAB 2: CHATBOT
# =========================================================
with tab_chat:
    st.title("🤖 Dandori Course Chatbot")
    st.write("Tell me what you're in the mood for, and I'll suggest a few classes.")

    if "chat_context" not in st.session_state:
        st.session_state.chat_context = {
            "locations": None,
            "price_filter": None,
        }

    if "recommender" not in st.session_state:
        try:
            with st.spinner("Loading chatbot AI model (first time may take 30-60 seconds)..."):
                st.session_state.recommender = CourseRecommender()
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {str(e)}")
            st.code(f"Error type: {type(e).__name__}\nDetails: {str(e)}")
            if "429" in str(e) or "Too Many Requests" in str(e):
                st.info("⚠️ Hugging Face rate limit exceeded. Please try again in a few minutes.")
                st.info("💡 Tip: Set HF_TOKEN in your environment to get higher rate limits.")
            st.info("Please try again in a few minutes or use the Search tab instead.")
            st.session_state.recommender = None

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey! What kind of class are you looking for today?"}
        ]

    # Initialize carousel index for each message if not present
    if "carousel_indices" not in st.session_state:
        st.session_state.carousel_indices = {}
    
    # Initialize expanded course tracking for chatbot
    if "chatbot_expanded" not in st.session_state:
        st.session_state.chatbot_expanded = {}

    # Render chat history (and matches inside assistant messages)
    for msg_idx, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.write(m["content"])

            if m["role"] == "assistant" and "recs" in m and m["recs"]:
                recs = m["recs"]
                num_recs = len(recs)
                
                if num_recs == 0:
                    continue
                
                # Initialize carousel index for this message
                carousel_key = f"carousel_{msg_idx}"
                if carousel_key not in st.session_state.carousel_indices:
                    st.session_state.carousel_indices[carousel_key] = 0
                
                current_idx = st.session_state.carousel_indices[carousel_key]
                
                # Check if any course is expanded for this message
                expanded_key = f"expanded_{msg_idx}"
                expanded_course_idx = st.session_state.chatbot_expanded.get(expanded_key, None)
                
                # If a course is expanded, show only that one
                if expanded_course_idx is not None and 0 <= expanded_course_idx < num_recs:
                    r = recs[expanded_course_idx]
                    
                    # Get full course details from dataframe
                    course_match = df[df["class_id"] == r["class_id"]]
                    if not course_match.empty:
                        course = course_match.iloc[0].to_dict()
                        
                        with st.container(border=True):
                            col_header, col_button = st.columns([5, 1])
                            with col_header:
                                st.markdown(f"### {course['title']}")
                            with col_button:
                                if st.button("← Back", key=f"back_chat_{msg_idx}"):
                                    st.session_state.chatbot_expanded[expanded_key] = None
                                    st.rerun()
                            
                            col_a, col_b = st.columns([2, 1])
                            with col_a:
                                st.markdown(f"**Instructor:** {course.get('instructor','') or 'Unknown'}")
                                st.markdown(f"**Location:** 📍 {course.get('location','') or 'Unknown'}")
                                st.markdown(f"**Course Type:** 📚 {course.get('course_type','') or 'Unknown'}")
                                cost = course.get("cost_gbp", None)
                                cost_display = f"£{cost:.2f}" if pd.notna(cost) else "Unknown"
                                st.markdown(f"**Cost:** 💷 {cost_display}")
                            with col_b:
                                st.markdown(f"**Class ID:** {course['class_id']}")
                            
                            st.markdown("---")
                            st.markdown("**Description:**")
                            st.write(course.get("description", ""))
                            
                            st.markdown("---")
                            c1, c2, c3 = st.columns(3)
                            
                            def render_list_or_text(val):
                                if isinstance(val, list):
                                    if not val:
                                        st.write("—")
                                    else:
                                        for item in val:
                                            st.markdown(f"- {item}")
                                else:
                                    st.write(val if val else "—")
                            
                            with c1:
                                st.markdown("**Learning Objectives:**")
                                render_list_or_text(course.get("learning_objectives", []))
                            
                            with c2:
                                st.markdown("**Skills Developed:**")
                                render_list_or_text(course.get("skills_developed", []))
                            
                            with c3:
                                st.markdown("**Provided Materials:**")
                                render_list_or_text(course.get("provided_materials", []))
                    continue
                
                # Show up to 4 courses at a time
                courses_per_page = 4
                total_pages = math.ceil(num_recs / courses_per_page)
                start_idx = current_idx * courses_per_page
                end_idx = min(start_idx + courses_per_page, num_recs)
                visible_recs = recs[start_idx:end_idx]
                
                # Navigation controls (only if more than 4 courses)
                if num_recs > courses_per_page:
                    col_left, col_center, col_right = st.columns([1, 6, 1])
                    
                    with col_left:
                        if st.button("◀", key=f"prev_{msg_idx}", disabled=(current_idx == 0)):
                            st.session_state.carousel_indices[carousel_key] = max(0, current_idx - 1)
                            st.rerun()
                    
                    with col_center:
                        st.markdown(f"<div style='text-align: center;'>Showing {start_idx + 1}-{end_idx} of {num_recs} courses</div>", unsafe_allow_html=True)
                    
                    with col_right:
                        if st.button("▶", key=f"next_{msg_idx}", disabled=(current_idx >= total_pages - 1)):
                            st.session_state.carousel_indices[carousel_key] = min(total_pages - 1, current_idx + 1)
                            st.rerun()
                
                # Display courses in horizontal layout
                cols = st.columns(len(visible_recs))
                
                # Add CSS for consistent card heights and alignment
                st.markdown(
                    """
                    <style>
                    /* Force equal height columns in chatbot results */
                    div[data-testid="stHorizontalBlock"] {
                        align-items: stretch !important;
                    }
                    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                        display: flex !important;
                        flex-direction: column !important;
                    }
                    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
                        flex: 1 !important;
                        display: flex !important;
                        flex-direction: column !important;
                    }
                    /* Make course title area fixed height */
                    .course-title {
                        min-height: 3em;
                        display: flex;
                        align-items: center;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                for col_idx, (col, r) in enumerate(zip(cols, visible_recs)):
                    with col:
                        # Highlight first result as best match
                        is_best_match = (start_idx + col_idx == 0)
                        
                        # Use custom styling for best match
                        if is_best_match:
                            with st.container(border=True):
                                st.markdown("🏆 **BEST MATCH**", help="This course best matches your query")
                                # Use fixed height div for title
                                title = r['title']
                                if len(title) > 45:
                                    title = title[:42] + "..."
                                st.markdown(f'<div class="course-title"><strong>{title}</strong></div>', unsafe_allow_html=True)
                                
                                cost = r.get('cost_gbp')
                                try:
                                    cost_display = f"£{float(cost):.2f}" if cost is not None else "Unknown"
                                except (ValueError, TypeError):
                                    cost_display = f"£{cost}" if cost else "Unknown"
                                st.markdown(f"📍 {r['location']}")
                                
                                # Show distance if available
                                if r.get('distance_miles') is not None:
                                    if r['distance_miles'] == 0.0:
                                        st.markdown(f"🗺️ In {r['location']}")
                                    else:
                                        st.markdown(f"🗺️ {r['distance_miles']:.1f} miles away")
                                else:
                                    # Add empty space to maintain alignment
                                    st.markdown("&nbsp;", unsafe_allow_html=True)
                                
                                st.markdown(f"💷 {cost_display}")
                                st.markdown(f"📚 {r['course_type']}")
                                st.caption(f"🆔 {r['class_id']}")
                                
                                # More Details button
                                if st.button("More Details", key=f"details_chat_{msg_idx}_{start_idx + col_idx}", use_container_width=True):
                                    st.session_state.chatbot_expanded[expanded_key] = start_idx + col_idx
                                    st.rerun()
                        else:
                            with st.container(border=True):
                                # Use fixed height div for title
                                title = r['title']
                                if len(title) > 45:
                                    title = title[:42] + "..."
                                st.markdown(f'<div class="course-title"><strong>{title}</strong></div>', unsafe_allow_html=True)
                                
                                cost = r.get('cost_gbp')
                                try:
                                    cost_display = f"£{float(cost):.2f}" if cost is not None else "Unknown"
                                except (ValueError, TypeError):
                                    cost_display = f"£{cost}" if cost else "Unknown"
                                st.markdown(f"📍 {r['location']}")
                                
                                # Show distance if available
                                if r.get('distance_miles') is not None:
                                    if r['distance_miles'] == 0.0:
                                        st.markdown(f"🗺️ In {r['location']}")
                                    else:
                                        st.markdown(f"🗺️ {r['distance_miles']:.1f} miles away")
                                else:
                                    # Add empty space to maintain alignment
                                    st.markdown("&nbsp;", unsafe_allow_html=True)
                                
                                st.markdown(f"💷 {cost_display}")
                                st.markdown(f"📚 {r['course_type']}")
                                st.caption(f"🆔 {r['class_id']}")
                                
                                # More Details button
                                if st.button("More Details", key=f"details_chat_{msg_idx}_{start_idx + col_idx}", use_container_width=True):
                                    st.session_state.chatbot_expanded[expanded_key] = start_idx + col_idx
                                    st.rerun()

    # ✅ chat_input MUST be last
    user_msg = st.chat_input("e.g. Something creative in Yorkshire under £80…")

    if user_msg:
        # Check if recommender is available
        if st.session_state.recommender is None:
            st.error("Chatbot is currently unavailable. Please try again later or use the Search tab.")
            st.stop()
        
        # Add user message and display it immediately
        st.session_state.messages.append({"role": "user", "content": user_msg})
        
        # Display the user message right away
        with st.chat_message("user"):
            st.write(user_msg)
        
        # Show a spinner while processing
        with st.spinner("Thinking..."):
            # Helper function to update stateful context
            def update_context(user_msg: str, recommender: CourseRecommender):
                ctx = st.session_state.chat_context

                # Extract structured info - try known locations first
                locs = recommender.extract_locations_from_text(user_msg)
                
                # If no known locations found, try to extract any location
                if not locs:
                    any_loc = recommender.extract_any_location_from_text(user_msg)
                    if any_loc:
                        # Verify it can be geocoded before using it
                        coords = recommender.get_location_coords(any_loc)
                        if coords:
                            locs = [any_loc]
                
                price = parse_price_filter(user_msg)

                # Overwrite if new info provided
                if locs:
                    ctx["locations"] = locs

                if price:
                    ctx["price_filter"] = price

                # Handle "instead" logic (location swap but keep budget)
                if "instead" in user_msg.lower() and locs:
                    ctx["locations"] = locs

                st.session_state.chat_context = ctx

            # Check for out of scope queries
            if is_out_of_scope(user_msg):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        "I'm mainly here to help you explore Dandori courses.\n\n"
                        "Try something like:\n"
                        "- \"Recommend me something creative in Brighton under £80\"\n"
                        "- \"How many courses are £60 and above in York?\""
                    )
                })
                st.rerun()
            
            # Small talk short-circuit (don't run retrieval)
            if SMALLTALK_RE.search(user_msg or ""):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Hey! 👋 Tell me what kind of class you're looking for — location and budget help too."
                })
                st.rerun()

            # Deterministic count handling
            answer = handle_count_question(user_msg, st.session_state.recommender)
            if answer is not None:
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()

            # Normal RAG flow with stateful context
            recommender = st.session_state.recommender
            
            # Update context with new information from user message
            update_context(user_msg, recommender)
            ctx = st.session_state.chat_context
            
            # Use context (remembers previous location/budget)
            locs = ctx["locations"]
            parsed_price = ctx["price_filter"]
            has_location = bool(locs)
            has_budget = bool(parsed_price)

            # Check if query is about a region rather than a specific location
            region = recommender.extract_region_from_text(user_msg)
            
            # Use first location as reference for proximity search (only if not a region query)
            reference_loc = locs[0] if locs and not region else None
            recs = recommender.retrieve(user_msg, n_results=8, reference_location=reference_loc, region=region)

            # Deterministic fallback when vector retrieval returns nothing
            if not recs:
                recs = recommender.fallback_search(
                    query=user_msg,
                    locations=locs if locs else None,
                    price_filter=parsed_price,
                    limit=8,
                )

            reply = recommender.respond_smart(
                user_query=user_msg,
                recs=recs,
                has_location=has_location or bool(region),  # Region counts as location context
                has_budget=has_budget,
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "recs": [r.__dict__ for r in recs]
            })
        
        st.rerun()
