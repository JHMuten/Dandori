import re
import math
import streamlit as st
import pandas as pd

from recommender import CourseRecommender

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
    re.I
)
PRICE_RANGE_DASH_RE = re.compile(
    r"\b£?\s*(\d+(?:\.\d+)?)\s*[-–]\s*£?\s*(\d+(?:\.\d+)?)\b",
    re.I
)
PRICE_ABOVE_RE = re.compile(
    r"\b(?:above|over|more than|greater than|at least|min(?:imum)?)\s*£?\s*(\d+(?:\.\d+)?)\b",
    re.I
)
PRICE_BELOW_RE = re.compile(
    r"\b(?:under|below|less than|at most|max(?:imum)?|up to)\s*£?\s*(\d+(?:\.\d+)?)\b",
    re.I
)
PRICE_EXACT_RE = re.compile(
    r"\b(?:exactly|at)\s*£?\s*(\d+(?:\.\d+)?)\b",
    re.I
)

def parse_price_filter(text: str):
    """
    Returns (mode, a, b)
    mode in {"between", "above", "below", "exact"}.
    """
    t = text or ""

    m = PRICE_BETWEEN_RE.search(t) or PRICE_RANGE_DASH_RE.search(t)
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return ("between", lo, hi)

    m = PRICE_ABOVE_RE.search(t)
    if m:
        return ("above", float(m.group(1)), None)

    m = PRICE_BELOW_RE.search(t)
    if m:
        return ("below", float(m.group(1)), None)

    m = PRICE_EXACT_RE.search(t)
    if m:
        return ("exact", float(m.group(1)), None)

    return None

def handle_count_question(user_text: str, recommender: CourseRecommender):
    """
    Deterministic handler for:
    - total courses
    - courses in a location
    - courses above/under/between/exact £ threshold
    """
    text = user_text or ""
    if not COUNT_Q_RE.search(text):
        return None

    # 1) Price-based counts (most specific)
    parsed = parse_price_filter(text)
    if parsed:
        mode, a, b = parsed

        if mode == "above":
            n = recommender.count_above_price(a)
            return f"There are **{n}** course(s) priced **above £{a:g}**."

        if mode == "below":
            n = recommender.count_below_price(a)
            return f"There are **{n}** course(s) priced **below £{a:g}**."

        if mode == "between":
            n = recommender.count_between_prices(a, b)
            return f"There are **{n}** course(s) priced **between £{a:g} and £{b:g}**."

        if mode == "exact":
            n = recommender.count_exact_price(a)
            return f"There are **{n}** course(s) priced **exactly £{a:g}**."

    # 2) Location-based count
    loc = recommender.extract_location_from_text(text)
    if loc:
        n = recommender.count_in_location(loc)
        return f"We currently have **{n}** course(s) listed in **{loc}**."

    # 3) Total count
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
        return pd.read_pickle("all_pdfs/courses.pkl")

    df = load_data().copy()

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

    colA, colB, colC = st.columns([3, 2, 1])

    with colA:
        if df["cost_gbp"].notna().any():
            min_cost = float(df["cost_gbp"].min())
            max_cost = float(df["cost_gbp"].max())
        else:
            min_cost, max_cost = 0.0, 0.0

        price_range = st.slider(
            "💷 Price range (£)",
            min_value=min_cost,
            max_value=max_cost,
            value=(min_cost, max_cost),
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

    current_filters = (tuple(location_search), tuple(course_type_search), title_search, keyword_search, price_range, sort_by)
    if "prev_filters" not in st.session_state:
        st.session_state.prev_filters = current_filters
    else:
        if st.session_state.prev_filters != current_filters:
            st.session_state.expanded_id = None
            st.session_state.page = 1
            st.session_state.prev_filters = current_filters

    # Apply filters
    filtered_df = df.copy()

    if location_search:
        filtered_df = filtered_df[filtered_df["location"].isin(location_search)]

    if course_type_search:
        filtered_df = filtered_df[filtered_df["course_type"].isin(course_type_search)]

    if title_search:
        filtered_df = filtered_df[filtered_df["title"].str.contains(title_search, case=False, na=False)]

    if keyword_search:
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
    st.markdown(f"### Found {len(filtered_df)} course(s)")
    st.markdown("---")

    page_size = 12
    total = len(filtered_df)
    total_pages = max(1, math.ceil(total / page_size))

    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="page")
    start = (page - 1) * page_size
    end = start + page_size

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
                    st.markdown(f"**Cost:** 💷 £{cost if pd.notna(cost) else 'Unknown'}")
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
                            st.markdown(f"💷 £{cost if pd.notna(cost) else 'Unknown'}")

                            if st.button("View Details", key=f"view_{course['class_id']}"):
                                st.session_state.expanded_id = course["class_id"]
                                st.rerun()

    # Sidebar
    with st.sidebar:
        st.image("dandori-logo.png", use_container_width=True)
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
    st.write("Tell me what you’re in the mood for, and I’ll suggest a few classes.")

    if "recommender" not in st.session_state:
        st.session_state.recommender = CourseRecommender()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey! What kind of class are you looking for today?"}
        ]

    # Render chat history (and matches inside assistant messages)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

            if m["role"] == "assistant" and "recs" in m and m["recs"]:
                st.markdown("### Matches I used")
                for r in m["recs"]:
                    with st.container(border=True):
                        st.markdown(f"**{r['title']}**")
                        st.markdown(f"📍 {r['location']} • 💷 £{r['cost_gbp']} • 📚 {r['course_type']}")
                        st.markdown(f"👤 {r['instructor']} • 🆔 {r['class_id']}")

    # ✅ chat_input MUST be last
    user_msg = st.chat_input("e.g. Something creative in Yorkshire under £80…")

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})

        # Deterministic count handling
        answer = handle_count_question(user_msg, st.session_state.recommender)
        if answer is not None:
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

        # Normal RAG flow
        recs = st.session_state.recommender.retrieve(user_msg, n_results=8)
        reply = st.session_state.recommender.respond(user_msg, recs)

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "recs": [r.__dict__ for r in recs]
        })
        st.rerun()