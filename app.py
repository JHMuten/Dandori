import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="Course Search", page_icon="📚", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_pickle('pdf-samples/courses.pkl')

df = load_data()

# Title
st.title("📚 Course Search")
st.markdown("---")

# Search filters
col1, col2, col3 = st.columns(3)

with col1:
    locations = ["All"] + sorted(df['location'].unique().tolist())
    location_search = st.selectbox("🌍 Filter by Location", locations)

with col2:
    title_search = st.text_input("📖 Search by Title", placeholder="e.g., Waffle")

with col3:
    keyword_search = st.text_input("🔍 Search by Keyword", placeholder="e.g., culinary")

# Filter dataframe
filtered_df = df.copy()

if location_search != "All":
    filtered_df = filtered_df[filtered_df['location'] == location_search]

if title_search:
    filtered_df = filtered_df[filtered_df['title'].str.contains(title_search, case=False, na=False)]

if keyword_search:
    # Search across multiple text columns
    mask = (
        filtered_df['title'].str.contains(keyword_search, case=False, na=False) |
        filtered_df['description'].str.contains(keyword_search, case=False, na=False) |
        filtered_df['skills_developed'].str.contains(keyword_search, case=False, na=False) |
        filtered_df['learning_objectives'].str.contains(keyword_search, case=False, na=False)
    )
    filtered_df = filtered_df[mask]

# Display results
st.markdown(f"### Found {len(filtered_df)} course(s)")
st.markdown("---")

if len(filtered_df) > 0:
    # Display courses in a grid (4 per row)
    courses_list = filtered_df.to_dict('records')
    
    for i in range(0, len(courses_list), 4):
        # Check if any course in this row is expanded
        row_has_expanded = any(st.session_state.get(f"expanded_{i}_{j}", False) for j in range(4) if i + j < len(courses_list))
        
        if not row_has_expanded:
            # Show normal grid view
            cols = st.columns(4)
            
            for j, col in enumerate(cols):
                if i + j < len(courses_list):
                    course = courses_list[i + j]
                    
                    with col:
                        with st.container(border=True):
                            st.markdown(f"**{course['title']}**")
                            st.markdown(f"📍 {course['location']}")
                            st.markdown(f"📚 {course['course_type']}")
                            st.markdown(f"💷 £{course['cost_gbp']}")
                            
                            if st.button("View Details", key=f"btn_{i}_{j}"):
                                st.session_state[f"expanded_{i}_{j}"] = True
                                st.rerun()
        else:
            # Show expanded view for the selected course
            for j in range(4):
                if i + j < len(courses_list) and st.session_state.get(f"expanded_{i}_{j}", False):
                    course = courses_list[i + j]
                    
                    with st.container(border=True):
                        col_header, col_button = st.columns([5, 1])
                        with col_header:
                            st.markdown(f"### {course['title']}")
                        with col_button:
                            if st.button("Hide Details", key=f"btn_{i}_{j}"):
                                st.session_state[f"expanded_{i}_{j}"] = False
                                st.rerun()
                        
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            st.markdown(f"**Instructor:** {course['instructor']}")
                            st.markdown(f"**Location:** 📍 {course['location']}")
                            st.markdown(f"**Course Type:** 📚 {course['course_type']}")
                            st.markdown(f"**Cost:** 💷 £{course['cost_gbp']}")
                        
                        with col_b:
                            st.markdown(f"**Class ID:** {course['class_id']}")
                        
                        st.markdown("---")
                        
                        st.markdown("**Description:**")
                        st.write(course['description'])
                        
                        st.markdown("---")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Learning Objectives:**")
                            if isinstance(course['learning_objectives'], list):
                                for obj in course['learning_objectives']:
                                    st.markdown(f"- {obj}")
                            else:
                                st.write(course['learning_objectives'])
                        
                        with col2:
                            st.markdown("**Skills Developed:**")
                            if isinstance(course['skills_developed'], list):
                                for skill in course['skills_developed']:
                                    st.markdown(f"- {skill}")
                            else:
                                st.write(course['skills_developed'])
                        
                        with col3:
                            st.markdown("**Provided Materials:**")
                            if isinstance(course['provided_materials'], list):
                                for material in course['provided_materials']:
                                    st.markdown(f"- {material}")
                            else:
                                st.write(course['provided_materials'])
                    
                    break
else:
    st.info("No courses found matching your search criteria. Try adjusting your filters.")

# Sidebar with stats
with st.sidebar:
    st.image("dandori-logo.png", use_container_width=True)
    st.metric("Courses Available", len(df))
    st.metric("Courses Found", len(filtered_df))
    
    st.markdown("---")
    st.markdown("### Available Locations")
    for loc in df['location'].unique():
        st.markdown(f"- {loc}")
