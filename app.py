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
    location_search = st.text_input("🌍 Search by Location", placeholder="e.g., Harrogate")

with col2:
    title_search = st.text_input("📖 Search by Title", placeholder="e.g., Waffle")

with col3:
    keyword_search = st.text_input("🔍 Search by Keyword", placeholder="e.g., culinary")

# Filter dataframe
filtered_df = df.copy()

if location_search:
    filtered_df = filtered_df[filtered_df['location'].str.contains(location_search, case=False, na=False)]

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
    for idx, row in filtered_df.iterrows():
        with st.expander(f"**{row['title']}** - {row['location']}", expanded=True):
            col_a, col_b = st.columns([2, 1])
            
            with col_a:
                st.markdown(f"**Instructor:** {row['instructor']}")
                st.markdown(f"**Location:** {row['location']}")
                st.markdown(f"**Course Type:** {row['course_type']}")
                st.markdown(f"**Cost:** £{row['cost_gbp']}")
                
            with col_b:
                st.markdown(f"**Class ID:** {row['class_id']}")
            
            st.markdown("**Description:**")
            st.write(row['description'])
            
            st.markdown("**Learning Objectives:**")
            st.write(row['learning_objectives'])
            
            st.markdown("**Skills Developed:**")
            st.write(row['skills_developed'])
            
            st.markdown("**Provided Materials:**")
            st.write(row['provided_materials'])
else:
    st.info("No courses found matching your search criteria. Try adjusting your filters.")

# Sidebar with stats
with st.sidebar:
    st.header("📊 Dataset Info")
    st.metric("Total Courses", len(df))
    st.metric("Filtered Results", len(filtered_df))
    
    st.markdown("---")
    st.markdown("### Available Locations")
    for loc in df['location'].unique():
        st.markdown(f"- {loc}")
