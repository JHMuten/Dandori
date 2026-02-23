#grounding.py

# Convert the pandas dataframe into a chroma database
# Then users can search semantically, rather than having to 
# match text exactly

import pandas as pd
import chromadb

def load_data():
    return pd.read_pickle('all_pdfs/courses.pkl')

df = load_data()

# 1. Initialize the client
client = chromadb.PersistentClient(path="data/courses_db")
collection = client.get_or_create_collection(name="course_catalog")

# 2. Loop through your DataFrame to add data
for _, row in df.iterrows():
    # Combine text-heavy columns for the search engine to "read"
    searchable_text = f"Title: {row['title']}. Objectives: {row['learning_objectives']}. {row['description']}"
    
    collection.add(
        documents=[searchable_text],
        ids=[str(row['class_id'])],
        metadatas=[{
            "instructor": row['instructor'],
            "location": row['location'],
            "cost": row['cost_gbp'],
            "type": row['course_type'],
            "title": row['title']
        }]
    )
