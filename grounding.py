#grounding.py

# Convert the pandas dataframe into a chroma database
# Then users can search semantically, rather than having to 
# match text exactly

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = "data/courses_db"
COLLECTION_NAME = "course_catalog"

def load_data():
    return pd.read_pickle("data/courses.pkl")

def compact_text(row) -> str:
    title = str(row.get("title", "")).strip()
    obj = str(row.get("learning_objectives", "")).strip()
    desc = str(row.get("description", "")).strip()

    if len(desc) > 500:
        desc = desc[:500] + "..."

    return f"Title: {title}\nObjectives: {obj}\nDescription: {desc}"

def main():
    df = load_data()

    # ✅ De-duplicate IDs BEFORE indexing
    df["class_id"] = df["class_id"].astype(str)
    before = len(df)
    df = df.drop_duplicates(subset=["class_id"], keep="last").reset_index(drop=True)
    after = len(df)
    print(f"✅ De-duplicated class_id: {before} -> {after}")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cpu"  # Use CPU for lower memory usage
    )

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    # ✅ Clear safely (avoids duplicates across runs)
    existing = collection.get(include=[])
    existing_ids = existing.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)

    batch_docs, batch_ids, batch_metas = [], [], []
    BATCH_SIZE = 64

    for _, row in df.iterrows():
        batch_docs.append(compact_text(row))
        batch_ids.append(str(row["class_id"]))
        batch_metas.append({
            "instructor": row.get("instructor", ""),
            "location": row.get("location", ""),
            "cost": row.get("cost_gbp", ""),
            "type": row.get("course_type", ""),
            "title": row.get("title", ""),
        })

        if len(batch_ids) >= BATCH_SIZE:
            collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
            batch_docs, batch_ids, batch_metas = [], [], []

    if batch_ids:
        collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)

    print("✅ Chroma DB rebuilt successfully (local embeddings).")

if __name__ == "__main__":
    main()
