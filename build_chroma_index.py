import os
import ast
import pandas as pd
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from google import genai

# ---------- config ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY missing. Put it in .env")

ai = genai.Client(api_key=GEMINI_API_KEY)

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "dandori_courses"
COURSES_CSV = "courses.csv"
BATCH_SIZE = 64


# ---------- helpers ----------
def clean_list_column(value) -> str:
    if pd.isna(value):
        return ""
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return "\n- " + "\n- ".join(parsed)
    except Exception:
        pass
    return str(value)


def make_retrieval_text(row: pd.Series) -> str:
    return f"""Title: {row['title']}
Instructor: {row['instructor']}
Location: {row['location']}
Course Type: {row['course_type']}
Cost: £{row['cost_gbp']}

Learning Objectives:{clean_list_column(row['learning_objectives'])}

Skills Developed:{clean_list_column(row['skills_developed'])}

Provided Materials:{clean_list_column(row['provided_materials'])}

Description:
{row['description']}

Class ID: {row['class_id']}
Source File: {row['file_name']}
"""


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Returns list of vectors (list[float]) in the same order as texts.
    """
    vectors = []
    for t in texts:
        resp = ai.models.embed_content(
            model="text-embedding-004",
            contents=t
        )
        vectors.append(resp.embeddings[0].values)
    return vectors


# ---------- main ----------
def main():
    df = pd.read_csv(COURSES_CSV)

    # IMPORTANT: Use Client + Settings (avoids Rust bindings on Windows)
    client = chromadb.PersistentClient(
    path=PERSIST_DIR,
    settings=Settings(
        anonymized_telemetry=False,
        chroma_api_impl="chromadb.api.segment.SegmentAPI"
        )
    )
    # MVP: rebuild collection each run
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_ids, batch_docs, batch_metas = [], [], []

    for _, row in df.iterrows():
        doc = make_retrieval_text(row)

        batch_ids.append(str(row["class_id"]))
        batch_docs.append(doc)
        batch_metas.append({
            "class_id": str(row["class_id"]),
            "title": str(row["title"]),
            "location": str(row["location"]),
            "course_type": str(row["course_type"]),
            "cost_gbp": float(row["cost_gbp"]),
            "file_name": str(row["file_name"]),
        })

        if len(batch_ids) >= BATCH_SIZE:
            embeddings = embed_documents(batch_docs)
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=embeddings,
            )
            batch_ids, batch_docs, batch_metas = [], [], []

    # final partial batch
    if batch_ids:
        embeddings = embed_documents(batch_docs)
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings,
        )



    print(f"✅ Indexed {len(df)} courses into Chroma at '{PERSIST_DIR}'")


if __name__ == "__main__":
    main()