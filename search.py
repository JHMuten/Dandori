#search.py
# For quick searches once the chromadb has been set up

import chromadb

# 1. Point to the EXACT same path you used in your creation script
# This connects to the existing database on your disk
client = chromadb.PersistentClient(path="data/courses_db")

# 2. Use 'get_collection' (if you are sure it exists) 
# or 'get_or_create_collection' (to be safe)
collection = client.get_collection(name="course_catalog")

# 3. Verify it worked by checking the count
print(f"Connected! Database contains {collection.count()} courses.")

# 4. Perform a search
results = collection.query(
    query_texts=["waflle"],
    n_results=2
)

# 5. Display the results nicely
for i in range(len(results['ids'][0])):
    print(f"\n--- Match {i+1} ---")
    print(f"Title: {results['metadatas'][0][i]['title']}")
    print(f"Instructor: {results['metadatas'][0][i]['instructor']}")
    print(f"Cost: £{results['metadatas'][0][i]['cost']}")