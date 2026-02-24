import pandas as pd

df = pd.read_pickle("data/courses.pkl")

# Save as CSV (recommended)
df.to_csv("courses.csv", index=False)

# OR save as JSON
df.to_json("courses.json", orient="records", indent=2)