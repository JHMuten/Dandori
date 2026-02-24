# duplicates.py

import pandas as pd


def main():
    df = pd.read_pickle("all_pdfs/courses.pkl")

    print(f"Total rows: {len(df)}")

    # Ensure class_id is treated as string
    df["class_id"] = df["class_id"].astype(str)

    duplicates = df[df.duplicated(subset=["class_id"], keep=False)]

    if duplicates.empty:
        print("✅ No duplicate class_id values found.")
    else:
        print(f"⚠ Found {duplicates['class_id'].nunique()} duplicated class_ids.")
        print("\nDuplicate entries:\n")
        print(duplicates[["class_id", "title"]].sort_values("class_id"))

        print("\nDuplicate counts:")
        print(duplicates["class_id"].value_counts())


if __name__ == "__main__":
    main()