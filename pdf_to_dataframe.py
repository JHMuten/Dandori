"""
PDF to Pandas DataFrame Converter - School of Dandori Course Data
=================================================================
Extracts structured course information from PDFs into a pandas DataFrame.

Requirements:
    pip install pandas pdfplumber

Usage (CLI):
    python pdf_to_dataframe.py file1.pdf file2.pdf ...
    python pdf_to_dataframe.py *.pdf

Usage (import):
    from pdf_to_dataframe import pdfs_to_dataframe
    df = pdfs_to_dataframe(["class_001.pdf", "class_008.pdf"])
    print(df)

DataFrame columns:
    file_name           - source filename
    class_id            - e.g. CLASS_4033
    title               - course title
    instructor          - instructor name
    location            - city/location
    course_type         - e.g. Culinary Arts, Fiber Arts
    cost_gbp            - cost as float (e.g. 75.0)
    learning_objectives - list of objective strings
    provided_materials  - list of material strings
    skills_developed    - list of skill tag strings
    description         - full course description text
"""

import re
import glob
import sys
from pathlib import Path

import pandas as pd
import pdfplumber


# --------------------------------------------------------------------------- #
# Skills parsing (uses word bounding boxes to detect column gaps)
# --------------------------------------------------------------------------- #

def _skills_from_page(page):
    """
    Parse 'Skills Developed' tags from page 2 using word x-positions.
    Skills are rendered as spaced tags; positional parsing correctly splits
    multi-word skills (e.g. 'Creative Cooking') from adjacent tags.
    """
    words = page.extract_words()
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    # Locate the "Skills Developed" header: find "Skills" followed immediately
    # by "Developed" on the same line.
    header_bottom = None
    for i, w in enumerate(sorted_words):
        if w["text"] == "Skills" and i + 1 < len(sorted_words):
            nxt = sorted_words[i + 1]
            if nxt["text"] == "Developed" and abs(nxt["top"] - w["top"]) < 4:
                header_bottom = w["bottom"]
                break

    if header_bottom is None:
        return []

    # Locate "Course Description" header (to know where skills section ends)
    section_top = 9999
    for i, w in enumerate(sorted_words):
        if w["text"] == "Course" and i + 1 < len(sorted_words):
            nxt = sorted_words[i + 1]
            if nxt["text"] == "Description" and abs(nxt["top"] - w["top"]) < 4:
                section_top = w["top"]
                break

    # Collect words between the two headers
    skill_words = [
        w for w in sorted_words
        if w["top"] >= header_bottom and w["bottom"] <= section_top
    ]
    if not skill_words:
        return []

    # Group words into lines (same vertical position ± 4 pts)
    lines, current_line = [], [skill_words[0]]
    for w in skill_words[1:]:
        if abs(w["top"] - current_line[-1]["top"]) < 4:
            current_line.append(w)
        else:
            lines.append(current_line)
            current_line = [w]
    lines.append(current_line)

    # Within each line, group consecutive words with small horizontal gaps
    # into a single skill tag; large gaps (>20 pts) separate different tags.
    skills = []
    for line in lines:
        groups, current_group = [], [line[0]]
        for w in line[1:]:
            if w["x0"] - current_group[-1]["x1"] > 20:
                groups.append(current_group)
                current_group = [w]
            else:
                current_group.append(w)
        groups.append(current_group)
        for group in groups:
            tag = " ".join(g["text"] for g in group).strip()
            if tag:
                skills.append(tag)

    return skills


# --------------------------------------------------------------------------- #
# Main extraction function
# --------------------------------------------------------------------------- #

def extract_course_data(pdf_path: str) -> dict:
    """
    Extract all structured fields from a single School of Dandori PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        dict matching the DataFrame columns described in the module docstring.
    """
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
        skills = _skills_from_page(pdf.pages[1]) if len(pdf.pages) > 1 else []

    full_text = "\n".join(pages_text)
    p1 = pages_text[0]

    data = {"file_name": Path(pdf_path).name}

    # --- Class ID ---
    m = re.search(r"Class ID:\s*(CLASS_\w+)", full_text)
    data["class_id"] = m.group(1) if m else None

    # --- Title (first non-empty line of page 1) ---
    p1_lines = [l.strip() for l in p1.splitlines() if l.strip()]
    data["title"] = p1_lines[0] if p1_lines else None

    # --- Instructor & Location ---
    # PDF layout: "Instructor: Location:\n<name>  <city>"
    # Location is always a single word; split on last whitespace token.
    m = re.search(r"Instructor:\s+Location:\s*\n(.+)", p1)
    if m:
        val = m.group(1).strip()
        parts = val.rsplit(" ", 1)
        data["instructor"] = parts[0].strip()
        data["location"]   = parts[1].strip() if len(parts) > 1 else None
    else:
        data["instructor"] = data["location"] = None

    # --- Course Type & Cost ---
    # PDF layout: "Course Type: Cost:\n<type>  £<amount>"
    m = re.search(r"Course Type:\s+Cost:\s*\n(.+)", p1)
    if m:
        val = m.group(1).strip()
        cost_m = re.search(r"(£[\d,.]+)", val)
        if cost_m:
            data["course_type"] = val[: val.index(cost_m.group(1))].strip()
            data["cost_gbp"]    = float(re.sub(r"[£,]", "", cost_m.group(1)))
        else:
            data["course_type"] = val
            data["cost_gbp"]    = None
    else:
        data["course_type"] = data["cost_gbp"] = None

    # --- Bullet-list sections (Learning Objectives & Provided Materials) ---
    def extract_bullet_section(text, header):
        pattern = rf"{re.escape(header)}\s*\n(.*?)(?=\n[A-Z][^\n]{{2,}}\n|\Z)"
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            return []
        return [i.strip() for i in re.findall(r"[•\-\*]\s*(.+)", m.group(1))]

    data["learning_objectives"] = extract_bullet_section(p1, "Learning Objectives")
    data["provided_materials"]  = extract_bullet_section(p1, "Provided Materials")

    # --- Skills Developed (positional word-gap parsing) ---
    data["skills_developed"] = skills

    # --- Course Description ---
    m = re.search(r"Course Description\s*\n(.*?)(?=Class ID:|\Z)", full_text, re.DOTALL)
    if m:
        desc = re.sub(r"(?<!\n)\n(?!\n)", " ", m.group(1).strip())
        data["description"] = desc.strip()
    else:
        data["description"] = None

    return data


# --------------------------------------------------------------------------- #
# Batch loader
# --------------------------------------------------------------------------- #

def pdfs_to_dataframe(pdf_paths: list) -> pd.DataFrame:
    """
    Parse multiple PDFs and return one DataFrame row per file.

    Args:
        pdf_paths: Iterable of PDF file path strings.

    Returns:
        pd.DataFrame with columns described in the module docstring.
    """
    records = []
    for path in pdf_paths:
        try:
            record = extract_course_data(path)
            records.append(record)
            print(f"  Parsed: {Path(path).name}")
        except Exception as exc:
            print(f"  ERROR parsing {path}: {exc}")
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Manually expand any glob patterns in argv (needed on Windows, harmless on Mac/Linux)
    raw_args = sys.argv[1:]
    if raw_args:
        paths = []
        for arg in raw_args:
            expanded = sorted(glob.glob(arg))
            if expanded:
                paths.extend(expanded)
            else:
                paths.append(arg)   # keep as-is so the error message is meaningful
    else:
        paths = sorted(glob.glob("*.pdf"))

    if not paths:
        print("Usage: python pdf_to_dataframe.py *.pdf")
        print("       python pdf_to_dataframe.py file1.pdf file2.pdf ...")
        sys.exit(1)

    print(f"Parsing {len(paths)} PDF(s)...\n")
    df = pdfs_to_dataframe(paths)

    print(f"\nDataFrame shape: {df.shape}")

    if df.empty:
        print("No records extracted — check that the paths above point to valid PDFs.")
        sys.exit(1)

    print("\n--- Key fields ---")
    print(df[["class_id", "title", "instructor", "location", "course_type", "cost_gbp"]].to_string(index=False))

    print("\n--- Skills (all rows) ---")
    for _, row in df.iterrows():
        print(f"  {row['class_id']}: {row['skills_developed']}")

    print("\n--- Learning objectives (row 0) ---")
    for obj in df.iloc[0]["learning_objectives"]:
        print(f"  - {obj}")

    # Save DataFrame to pickle
    pickle_path = Path(paths[0]).parent / "courses.pkl"
    df.to_pickle(pickle_path)
    print(f"\nDataFrame saved to: {pickle_path}")
