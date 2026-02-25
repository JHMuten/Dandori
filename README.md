# Course Search Application

A Streamlit web application for searching and browsing course data extracted from PDF files.

## Features

- Search courses by location (dropdown filter)
- Search courses by title
- Search courses by keyword (exact text matching or semantic search)
- Semantic search using ChromaDB (finds courses by meaning, not just exact words)
- Filter by course type and price range
- Sort results by title, location, or cost
- View detailed course information including instructor, cost, learning objectives, and materials

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Extracting Data from PDFs

First, you'll need to extract course data from your PDF files. Provide the path to your PDF files as an argument:

```bash
python pdf_to_dataframe.py path/to/your/pdfs/*.pdf
```

Or on Windows:

```cmd
python pdf_to_dataframe.py path\to\your\pdfs\*.pdf
```

This will process all specified PDF files and create a `courses.pkl` file in the same directory as the PDFs.

### Setting up Semantic Search (Optional)

To enable semantic search functionality, create a ChromaDB database from your course data:

```bash
python grounding.py
```

This creates a vector database in the `data/courses_db` directory that enables semantic search (finding courses by meaning rather than exact text matches).

### Adding Geographic Coordinates (Optional)

To enrich your course data with latitude/longitude coordinates for location-based features:

```bash
python geocode_locations.py
```

This script will:
- Replace vague location names (like "District", "Gardens") with "TBC"
- Geocode all valid UK locations using the Nominatim API
- Add `latitude` and `longitude` columns to your dataset
- Save the updated data to `data/courses.pkl`
- Create a reference file at `data/location_coordinates.csv`

**Note:** This process takes ~30 seconds due to Nominatim's rate limit (1 request/second). The coordinates are cached in the pickle file, so you only need to run this once.

### Running the Application

Once you have the `courses.pkl` file, update the path in `app.py` (line 13) to point to your pickle file location, then run:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Data

The application loads course data from a pickle file containing extracted information from PDF course catalogs. You must first run `pdf_to_dataframe.py` on your PDF files to generate this pickle file.

For semantic search functionality, run `grounding.py` to create a ChromaDB vector database. This enables intelligent search that understands meaning (e.g., searching for "baking" will find "waffle" courses).

## Project Structure

- `app.py` - Main Streamlit application with semantic search support
- `pdf_to_dataframe.py` - Script for extracting data from PDFs
- `grounding.py` - Script for creating ChromaDB vector database for semantic search
- `search.py` - Example script showing how to query the ChromaDB database
- `requirements.txt` - Python dependencies
- `data/` - Directory containing the ChromaDB vector database (created by grounding.py)
