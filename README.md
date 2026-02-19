# Course Search Application

A Streamlit web application for searching and browsing course data extracted from PDF files.

## Features

- Search courses by location (dropdown filter)
- Search courses by title
- Search courses by keyword (searches across title, description, skills, and learning objectives)
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

### Running the Application

Once you have the `courses.pkl` file, update the path in `app.py` (line 9) to point to your pickle file location, then run:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Data

The application loads course data from a pickle file containing extracted information from PDF course catalogs. You must first run `pdf_to_dataframe.py` on your PDF files to generate this pickle file.

## Project Structure

- `app.py` - Main Streamlit application
- `pdf-samples/` - Directory containing PDF files and the pickled dataframe
- `pdf_to_dataframe.py` - Script for extracting data from PDFs
- `requirements.txt` - Python dependencies
