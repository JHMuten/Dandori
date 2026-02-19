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

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Data

The application loads course data from `pdf-samples/courses.pkl`, which contains extracted information from PDF course catalogs.

## Project Structure

- `app.py` - Main Streamlit application
- `pdf-samples/` - Directory containing PDF files and the pickled dataframe
- `pdf_to_dataframe.py` - Script for extracting data from PDFs
- `requirements.txt` - Python dependencies
