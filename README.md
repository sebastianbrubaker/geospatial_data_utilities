# Geospatial Data Utilities
---
**These Python modules offer various utilities for finding and processing/normalizing geospatial data that can be easily orchestrated into a pipeline**
## `normalization_utilities.py`
This module provides batch processing utilities for spatial data, including:
- Extracting .kml files from .kmz archives
- Converting .kml files to shapefiles and GeoJSON
- Appending a user-defined ID field to attribute data and filenames
- Batch reprojection of shapefiles to a specified coordinate reference system (CRS)

## `search_utilities.py`
This module provides a toolkit for recursively searching, matching, and organizing files 
with support for spatial data formats, semantic similarity search, and fuzzy matching.

Core features include:
- Recursive file system crawling with optional depth control.
- File type search (including inside .zip archives).
- Semantic similarity search:
    - Text-based (SentenceTransformer models).
    - Image-based (CLIP embeddings).
- Fuzzy text search for approximate keyword matching.
- Batch search execution across multiple files.
- Utility functions for:
    - Flattening tabular data for search.
    - Extracting specific files from zips.
    - Rendering PDFs to images.
    - Path normalization and Windows max path length bypass.
- Directory creation and cleanup helpers for structured staging of results.

## 'search.ipynb'
This notebook is an example of how the results from 'search_utilities.crawl_dir()' can be searched.

## 'control.ipynb'
This notebook is an example of how functions from `search_utilities.py` and `search_utilities.py` can be strung together to automate the preperation of legacy geospatial data for BC EAO projects.