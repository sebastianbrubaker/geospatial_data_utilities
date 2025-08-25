# Author: Sebastian Brubaker

"""
Data Search Utilities

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
"""

import os
import re
import numpy as np
import pandas as pd
from pandas import DataFrame
from fuzzywuzzy import fuzz
import shutil
import zipfile
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import fitz
import logging
from tqdm import tqdm
import torch
from glob import glob


# --- Globals --- #


LOGGER = logging.getLogger("logger")

SPATIAL_KEYWORDS = {
    # General coordinate terms
    "latitude", "longitude",
    "x coordinate", "y coordinate",
    #"x_coord", "y_coord", "xcoord", "ycoord",

    # # Projected coordinate systems
    "easting", "northing",
    "utm easting", "utm northing",
    # "utm_x", "utm_y",
    # "utm zone",

    # # Geographic coordinate systems
    "wgs84", "nad83", "coordinate system", "crs",

    # # Combined or embedded fields
    # "coordinates", "location", "lat/lon", "latlong", "longlat",
    # "point", "geometry", "spatial location", "spatial reference",

    # # Colloquial or ambiguous terms (used cautiously)
    # "site location", "map reference", "gps", "gps coordinates",
    # "geo location", "grid reference", "grid coordinates",

    # # Specific to BC/Canada context
    # "bc albers", "albers", "epsg", "nadt83", "natural resource coordinates",

    # # Other possible indicators
    # "zone", "mgrs", "map sheet", "tile id"
}

ALL_TARGET_EXTENSIONS = {'.zip', '.shp', '.gdb', '.geojson', '.kml', '.kmz',
                        '.dwg', '.png', '.pdf', '.jpg', '.csv', '.xls', '.xlsx',
                        '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx', '.qix'}

SPATIAL_EXTENSIONS = {'.shp', '.gdb', '.geojson', '.kml', '.kmz',
                        '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx', '.qix'}

SHAPEFILE_EXTENSIONS = {".shp", ".shx", ".dbf", ".prj", ".cpg", 
                        ".sbn", ".sbx", ".qix", ".aih", ".ain"}

TABULAR_EXTENSIONS = {'.csv', '.xls', '.xlsx'}

IMAGE_EXTENSIONS = {'.png', '.jpg'}

TXT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
IMG_MODEL = SentenceTransformer("clip-ViT-B-32")


# --- Search Utilities --- #


def search_type(search_root:str, target_extensions:set[str], bypass_path_len:bool=True) -> DataFrame:
    """
    Recursively searches a directory tree for files with specific extensions.

    Parameters
    ----------
    search_root : str
        Path where the recursive search begins.
    target_extensions : set[str]
        Iterable of file extensions to match. Extensions should include a leading
        dot and are matched case-insensitively (e.g. {'.shp', '.csv'}).
    bypass_path_len : bool, optional
        If True, will prepend the Windows long-path prefix to paths to avoid
        `MAX_PATH` issues on Windows. Default is True.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per matched file and the following columns:
            - path (str): Full, absolute file path.
            - extension (str): File extension (lowercased).
            - filename (str): The file's base name.
            - folder (str): Parent directory of the file.
    """

    project_files = []

    if bypass_path_len:
        search_root = prepend_max_path_bypass(search_root)

    for root, dirs, files in os.walk(search_root):
        if bypass_path_len:
            root = prepend_max_path_bypass(root)

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in target_extensions:
                file_path = os.path.join(root, file)
                project_files.append({
                    'path': file_path,
                    'extension': ext,
                    'filename': file,
                    'folder': root
                })

    return pd.DataFrame(project_files)


def search_type_zips(zip_folder: str, target_extensions: set[str]) -> pd.DataFrame:
    """
    Scans all .zip files within a folder and reports members whose extensions match
    the provided target_extensions.

    Parameters
    ----------
    zip_folder : str
        Directory containing .zip files to scan.
    target_extensions : set[str]
        File extensions to match inside archive members (leading dot expected).

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per matching archive member and columns:
            - path (str): A synthetic path combining the zip file path and member path.
            - zip_path (str): Filesystem path to the .zip archive.
            - member (str): The internal member path inside the zip.
            - extension (str): Member file extension (lowercased).
            - filename (str): Basename of the member.
    """

    zip_hits = []

    for zip_name in os.listdir(zip_folder):
        
        if zip_name.lower().endswith('.zip'):
            zip_path = os.path.join(zip_folder, zip_name)

            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    for member in z.namelist():
                        ext = os.path.splitext(member)[1].lower()
                        if ext in target_extensions:
                            zip_hits.append({
                            'path': os.path.join(zip_path, member),
                            "zip_path": zip_path,
                            "member": member,
                            'extension': ext,
                            'filename': os.path.basename(member),
                        })
                        
            except Exception as e:
                LOGGER.error(f"Failed to read {zip_path}: {e}")

    return pd.DataFrame(zip_hits)


def crawl_dir(root: str, results: list[dict]=None, max_depth: int=-1, cur_depth: int=0) -> list[dict]:
    """
    Recursively traverses a directory and returns metadata for each item encountered.

    Parameters
    ----------
    root : str
        Directory path to start crawling.
    results : list[dict], optional
        List to append results to. If None, a new list is created and returned.
    max_depth : int, optional
        Maximum recursion depth. Use -1 for unlimited depth (default: -1).
    cur_depth : int, optional
        Current recursion depth; callers should not set this (internal use).

    Returns
    -------
    list[dict]
        A list of dictionaries describing each entry. Each dictionary contains:
            - full_path (str): Full path to the entry.
            - item_name (str): The name of the entry (basename).
            - extension (str): The entry's file extension (lowercased) or '' for folders.
            - is_folder (bool): True if the entry is a directory.
    """

    if results is None:
        results = []

    if max_depth != -1 and cur_depth > max_depth:
        return results
    
    for entry in os.scandir(prepend_max_path_bypass(root)):
        entry_path = prepend_max_path_bypass(entry.path) # Bypass max path length
        results.append({
                        "full_path": entry_path,
                        "item_name": entry.name,
                        "extension": os.path.splitext(entry.name)[-1].lower(),
                        "is_folder": entry.is_dir(),
                        })
        
        if entry.is_dir():
            crawl_dir(entry_path, results, max_depth=max_depth, cur_depth=cur_depth + 1)
    
    return results
        

def fuzzy_match(string: str, keywords: list[str], threshold: int=85) -> bool:
    """
    Determines if a given string approximately matches any keywords using fuzzy matching.

    Parameters
    ----------
    string : str
        Input string to test. Non-string inputs return False.
    keywords : list[str]
        Sequence of words to match against.
    threshold : int, optional
        Minimum fuzzy match score (0-100) to consider as a match. Default is 85.

    Returns
    -------
    bool
        True if any keyword approximately matches the input string, otherwise False.
    """

    if not isinstance(string, str):
        return False

    for keyword in keywords:
        if fuzz.token_set_ratio(string.lower(), keyword.lower()) >= threshold:
            return True
        
    return False


def fuzzy_search_file(file_path: str, keywords: list[str], threshold: int=85) -> bool:
    """
    Scans a tabular file (.csv, .xls, .xlsx) for fuzzy keyword matches across headers
    and cell values.

    Parameters
    ----------
    file_path : str
        Path to the tabular file to scan.
    keywords : list[str]
        Keywords to fuzzy-match against column names and cell values.
    threshold : int, optional
        Fuzzy match score threshold (0-100). Default is 85.

    Returns
    -------
    bool
        True if any header or cell value matches any keyword approximately; False otherwise.

    Notes
    -----
    - Non-supported file extensions raise a ValueError which is caught and logged;
      in such cases the function returns False.
    - All data is read as strings to keep comparisons consistent.
    """

    ext = os.path.splitext(file_path)[-1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, dtype=str)
        elif ext in {".xls", ".xlsx"}:
            df = pd.read_excel(file_path, dtype=str)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        LOGGER.error(f"Failed to open {file_path}: {e}")
        return False
    
    # Scan file for keyword matches
    for col in df.columns:
        if fuzzy_match(col, keywords, threshold):
            return True
        for val in df[col].dropna().astype(str):
            if fuzzy_match(val, keywords, threshold):
                return True

    return False


def semantic_match(strings: list[str], keywords: list[str], model=TXT_MODEL) -> float:
    """
    Computes the maximum cosine similarity between two collections of strings using the
    provided SentenceTransformer model.

    Parameters
    ----------
    strings : list[str]
        List of strings to compare (e.g. file headers, folder names).
    keywords : iterable[str]
        Strings to compare against.
    model : SentenceTransformer, optional
        Pre-loaded SentenceTransformer model used to create embeddings. Defaults to TXT_MODEL.

    Returns
    -------
    float
        The maximum cosine similarity score observed between any pair of string and keyword.

    Notes
    -----
    - Input collections are cleaned (duplicated removed, lowercased & stripped) before
      embedding to avoid redundant comparisons.
    """

    keywords = prep_1D_collection(keywords)
    strings = prep_1D_collection(strings)
    data_embeddings = model.encode(strings, convert_to_tensor=True)
    keyword_embeddings = model.encode(keywords, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(data_embeddings, keyword_embeddings)
    return cosine_scores.max().item()


def semantic_batch_match(items: list[str], query: str, model=TXT_MODEL, clean_strings: bool=True) -> list[float]:
    """
    Computes similarity scores between a list of items and a single textual query.

    Parameters
    ----------
    items : list[str]
        Items to compare (e.g. folder names, filenames).
    query : str
        A single search query string.
    model : SentenceTransformer or str, optional
        If a string is provided the function will instantiate a SentenceTransformer
        with that name. Otherwise, provide a preloaded SentenceTransformer.
    clean_strings : bool, optional
        If True, items and query will be stripped and lowercased prior to encoding.

    Returns
    -------
    list[float]
        Similarity scores (one per item) where higher values indicate greater similarity
        to the query. The order corresponds to the order of `items`.

    Notes
    -----
    - The function returns scores as a Python list of floats for convenience.
    """

    if type(model) == str:
        model = SentenceTransformer(model) # User specifed model

    if clean_strings:
        items = [str(i).strip().lower() for i in items]
        query = query.strip().lower()

    item_embeddings = model.encode(items, convert_to_tensor=True)
    query_embedding = model.encode([query], convert_to_tensor=True)

    cosine_scores = model.similarity(item_embeddings, query_embedding)
    return cosine_scores.squeeze().cpu().numpy().tolist()


def semantic_search_dir(dir_tree: pd.DataFrame, search_term: str, search_field: str = "item_name",
                        model=TXT_MODEL) -> pd.DataFrame:
    """
    Performs a semantic search against a DataFrame that represents filesystem entries
    (such as what `crawl_dir` returns).

    Parameters
    ----------
    dir_tree : pandas.DataFrame
        DataFrame containing at least the column specified by `search_field`.
    search_term : str
        Query string to compare against each entry in `search_field`.
    search_field : str, optional
        Column name in `dir_tree` to compare to `search_term` (default: "item_name").
    model : SentenceTransformer, optional
        Model used for embedding text. Defaults to TXT_MODEL.

    Returns
    -------
    pandas.DataFrame
        A copy of `dir_tree` with a new column "similarity" containing similarity
        scores. The returned DataFrame is sorted in descending order by "similarity".
    """

    ret_df = dir_tree.copy()
    folder_names = ret_df[search_field].fillna("").astype(str).tolist()
    scores = semantic_batch_match(folder_names, search_term, model=model)
    ret_df["similarity"] = scores

    # Sort by score
    ret_df = ret_df.sort_values(by="similarity", ascending=False)

    return ret_df


def semantic_search_file(file_path: str, keywords: set[str]=SPATIAL_KEYWORDS, both_ways: bool=True, n: int=10) -> float:
    """
    Computes the semantic similarity between a tabular file's headers / first rows and
    a set of keywords. The function reads the first `n` rows (and optionally the first
    `n` columns when `both_ways=True`) and returns the maximum similarity score.

    Parameters
    ----------
    file_path : str
        Path to a tabular file (.csv, .xls, .xlsx).
    keywords : set[str], optional
        Keywords to compare against. Defaults to SPATIAL_KEYWORDS.
    both_ways : bool, optional
        If True, both row-wise and column-wise (transposed) summaries will be used.
    n : int, optional
        Number of rows / columns to include in the flattened representation.

    Returns
    -------
    float
        Maximum cosine similarity score between the file-derived strings and keywords.

    See Also
    --------
    - flatten_tabular_rows: helper that extracts headers, sheet names and sample values.
    - semantic_match: performs the actual embedding + comparison.
    """
    strings = prep_1D_collection(flatten_tabular_rows(file_path, both_ways, n))
    return semantic_match(strings, keywords)


def semantic_search_images(
    img_paths: list[str],
    query: str | Image.Image,
    model,
    top_k: int=3,
    batch_size: int=64,
) -> list[dict]:
    """
    Performs a batched semantic search across images using an image-capable
    SentenceTransformer (e.g., CLIP-based models).

    Parameters
    ----------
    img_paths : list[str]
        Paths to candidate images (corpus).
    query : str or PIL.Image.Image
        Textual or image query to rank corpus images against.
    model : SentenceTransformer
        A SentenceTransformer model capable of encoding images (and queries).
    top_k : int, optional
        Number of top results to return per query. Defaults to 3.
    batch_size : int, optional
        Number of images to encode per batch to control memory use.

    Returns
    -------
    list[dict]
        List of top result dictionaries containing:
            - file_path (str): Path to the matched image.
            - score (float): Similarity score.

    Notes
    -----
    - Failures reading or encoding specific images are logged and those images are
      skipped. If no images can be encoded the function returns None.
    """

    img_embs = []

    for i in tqdm(range(0, len(img_paths), batch_size), desc="Encoding images"):
        batch_paths = img_paths[i:i+batch_size]

        batch_imgs = []
        for path in batch_paths:
            try:
                batch_imgs.append(Image.open(path).convert("RGB"))
            except Exception as e:
                LOGGER.error(f"Failed to read {path}: {e}")

        with torch.no_grad():
            try:
                batch_emb = model.encode(batch_imgs, convert_to_tensor=True)
            except Exception as e:
                LOGGER.error(f"Failed to encode batch: {e}")
        
        img_embs.append(batch_emb)

    # Stack all batches
    try:
        corpus_embeddings = torch.cat(img_embs, dim=0)

        # Encode query
        query_emb = model.encode([query], convert_to_tensor=True)

        # Search
        hits = util.semantic_search(query_embeddings=query_emb, corpus_embeddings=corpus_embeddings, top_k=top_k)[0]

        # Build result
        results = [
            {
                "file_path": img_paths[hit["corpus_id"]],
                "score": float(hit["score"]),
            }
            for hit in hits
        ]
        
        return results
    
    except Exception as e:
        LOGGER.error(f"Failed to search images: {e}")
        return None


def get_descendants(root: str, dir_tree: pd.DataFrame) -> list[dict]:
    """
    Returns all entries in `dir_tree` whose `full_path` is a descendant of `root`.

    Parameters
    ----------
    root : str
        The ancestor directory path for which descendants will be returned.
    dir_tree : pandas.DataFrame
        DataFrame containing a column `full_path` (as produced by `crawl_dir`).

    Returns
    -------
    list[dict]
        A list of dicts (rows) from `dir_tree` that are descendants of `root`.

    Notes
    -----
    - The function normalizes `root` and ensures trailing separators so partial
      matches (e.g. '/data' matching '/data2') are avoided.
    """

    # Ensure root ends with a separator for exact matching
    root = os.path.normpath(root)
    if not root.endswith(os.sep):
        root = root + os.sep

    # Find all entries whose full_path starts with the root path (excluding the root itself)
    descendants = dir_tree[
        (dir_tree["full_path"].str.startswith(root)) &
        (dir_tree["full_path"] != root.rstrip(os.sep))
    ]

    return descendants.to_dict(orient="records")


# --- Formatting and Exporting Tools --- #


def strip_prefix(path: str) -> str:
    """
    Removes any leading path prefixes and returns a path starting from the drive
    letter (e.g. 'C:') if present.

    Parameters
    ----------
    path : str
        File path to normalize.

    Returns
    -------
    str
        Path trimmed to start at the first found drive-letter pattern (e.g. 'C:')
        if such a pattern exists; otherwise the original path is returned.
    """

    match = re.search(r"[A-Z]:", path, re.IGNORECASE)

    if match:
        return path[match.start():]
    return path


def prepend_max_path_bypass(path: str) -> str:
    """
    Prepends the Windows long-path prefix ("\\?\\") to a path if it is not
    already present. This helps avoid Windows' MAX_PATH limitations when using
    the Win32 API via Python file operations.

    Parameters
    ----------
    path : str
        File path to modify.

    Returns
    -------
    str
        Path with the long-path prefix added when necessary. If the prefix is
        already present the input path is returned unchanged.
    """

    if path.startswith("\\\\?\\"):
        return path
    return r"\\?\{}".format(path)


def unzip(zip_path: str, out_dir: str) -> None:
    """
    Extracts an entire zip archive into `out_dir`.

    Parameters
    ----------
    zip_path : str
        Path to the .zip archive.
    out_dir : str
        Directory where archive contents will be extracted. The directory will be
        created if it does not already exist.
    """

    with zipfile.ZipFile(zip_path, 'r') as zip:
        zip.extractall(out_dir)


def extract_member_from_zip(zip_path: str, member: str, out_dir: str):
    """
    Extracts a single member file from a zip archive and writes it into `out_dir`.

    Parameters
    ----------
    zip_path : str
        Path to the .zip archive.
    member : str
        Internal path to the member inside the zip to extract.
    out_dir : str
        Output directory for the extracted member. The directory will be created
        implicitly by the open() call if it exists; callers should ensure it exists.
    """

    filename = os.path.basename(member)
    output_path = os.path.join(out_dir, filename)
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(member) as source, open(output_path, 'wb') as target:
            target.write(source.read())


def extract_members_from_zip(zip_paths: list[str], members: list[str], out_dir: str):
    """
    Extracts a list of members from a sequence of zip archives. The two lists are
    paired element-wise (zip_paths[i] -> members[i]).

    Parameters
    ----------
    zip_paths : list[str]
        List of zip archive paths.
    members : list[str]
        List of member paths corresponding to each archive.
    out_dir : str
        Output directory to write the extracted files.
    """

    for zip_path, member in zip(zip_paths, members):
        extract_member_from_zip(zip_path, member, out_dir)


def flatten_df(df: pd.DataFrame, n: int=10, transpose: bool=False) -> list[str]:
    """
    Flattens the first `n` rows (or columns when transpose=True) of a DataFrame
    into a list of strings including headers and cell values. Non-string values
    are cast to strings to preserve content for search.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to flatten.
    n : int, optional
        Number of rows (or columns) to include. Defaults to 10.
    transpose : bool, optional
        If True, the function will flatten the first `n` columns (using the
        DataFrame's index as header-like identifiers). Default is False.

    Returns
    -------
    list[str]
        Concatenation of header/index labels and flattened cell values as strings.
    """

    try:
        df = df.astype(str)

        if not transpose:
            df_subset = df.head(n)
            header_list = df_subset.columns.astype(str).tolist()
            value_list = df_subset.to_numpy().flatten().tolist()
        else:
            df_subset = df.iloc[:, :n]
            header_list = df_subset.index.astype(str).tolist()
            value_list = df_subset.to_numpy().flatten().tolist()

        return header_list + value_list

    except Exception as e:
        LOGGER.error(f"Error flattening DataFrame: {e}")
        return []
    

def flatten_tabular_rows(data: str, both_ways: bool=True, n: int=10) -> list[str]:
    """
    Extracts header, sheet names (for Excel), and the first `n` rows/columns of
    a tabular file and returns them as a flat list of strings suitable for
    semantic or fuzzy searching.

    Parameters
    ----------
    data : str
        Path to the tabular file (.csv, .xls, .xlsx).
    both_ways : bool, optional
        If True, includes both row-wise and column-wise (transposed) content.
    n : int, optional
        Number of rows / columns to include from each sheet. Defaults to 10.

    Returns
    -------
    list[str]
        List containing the filename, sheet names (if any), headers, and sampled
        cell values as strings.
    """

    ext = os.path.splitext(data)[1].lower()
    f_name = os.path.basename(data)
    flat_list = []

    try:
        if ext == '.csv':
            df = pd.read_csv(data, encoding="utf-8")
            flat_list.append(f_name)
            flat_list.extend(flatten_df(df, n))

            if both_ways:
                flat_list.extend(flatten_df(df, n, transpose=True))

        elif ext in {'.xls', '.xlsx', '.xlsm'}:
            sheet_dict = pd.read_excel(data, sheet_name=None, dtype=str)
            flat_list.append(f_name)

            for sheet_name, df in sheet_dict.items():
                flat_list.append(sheet_name)
                flat_list.extend(flatten_df(df, n))

                if both_ways:
                    flat_list.extend(flatten_df(df, n, transpose=True))

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except Exception as e:
        LOGGER.error(f"Failed to flatten {data}: {e}")

    return flat_list


def prep_1D_collection(collection: list[str]) -> list:
    """
    Cleans an iterable of strings for semantic processing by removing duplicates
    and normalizing case and whitespace.

    Parameters
    ----------
    collection : iterable[str]
        Iterable of values (ideally strings) to normalize.

    Returns
    -------
    list[str]
        List of unique, lowercased, stripped strings derived from `collection`.
    """

    cleaned = {str(i).strip().lower() for i in collection if isinstance(i, str)}
    return list(cleaned)


def unique_folders(csv_path: str, out_path: str) -> None:
    """
    Reads a hits CSV generated by `search_type` and writes a CSV that contains
    a single column of unique folder paths.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file containing a `folder` column.
    out_path : str
        Destination CSV file to write unique folder values to.

    Returns
    -------
    None
    """
    df = pd.read_csv(csv_path)
    filtered_df = df["folder"].drop_duplicates()
    filtered_df.to_csv(out_path)


def gather_files(file_paths: list[str], target_folder: str, file_ext: set=None,
                  glob_files: bool=False, glob_type: set[str]=None,
                  ) -> None:
    """
    Copies files listed in `file_paths` into `target_folder`, optionally: filtering
    by extension and copying sidecar files (e.g. a .shp files .shx sidecar file).

    Parameters
    ----------
    file_paths : list[str]
        Paths of files to copy.
    target_folder : str
        Destination directory where files will be copied. Created if it doesn't exist.
    file_ext : set[str] or None, optional
        If provided, only files whose extension (lowercased) is in this set are
        copied. Default is None (copy all provided files).
    glob_files : bool, optional
        If True files with file type matching that specified by 'glob_type' will be copied
        along with their sidecar files.
    glob_type : set[str], optional
        The types of files to consider for sidecar copying if 'glob_files' is True. '.shp' by default.
    
    Returns
    -------
    None
    """
    if glob_type == None:
        glob_type = {'.shp'}

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file_path in file_paths:

        ext = os.path.splitext(file_path)[-1].lower()

        # Skip files that don't match extension 
        if file_ext and ext not in file_ext:
            continue

        # Sidecar copying logic    
        if ext in glob_type and glob_files:
            base_no_ext = os.path.splitext(file_path)[0]

            pattern = f"{base_no_ext}.*"
            sidecar_files = glob(pattern)

            for sf in sidecar_files:
                try:
                    dest = os.path.join(target_folder, os.path.basename(sf))
                    shutil.copy2(sf, dest)
                    LOGGER.debug(f"Copied sidecar: {os.path.basename(sf)}")
                except Exception as e:
                    LOGGER.error(f"Failed to copy sidecar {sf}: {e}")
            continue  # skip to next file_path after copying all sidecars

        # Normal file copying logic 
        try:
            filename = os.path.basename(file_path)
            destination = os.path.join(target_folder, filename)
            shutil.copy2(file_path, destination)
            LOGGER.debug(f"Copied: {filename}")
        except Exception as e:
            LOGGER.error(f"Failed to copy {file_path}: {e}")


def render_pdf_pages(pdf_path: str, out_dir: str = "rendered_pages", zoom: float = 2.0) -> None:
    """
    Renders each page of a PDF file into PNG images and saves them into `out_dir`.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file to render.
    out_dir : str, optional
        Directory where PNGs will be written. Created if it does not exist.
    zoom : float, optional
        Zoom factor used when rasterizing pages to increase output resolution.

    Returns
    -------
    None
    """

    os.makedirs(out_dir, exist_ok=True)
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                try:
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    output_path = os.path.join(out_dir, f"{os.path.basename(pdf_path)}_page_{i+1}.png")
                    pix.save(output_path)
                except Exception as e:
                    LOGGER.error(f"Failed to render page {i} of {pdf_path}: {e}")
    except Exception as e:
        LOGGER.error(f"Failed to open PDF {pdf_path}: {e}")
        

def clean_dir(dir_path: str) -> None:
    """
    Removes all files and subdirectories contained within `dir_path`.

    Parameters
    ----------
    dir_path : str
        Directory to clean. The function will remove files and recursively delete
        subdirectories. It will not remove `dir_path` itself.

    Returns
    -------
    None
    """

    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def make_dump_dir(
                dest_dir: str="",
                root_name: str="data1",
                all_files: str="files",
                shapefiles: str="shapefiles",
                prepped_shapefiles: str="prepped_shapefiles",
                kml_files: str="kmls",
                kmz_files: str="kmzs",
                zip_files: str="zips",
                tabular_files: str="tabular",
                spatial_tabular: str="spatial_tabular",
                images: str="images",
                image_hits: str="image_hits",
                pdf_images: str="pdf_images",
                pdfs: str="pdfs",
                ) -> None:
    """
    Creates a structured directory tree used for dumping gathered project files.

    Parameters
    ----------
    dest_dir : str, optional
        Parent directory where the root project directory will be created. Defaults
        to the current working directory when an empty string is provided.
    root_name : str, optional
        Name of the project root folder created under `dest_dir`.
    all_files, shapefiles, ... : str, optional
        Names for each child folder created beneath the project root. The function
        returns a dictionary mapping logical keys to full folder paths.

    Returns
    -------
    dict
        Mapping of logical folder keys (e.g. 'root', 'images', 'tabular_files') to
        their corresponding absolute paths on disk.
    """

    root_path = os.path.join(dest_dir, root_name)
    paths = {
        "root": root_path,
        "all_files": os.path.join(root_path, all_files),
        "shapefiles": os.path.join(root_path, shapefiles),
        "prepped_shapefiles": os.path.join(root_path, prepped_shapefiles),
        "kml_files": os.path.join(root_path, kml_files),
        "kmz_files": os.path.join(root_path, kmz_files),
        "zip_files": os.path.join(root_path, zip_files),
        "tabular_files": os.path.join(root_path, tabular_files),
        "spatial_tabular": os.path.join(root_path, spatial_tabular),
        "images": os.path.join(root_path, images),
        "image_hits": os.path.join(root_path, image_hits),
        "pdf_images": os.path.join(root_path, pdf_images),
        "pdfs": os.path.join(root_path, pdfs),
    }
    for folder_path in paths.values():
        os.makedirs(folder_path, exist_ok=True)
    return paths


def make_staging_dir(dest_dir: str, root_name: str, drive_location: str,
                     meta_folder: str="meta",
                     image_folder: str="images",
                     shapefile_folder: str="shapefiles",
                     drive_location_file: str="drive_location.txt",
                     ) -> dict:
    """
    Creates a project-stage folder structure and writes a small metadata file with
    the original drive location.

    Parameters
    ----------
    dest_dir : str
        Parent directory where the project folder will be created.
    root_name : str
        Name of the project folder to create.
    drive_location : str
        Value written to the drive location file inside the meta folder.
    meta_folder, image_folder, shapefile_folder, drive_location_file : str, optional
        Names of subfolders and metadata filename used in the staging folder.

    Returns
    -------
    dict
        Mapping describing the key paths that were created (prj_folder, meta_folder,
        drive_location_file, image_folder, shapefile_folder).
    """

    # Define paths
    prj_folder = os.path.join(dest_dir, root_name)
    meta_folder = os.path.join(prj_folder, meta_folder)
    drive_location_file = os.path.join(meta_folder, drive_location_file)
    image_folder = os.path.join(prj_folder, image_folder)
    shapefile_folder = os.path.join(prj_folder, shapefile_folder)

    paths = {
            "prj_folder": prj_folder,
            "meta_folder": meta_folder,
            "drive_location_file": drive_location_file,
            "image_folder": image_folder,
            "shapefile_folder": shapefile_folder,
    }

    # Create dirs
    for key, path in paths.items():
        if key != "drive_location_file":
            os.makedirs(path, exist_ok=True)

    # Write drive_location to txt
    with open(drive_location_file, "w") as file:
        file.write(drive_location)

    return paths


# --- Batch Processing --- #


def batch_search_files(
    path_list: list[str],
    search_fn: callable,
    *args,
    **kwargs
) -> pd.DataFrame:
    """
    Applies `search_fn` to each path in `path_list` and collects results into a
    DataFrame. The `search_fn` should accept a single file path as its first
    argument and return a truthy value indicating a match or a numeric score.

    Parameters
    ----------
    path_list : list[str]
        List of file paths to evaluate.
    search_fn : callable
        Function that will be called as `search_fn(file_path, *args, **kwargs)`.
    *args, **kwargs : any
        Additional arguments forwarded to `search_fn`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with rows for each file and columns:
            - path, extension, filename, folder, search_result
    """

    results = []

    for file_path in path_list:
        LOGGER.info(f"Searching: {os.path.basename(file_path)}")
        try:
            match = search_fn(file_path, *args, **kwargs)
        except Exception as e:
            LOGGER.error(f"Error searching {file_path}: {e}")
            match = False

        results.append({
            "path": file_path,
            "extension": os.path.splitext(file_path)[1].lower(),
            "filename": os.path.basename(file_path),
            "folder": os.path.dirname(file_path),                
            "search_result": match
        })

    return pd.DataFrame(results)


def get_dir_item_paths(dir_path:str) -> list[str]:
    """
    Returns a list of full paths for items contained directly within `dir_path`.

    Parameters
    ----------
    dir_path : str
        Directory to list.

    Returns
    -------
    list[str]
        List of absolute paths for each item in `dir_path`.
    """

    items = os.listdir(dir_path)
    return [os.path.join(dir_path, item) for item in items]