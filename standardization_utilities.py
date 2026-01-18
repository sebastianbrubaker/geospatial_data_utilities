# Author: Sebastian Brubaker

"""
Spatial Data Pipeline Utilities

This module provides batch processing utilities for spatial data, including:
    - Extracting .kml files from .kmz archives
    - Converting .kml files to shapefiles and GeoJSON
    - Appending a user-defined ID field to attribute data and filenames
    - Batch reprojection of shapefiles to a specified coordinate reference system (CRS)
"""

import os
import re
import geopandas as gpd
from geopandas import GeoDataFrame
import zipfile
import logging


# --- Globals --- #


LOGGER = logging.getLogger("logger")


# --- File Conversion Utilities --- #


def unzip(zip_path: str, out_dir: str) -> None:
    """
    Extract the contents of a zipfile to a specified directory.

    Parameters
    ----------
    zip_path : str
        Path to the zip file.
    out_dir : str
        Directory where contents will be extracted.
    """

    with zipfile.ZipFile(zip_path, 'r') as zip:
        zip.extractall(out_dir)


def kmz_to_kml(kmz_path: str, out_dir: str) -> str:
    """
    Extract the .kml file from a .kmz archive.

    Parameters
    ----------
    kmz_path : str
        Path to the .kmz file.
    out_dir : str
        Directory to extract the contents into.

    Returns
    -------
    str
        Full path to the extracted .kml file.

    Raises
    ------
    FileNotFoundError
        If no .kml file is found in the archive.

    Notes
    -----
    Each .kmz file is assumed to contain a single .kml file.
    """

    unzip(kmz_path, out_dir)

    kml_files = [f for f in os.listdir(out_dir) if f.lower().endswith('.kml')]

    if not kml_files:
        raise FileNotFoundError("No .kml file found inside the .kmz archive.")
    
    return os.path.join(out_dir, kml_files[0])


# --- Data Formatting Utilities --- #


def append_id(gdf: GeoDataFrame, id_field_name: str, uid: int) -> GeoDataFrame:
    """
    Append an ID column to the GeoDataFrame and populate it with the given ID.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to modify.
    id_field_name : str
        Name of the ID column to add.
    uid : int
        ID value to populate.

    Returns
    -------
    geopandas.GeoDataFrame
        Modified GeoDataFrame with the new ID field.
    """ 

    if id_field_name in gdf.columns:
        LOGGER.warning(f"WARNING: {id_field_name} is already in GeoDataFrame. Values overwritten.")

    gdf[id_field_name] = uid
    return gdf


def parse_by_geom(gdf: GeoDataFrame) -> dict[str, GeoDataFrame]:
    """
    Split a GeoDataFrame into separate GeoDataFrames by geometry type.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame.

    Returns
    -------
    dict of {str: geopandas.GeoDataFrame}
        Dictionary where keys are geometry types and values are GeoDataFrames filtered by that type.
    """
    return {
        geom_type: gdf[gdf.geom_type == geom_type].copy().reset_index(drop=True)
        for geom_type in gdf.geom_type.unique()
    }


def reproject_shp(shp_path: str, out_path: str, epsg: int=3005) -> None:
    """
    Reproject a shapefile to a specified coordinate reference system (CRS).

    Parameters
    ----------
    shp_path : str
        Input shapefile path.
    out_path : str
        Output file path.
    epsg : int, optional
        EPSG code of target CRS (default is 3005, BC Albers).

    Returns
    -------
    None
    """

    gdf = gpd.read_file(shp_path)

    if not gdf.crs:
        LOGGER.warning(f"{shp_path} CRS not set")
        return None
    
    gdf = gdf.to_crs(epsg=epsg) # Reproject crs
    gdf.to_file(out_path)


def clean_filename(filename: str) -> str:
    """
    Replace spaces and special characters in filename with underscores, preserving extension.

    Parameters
    ----------
    filename : str
        Filename to clean.

    Returns
    -------
    str
        Cleaned filename.
    """

    name, ext = os.path.splitext(filename)
    name = re.sub(r'[^A-Za-z0-9_\-]', '_', name)
    return f"{name}{ext}"


# --- Export Functions --- #


def gdf_to_file(gdf: GeoDataFrame, file_path: str) -> None:
    """
    Save a GeoDataFrame to a file. File type is inferred from extension.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to export.
    file_path : str
        Path to output file, including extension.
    """

    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    safe_base_name = clean_filename(base_name)
    safe_path = os.path.join(dir_name, safe_base_name)
    gdf.to_file(safe_path)


def gdfs_to_files(gdf_dict: dict[str, GeoDataFrame], out_dir: str, base_name: str) -> None:
    """
    Export multiple GeoDataFrames to files in a specified directory.

    Parameters
    ----------
    gdf_dict : dict of {str: geopandas.GeoDataFrame}
        Dictionary mapping geometry types to GeoDataFrames.
    out_dir : str
        Directory to write output files.
    base_name : str
        Base name for each output file.
    """ 

    for geom_type, gdf in gdf_dict.items():
        safe_geom = geom_type.replace(" ", "_").replace("/", "_")
        out_filename = clean_filename(f"{base_name}_{safe_geom}.shp")
        out_path = os.path.join(out_dir, out_filename)
        gdf_to_file(gdf, out_path)


def kml_to_shapefiles(kml_path: str, out_dir: str, base_name: str) -> None:
    """
    Convert a KML file to one or more shapefiles by geometry type.

    Parameters
    ----------
    kml_path : str
        Path to the .kml file.
    out_dir : str
        Directory to write output shapefiles.
    base_name : str
        Base name for output files.
    """

    gdf = gpd.read_file(kml_path)
    gdfs = parse_by_geom(gdf)
    gdfs_to_files(gdfs, out_dir, base_name)


# --- ID Tagging Utilities --- #


def append_id_to_shapefile(shp_path: str, out_dir: str, field_name: str, uid: int) -> None:
    """
    Append an ID field to a shapefile and save the result with the ID in the filename.

    Parameters
    ----------
    shp_path : str
        Path to the input shapefile.
    out_dir : str
        Directory to save the updated shapefile.
    field_name : str
        Name of the new ID field.
    uid : int
        ID value to assign to all features.
    """ 

    try:
        gdf = gpd.read_file(shp_path)

    except Exception as e:
        LOGGER.error(f"Failed to read {shp_path}: {e}.")
        return
    
    modified_gdf = append_id(gdf, field_name, uid)

    base_name = os.path.splitext(os.path.basename(shp_path))[0]
    safe_basename = clean_filename(f"{base_name}_{uid}.shp")
    out_path = os.path.join(out_dir, safe_basename)

    gdf_to_file(modified_gdf, out_path)


def append_id_to_shapefiles(shp_dir: str, out_dir: str, field_name: str, uid: int) -> None:
    """
    Append an ID field to all shapefiles in a directory.

    Each output file is saved to the specified directory with the ID value in the filename.

    Parameters
    ----------
    shp_dir : str
        Directory containing input shapefiles.
    out_dir : str
        Directory to save updated shapefiles.
    field_name : str
        Name of the new ID field.
    uid : int
        ID value to assign to all features.
    """

    shp_paths = [
        os.path.join(shp_dir, file)
        for file in os.listdir(shp_dir) 
        if file.lower().endswith(".shp")
    ]

    for shp in shp_paths:
        append_id_to_shapefile(shp, out_dir, field_name, uid)


# --- Batch Conversion Functions --- #


def kmzs_to_kmls(kmz_dir: str, out_dir: str) -> str:
    """
    Batch extract .kml files from .kmz files in a directory.

    Each .kml is renamed to match its .kmz filename and saved to the output directory.

    Parameters
    ----------
    kmz_dir : str
        Directory containing .kmz files.
    out_dir : str
        Directory to save extracted .kml files.

    Returns
    -------
    str
        Path to the output directory.

    Notes
    -----
    Each .kmz file is assumed to contain exactly one .kml file.
    """

    for filename in os.listdir(kmz_dir):
        if filename.lower().endswith('.kmz'):
            kmz_path = os.path.join(kmz_dir, filename)
            try:
                kml_path = kmz_to_kml(kmz_path, out_dir)
                base_name = os.path.splitext(filename)[0]
                out_kml_path = os.path.join(out_dir, f"{base_name}.kml")
                os.replace(kml_path, out_kml_path)
            except Exception as e:
                LOGGER.error(f"Error extracting {filename}: {e}")

    return out_dir


def kmls_to_shapefiles(kml_dir: str, out_dir: str) -> None:
    """
    Batch convert .kml files in a directory to shapefiles.

    Parameters
    ----------
    kml_dir : str
        Directory containing .kml files.
    out_dir : str
        Directory to save resulting shapefiles.
    """ 

    for filename in os.listdir(kml_dir):
        if filename.lower().endswith('.kml'):
            kml_path = os.path.join(kml_dir, filename)
            base_name = os.path.splitext(filename)[0]
            kml_to_shapefiles(kml_path, out_dir=out_dir, base_name=base_name)


def reproject_shps(input_root: str, output_root: str, epsg: int=3005) -> None:
    """
    Reproject all shapefiles in a directory (recursively) to a specified EPSG code
    and save them to an output directory.

    Parameters
    ----------
    input_root : str
        Root directory to search for .shp files.
    output_root : str
        Directory where reprojected files will be saved.
    epsg : int, optional
        EPSG code for the target projection (default is 3005 - BC Albers)
    """

    for dirpath, _, filenames in os.walk(input_root):
        for file in filenames:
            if file.lower().endswith(".shp"):
                in_shp_path = os.path.join(dirpath, file)
                out_shp_path = os.path.join(output_root, file)

                try:
                    reproject_shp(in_shp_path, out_shp_path, epsg)
                    LOGGER.info(f"Reprojected: {file}")
                except Exception as e:
                    LOGGER.error(f"Failed to reproject {file}: {e}")
