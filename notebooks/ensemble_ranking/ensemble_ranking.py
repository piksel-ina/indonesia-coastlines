#!/usr/bin/env python
# coding: utf-8

# # Indonesian ensemble rankings creation

import pandas as pd
import geopandas as gpd
from eo_tides.validation import tide_correlation
from datacube import Datacube

import os
os.environ["USE_PYGEOS"] = "0"

from odc.stac._mdtools import _normalize_geometry
from collections import Counter
import xarray as xr
import numpy as np
#from odc.geo.geom import point


def http_to_s3_url(http_url):
    """Convert a USGS HTTP URL to an S3 URL"""
    s3_url = http_url.replace(
        "https://landsatlook.usgs.gov/data", "s3://usgs-landsat"
    ).rstrip(":1")
    return s3_url
    
def mostcommon_crs(datasets):
    crs_counts = Counter(dataset.metadata_doc["crs"] for dataset in datasets)
    return crs_counts.most_common(1)[0][0]

    
def load_index(
    dc,
    time = None,
    lon = None,
    lat = None,
    geopolygon = None,
    mask_geopolygon = False,
    crs = None,
    resolution = 30,
    resampling = "cubic",
    max_cloud_cover = 60,
    load_ls = True,
    load_s2 = False,
    chunks={"x": 2048, "y": 2048},
    index="ndwi",
):
    """Load an NDWI or MNDWI time-series from Landsat and/or Sentinel-2 from datacube.
    adapted from load_ndwi_mpc()

    Parameters
    ----------
    time : tuple, optional
        The time range to load data for as a tuple of strings (e.g.
        `("2020", "2021")`. If not provided, data will be loaded for
        all available timesteps.
    lon, lat : tuple, optional
        Tuples defining the spatial x and y extent to load in degrees.
    geopolygon : multiple types, optional
        Load data into the extents of a geometry. This could be an
        odc.geo Geometry, a GeoJSON dictionary, Shapely geometry, GeoPandas
        DataFrame or GeoSeries. GeoJSON and Shapely inputs are assumed to
        be in EPSG:4326 coordinates.
    mask_geopolygon : bool, optional
        Whether to mask pixels as nodata if they are outside the extent
        of a provided geopolygon. Defaults to False.
    crs : str, optional
        The Coordinate Reference System (CRS) to load data into. Defaults
        to None, which will attempt to load data into its most common native
        CRS to minimise resampling.
    resolution : int, optional
        Spatial resolution to load data in. Defaults to 30 metres.
    resampling : str, optional
        Resampling method used for surface reflectance bands. Defaults
        to "cubic"; "nearest" will always be used for categorical cloud
        masking bands.
    max_cloud_cover : int, optional
        The maximum threshold of cloud cover to load. Defaults to 60%.
    load_ls : bool, optional
        Whether to query and load Landsat data.
    load_s2 : bool, optional
        Whether to query and load Sentinel-2 data.
    chunks : dictionary, optional
        Dask chunking used to load data as lazy Dask backed arrays.
        Defaults to `{"x": 2048, "y": 2048}`.

    Returns
    -------
    satellite_ds : xarray.Dataset
        The loaded dataset as an `xarray.Dataset`, containing a single
        "ndwi" `xarray.DataArray`.

    """
    # Assemble parameters used for querying and loading
    query_params = {
        "time": time,
        "geopolygon": geopolygon,
    }
    load_params = {
        "resolution": resolution,
        "dask_chunks": chunks,
        "group_by": "solar_day",
        "resampling": {"qa_pixel": "nearest", "scl": "nearest", "*": resampling},
    }

    # List to hold outputs for each sensor (Landsat, Sentinel-2)
    output_list = []
    if dc is None: dc = Datacube()

    if load_ls:
        # Load Landsat
        datasets = dc.find_datasets(
            product=["ls8_c2l2_sr", "ls9_c2l2_sr"],
            cloud_cover=(0, max_cloud_cover),
            landsat_collection_category=["T1"],
            **query_params
        )
        print(f"Found {len(datasets)} Landsat datasets")

        if crs is None:
            crs = mostcommon_crs(datasets)

        if index == "mndwi":
            band2 = "swir16"
        elif index == "ndwi":
            band2 = "nir08"
        else:
            err_msg = "index can only be mndwi or ndwi"
            raise Exception(err_msg)

        ds_ls = dc.load(
            datasets=datasets,
            measurements=["green", band2, "qa_pixel"],
            output_crs=crs,
            patch_url=http_to_s3_url,
            driver="rio",
            skip_broken_datasets=True,
            **query_params,
            **load_params,
        )

        # Apply simple Landsat cloud mask
        cloud_mask = (
            # Bit 3: high confidence cloud, bit 4: high confidence shadow
            # https://medium.com/analytics-vidhya/python-for-geosciences-
            # raster-bit-masks-explained-step-by-step-8620ed27141e
            np.bitwise_and(ds_ls.qa_pixel, 1 << 3) | np.bitwise_and(
                ds_ls.qa_pixel, 1 << 4)
        ) == 0
        ds_ls = ds_ls.where(cloud_mask).drop_vars("qa_pixel")

        # Rescale to between 0.0 and 1.0
        ds_ls = (ds_ls.where(ds_ls != 0) * 0.0000275 + -0.2).clip(0, 1)

        # Convert to NDWI
        ndwi_ls = (ds_ls.green - ds_ls[band2]) / (ds_ls.green + ds_ls[band2])
        output_list.append(ndwi_ls)

    if load_s2:
        # Load Sentinel-2
        datasets = dc.find_datasets(
            product = ["s2_l2a"],
            cloud_cover=(0, max_cloud_cover),
            **query_params,
        )
        print(f"Found {len(datasets)} Sentinel-2 datasets")

        if crs is None:
            crs = mostcommon_crs(datasets)

        if index == "mndwi":
            band2 = "swir16"
        elif index == "ndwi":
            band2 = "nir"
        else:
            err_msg = "index can only be mndwi or ndwi"
            raise Exception(err_msg)


        ds_s2 = dc.load(
            datasets=datasets,
            measurements = ["green", band2, "scl"],
            output_crs=crs,
            driver="rio",
            **query_params,
            **load_params,
        )

        # Apply simple Sentinel-2 cloud mask
        # 1: defective, 3: shadow, 8:medium probability cloud, 9: high confidence cloud
        cloud_mask = ~ds_s2.scl.isin([1, 3, 8, 9])
        ds_s2 = ds_s2.where(cloud_mask).drop_vars("scl")

        # Rescale to between 0.0 and 1.0
        ds_s2 = (ds_s2.where(ds_s2 != 0) * 0.0001).clip(0, 1)

        # Convert to NDWI
        ndwi_s2 = (ds_s2.green - ds_s2[band2]) / (ds_s2.green + ds_s2[band2])
        output_list.append(ndwi_s2)

    # Merge into a single dataset
    ndwi = xr.concat(output_list, dim="time").sortby("time").to_dataset(name=index)

    # Optionally mask areas outside of supplied geopolygon (this has to be
    # applied here because applying it at the `stac_load` level converts
    # cloud masking bands to "float32".
    if mask_geopolygon & (geopolygon is not None):
        geopolygon = _normalize_geometry(geopolygon)
        ndwi = ndwi.odc.mask(poly=geopolygon)

    return ndwi
    

def corr_to_rankings(saved_corr):
    """
    Change the shape of the data to suit the tide model functions from eo-tides
    """
    
    df_reset = saved_corr.reset_index()

    # --- rank columns: rank_{model} ---
    rank_wide = (
        df_reset.pivot(index=["x", "y", "point_idx"], columns="tide_model", values="rank")
        .rename(columns=lambda m: f"rank_{m}")
    )

    # --- correlation columns: corr_{model} ---
    corr_wide = (
        df_reset.pivot(index=["x", "y", "point_idx"],
                       columns="tide_model", values="correlation")
        .rename(columns=lambda m: f"corr_{m}")
    )

    # --- top and worst model per x,y ---
    idx_top = df_reset.groupby(["x", "y", "point_idx"])["correlation"].idxmax()
    idx_worst = df_reset.groupby(["x", "y", "point_idx"])["correlation"].idxmin()

    top_model = (
        df_reset.loc[idx_top, ["x", "y", "point_idx", "tide_model",
                               "correlation", "valid_perc"]]
        .rename(columns={
            "tide_model": "top_model",     # best model
            "correlation": "top_correlation",     # value used for ranking
            "rank_valid_perc": "valid_perc"  # single valid_perc
        })
        .set_index(["x", "y", "point_idx"])
    )

    worst_model = (
        df_reset.loc[idx_worst, ["x", "y", "point_idx", "tide_model", "correlation"]]
        .rename(columns={
            "tide_model": "worst_model",
            "correlation": "worst_correlation"
        })
        .set_index(["x", "y", "point_idx"])
    )

    # --- combine everything ---
    out = pd.concat(
        [rank_wide, corr_wide, top_model, worst_model],
        axis=1
    ).reset_index()

    # --- add geometry ---
    model_rankings_gdf = gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["x"], out["y"]),
        crs="EPSG:4326",
    )

    return model_rankings_gdf

