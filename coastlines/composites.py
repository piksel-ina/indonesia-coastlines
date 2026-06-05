import os
import sys

import click
import numpy as np
import xarray as xr

from odc.geo.geobox import GeoBox

import datacube
from datacube.utils.dask import start_local_dask

from eo_tides.eo import pixel_tides
from odc.algo import (
    int_geomedian,
    enum_to_bool,
    mask_cleanup,
    to_f32,
    keep_good_only,
)

from coastlines.utils import (
    #    CoastlinesException,
    #    click_config_path,
    #    click_output_location,
    #    click_output_version,
    #    click_overwrite,
    #    click_study_area,
    configure_logging,
    get_study_site_geometry,
    #    is_s3,
    #    load_config,
    #    parallel_apply,
    #    tide_cutoffs,
)


def rename_bands(ds, old_string, new_string):
    # Create a new dataset with renamed bands
    ds_renamed = ds.rename(
        {band: band.replace(old_string, new_string) for band in ds.data_vars}
    )
    return ds_renamed


def rename_add_prefix(ds, prefix, exclude_prefix="qa"):
    # Create a new dataset with renamed bands
    ds_renamed = ds.rename(
        {band: f"{prefix}_{band}" for band in ds.data_vars if not band.startswith(
            exclude_prefix)}
    )
    return ds_renamed


def load_s2(
    dc,
    geobox,
    time_range=("2019", "2021"),
    resolution=10,
    crs="EPSG:6933",
    include_coastal_aerosol=False,
    max_cloudcover=80,
    cloud_filters={"cloud medium probability": [["opening", 2], ["dilation", 5]],
                   "cloud high probability": [["opening", 2], ["dilation", 5]],
                   "thin cirrus": [["dilation", 5]]},
    dask_chunks=None,
    dtype="float32",
    log=None,
    run_id="",
    **query,
):
    """Loads cloud-masked Sentinel-2 satellite data for a given
    study area and time range.

    Parameters
    ----------
    dc : datacube.Datacube()
        A datacube instance to load data from.
    geom : Geometry
        A datacube Geometry object defining a custom spatial extent of
        interest.
    time_range : tuple, optional
        A tuple containing the start and end date for the time range of
        interest, in the format (start_date, end_date). The default is
        ("2019", "2021").
    resolution : int or float, optional
        The spatial resolution (in metres) to load data at. The default
        is 10.
    crs : str, optional
        The coordinate reference system (CRS) to project data into. The
        default is "EPSG:6933".
    max_cloudcover : float, optional
        The maximum cloud cover metadata value used to load data.
        Defaults to 80 (i.e. 80% cloud cover).
    include_coastal_aerosol : bool, optional
        Whether to load data from the Sentinel-2 coastal aerosol band.
        Defaults to False.
    dask_chunks : dict, optional
        Optional custom Dask chunks to load data with. Defaults to None,
        which will use '{"x": 3200, "y": 3200}'.
    dtype : str, optional
        Desired data type for output data. Valid values are "int16"
        (default) and "float32".
    log : logging.Logger, optional
        Logger object, by default None.
    run_id : str, optional
        run id, by default ''. added log statements.
    **query :
        Optional datacube.load keyword argument parameters used to
        query data.

    Returns
    -------
    satellite_ds : xarray.Dataset
        An xarray dataset containing the loaded Sentinel-2
        data.
    """

    # Set spectral bands to load
    s2_bands = [
        "blue",
        "green",
        "red",
        "rededge1",
        "rededge2",
        "rededge3",
        "nir",
        "nir08",
        "swir16",
        "swir22",
        "scl",
    ]
    if include_coastal_aerosol:
        s2_bands = ["coastal"] + s2_bands

    # Set up load params
    load_params = {
        "like": geobox,
        "dask_chunks": {"x": 3200, "y": 3200} if dask_chunks is None else dask_chunks,
        "resampling": {
            "*": "bilinear",
            "scl": "nearest",
        },
        "group_by": "solar_day",
        "driver": "rio",
    }

    dss_s2 = dc.find_datasets(
        product=["s2_l2a"],
        like=geobox,
        time=time_range,
        cloud_cover=(0, max_cloudcover),
        **query,
    )

    if log:
        log.info(f"{run_id}: Found {len(dss_s2)} Sentinel-2 datasets")

    # Load datasets
    ds = dc.load(
        datasets=dss_s2,
        measurements=s2_bands,
        **load_params,
    )

    # Set cloud mask
    ds.scl.attrs['flags_definition'] = dc.list_measurements(
    ).loc[('s2_l2a', 'scl')].flags_definition
    # Erase Data Pixels for which mask == nodata
    mask = ds["scl"]
    bad = enum_to_bool(mask, ("no data",))
    for cloud_class, c_filter in cloud_filters.items():
        if not isinstance(cloud_class, tuple):
            cloud_class = (cloud_class,)
        cloud_mask = enum_to_bool(mask, cloud_class)
        cloud_mask_buffered = mask_cleanup(cloud_mask, mask_filters=c_filter)
        bad = cloud_mask_buffered | bad

    ds = ds.where(~bad).drop_vars("scl")

    # Optionally convert to float, setting all nodata pixels to `np.nan`
    # (required for NDWI, so will be applied even if `dtype="int16"`)
    if (dtype == "float32"):
        ds = to_f32(ds)

    return ds, dss_s2


def tidal_thresholds(
    tides_highres,
    threshold_lowtide=0.15,
    threshold_hightide=0.85,
    min_obs=0,
):
    # Calculate per-pixel integer rankings for each tide height
    rank_n = tides_highres.rank(dim="time")

    # Calculate pixel-based low and high ranking thresholds from
    # max ranking. Max ranking needs to be rounded up to the nearest
    # integer using "ceil" as xarray will give multiple observation
    # an average rank (e.g. 50.5) value if they are both identical.
    # Additionally: to ensure we capture all matching values, Low
    # threshold needs to be rounded up ("ceil"), and high tide
    # rounded down ("floor").
    rank_max = np.ceil(rank_n.max(dim="time"))
    rank_thresh_low = np.ceil(rank_max * threshold_lowtide)
    rank_thresh_high = np.floor(rank_max * threshold_hightide)

    # Update thresholds to ensure minimum number of valid observations
    if min_obs > 0:
        rank_thresh_low = np.maximum(rank_thresh_low, min_obs)
        rank_thresh_high = np.minimum(rank_thresh_high, rank_max - min_obs)

    # Calculate tide thresholds by masking tides by ranking threshold
    tide_thresh_low = tides_highres.where(
        rank_n <= rank_thresh_low).max(dim="time")
    tide_thresh_high = tides_highres.where(
        rank_n >= rank_thresh_high).min(dim="time")

    return tide_thresh_low, tide_thresh_high


def tidal_composites(
    satellite_ds,
    threshold_lowtide=0.15,
    threshold_hightide=0.85,
    min_obs=0,
    eps=1e-4,
    cpus=None,
    max_iters=10000,
    tide_model="EOT20",
    tide_model_dir="/var/share/tide_models",
    run_id=None,
    log=None,
):
    """Calculates Geometric Median composites of the coastal zone at low
    and high tide using satellite imagery and tidal modeling.

    This function uses tools from `odc.algo` to keep data in its
    original integer datatype throughout the analysis to minimise
    memory usage. Modelled tide data and nodata pixels are used
    to filter satellite data to low and high tide images prior to
    loading it into memory, allowing more efficient processing.

    Pixel-based implementation of the method originally published in:

    Sagar, S., Phillips, C., Bala, B., Roberts, D., & Lymburner, L.
    (2018). Generating Continental Scale Pixel-Based Surface Reflectance
    Composites in Coastal Regions with the Use of a Multi-Resolution
    Tidal Model. Remote Sensing, 10, 480. https://doi.org/10.3390/rs10030480

    Parameters
    ----------
    satellite_ds : xarray.Dataset
        A satellite data time series containing spectral bands.
    threshold_lowtide : float, optional
        Quantile used to identify low tide observations, by default 0.15.
    threshold_hightide : float, optional
        Quantile used to identify high tide observations, by default 0.85.
    min_obs : int, optional
        Minimum number of clear observations to enforce when calculating tide
        height thresholds. Defaults to 0, which will not apply any minimum.
    eps: float, optional
        Termination criteria passed on to the geomedian algorithm.
    cpus: int, optional
        Requested number of cpus which is passed on to the geomedian function.
    max_iters : int, optional
        Maximum number of iterations done per output pixel in the
        geomedian calculation. This can be set to a low value (e.g. 10)
        to increase the processing speed of test runs.
    tide_model : str, optional
        The tide model or a list of models used to model tides, as
        supported by the `eo-tides` Python package. Options include:
        - "EOT20" (default)
        - "TPXO10-atlas-v2-nc"
        - "FES2022"
        - "FES2022_extrapolated"
        - "FES2014"
        - "FES2014_extrapolated"
        - "GOT5.6"
        - "ensemble" (experimental: combine all above into single ensemble)
    tide_model_dir : str, optional
        The directory containing tide model data files. Defaults to
        "/var/share/tide_models"; for more information about the
        directory structure, refer to `eo-tides.utils.list_models`.
    run_id : string, optional
        An optional string giving the name of the analysis; used to
        prefix log entries.
    log : logging.Logger, optional
        Logger object, by default None.

    Returns
    -------
    ds_lowtide : xarray.Dataset
        xarray.Dataset object containing a geomedian of the observations
        with the lowest X quantile tide heights for each pixel.
    ds_hightide : xarray.Dataset
        xarray.Dataset object containing a geomedian of the observations
        with the highest X quantile tide values for each pixel.

    """
    # Set up logs if no log is passed in
    if log is None:
        log = configure_logging()

    # Use run ID name for logs if it exists
    run_id = "Processing" if run_id is None else run_id

    # Model tides into for spatial extent and timesteps in satellite data
    log.info(f"{run_id}: Modelling tide heights for each pixel")
    tides_highres = pixel_tides(
        data=satellite_ds,
        model=tide_model,
        resample=True,
        directory=tide_model_dir,
    )
    # Identify nodata pixels in satellite data array by loading only
    # a single band into memory
    log.info(f"{run_id}: Loading red band to identify nodata pixels")
    nodata = satellite_ds.red.nodata
    nodata_array = (satellite_ds.red != nodata).compute()

    # Calculate the total clear pixel count for each pixel
    qa_count_clear_total = nodata_array.sum(dim="time").astype("int16")

    # Mask tides to make nodata match satellite data array
    tides_highres = tides_highres.where(nodata_array)

    # Calculate low and high tide thresholds from masked tide data
    log.info(
        f"{run_id}: Calculating low and high tide thresholds with minimum {min_obs} observations"
    )
    low_threshold, high_threshold = tidal_thresholds(
        tides_highres=tides_highres,
        threshold_lowtide=threshold_lowtide,
        threshold_hightide=threshold_hightide,
        min_obs=min_obs,
    )

    # Create masks for selecting satellite observations below and above the
    # low and high tide thresholds
    low_mask = tides_highres <= low_threshold
    high_mask = tides_highres >= high_threshold

    # Keep only scenes with at least 1% valid data to speed up geomedian
    low_keep = low_mask.mean(dim=["x", "y"]) >= 0.01
    high_keep = high_mask.mean(dim=["x", "y"]) >= 0.01
    ds_low = satellite_ds.sel(time=low_keep)
    ds_high = satellite_ds.sel(time=high_keep)

    # Load low and high subsets of data into memory
    log.info(
        f"{run_id}: Loading {len(ds_low.time)} low tide satellite images into memory"
    )
    ds_low.load()
    log.info(
        f"{run_id}: Loading {len(ds_high.time)} high tide satellite images into memory"
    )
    ds_high.load()

    # Use `keep_good_only` to set any pixels outside of the tide masks to nodata
    ds_low_masked = keep_good_only(x=ds_low, where=low_mask.sel(time=low_keep))
    ds_high_masked = keep_good_only(
        x=ds_high, where=high_mask.sel(time=high_keep))

    # Calculate low and high tide geomedians
    num_threads = cpus if cpus is not None else os.cpu_count() - 2
    log.info(f"{run_id}: Running low tide geomedian with {num_threads} threads")
    ds_lowtide = int_geomedian(
        ds=ds_low_masked,
        maxiters=max_iters,
        num_threads=num_threads,
        eps=eps,
    )
    log.info(f"{run_id}: Running high tide geomedian with {num_threads} threads")
    ds_hightide = int_geomedian(
        ds=ds_high_masked,
        maxiters=max_iters,
        num_threads=num_threads,
        eps=eps,
    )

    # Calculate clear count (both low and high tide clear counts
    # are identical, so we can just use one)
    log.info(f"{run_id}: Calculating clear counts")
    ds_lowtide["qa_count_clear_low"] = (
        (ds_low_masked.red != nodata).sum(dim="time").astype("int16")
    )
    # low and high tide clear counts are similar but not identical?
    ds_hightide["qa_count_clear_high"] = (
        (ds_high_masked.red != nodata).sum(dim="time").astype("int16")
    )

    # Add the total count clear (Only add once)
    ds_lowtide["qa_count_clear_total"] = qa_count_clear_total

    # Add low and high tide thresholds to the output datasets
    ds_lowtide["qa_low_threshold"] = low_threshold
    ds_hightide["qa_high_threshold"] = high_threshold

    return ds_lowtide, ds_hightide

### TODO: Make export work ###

@click.command()
@click.option(
    "--study_area",
    type=str,
    required=True,
    help="A string providing tile index (e.g. in the form 'x,y') to run the analysis on.",
)
@click.option(
    "--grid_path",
    type=str,
    required=True,
    help="Path to grid definition file.",
)
@click.option(
    "--start_date",
    type=str,
    required=True,
    help="The start date of satellite data to load from the "
    "datacube. This can be any date format accepted by datacube. "
    "For DEA Tidal Composites, this is set to provide a three year window "
    "centred over `label_date` below.",
)
@click.option(
    "--end_date",
    type=str,
    required=True,
    help="The end date of satellite data to load from the "
    "datacube. This can be any date format accepted by datacube. "
    "For DEA Tidal Composites, this is set to provide a three year window "
    "centred over `label_date` below.",
)
@click.option(
    "--label_date",
    type=str,
    required=True,
    help="The date used to label output arrays, and to use as the date "
    "assigned to the dataset when indexed into Datacube.",
)
@click.option(
    "--output_version",
    type=str,
    required=True,
    help="The version number to use for output files and metadata (e.g. '0.0.1').",
)
@click.option(
    "--output_dir",
    type=str,
    default="data/processed/",
    help="The directory/location to output data and metadata; supports "
    "both local disk and S3 locations. Defaults to 'data/processed/'.",
)
@click.option(
    "--resolution",
    type=int,
    default=10,
    help="The spatial resolution in metres used to load satellite "
    "data and produce tidal composite outputs. Defaults to 10 metre "
    "Sentinel-2 resolution.",
)
@click.option(
    "--threshold_lowtide",
    type=float,
    default=0.15,
    help="The quantile used to identify low tide observations. Defaults to 0.15.",
)
@click.option(
    "--threshold_hightide",
    type=float,
    default=0.85,
    help="The quantile used to identify high tide observations. Defaults to 0.85.",
)
@click.option(
    "--min_obs",
    type=int,
    default=0,
    help="Minimum number of clear observations to enforce when calculating tide "
    "height thresholds. Defaults to 0, which will not apply any minimum.",
)
@click.option(
    "--include_coastal_aerosol/--no-include_coastal_aerosol",
    type=bool,
    default=True,
    help="Whether to include the coastal aerosol band. Defaults to True",
)
@click.option(
    "--eps",
    type=float,
    default=1e-4,
    help="Termination criteria passed on to the geomedian algorithm.",
)
@click.option(
    "--cpus",
    type=int,
    default=None,
    help="Requested number of CPUs which is passed on to the geomedian function.",
)
@click.option(
    "--max_iters",
    type=int,
    default=1000,
    help="Maximum number of iterations done per output pixel in the "
    "geomedian calculation. This can be set to a low value (e.g. 10) "
    "to increase the processing speed of test runs.",
)
@click.option(
    "--tide_model",
    type=str,
    multiple=True,
    default=["FES2022"],
    help="The model used for tide modelling, as supported by the "
    "`eo-tides` Python package. Options include 'EOT20' (default), "
    "'TPXO10-atlas-v2-nc', 'FES2022', 'FES2014', 'GOT5.6', 'ensemble'.",
)
@click.option(
    "--tide_model_dir",
    type=str,
    default="data/coastlines/tide_models",
    help="The directory containing tide model data files. Defaults to "
    "'data/coastlines/tide_models'; for more information about the required "
    "directory structure, refer to `eo-tides.utils.list_models`.",
)
@click.option(
    "--overwrite/--no-overwrite",
    type=bool,
    default=True,
    help="Whether to overwrite tile data if it already exists.",
)
def tidal_composites_cli(
    study_area,
    grid_path,
    start_date,
    end_date,
    label_date,
    output_version,
    output_dir,
    resolution,
    threshold_lowtide,
    threshold_hightide,
    min_obs,
    include_coastal_aerosol,
    eps,
    cpus,
    max_iters,
    tide_model,
    tide_model_dir,
    overwrite,
):
    # Create sample filename to test if data exists on file system
    filename = f"{output_dir}s2_tidal_composites/{output_version.replace('.', '-')}/{study_area[:4]}/{study_area[4:]}/{label_date}--P1Y/s2_tidal_composites_{study_area}_{label_date}--P1Y_final.stac-item.json"

    process_tile = True
    if overwrite:
        process_tile = True
    elif os.path.exists(filename):
        process_tile = False

    # Create a unique run ID based on input params and use for logs
    input_params = locals()
    run_id = f"[{output_version}] [{label_date}] [{study_area}]"
    log = configure_logging(run_id)

    # Record params in logs
    log.info(f"{run_id}: Using parameters {input_params}")

    if process_tile:
        try:
            # Create local dask cluster to improve data load time
            _ = start_local_dask(
                n_workers=4, threads_per_worker=8, mem_safety_margin="2G")

            # Connect to datacube to load data
            dc = datacube.Datacube(app="Composites_CLI")

            output_crs = "epsg:6933"
            geometry = get_study_site_geometry(grid_path, study_area)
            # turn into odc.geo.geobox.GeoBox
            geobox = GeoBox.from_bbox(geometry.to_crs(
                output_crs).bounds.values[0], crs=output_crs, resolution=resolution)

            # Load satellite data and dataset IDs for metadata
            # Use `filter_granules` predicate function to drop list of
            # custom Sentinel-2 MGRS granules with poor data coverage
            satellite_ds, dss_s2 = load_s2(
                dc=dc,
                geobox=geobox,
                time_range=(start_date, end_date),
                resolution=resolution,
                crs=output_crs,
                include_coastal_aerosol=include_coastal_aerosol,
                max_cloudcover=60,
                dtype="int16",
                log=log,
                run_id=run_id,
            )
            log.info(
                f"{run_id}: Found {len(satellite_ds.time)} satellite data timesteps"
            )

            # Fail early if not enough observations
            if len(satellite_ds.time) < 50:
                raise Exception(
                    "Insufficient satellite data available to process composites; skipping."
                )

            # Calculate high and low tide geomedian composites
            log.info(f"{run_id}: Running Tidal Composites workflow")
            ds_lowtide, ds_hightide = tidal_composites(
                satellite_ds=satellite_ds,
                threshold_lowtide=threshold_lowtide,
                threshold_hightide=threshold_hightide,
                min_obs=min_obs,
                eps=eps,
                cpus=cpus,
                max_iters=max_iters,
                tide_model=tide_model,
                tide_model_dir=tide_model_dir,
                run_id=run_id,
                log=log,
            )

            # Rename low and high tide bands to add "low"/"high"
            ds_hightide = rename_add_prefix(ds_hightide, "high")
            ds_lowtide = rename_add_prefix(ds_lowtide, "low")

            # Concatenate into a single output dataset
            ds_tidalcomposites = xr.merge(
                [ds_lowtide, ds_hightide], compat='no_conflicts')

            # Ensure spatial information is still attached
            # ds_tidalcomposites = odc.geo.xr.assign_crs(
            #    ds_tidalcomposites, satellite_ds.odc.crs
            # )

            custom_dtypes = {
                "low_coastal": (np.int16, -999),
                "low_blue": (np.int16, -999),
                "low_green": (np.int16, -999),
                "low_red": (np.int16, -999),
                "low_rededge1": (np.int16, -999),
                "low_rededge2": (np.int16, -999),
                "low_rededge3": (np.int16, -999),
                "low_nir": (np.int16, -999),
                "low_nir08": (np.int16, -999),
                "low_swir16": (np.int16, -999),
                "high_coastal": (np.int16, -999),
                "high_blue": (np.int16, -999),
                "high_green": (np.int16, -999),
                "high_red": (np.int16, -999),
                "high_rededge1": (np.int16, -999),
                "high_rededge2": (np.int16, -999),
                "high_rededge3": (np.int16, -999),
                "high_nir": (np.int16, -999),
                "high_nir08": (np.int16, -999),
                "high_swir16": (np.int16, -999),
                "high_swir22": (np.int16, -999),
                "qa_low_threshold": (np.float32, np.nan),
                "qa_high_threshold": (np.float32, np.nan),
                "qa_count_clear": (np.int16, -999),
                "qa_count_clear_total": (np.int16, -999),
            }

            # Sets correct dtypes and nodata
            ds_prepared = prepare_for_export(
                ds_tidalcomposites,
                custom_dtypes=custom_dtypes,
                log=log,
            )

            # Calculate additional tile-level tidal metadata and graph.
            metadata_dict, tide_graph_fig = tidal_metadata(
                product_family="tidal_composites",
                threshold_lowtide=threshold_lowtide,
                threshold_hightide=threshold_hightide,
                data=satellite_ds,
                modelled_freq="30min",
                model=tide_model,
                directory=tide_model_dir,
            )

            # Export data and metadata
            export_dataset_metadata(
                ds_prepared,
                year=label_date,
                study_area=study_area,
                output_location=output_dir,
                s2_lineage=dss_s2,
                dataset_version=output_version,
                product_family="tidal_composites",
                odc_product="s2_tidal_composites",
                thumbnail_bands=["low_red", "low_green", "low_blue"],
                tide_graph_fig=tide_graph_fig,
                additional_metadata=metadata_dict,
                run_id=run_id,
                log=log,
            )

            # Close dask client
            client.close()
            log.info(f"{run_id}: Completed Tidal Composites workflow")

        except Exception as e:
            log.exception(f"{run_id}: Failed to run process with error {e}")
            sys.exit(1)
    else:
        log.info(f"{run_id}: Skipping as overwrite==False")


if __name__ == "__main__":
    tidal_composites_cli()
