import openeo
from openeo.processes import array_modify, array_concat, ProcessBuilder,array_create
from openeo.extra.spectral_indices import compute_indices, compute_and_rescale_indices, append_index
from openeo_classification.connection import connection
from datetime import timedelta, date

temporal_partition_options = {
        "indexreduction": 0,
        "temporalresolution": "None",
        "tilesize": 256
}

creo_partition_options = {
        "indexreduction": 7,
        "temporalresolution": "ByDay",
        "tilesize": 256
}
job_options = {
        "driver-memory": "2G",
        "driver-memoryOverhead": "4G",
        "driver-cores": "2",
        "executor-memory": "2G",
        "executor-memoryOverhead": "2G",
        "executor-cores": "1",
        "max-executors": "100"
}



def _calculate_intervals(start_date, end_date, stepsize = 10, overlap = 10):
    between_steps = timedelta(days=stepsize)
    within_steps = timedelta(days=stepsize+overlap)
    date1 = start_date
    date2 = start_date + within_steps
    tot_intervals = []
    while date2 < end_date:
        tot_intervals.append([date1.isoformat(),date2.isoformat()])
        date1 += between_steps
        date2 += between_steps
    return tot_intervals


def load_features(year, connection_provider = connection, provider = "Terrascope", sampling=False):
    start_date = date(year,3,15)
    end_date = date(year,10,31)
    stepsize_s2 = 10
    stepsize_s1 = 12
    # idx_dekad, idx_list, s2_list = sentinel2_features(start_date, end_date, connection_provider, provider, sampling=sampling, stepsize=stepsize_s2)
    idx_dekad = sentinel2_features(start_date, end_date, connection_provider, provider, sampling=sampling, stepsize=stepsize_s2)

    # dem = load_dem(idx_dekad, connection_provider)

    idx_features = compute_statistics(idx_dekad, start_date, end_date, stepsize=stepsize_s2).linear_scale_range(0,30000,0,30000)

    s1_dekad = sentinel1_features(start_date, end_date, connection_provider, provider, orbitDirection="ASCENDING", sampling=sampling, stepsize=stepsize_s1)
    s1_dekad = s1_dekad.resample_cube_spatial(idx_dekad)

    s1_features = compute_statistics(s1_dekad, start_date, end_date, stepsize=stepsize_s1).linear_scale_range(0,30000,0,30000)

    features = idx_features.merge_cubes(s1_features)

    # base_features = idx_dekad.merge_cubes(s1_dekad)
    # base_features = base_features.rename_labels("bands", s2_list + idx_list + ["ratio", "VV", "VH"])
    # features = compute_statistics(base_features, stepsize=).linear_scale_range(0,30000,0,30000)
    # return s2, idx_dekad, s1, s1_dekad, features
    return features

# def load_dem(cube, connection_provider):
#     c = connection_provider()
#     dem = c.load_collection("COPERNICUS_30",
#                            temporal_extent=temp_ext_s2)
#     return dem.max_time().resample_cube_spatial(cube)

def sentinel2_features(start_date, end_date, connection_provider, provider, sampling, stepsize=10, overlap=10, reducer="median"):
    temp_ext_s2 = [start_date.isoformat(), end_date.isoformat()]
    props = {}
    s2_id = "SENTINEL2_L2A"
    if (provider.upper() == "TERRASCOPE"):
        s2_id = "TERRASCOPE_S2_TOC_V2"
        props = {
            "eo:cloud_cover": lambda v: v == 80
        }
    elif (provider.upper() == "CREODIAS"):
        props = {
            "provider:backend": lambda v: v == "creodias",
            "eo:cloud_cover": lambda v: v == 80,
        }
    c = connection_provider()
    s2 = c.load_collection(s2_id,
                           temporal_extent=temp_ext_s2,
                           bands=["B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "SCL"],
                           properties=props)

    if(provider.lower()=="creodias"):
        s2._pg.arguments['featureflags'] = creo_partition_options

    if not sampling:
        s2 = cropland_mask(s2, c, provider)

    s2 = s2.process("mask_scl_dilation", data=s2, scl_band_name="SCL").filter_bands(s2.metadata.band_names[:-1])

    idx_list = ["NDVI", "NDMI", "NDGI", "NDRE1", "NDRE2", "NDRE5", "ANIR"]
    s2_list = ["B06", "B12"]
    # s2_list = ["B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]
    index_dict = {
        "collection": {
            "input_range": [0, 8000],
            "output_range": [0, 30000]
        },
        "indices": {
            index: {"input_range": [-1, 1], "output_range": [0, 30000]} for index in idx_list[:-1]
        }
    }
    index_dict["indices"]["ANIR"] = {"input_range": [0,1], "output_range": [0,30000]}
    indices = compute_and_rescale_indices(s2, index_dict, True).filter_bands(s2_list + idx_list)

    # months = ["03","04","05","06","07","08","09","10","11"]
    # days = ["01","11","21"]
    # idx_dekad = indices.aggregate_temporal(intervals=[[str(year)+"-"+months[i//len(days)]+"-"+days[i%len(days)], str(year)+"-"+months[(i+2)//len(days)]+"-"+days[(i+2)%len(days)]] for i in range(len(days)*len(months))[:-2]][:-1], reducer="median")
    idx_dekad = indices.aggregate_temporal(
        intervals=_calculate_intervals(start_date, end_date, stepsize = stepsize, overlap=overlap),
        reducer=reducer
    )

    # idx_dekad = indices.aggregate_temporal_period(period="dekad", reducer="median")
    idx_dekad = idx_dekad.apply_dimension(dimension="t", process="array_interpolate_linear")#.filter_temporal([str(year) + "-03-01", str(year) + "-11-01"])
    # return idx_dekad, idx_list, s2_list
    return idx_dekad


def sentinel1_features(start_date, end_date, connection_provider = connection, provider = "Terrascope", relativeOrbit=None, orbitDirection = None, sampling=False, stepsize=12, overlap=6, reducer="mean"):
    """
    Retrieves and preprocesses Sentinel-1 data into a cube with 10-daily periods (dekads).

    @param year:
    @param connection_provider:
    @param provider:
    @return:
    """

    s1 = sentinel1_inputs(start_date, end_date, connection_provider, provider, orbitDirection, relativeOrbit,sampling)
    # s1_dekad = s1.aggregate_temporal_period(period="dekad", reducer="median")
    s1_dekad = s1.aggregate_temporal(
        intervals=_calculate_intervals(start_date, end_date, stepsize = stepsize, overlap=overlap),
        reducer=reducer
    )

    s1_dekad = s1_dekad.apply_dimension(dimension="t", process="array_interpolate_linear")
    return s1_dekad

def cropland_mask(cube_to_mask, connection, provider):
    if (provider.lower() == "terrascope"):
        wc = connection.load_collection("ESA_WORLDCOVER_10M_2020_V1", bands=["MAP"],
                                        temporal_extent=["2020-12-30", "2021-01-01"])
        cube_to_mask = cube_to_mask.mask((wc.band("MAP") != 40).min_time().resample_cube_spatial(cube_to_mask))
    return cube_to_mask


def compute_statistics(base_features, start_date, end_date, stepsize):
    """
    Computes statistics over a datacube.
    For correct statistics, the datacube needs to be preprocessed to contain observation at equitemporal intervals, without nodata values.

    @param base_features:
    @return:
    """
    def computeStats(input_timeseries: ProcessBuilder, sample_stepsize, offset):
        tsteps = list([input_timeseries.array_element(offset + sample_stepsize * index) for index in range(0, 6)])
        return array_concat(
            array_concat(input_timeseries.quantiles(probabilities=[0.25, 0.5, 0.75]), input_timeseries.sd()), tsteps)

    tot_samples = (end_date - start_date).days // stepsize
    nr_tsteps = 6
    sample_stepsize = tot_samples // nr_tsteps
    offset = int(sample_stepsize/2 + (tot_samples%nr_tsteps)/2)

    features = base_features.apply_dimension(dimension='t', target_dimension='bands', process=lambda x: computeStats(x, sample_stepsize, offset))#.apply(lambda x: x.linear_scale_range(-500, 500, -50000, 50000))
    tstep_labels = ["t" + str(offset + sample_stepsize * index) for index in range(0, 6)]
    all_bands = [band + "_" + stat for band in base_features.metadata.band_names for stat in
                 ["p25", "p50", "p75", "sd"] + tstep_labels]
    features = features.rename_labels('bands', all_bands)
    return features


def sentinel1_inputs(start_date, end_date, connection_provider, provider= "Terrascope", orbitDirection=None, relativeOrbit=None,sampling=False):
    c = connection_provider()
    temp_ext_s1 = [start_date.isoformat(), end_date.isoformat()]
    if (provider.upper() == "TERRASCOPE"):
        s1_id = "S1_GRD_SIGMA0_ASCENDING"
    else:
        s1_id = "SENTINEL1_GRD"
    properties = {
        #    "provider:backend": lambda v: v == "creo",
    }
    if relativeOrbit is not None:
        properties["relativeOrbitNumber"] = lambda p: p == relativeOrbit
    if orbitDirection is not None:
        properties["orbitDirection"] = lambda p: p == orbitDirection

    if provider.upper()=="SENTINELHUB":
        properties["polarization"] = lambda p: p == "DV"

    s1 = c.load_collection(s1_id,
                           temporal_extent=temp_ext_s1,
                           bands=["VH", "VV"],
                           properties=properties
                           )
    # s1._pg.arguments['featureflags'] = temporal_partition_options
    if (provider.upper() != "TERRASCOPE"):
        s1 = s1.sar_backscatter(coefficient="sigma0-ellipsoid",options={"implementation_version":"2","tile_size":256, "otb_memory":256})

    if not sampling:
        s1 = cropland_mask(s1, c, provider)
    # Observed Ranges:
    # VV: 0 - 0.3 - Db: -20 .. 0
    # VH: 0 - 0.3 - Db: -30 .. -5
    # Ratio: 0- 1
    #S1_GRD = S1_GRD.apply(lambda x: 10 * x.log(base=10))
    s1 = s1.apply_dimension(dimension="bands",
                            process=lambda x:array_create([30.0 * x[0] / x[1],30.0+10.0 * x[0].log(base=10),30.0+10.0*x[1].log(base=10)]))
    s1 = s1.rename_labels("bands", ["ratio"] + s1.metadata.band_names)
    # scale to int16
    s1 = s1.linear_scale_range(0, 30, 0,30000)
    return s1

