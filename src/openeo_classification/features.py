from datetime import timedelta, date

from openeo.extra.spectral_indices.spectral_indices import compute_and_rescale_indices
from openeo.processes import array_concat, ProcessBuilder, array_create, if_, is_nodata

from openeo_classification.connection import connection
import scipy.signal
import numpy as np

temporal_partition_options = {
        "indexreduction": 0,
        "temporalresolution": "None",
        "tilesize": 256
}

creo_partition_options = {
        "indexreduction": 12,
        "temporalresolution": "ByDay",
        "tilesize": 256
}
job_options = {
        "driver-memory": "4G",
        "driver-memoryOverhead": "4G",
        "driver-cores": "2",
        "executor-memory": "2G",
        "executor-memoryOverhead": "2G",
        "executor-cores": "1",
        "max-executors": "15",
        "soft-errors": "true"
}

creo_job_options = {
        "driver-memory": "4G",
        "driver-memoryOverhead": "2G",
        "driver-cores": "1",
        "executor-memory": "2000m",
        "executor-memoryOverhead": "3500m",
        "executor-cores": "4",
        "executor-request-cores": "400m",
        "max-executors": "20"
    }

creo_job_options_production = {
        "driver-memory": "1G",
        "driver-memoryOverhead": "512m",
        "driver-cores": "1",
        "executor-memory": "2200m",
        "executor-memoryOverhead": "3000m",
        "executor-cores": "2",
        "executor-request-cores": "400m",
        "max-executors": "29"
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


def load_features(year, connection_provider = connection, provider = "Terrascope", processing_opts={}, sampling=False):
    start_date = date(int(year),3,15)
    end_date = date(int(year),10,31)
    stepsize_s2 = 10
    stepsize_s1 = 12
    # idx_dekad, idx_list, s2_list = sentinel2_features(start_date, end_date, connection_provider, provider, sampling=sampling, stepsize=stepsize_s2)

    s2_list = ["B06", "B12"]
    # s2_list = ["B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]


    idx_dekad = sentinel2_features(start_date, end_date, connection_provider, provider, s2_list=s2_list, processing_opts=processing_opts, sampling=sampling, stepsize=stepsize_s2)

    # dem = load_dem(idx_dekad, connection_provider)

    idx_features = compute_statistics(idx_dekad, start_date, end_date, stepsize=stepsize_s2).linear_scale_range(0,30000,0,30000)

    orbitDir = "ASCENDING"
    if provider.lower() == "creodias":
        orbitDir = orbitDir.lower()
    s1_dekad = sentinel1_features(start_date, end_date, connection_provider, provider, processing_opts=processing_opts, orbitDirection=orbitDir, sampling=sampling, stepsize=stepsize_s1)
    s1_dekad = s1_dekad.resample_cube_spatial(idx_dekad)

    s1_features = compute_statistics(s1_dekad, start_date, end_date, stepsize=stepsize_s1).linear_scale_range(0,30000,0,30000)

    # if(not sampling):
    #     #reduced input features for model
    #     idx_features = idx_features.filter_bands([ 'B06_p25', 'B06_p50', 'B06_p75', 'B06_sd', 'B06_t4', 'B06_t7', 'B06_t10', 'B06_t13', 'B06_t16', 'B06_t19', 
    #                     'B12_p25', 'B12_p50', 'B12_p75', 'B12_sd', 'B12_t4', 'B12_t7', 'B12_t10', 'B12_t13', 'B12_t16', 'B12_t19', 
    #                     'NDVI_p25', 'NDVI_p50', 'NDVI_p75', 'NDVI_sd', 
    #                     'NDGI_p25', 'NDGI_p50', 'NDGI_p75', 'NDGI_sd', 'NDGI_t4', 'NDGI_t7', 'NDGI_t10', 'NDGI_t13', 'NDGI_t16', 'NDGI_t19', 
    #                     'NDRE1_p25', 'NDRE1_p50', 'NDRE1_p75', 'NDRE1_sd', 
    #                     'NDRE2_p25', 'NDRE2_p50', 'NDRE2_p75', 'NDRE2_sd', 
    #                     'NDRE5_p25', 'NDRE5_p50', 'NDRE5_p75', 'NDRE5_sd', 
    #                     'ANIR_p25', 'ANIR_p50', 'ANIR_p75', 'ANIR_sd', 'ANIR_t4', 'ANIR_t7', 'ANIR_t10', 'ANIR_t13', 'ANIR_t16', 'ANIR_t19'])
    #     s1_features = s1_features.filter_bands([ 'ratio_p25', 'ratio_p50', 'ratio_p75', 'ratio_sd', 
    #                     'VV_p25', 'VV_p50', 'VV_p75', 'VV_sd', 'VV_t2', 'VV_t5', 'VV_t8', 'VV_t11', 'VV_t14','VV_t17', 
    #                     'VH_p25', 'VH_p50', 'VH_p75', 'VH_sd', 'VH_t2', 'VH_t5', 'VH_t8', 'VH_t11', 'VH_t14', 'VH_t17'])
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

def sentinel2_features(start_date, end_date, connection_provider, provider, index_dict=None, s2_list=[], processing_opts={}, sampling=False, stepsize=10, overlap=10, reducer="median", luc=False,cloud_procedure="sen2cor"):
    if index_dict == None:
        idx_list = ["NDVI", "NDMI", "NDGI", "NDRE1", "NDRE2", "NDRE5"]# if sampling else ["NDVI", "NDGI", "NDRE1", "NDRE2", "NDRE5"]
        index_dict = {
            "collection": {
                "input_range": [0, 8000],
                "output_range": [0, 30000]
            },
            "indices": {
                index: {"input_range": [-1, 1], "output_range": [0, 30000]} for index in idx_list
            }
        }
        index_dict["indices"]["ANIR"] = {"input_range": [0,1], "output_range": [0,30000]}
        print(index_dict)

    temp_ext_s2 = [start_date.isoformat(), end_date.isoformat()]
#     props = {
#         "eo:cloud_cover": lambda v: v == 80
#     }
    props = {
        "eo:cloud_cover": lambda v: v <= 80
    }
    s2_id = "SENTINEL2_L2A"
    if not luc:
        if (provider.upper() == "TERRASCOPE"):
            s2_id = "TERRASCOPE_S2_TOC_V2"
            props = {
                "eo:cloud_cover": lambda v: v == 80 ## moet sowieso <= zijn?
            }
        elif (provider.upper() == "CREODIAS"):
            s2_id = "SENTINEL2_L2A"
            props = {
               # "provider:backend": lambda v: v == "creodias",
                "eo:cloud_cover": lambda v: v == 80
            }
    bands = ["B03","B04","B05","B06","B07","B08","B11","B12","SCL"]

    c = connection_provider()
    s2 = c.load_collection(s2_id,
                           temporal_extent=temp_ext_s2,
                           # bands=["B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "SCL"],
                           bands=bands,
                           properties=props)

    if(provider.lower()=="creodias"):
        s2._pg.arguments['featureflags'] = creo_partition_options
    else:
        s2._pg.arguments['featureflags'] = processing_opts

    s2._pg.arguments['featureflags']['tilesize'] = processing_opts.get("tilesize",256)
    s2._pg.arguments['featureflags']['experimental'] = True

    if not sampling:
        s2 = cropland_mask(s2, c, provider)
    if cloud_procedure=="scl":
        g = scipy.signal.windows.gaussian(11, std=1.6)
        kernel = np.outer(g, g)
        kernel = kernel / kernel.sum()
        classification = s2.band("SCL")
        mask = ~ ((classification == 4) | (classification == 5)) #only select the vegetation and bare soil classes
        mask = mask.apply_kernel(kernel)
        mask = mask > 0.1
        s2 = s2.mask(mask)
    else:
        s2 = s2.process("mask_scl_dilation", data=s2, scl_band_name="SCL", kernel2_size=101).filter_bands(s2.metadata.band_names[:-1])

    print(list(index_dict["indices"].keys()))
    indices = compute_and_rescale_indices(s2, index_dict, True).filter_bands(s2_list + list(index_dict["indices"].keys()))

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


def sentinel1_features(start_date, end_date, connection_provider = connection, provider = "Terrascope", processing_opts={}, relativeOrbit=None, orbitDirection = None, sampling=False, stepsize=12, overlap=6, reducer="mean"):
    """
    Retrieves and preprocesses Sentinel-1 data into a cube with 10-daily periods (dekads).

    @param year:
    @param connection_provider:
    @param provider:
    @return:
    """

    s1 = sentinel1_inputs(start_date, end_date, connection_provider, provider, processing_opts, orbitDirection, relativeOrbit,sampling)
    # s1_dekad = s1.aggregate_temporal_period(period="dekad", reducer="median")
    s1_dekad = s1.aggregate_temporal(
        intervals=_calculate_intervals(start_date, end_date, stepsize = stepsize, overlap=overlap),
        reducer=reducer
    )

    s1_dekad = s1_dekad.apply_dimension(dimension="t", process="array_interpolate_linear")
    return s1_dekad

def cropland_mask(cube_to_mask, connection, provider="terrascope"):
    if (provider.lower() != "creodias"):
        wc = connection.load_collection("ESA_WORLDCOVER_10M_2020_V1", bands=["MAP"],
                                        temporal_extent=["2020-12-30", "2021-01-01"])
        worldcover_band = wc.band("MAP")
        mask = ( (worldcover_band != 30) & (worldcover_band != 40)).min_time()
        if(cube_to_mask is not None):
            return cube_to_mask.mask(mask.resample_cube_spatial(cube_to_mask))
        else:
            return mask
    else:
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
        tsteps[1] = if_(is_nodata(tsteps[1]), tsteps[2], tsteps[1])
        tsteps[4] = if_(is_nodata(tsteps[4]), tsteps[3], tsteps[4])
        tsteps[0] = if_(is_nodata(tsteps[0]), tsteps[1], tsteps[0])
        tsteps[5] = if_(is_nodata(tsteps[5]), tsteps[4], tsteps[5])
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


def sentinel1_inputs(start_date, end_date, connection_provider, provider= "Terrascope", processing_opts:dict={}, orbitDirection=None, relativeOrbit=None,sampling=False):
    c = connection_provider()
    temp_ext_s1 = [start_date.isoformat(), end_date.isoformat()]
    if (provider.upper() == "TERRASCOPE"):
        s1_id = "S1_GRD_SIGMA0_ASCENDING"
    else:
        s1_id = "SENTINEL1_GRD"
    properties = {
        #    "provider:backend": lambda v: v == "creo",
    }
    # if relativeOrbit is not None: ### DEZE ZORGT VOOR VERSCHILLEN !
    #     properties["relativeOrbitNumber"] = lambda p: p == relativeOrbit
    if orbitDirection is not None:
         properties["orbitDirection"] = lambda p: p == orbitDirection

    if provider.upper()=="SENTINELHUB":
         properties["polarization"] = lambda p: p == "DV"

    if provider.upper()=="CREODIAS":
        #in 2021 the interpretation of timeliness changed
        #https://sentinels.copernicus.eu/web/sentinel/-/copernicus-sentinel-1-nrt-3h-and-fast24h-products
         properties["timeliness"] = lambda p: p == "NRT-3h|Fast-24h"

    s1 = c.load_collection(s1_id,
                           temporal_extent=temp_ext_s1,
                           bands=["VH", "VV"],
                           properties=properties
                           )

    if (provider.lower() == "creodias"):
        s1._pg.arguments['featureflags'] = creo_partition_options
    else:
        if sampling:
            s1._pg.arguments['featureflags'] = temporal_partition_options
        else:
            s1._pg.arguments['featureflags'] = processing_opts

    s1._pg.arguments['featureflags']['experimental'] = False


    if (provider.upper() != "TERRASCOPE"):
        impl_version = "1" if sampling else "2"
        s1 = s1.sar_backscatter(coefficient="sigma0-ellipsoid",options={"implementation_version":impl_version,"tile_size":processing_opts.get("tilesize",256), "otb_memory":128,"debug":True})

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
