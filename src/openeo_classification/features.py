import openeo
from openeo.processes import array_modify, array_concat, ProcessBuilder
from openeo.extra.spectral_indices import compute_indices, compute_and_rescale_indices
from .connection import connection

temporal_partition_options = {
        "indexreduction": 0,
        "temporalresolution": "None",
        "tilesize": 256
}
job_options = {
        "driver-memory": "2G",
        "driver-memoryOverhead": "1G",
        "driver-cores": "2",
        "executor-memory": "2G",
        "executor-memoryOverhead": "2G",
        "executor-cores": "1",
        "max-executors": "100"
}

def load_features(year, connection_provider = connection, provider = "Terrascope"):
    temp_ext_s2 = [str(year - 1) + "-09-01", str(year + 1) + "-04-30"]
    temp_ext_s1 = [str(year) + "-01-01", str(year) + "-12-31"]

    c = connection_provider()
    s2 = c.load_collection("SENTINEL2_L2A",
                                    temporal_extent=temp_ext_s2,
                                    bands=["B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "SCL"],
                                    properties= {"provider:backend": lambda v: v == "creo"})
    s2._pg.arguments['featureflags'] = temporal_partition_options
    s2 = s2.process("mask_scl_dilation", data=s2, scl_band_name="SCL").filter_bands(s2.metadata.band_names[:-1])

    if(provider.upper() == "TERRASCOPE"):
        s1_id = "S1_GRD_SIGMA0_ASCENDING"
    else:
        s1_id = "SENTINEL1_GRD"
    s1 = c.load_collection(s1_id,
                            temporal_extent=temp_ext_s1,
                            bands=["VH", "VV"]
                            )
    s1._pg.arguments['featureflags'] = temporal_partition_options

    if (provider.upper() != "TERRASCOPE"):
        s1 = s1.sar_backscatter(coefficient="sigma0-ellipsoid")

    s1 = s1.apply_dimension(dimension="bands",
                            process=lambda x: array_modify(data=x, values=x.array_element(0) / x.array_element(1),
                                                           index=0))
    s1 = s1.linear_scale_range(0, 1, 0, 250)#apply_dimension(dimension="bands", process=lambda x: lin_scale_range(x, 0, 1, 0, 250))

    idx_list = ["NDVI", "NDMI", "NDGI", "NDRE1", "NDRE2", "NDRE5"]#, "ANIR"
    s2_list = ["B06", "B12"]

    s1_dekad = s1.aggregate_temporal_period(period="dekad", reducer="mean")
    s1_dekad = s1_dekad.apply_dimension(dimension="t", process="array_interpolate_linear")
    s1_dekad = s1_dekad.resample_cube_spatial(s2)

    index_dict = {
        "collection": {
            "input_range": None,
            "output_range": None
        },
        "indices": {
            index: {"input_range": [0,1], "output_range": [0,250]} for index in idx_list
        }
    }

    indices = compute_and_rescale_indices(s2, index_dict, True).filter_bands(s2_list + idx_list)
    idx_dekad = indices.aggregate_temporal_period(period="dekad", reducer="mean")
    idx_dekad = idx_dekad.apply_dimension(dimension="t", process="array_interpolate_linear").filter_temporal(
        [str(year) + "-01-01", str(year) + "-12-31"])

    base_features = idx_dekad.merge_cubes(s1_dekad)
    base_features = base_features.rename_labels("bands", s2_list + idx_list + ["ratio", "VV", "VH"])

    def computeStats(input_timeseries: ProcessBuilder):
        tsteps = list([input_timeseries.array_element(6 * index) for index in range(0, 6)])
        return array_concat(
            array_concat(input_timeseries.quantiles(probabilities=[0.1, 0.5, 0.9]), input_timeseries.sd()), tsteps)

    features = base_features.apply_dimension(dimension='t', target_dimension='bands', process=computeStats).apply(
        lambda x: x.linear_scale_range(0, 250, 0, 250))

    tstep_labels = ["t" + str(6 * index) for index in range(0, 6)]
    all_bands = [band + "_" + stat for band in base_features.metadata.band_names for stat in
                 ["p10", "p50", "p90", "sd"] + tstep_labels]
    features = features.rename_labels('bands', all_bands)
    return features