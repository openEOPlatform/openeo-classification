from pathlib import Path

import geopandas as gpd

from shapely.geometry import LineString
import pandas as pd
import numpy as np
import h3.api.basic_int as h3

def split_lucas(working_dir=Path("."), filter_LC=[]):
    df = gpd.read_file("https://artifactory.vgt.vito.be/auxdata-public/openeo/2018_EU_LUCAS_POINT_110.gpkg")
    #croptype filter
    df = df[df.LC.isin(filter_LC)]

    cellsh3 = df.geometry.centroid.apply(lambda x: h3.geo_to_h3(x.y, x.x, 10))
    df['s2cell'] = cellsh3
    df.index = cellsh3
    df.sort_index(inplace=True)
    df = df[['POINT_ID',  's2cell',  'LC1_LABEL', 'LC', 'geometry']]

    def dist(x):
        return h3.point_dist(h3.h3_to_geo(x.astype(np.int64)[-1] - 1), h3.h3_to_geo(x.astype(np.int64)[0] - 1))

    distance = df.s2cell.rolling(window=2).apply(lambda x: dist(x), raw=True)
    breakpoints = df.s2cell[distance > 200]

    # out,bins = pd.qcut(df.s2cell,q=int(len(cells)/500),retbins=True)
    cuts = breakpoints.values  # np.concatenate((breakpoints.values))
    cuts.sort()
    categories = pd.cut(df.index, bins=(cuts - 1), right=False)
    grouped = df.groupby(categories)
    count = grouped.POINT_ID.agg('count')
    print(len(count))
    print(len(cuts))
    count.name = "COUNT"
    count = count.reset_index().COUNT

    union = grouped.geometry.aggregate(lambda s: s.unary_union)
    polys = union.convex_hull
    polys.name = "GEOM"

    filenames = []
    for name, group in grouped:
        f = working_dir / f"group_{name}.json"
        filenames.append(str(f))
        group.to_file(f)

    splits_frame = gpd.GeoDataFrame({"COUNT": count, "FILENAME": pd.Series(filenames)}, geometry=polys.reset_index().GEOM)
    splits_frame.to_file(working_dir/ "lucas_split_overview.json", index=False)
    return splits_frame
