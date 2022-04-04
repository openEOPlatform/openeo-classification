import json
import openeo_classification
from importlib.resources import open_text

from openeo.util import rfc3339
from openeogeotrellis.catalogs.base import CatalogStatus
from openeogeotrellis.catalogs.creo import CreoCatalogClient

tiles_latvia = ["34VEH", "34UEG","34VFH","34UFG","34UFF","35VLC","35ULB","35ULA","35UMB"]

# TODO: invalid reference (https://github.com/openEOPlatform/openeo-classification/issues/2)
with open_text(openeo_classification,"result.json") as f:

    data = json.load(f)
    result = data["features"]

catalog = CreoCatalogClient(CreoCatalogClient.MISSION_SENTINEL2, level=CreoCatalogClient.LEVEL2A)
products = catalog.query(rfc3339.parse_date_or_datetime("2018-10-01"), rfc3339.parse_date_or_datetime("2019-01-01"), tiles_latvia, cldPrcnt=60)
print(products)

to_order = [p for p in products if p.getStatus() == CatalogStatus.ORDERABLE]

print(to_order)
catalog.order(to_order)
