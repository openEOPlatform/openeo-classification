import json
from importlib.resources import open_text


def read_json_resource(package: str, resource: str) -> dict:
    with open_text(package=package, resource=resource) as f:
        return json.load(f)
