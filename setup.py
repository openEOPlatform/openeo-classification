import re
from pathlib import Path

from setuptools import setup, find_packages

tests_require = [
    "pytest",
]


def get_version(path: str = "src/openeo_classification/_version.py"):
    # Single-sourcing package version (https://packaging.python.org/en/latest/guides/single-sourcing-package-version/#single-sourcing-the-package-version)
    try:
        regex = re.compile(r"^__version__\s*=\s*(?P<q>['\"])(?P<v>.+?)(?P=q)\s$", flags=re.MULTILINE)
        with (Path(__file__).absolute().parent / path).open("r") as f:
            return regex.search(f.read()).group("v")
    except Exception:
        raise RuntimeError("Failed to find version string.")


setup(
    name="openeo_classification",
    version=get_version(),
    description='openEO based classification workflows &amp; utilities, for landcover and crop mapping usecases.',
    packages=find_packages(
        where="src"
    ),
    package_dir={"": "src"},
    package_data={
        "openeo_classification.resources": [
            "grids/*.geojson",
        ]
    },
    install_requires=[
        "setuptools",
        "openeo>=0.9.0a1",
        "geopandas",
        "ipywidgets",
        "rasterio",
        "utm",
        "scikit-learn",
    ],
    tests_require=tests_require,
    extras_require={"dev": tests_require},
)
