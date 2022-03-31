from setuptools import setup, find_packages

tests_require = [
    "pytest",
]

setup(
    name="openeo-classification",
    version="0.1.0",
    description='openEO based classification workflows &amp; utilities, for landcover and crop mapping usecases.',
    packages=find_packages(
        where="src"
    ),
    package_dir={"": "src"},
    include_package_data=True,
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
