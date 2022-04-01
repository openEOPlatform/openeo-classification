__version__ = "0.1.1a1"


def get_version() -> str:
    """
    Get installed/available version of given package/distribution (best-effort style):
    might contain additional build information on top of basic `__version__` string
    """
    # TODO: move this functionality to openeo package?
    try:
        # Try to find version of installed distribution (which might contain additional build info)
        try:
            from importlib.metadata import version
        except ImportError:
            try:
                from importlib_metadata import version
            except ImportError:
                import pkg_resources
                version = lambda d: pkg_resources.get_distribution(d).version
        return version("openeo_classification")
    except Exception:
        # If all above failed: use hardcoded version from source.
        return __version__
