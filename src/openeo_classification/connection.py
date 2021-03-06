from functools import partial, lru_cache

import openeo
from openeo import Connection

_default_url = "openeo-dev.vito.be"

def set_backend(url):
    _default_url = url


@lru_cache(maxsize=None)
def _cached_connection(url) -> Connection:
    return openeo.connect(url)


def connection(url = _default_url) -> Connection:
    """
    Returns an authenticated openEO connection.
    Connects to openEO platform by default, but others can be used as well by specifying the url.

    @param url:
    @return:
    """
    c = _cached_connection(url)
    c.authenticate_oidc()
    return c

terrascope_dev = partial(connection,"openeo-dev.vito.be")
creo = partial(connection,"openeo.creo.vito.be")
creo_new = partial(connection,"https://openeo-dev.creo.vito.be/openeo/1.1.0/")
