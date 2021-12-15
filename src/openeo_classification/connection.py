import openeo
from openeo import Connection
from functools import cache,partial


_default_url = "openeo.vito.be"

def set_backend(url):
    _default_url = url

@cache
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