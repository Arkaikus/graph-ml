import logging
from hashlib import md5
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

import pandas as pd

logger = logging.getLogger(__name__)


class USGS:
    query_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    supported_kwargs = [
        "format",
        "endtime",
        "starttime",
        "updatedafter",
        "minlatitude",
        "minlongitude",
        "maxlatitude",
        "maxlongitude",
        "latitude",
        "longitude",
        "maxradius",
        "maxradiuskm",
        "catalog",
        "contributor",
        "eventid",
        "includeallmagnitudes",
        "includeallorigins",
        "includearrivals",
        "includedeleted",
        "includesuperseded",
        "limit",
        "maxdepth",
        "maxmagnitude",
        "mindepth",
        "minmagnitude",
        "offset",
        "orderby",
        "alertlevel",
        "eventtype",
    ]

    def __init__(self, latitude: tuple[float, float], longitude: tuple[float, float]) -> None:
        """
        :param latitude: (min_latitude, max_latitude)
        :param longitude: (min_longitude, max_longitude)
        """
        if not isinstance(latitude, (list, tuple)) or len(latitude) != 2:
            raise ValueError("latitude must be a sequence of 2 floats (min, max)")
        if not isinstance(longitude, (list, tuple)) or len(longitude) != 2:
            raise ValueError("longitude must be a sequence of 2 floats (min, max)")
        self.minlatitude, self.maxlatitude = float(latitude[0]), float(latitude[1])
        self.minlongitude, self.maxlongitude = float(longitude[0]), float(longitude[1])

    def download(
        self,
        format: str = "csv",
        starttime: str = "1975-01-01",
        orderby: str = "time-asc",
        eventtype: str = "earthquake",
        force_download: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        params = {
            "format": format,
            "starttime": starttime,
            "minlatitude": self.minlatitude,
            "maxlatitude": self.maxlatitude,
            "minlongitude": self.minlongitude,
            "maxlongitude": self.maxlongitude,
            "orderby": orderby,
            "eventtype": eventtype,
        }
        for k, v in kwargs.items():
            if k in self.supported_kwargs:
                params[k] = v
        query = urlencode(params)
        query_hash = md5(query.encode("utf-8")).hexdigest()
        file_path = Path(f"csv/{query_hash}.csv")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if file_path.exists() and not force_download:
                return pd.read_csv(file_path)
            data = pd.read_csv(self.query_url + "?" + query)
            data.to_csv(file_path, index=False)
            return data
        except (HTTPError, URLError, OSError, ConnectionError) as e:
            logger.exception("USGS request failed: %s", e)
            return pd.DataFrame()
        except Exception as e:
            logger.exception("Unexpected error loading USGS data: %s", e)
            raise
