import os
import traceback
from hashlib import md5
from urllib.parse import urlencode
from pathlib import Path

import pandas as pd


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

    def __init__(self, latitude, longitude) -> None:
        """
        :param latitude: tuple of 2, (minlatitude, maxlatitude)
        :param longitude: tuple of 2, (minlongitude, maxlongitude)
        """
        assert isinstance(latitude, (list, tuple)) and len(latitude) == 2
        assert isinstance(longitude, (list, tuple)) and len(longitude) == 2
        self.minlatitude, self.maxlatitude = latitude
        self.minlongitude, self.maxlongitude = longitude

    def download(
        self,
        format="csv",
        starttime="1975-01-01",
        orderby="time-asc",
        evettype="earthquake",
        force_download=False,
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
            "eventtype": evettype,
        }
        [params.update({k: v}) for k, v in kwargs.items() if k in self.supported_kwargs]
        query = urlencode(params)
        query_hash = md5(query.encode("utf-8")).hexdigest()
        file_path = Path(f"csv/{query_hash}.csv")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if file_path.exists() and not force_download:
                data = pd.read_csv(file_path)
            else:
                data = pd.read_csv(self.query_url + "?" + query)
                data.to_csv(file_path, index=False)

            return data
        except:
            traceback.print_exc()
            return pd.DataFrame()
