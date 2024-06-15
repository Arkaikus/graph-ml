import logging
import os

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)


def read_coordinates(latitude=None, longitude=None):
    load_dotenv(override=True)
    latitude = latitude or os.getenv("LATITUDE", "").split(",")
    longitude = longitude or os.getenv("LONGITUDE", "").split(",")
    return latitude, longitude
