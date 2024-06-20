import geopy
import geopy.distance as gpd
import pandas as pd

from .hash import Hashable


class Grid(Hashable):
    def __init__(self, latitude, longitude, distance=100, **kwargs):
        """
        Takes the latitude (min, max) and longitude (min, max) and the distance size of each node in km
        """
        assert isinstance(latitude, (list, tuple)) and len(latitude) == 2
        assert isinstance(longitude, (list, tuple)) and len(longitude) == 2

        self.min_latitude, self.max_latitude = latitude
        self.min_longitude, self.max_longitude = longitude
        self.distance = distance

        self.origin = (self.min_latitude, self.min_longitude)
        self.width = gpd.distance(self.origin, (self.min_latitude, self.max_longitude)).km
        self.height = gpd.distance(self.origin, (self.max_latitude, self.min_longitude)).km
        # overestimate width and height to be evenly divided by cellsize ex:
        # 659 => 700 if cellsize = 50
        # 659 => 660 if cellsize = 60
        # 659 => 700 if cellsize = 100
        self.width = ((self.width + self.distance) // self.distance) * self.distance
        self.height = ((self.height + self.distance) // self.distance) * self.distance
        self.total_positions = int((self.width * self.height) // (self.distance**2))
        self.width_positions = self.width // self.distance
        self.height_positions = self.height // self.distance

    def to_node(self, latitude, longitude):
        """
        This function calculates to which node is located an event given the latitude and longitude

        Parameters
        ----------
        latitude : float
            the latitude in grades of a seismic event
        longitude : float
            the longitude in grades of a seismic event

        Returns
        -------
        position : int
            the corresponding position given the distance to the minimumn latitude and longitude
        """

        latitude_step = (latitude, self.min_longitude)
        longitude_step = (self.min_latitude, longitude)
        vertical_distance_to_origin = round(gpd.distance(self.origin, latitude_step).km, 0)
        horizontal_distance_to_origin = round(gpd.distance(self.origin, longitude_step).km, 0)

        horizontal_position = horizontal_distance_to_origin // self.distance
        vertical_position = vertical_distance_to_origin // self.distance
        return int(horizontal_position + (vertical_position * self.width_positions))

    def to_coordinate(self, node, offset_x=0, offset_y=0, offset=0):
        """returns the latitude and longitude for a given node with an offset in (km)"""
        assert node in range(0, self.total_positions), (node, self.total_positions)

        point = geopy.Point(self.origin)
        x_position = node % self.width_positions
        y_position = node // self.width_positions
        x_delta = (x_position * self.distance) + offset_x + offset
        y_delta = (y_position * self.distance) + offset_y + offset

        point = gpd.geodesic(kilometers=x_delta).destination(point=point, bearing=90)
        point = gpd.geodesic(kilometers=y_delta).destination(point=point, bearing=0)
        return (point.latitude, point.longitude)

    def apply_node(self, data: pd.DataFrame):
        """
        Returns
        -------
        apply : function
            lambda function for pandas apply usage, the argument is a pandas dataframe row
        """
        assert "latitude" in data, "[latitude] is required for position tagging"
        assert "longitude" in data, "[longitude] is required for position tagging"

        def parse_row(row):
            return self.to_node(row["latitude"], row["longitude"])

        return data.apply(parse_row, axis=1)

    @property
    def center(self):
        """Center of the grid in latitude and longitude"""
        center_lat = (self.max_latitude - self.min_latitude) / 2
        center_lon = (self.max_longitude - self.min_longitude) / 2
        return dict(
            lat=self.min_latitude + center_lat,
            lon=self.min_longitude + center_lon,
        )

    # bearing values: 0 – North, 90 – East, 180 – South, 270 or -90 – West
    def latitude_steps(self):
        """returns the longitude steps from origin (goes south-north)"""
        d = gpd.geodesic(kilometers=self.distance)
        steps = [self.min_latitude]
        start = geopy.Point(self.origin)
        for _ in range(int(self.height // self.distance)):
            start = d.destination(point=start, bearing=0)
            steps.append(start[0])

        return steps

    def longitude_steps(self):
        """returns the longitude steps from origin (goes west-east)"""
        d = gpd.geodesic(kilometers=self.distance)
        steps = [self.min_longitude]
        start = geopy.Point(self.origin)
        for _ in range(int(self.width // self.distance)):
            start = d.destination(point=start, bearing=90)
            steps.append(start[1])

        return steps
