import logging

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .grid import Grid
from .hash import Hashable
from .store import Store

logger = logging.getLogger(__name__)


class Data(Hashable):
    """
    Wrapper class to clean and normalize the csv catalog
    """

    def __init__(
        self,
        raw_data: pd.DataFrame,
        numeric_columns: list,
        time_column: bool = False,
        drop_time_column: bool = False,
        delta_time: bool = False,
        delta_type: str = "timedelta64[D]",
        min_year: int = 1973,
        min_magnitude: float = 0,
        zero_columns: list = [],
    ):
        """
        Parameters
        ----------
        numeric_columns : list
            list of the csv numeric columns to use.
        time_column : bool, default=False
            whether the csv file has the date/time column available.
        drop_time_column : bool, default=False
            whether to drop the date/time column from the dataframe after cleaning.
        delta_time : bool, default=False
            if time_column is set create a new column with the time difference between events in delta_type timespan.
        delta_type : str, default='timedelta64[D]'
            difference in time between events in delta_time.
        min_year : bool
            if time_column is true filters events with year greater than min_year
        min_magnitude : float
            if set filters events with magnitude > min_magnitude
        """
        self.raw_data = raw_data
        self.numeric_columns = numeric_columns
        self.time_column = time_column
        self.drop_time_column = drop_time_column
        self.delta_time = delta_time
        self.delta_type = delta_type
        self.min_year = min_year
        self.min_magnitude = min_magnitude
        self.zero_columns = zero_columns

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This method preprocess the data argument
        - Coerce numeric columns with pd.to_numeric
        - If min_magnitude is set it filters events with magnitude > self.min_magnitude
        - If time_column=True uses pd.to_datetime to data["time"]
        - If time_column=True filters events with year greater than self.min_year
        - If time_column=True and delta_time=True calculates the time difference between events in days by default

        """
        processed_data = data.copy()
        for column in self.numeric_columns:
            assert column in processed_data.columns, f"[{column}] is not in the dataframe"

        # Treat columns as numeric values and coerce NaN values
        processed_data[self.numeric_columns] = processed_data[self.numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        # Filter events with magnitude > self.min_magnitude
        if "mag" in self.numeric_columns:
            processed_data = processed_data[processed_data["mag"] > self.min_magnitude]

        if self.time_column:
            assert "time" in processed_data.columns, "[time] column is not in the dataframe"
            processed_data = processed_data[["time"] + self.numeric_columns]
            processed_data["time"] = pd.to_datetime(processed_data["time"], errors="coerce")
            # Filter events with year greater than self.min_year
            if self.min_year:
                processed_data["event_year"] = processed_data["time"].dt.year
                processed_data = processed_data[processed_data["event_year"] > self.min_year]
                processed_data.drop("event_year", axis=1, inplace=True)

            # If delta_time=True calculates the time difference between events in days by default
            if self.delta_time:
                self.numeric_columns.append("delta")
                processed_data["delta"] = (
                    (processed_data["time"] - processed_data["time"].shift()).astype(self.delta_type).fillna(0)
                )
                processed_data["delta"] = pd.to_numeric(processed_data["delta"])
                if "delta" not in self.zero_columns:
                    self.zero_columns.append("delta")

            if self.drop_time_column:
                processed_data.drop("time", axis=1, inplace=True)
        else:
            processed_data = processed_data[self.numeric_columns]

        return processed_data.dropna().reset_index(drop=True)

    def normalize(self, clean_data: pd.DataFrame, scaler: MinMaxScaler):
        """
        This method will apply sklearn.MinMaxScaler.fit_transform to the 'numeric_columns'
        of the data argument, the minmaxscaler will be set for later use of inverse_transform

        :params data: dataframe to be normalized
        :params scaler: sklearn MinMaxScaler or similar
        """
        data = clean_data.copy()
        scalers = {}
        for column in data.columns:
            if column in self.numeric_columns:
                data[column] = scaler.fit_transform(data[[column]])
                scalers[column] = scaler

        return data, scalers

    def denormalize(self, normal_data: pd.DataFrame, scalers: dict) -> pd.DataFrame:
        """
        This method returns the inverse transform for each scaled column
        """
        data = normal_data.copy()

        for column in self.numeric_columns:
            scaler: MinMaxScaler = scalers[column]
            data[column] = scaler.inverse_transform(data[[column]])

        return data

    def process(self, grid: Grid = None, scaler: MinMaxScaler = None):
        """
        Runs the cleaning and normalizaiton, against the wrapped data
        :param grid: (Grid) instance of grid object to handle node tagging
        :param notmalize: to run
        """
        data = self.clean(self.raw_data)
        scalers = None

        if grid:
            data["node"] = grid.apply_node(data)
            data = data.astype(dict(node=int))

        if scaler:
            # Insert event with minimum values before index 0
            min_values_row = data.min().to_frame().transpose()
            min_values_row["latitude"] = grid.min_latitude
            min_values_row["longitude"] = grid.min_longitude

            for column in self.zero_columns:
                # columns that actually real minimum value is 0
                # this helps offseting the minmax scaler
                # example: if magnitude is between 4.0 and 7.0, 4.0 isn't the real minimum
                # thus minmaxscaler cannot treat 4.0 as 0.0
                min_values_row[column] = 0
            data = pd.concat([min_values_row, data], ignore_index=True)
            data, scalers = self.normalize(data, scaler)
            return data.drop(0).reset_index(drop=True), scalers

        return data.reset_index(drop=True), scalers
