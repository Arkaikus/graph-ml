import logging
from functools import cache

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from .grid import Grid
from .hash import Hashable

logger = logging.getLogger(__name__)


class EarthquakeData(Hashable):
    """
    Wrapper class to clean and normalize the csv catalog
    """

    def __init__(
        self,
        raw_data: pd.DataFrame,
        numeric_columns: list,
        target: str,
        time_column: bool = False,
        drop_time_column: bool = False,
        delta_time: bool = False,
        delta_type: str = "timedelta64[s]",
        min_year: int = 1973,
        min_magnitude: float = 0,
        zero_columns: list = [],
        scaler_mode="standard",
        min_latitude=None,
        min_longitude=None,
        grid: Grid = None,
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
        delta_type : str, default='timedelta64[s]'
            difference in time between events in delta_time.
        min_year : bool
            if time_column is true filters events with year greater than min_year
        min_magnitude : float
            if set filters events with magnitude > min_magnitude
        """
        self.raw_data = raw_data
        self.numeric_columns = numeric_columns
        self.target = target
        self.time_column = time_column
        self.drop_time_column = drop_time_column
        self.delta_time = delta_time
        self.delta_type = delta_type
        self.min_year = min_year
        self.min_magnitude = min_magnitude
        self.zero_columns = zero_columns

        modes = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
        }
        self.scaler_mode = scaler_mode
        self.scaler_class = modes.get(scaler_mode)
        self.min_latitude = min_latitude
        self.min_longitude = min_longitude
        self.grid = grid
        self.scalers = {}
        assert min_latitude and min_latitude, "please provide min_latitude and min_longitude in .env or __init__"

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
                delta_values = processed_data["time"] - processed_data["time"].shift()
                delta_values.at[0] = pd.Timedelta(0)
                processed_data["delta"] = pd.to_numeric(delta_values.astype("timedelta64[s]"))
                if "delta" not in self.zero_columns:
                    self.zero_columns.append("delta")

            if self.drop_time_column:
                processed_data.drop("time", axis=1, inplace=True)
        else:
            processed_data = processed_data[self.numeric_columns]

        return processed_data.dropna().reset_index(drop=True)

    def normalize(self, clean_data: pd.DataFrame):
        """
        This method will apply sklearn.MinMaxScaler.fit_transform to the 'numeric_columns'
        of the data argument, the minmaxscaler will be set for later use of inverse_transform

        :params data: dataframe to be normalized
        :params scaler: sklearn MinMaxScaler or similar
        """
        data = clean_data.copy()
        if self.scaler_class:
            # Insert event with minimum values before index 0
            min_values_row = data.min().to_frame().transpose()
            min_values_row["latitude"] = float(self.min_latitude)
            min_values_row["longitude"] = float(self.min_longitude)

            for column in self.zero_columns:
                # columns that actually real minimum value is 0
                # this helps offseting the minmax scaler
                # example: if magnitude is between 4.0 and 7.0, 4.0 isn't the real minimum
                # thus minmaxscaler cannot treat 4.0 as 0.0
                min_values_row[column] = 0

            data = pd.concat([min_values_row, data], ignore_index=True)
            for column in data.columns:
                if column in self.numeric_columns:
                    scaler = self.scaler_class()
                    data[column] = scaler.fit_transform(data[[column]])
                    self.scalers[column] = scaler

            return data.drop(0).reset_index(drop=True)
        else:
            logger.warning("No scaler class detected")

        return data

    @cache
    def process(self):
        """
        Runs the cleaning and normalizaiton, against the wrapped data
        :param grid: (Grid) instance of grid object to handle node tagging
        :param notmalize: to run
        """
        data = self.clean(self.raw_data)

        if self.grid:
            data["node"] = self.grid.apply_node(data)
            data = data.astype(dict(node=int))

        return self.normalize(data)

    def to_sequences(self, sequence_size):
        """
        Processes the raw data and returns a two numpy arrays,
        one with sequences of shape (n-1, seq_size, feature_size)
        and target of shape (n-1,)
        """
        data = self.process()
        inputs, outputs = [], []
        logger.debug("Dataframe shape %s", data.shape)
        logger.debug("Sequence length %s", sequence_size)
        logger.debug("Available sequences %s", data.shape[0] // sequence_size)
        assert data.shape[0] - sequence_size > 1, "sequence can't be less than available data"
        for index in range(data.shape[0] - sequence_size - 1):
            sequence = data.iloc[index : index + sequence_size].values
            target_value = data.at[index + sequence_size, self.target]
            inputs.append(sequence)
            outputs.append(target_value)

        return np.array(inputs), np.array(outputs).reshape(-1, 1)

    def train_test_split(self, sequence_size, test_size: float, test=False):
        sequences, targets = self.to_sequences(sequence_size)
        (
            train_sequences,
            test_sequences,
            train_targets,
            test_targets,
        ) = train_test_split(sequences, targets, test_size=test_size, shuffle=False)

        logger.debug("Train records(%s): %s", 1 - test_size, train_sequences.shape[0])
        logger.debug("Test records(%s): %s", test_size, train_targets.shape[0])
        logger.debug("Total: %s", sequences.shape[0])
        logger.debug("Features: %s", tuple(sequences.shape[-1]))
        logger.debug("Target: %s", self.target)

        if test:
            return test_sequences, test_targets
        else:
            return train_sequences, train_targets
