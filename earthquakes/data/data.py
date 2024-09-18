from __future__ import annotations
import logging
from functools import cache

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from .grid import Grid
from .hash import Hashable

logger = logging.getLogger(__name__)


class EarthquakeData(Hashable):
    """
    Wrapper class to clean and normalize the csv catalog
    """

    modes = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
    }

    def __init__(
        self,
        raw_data: pd.DataFrame,
        features: list,
        targets: list[str],
        time_column: bool = False,
        drop_time_column: bool = False,
        delta_time: bool = False,
        delta_type: str = "timedelta64[s]",
        min_year: int = 1973,
        min_magnitude: float = 0,
        max_magnitude: float = 6,
        zero_columns: list = [],
        min_latitude=None,
        min_longitude=None,
        grid: Grid = None,
    ):
        """
        Parameters
        ----------
        features : list
            list of the csv numeric columns to use that are numeric.
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
        self.features = features
        self.targets = targets
        self.time_column = time_column
        self.drop_time_column = drop_time_column
        self.delta_time = delta_time
        self.delta_type = delta_type
        self.min_year = min_year
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.zero_columns = zero_columns
        self.min_latitude = min_latitude
        self.min_longitude = min_longitude
        self.grid = grid
        assert min_latitude and min_latitude, "please provide min_latitude and min_longitude in .env or __init__"

    @property
    def data(self) -> pd.DataFrame:
        if not hasattr(self, "_processed_data"):
            self._processed_data = self.__process()

        return self._processed_data

    def clean(self) -> pd.DataFrame:
        """
        This method preprocess the data argument
        - Coerce numeric columns with pd.to_numeric
        - If min_magnitude is set it filters events with magnitude > self.min_magnitude
        - If time_column=True uses pd.to_datetime to data["time"]
        - If time_column=True filters events with year greater than self.min_year
        - If time_column=True and delta_time=True calculates the time difference between events in days by default

        """
        processed_data = self.raw_data.copy()
        for column in self.features:
            assert column in processed_data.columns, f"[{column}] is not in the dataframe"

        # Treat columns as numeric values and coerce NaN values
        processed_data[self.features] = processed_data[self.features].apply(pd.to_numeric, errors="coerce")

        # Filter events with magnitude > self.min_magnitude
        if "mag" in self.features:
            mag_mask = (processed_data["mag"] > self.min_magnitude) & (processed_data["mag"] < self.max_magnitude)
            processed_data = processed_data[mag_mask]
            processed_data = processed_data[processed_data["magType"].isin(("mb",))]

        if self.time_column:
            assert "time" in processed_data.columns, "[time] column is not in the dataframe"
            processed_data = processed_data[["time"] + self.features]
            processed_data["time"] = pd.to_datetime(processed_data["time"], errors="coerce")
            # Filter events with year greater than self.min_year
            if self.min_year:
                processed_data["event_year"] = processed_data["time"].dt.year
                processed_data = processed_data[processed_data["event_year"] > self.min_year]
                processed_data.drop("event_year", axis=1, inplace=True)

            # If delta_time=True calculates the time difference between events in days by default
            if self.delta_time:
                self.features.append("delta")
                # delta_values = processed_data["time"] - processed_data["time"].shift()
                # delta_values.at[0] = pd.Timedelta(0)
                delta_values = processed_data["time"].diff().fillna(pd.Timedelta(seconds=0))
                processed_data["delta"] = pd.to_numeric(delta_values.astype("timedelta64[s]"))
                if "delta" not in self.zero_columns:
                    self.zero_columns.append("delta")

            if self.drop_time_column:
                processed_data.drop("time", axis=1, inplace=True)
        else:
            processed_data = processed_data[self.features]

        return processed_data.dropna().reset_index(drop=True)

    def normalize(self, clean_data: pd.DataFrame, mode="standard"):
        """
        This method will apply sklearn.MinMaxScaler.fit_transform to the 'numeric_columns'
        of the data argument, the minmaxscaler will be set for later use of inverse_transform

        :params data: dataframe to be normalized
        :params scaler: sklearn MinMaxScaler or similar
        """
        data = clean_data.copy()
        # scaler_class = self.modes.get(mode)
        # scaler: MinMaxScaler | StandardScaler | None = scaler_class() if scaler_class else None
        if mode == "minmax":
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
            scaler = MinMaxScaler()
            data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
            return data.drop(0).reset_index(drop=True)
        elif mode == "standard":
            scaler = StandardScaler()
            data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

            # min_value = np.min(data)
            # data = data - min_value # Shift to positive values
        else:
            logger.warning("No scaler class detected")

        return data

    def __process(self):
        """
        Runs the cleaning and normalizaiton, against the wrapped data
        :param grid: (Grid) instance of grid object to handle node tagging
        :param notmalize: to run
        """

        data = self.clean()
        assert all(t in data for t in self.targets)

        if self.grid:
            data["node"] = self.grid.apply_node(data)
            data = data.astype(dict(node=int))

        # data["latitude_longitude"] = data["latitude"] * data["longitude"]
        # self.features.append("latitude_longitude")
        return data

    def to_sequences(self, data: pd.DataFrame, lookback):
        """
        Processes the raw data and returns a two numpy arrays,
        one with shape (len(data) -lookback, S, F)
        and target of shape (len(data) -lookback,)

        where S is the number of sequences by feature
        F is the number of features in a sequence, aka lookback

        example:
        data.columns = ['latitude','longitude','magnitude']
        lookback = 10

        there will be 3 sequences of size 10 per window
        number of windows = len(data) - lookback

        [
            [# first window
              # [<---- lookback --->] # size of sequence
                [1,2,3,4,5,6,7,8,9,0] # sequence 1 of latitude
                [1,2,3,4,5,6,7,8,9,0] # sequence 2 of longitude
                [1,2,3,4,5,6,7,8,9,0] # sequence 3 of magnitude
            ],
            ...
        ]
        """
        input_chunks = []
        output_chunks = []
        sequences = data.shape[0] - lookback

        for i in range(sequences):
            end = i + lookback
            input_chunk = data.iloc[i:end][self.features]
            output_chunk = data.iloc[i + 1 : end + 1][self.targets]
            input_chunks.append(input_chunk)
            output_chunks.append(output_chunk)

        inputs = np.stack(input_chunks)
        outputs = np.array(output_chunks)
        return np.transpose(inputs, (0, 2, 1)), outputs  # [:, -1, :]

    @cache
    def train_test_split(self, sequence_size, test_size: float, scaler="standard", torch_tensor=True):
        """
        Calculates the rolling window sequences and splits the data into train, validation and test sets
        turns (9999, 5) where 9999 is the number of records and 5 the number of features
        into (9994, 3, 5) and (9994, 1) if 3 is the sequence size
        then applies a scaler to the data

        """

        data = self.normalize(self.data, mode=scaler)
        sequences, targets = self.to_sequences(data, sequence_size)
        if torch_tensor:
            sequences = torch.Tensor(sequences).to(torch.float32)
            targets = torch.Tensor(targets).to(torch.float32)
        X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=test_size, shuffle=False)
        return ((X_train, y_train), (X_test, y_test))
