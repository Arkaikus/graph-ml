import logging

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .store import Store
from .grid import Grid
from .hash import Hashable

logger = logging.getLogger(__name__)


class DataBuilder(Hashable):
    """
    This class does the cleaning and processing for the .csv catalog
    """

    def __init__(
        self,
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
        self.numeric_columns = numeric_columns
        self.time_column = time_column
        self.drop_time_column = drop_time_column
        self.delta_time = delta_time
        self.delta_type = delta_type
        self.min_year = min_year
        self.min_magnitude = min_magnitude
        self.zero_columns = zero_columns
        self.is_fitted = False
        self.scalers = {}
        self.cache_folder = "preprocess_" + self.hash

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

    def normalize(self, data: pd.DataFrame, scaler=MinMaxScaler) -> pd.DataFrame:
        """
        This method will apply sklearn.MinMaxScaler.fit_transform to the 'numeric_columns'
        of the data argument, the minmaxscaler will be set for later use of inverse_transform
        """
        processed_data = data.copy()
        store = Store(self.cache_folder, "scalers_" + self.hash)
        stored_data = store.load()
        if stored_data is not None:
            scalers, columns, is_fitted = stored_data
            self.scalers = scalers
            self.numeric_columns = columns
            self.is_fitted = is_fitted
            for column in self.scalers:
                processed_data[column] = pd.DataFrame(self.scalers[column].transform(processed_data[[column]]))

            return processed_data

        for column in processed_data.columns:
            if column in self.numeric_columns:
                column_scaler = scaler()
                processed_data[column] = column_scaler.fit_transform(processed_data[[column]])
                self.scalers[column] = column_scaler

        self.is_fitted = True
        store.save((self.scalers, self.numeric_columns, self.is_fitted))
        return processed_data

    def denormalize(self, data: pd.DataFrame, columns=[]) -> pd.DataFrame:
        """
        This method returns the inverse transform of minmaxscaler for the
        'numeric_columns' in the data argument
        """
        processed_data = data.copy()
        store = Store(self.cache_folder, "scalers_" + self.hash)
        stored_data = store.load()
        if stored_data is not None:
            scalers, _columns, is_fitted = stored_data
            self.scalers = scalers
            self.numeric_columns = _columns
            self.is_fitted = is_fitted

        if self.is_fitted:
            for column in columns or _columns:
                processed_data[column] = self.scalers[column].inverse_transform(data[[column]])

            return processed_data

        raise Exception("Please run normalize(data) method before trying to denormalize the data")

    def build_data(
        self,
        raw_data: pd.DataFrame,
        grid: Grid,
        normalize=True,
    ) -> pd.DataFrame:
        """
        Takes a dataframe with raw_data and applies the cleaning and normalizaiton
        """
        prefix = "normalized" if normalize else "cleaned"
        file_name = f"{prefix}_{self.hash}_{grid.hash}"
        store = Store(prefix + "_data", file_name)
        if not store.empty:
            return store.data
        else:
            data = raw_data.copy()
            data = self.clean(data)
            data["node"] = grid.apply_node(data)

            if normalize:
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
                data = self.normalize(data).drop(0)

            data = data.astype(dict(node=int)).reset_index(drop=True)
            store.save(data)
            return data
