from __future__ import annotations
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from .grid import Grid
from .hash import Hashable
from .graphs import nodes2graph, networkx_property

logger = logging.getLogger(__name__)


@dataclass
class EarthquakeData(Hashable):
    """
    Wrapper class to clean and normalize the csv catalog
    """

    raw_data: pd.DataFrame
    features: list
    targets: list[str]
    # zero_columns: list = field(default_factory=list)
    time_column: bool = True
    drop_time_column: bool = True
    delta_time: bool = True
    delta_type: str = "timedelta64[s]"
    min_year: int = 1973
    min_magnitude: float = 0
    max_magnitude: float = 10
    grid: Grid = None
    network_features: list = field(default_factory=list)
    network_lookback: int = 5

    def __post_init__(self):
        if not "time" in self.raw_data.columns:
            self.time_column = False
            self.delta_time = False
            self.drop_time_column = False

    @classmethod
    def from_path(cls, file_path: str, **kwargs):
        return cls(pd.read_csv(file_path), **kwargs)

    @property
    def data(self) -> pd.DataFrame:
        """cleans the data and tags the nodes if self.grid available"""
        data = self.clean()
        assert all(t in data for t in self.targets)

        if self.grid:
            data["node"] = self.grid.apply_node(data)
            data = data.astype(dict(node=int))

        return data

    @property
    def normalized_data(self) -> pd.DataFrame:
        return self.normalize(self.data)

    def clean(self) -> pd.DataFrame:
        """
        This method preprocess the data argument
        - Coerce numeric columns with pd.to_numeric
        - If min_magnitude is set it filters events with magnitude > self.min_magnitude
        - If time_column=True uses pd.to_datetime to data["time"]
        - If time_column=True filters events with year greater than self.min_year
        - If time_column=True and delta_time=True calculates the time difference between events in days by default

        """
        if not hasattr(self, "processed_data"):

            processed_data = self.raw_data.copy()
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
                if self.delta_time and not "delta" in self.features:
                    self.features.append("delta")
                    # delta_values = processed_data["time"] - processed_data["time"].shift()
                    # delta_values.at[0] = pd.Timedelta(0)
                    delta_values = processed_data["time"].diff().fillna(pd.Timedelta(seconds=0))
                    processed_data["delta"] = pd.to_numeric(delta_values.astype("timedelta64[s]"))
                    # if "delta" not in self.zero_columns:
                    #     self.zero_columns.append("delta")

                if self.drop_time_column and "time":
                    processed_data.drop("time", axis=1, inplace=True)
            else:
                processed_data = processed_data[self.features]

            self.processed_data = processed_data.dropna().reset_index(drop=True)

        return self.processed_data

    def normalize(self, clean_data: pd.DataFrame, mode="standard"):
        """
        This method will apply sklearn.MinMaxScaler.fit_transform to the 'numeric_columns'
        of the data argument, the minmaxscaler will be set for later use of inverse_transform

        :params data: dataframe to be normalized
        :params scaler: sklearn MinMaxScaler or similar
        """
        data = clean_data.copy()
        if mode == "standard":
            scaler = StandardScaler()
            data[self.features] = pd.DataFrame(
                scaler.fit_transform(data[self.features]),
                columns=self.features,
            )
        else:
            logger.warning("No scaler class detected")

        return data

    def to_sequences(
        self,
        data: pd.DataFrame,
        lookback,
        features: list = None,
        targets: list = None,
        network_features: list = None,
        network_lookback: int = 5,
        notebook=False,
    ):
        """
        Processes the raw data and returns a two numpy arrays,
        one with shape (len(data) -lookback, S, F)
        and target of shape (len(data) -lookback, lookback)

        where S is the number of sequences, each sequence holds a feature
        F is the number of feature values in the sequence, aka lookback

        target holds the next event window of size lookback

        example:
        data.columns = ['latitude','longitude','magnitude']
        lookback = 10

        there will be 3 sequences of size 10 per window
        number of windows = len(data) - lookback

        output1 = [
            [# first window
              # [<---- lookback --->] # size of sequence
                [1,2,3,4,5,6,7,8,9,10] # sequence 1 of latitude
                [1,2,3,4,5,6,7,8,9,10] # sequence 2 of longitude
                [1,2,3,4,5,6,7,8,9,10] # sequence 3 of magnitude
            ],
            ...
        ]
        output2 = [
            [# first target window
              # [<---- lookback --->] # size of sequence
                [2,3,4,5,6,7,8,9,10,11] # sequence 1 of target
                ...
            ],
            ...
        ]

        for 100 events output would be (90, 3, 10) and (90, 10)
        """
        sequences = data.shape[0] - lookback
        input_chunks = [None] * sequences
        output_chunks = [None] * sequences
        _features = features or self.features
        if "node" in data:
            if "node" not in _features:
                _features = _features + ["node"]
            max_nodes = int(data["node"].max() + 1)
            nx_features = network_features or self.network_features
            nx_lookback = network_lookback or self.network_lookback

        if notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        def worker(start, end):
            output_chunk = data.iloc[start + 1 : end + 1][targets or self.targets]
            if "node" in data and nx_features:
                input_chunk = data.iloc[start:end][_features]
                graph = nodes2graph(input_chunk["node"].values, max_nodes, nx_lookback)
                for feature in nx_features or []:
                    property_df = networkx_property(graph, feature)
                    input_chunk = pd.merge(
                        input_chunk.copy(),
                        property_df,
                        on="node",
                        how="left",
                    )

                input_chunk.drop(columns=["node"], inplace=True)
            else:
                input_chunk = data.iloc[start:end][_features]

            return start, input_chunk, output_chunk

        with ThreadPoolExecutor(8) as exc:
            futures = [exc.submit(worker, i, i + lookback) for i in range(sequences)]
            for future in tqdm(as_completed(futures), total=sequences):
                i, input_chunk, output_chunk = future.result()
                input_chunks[i] = input_chunk.values
                output_chunks[i] = output_chunk.values

        inputs = np.stack(input_chunks)
        outputs = np.array(output_chunks)
        return np.transpose(inputs, (0, 2, 1)), outputs

    def split(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        test_size: float,
        torch_tensor=True,
        shuffle=False,
        **kwargs,
    ):
        """
        Returns the train test split of sequences and targets, with the option to convert to torch tensors
        :params sequences: ndarray, sequences of the data
        :params targets: ndarray, target values
        :params torch_tensor: bool, whether to convert the numpy arrays to torch tensors
        :params shuffle: bool, whether to shuffle the data before splitting
            false by default, due to the nature of time series data
        """
        if torch_tensor:
            sequences = torch.Tensor(sequences).to(torch.float32)
            targets = torch.Tensor(targets).to(torch.float32)

        return train_test_split(
            sequences,
            targets,
            test_size=test_size,
            shuffle=shuffle,
            **kwargs,
        )

    def cut(self, df: pd.DataFrame, quantiles: int | list = 4):
        """
        Uses pandas quantile cut to bin the features
        returns the dataframe with the binned features
        and the list of binned features
        """
        for f in self.features:
            df[f"{f}_binned"] = pd.qcut(df[f], q=quantiles, labels=False)

        return df, [f"{f}_binned" for f in self.features]

    def one_hot(self, df: pd.DataFrame, suffix="_binned"):
        return (
            pd.get_dummies(
                df,
                columns=[f"{f}{suffix}" for f in self.features],
                prefix=self.features,
            )
            .drop(columns=self.features)
            .astype(int)
        )

    def categorical(self, quantiles: int | list = 4):
        """
        Applies quantile cut to the data and returns the one hot encoded
        dataframe plus the nnormalized features
        it also returns the binned features
        """
        data, bin_cols = self.cut(self.data, quantiles=quantiles)
        one_hot = self.one_hot(data)
        concat = pd.concat((one_hot, data.drop(columns=["node"], errors="ignore")), axis=1)
        nobins = list(set(concat.columns) - set(bin_cols))
        return concat[nobins], concat[bin_cols]
