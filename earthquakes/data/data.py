from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .graphs import networkx_property, nodes2graph
from .grid import Grid
from .hash import Hashable

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
        if "time" not in self.raw_data.columns:
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
            # Effective feature list for this run (may include "delta"); do not mutate self.features
            self._features_used = list(self.features)
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
                processed_data = processed_data[["time"] + self._features_used]
                processed_data["time"] = pd.to_datetime(processed_data["time"], errors="coerce")
                # Filter events with year greater than self.min_year
                if self.min_year:
                    processed_data["event_year"] = processed_data["time"].dt.year
                    processed_data = processed_data[processed_data["event_year"] > self.min_year]
                    processed_data.drop("event_year", axis=1, inplace=True)

                # If delta_time=True add time difference; extend _features_used without mutating self.features
                if self.delta_time and "delta" not in self._features_used:
                    delta_values = processed_data["time"].diff().fillna(pd.Timedelta(seconds=0))
                    processed_data["delta"] = pd.to_numeric(delta_values.astype("timedelta64[s]"))
                    self._features_used = list(self.features) + ["delta"]

                if self.drop_time_column and "time" in processed_data.columns:
                    processed_data = processed_data.drop("time", axis=1)
            else:
                processed_data = processed_data[self._features_used]

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
        features = getattr(self, "_features_used", self.features)
        if mode == "standard":
            scaler = StandardScaler()
            data[features] = pd.DataFrame(
                scaler.fit_transform(data[features]),
                columns=features,
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
            node_col = data["node"]
            if isinstance(node_col, pd.DataFrame):
                node_col = node_col.iloc[:, 0]
            max_nodes = int(np.asarray(node_col).max()) + 1
            nx_features = network_features or self.network_features
            nx_lookback = network_lookback or self.network_lookback

        if notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        def worker(start, end):
            output_chunk = data.iloc[start + 1 : end + 1][targets or self.targets]
            if "node" in data and nx_features:
                input_chunk = data.iloc[start:end][_features].copy()
                # Ensure single node column (deduplicate if categorical produced duplicates)
                input_chunk = input_chunk.loc[:, ~input_chunk.columns.duplicated()]
                node_col = input_chunk["node"]
                if isinstance(node_col, pd.DataFrame):
                    node_col = node_col.iloc[:, 0]
                nodes_arr = np.asarray(node_col, dtype=np.int64).flatten()
                graph = nodes2graph(nodes_arr, max_nodes, nx_lookback)
                for feature in nx_features or []:
                    property_df = networkx_property(graph, feature)
                    input_chunk = pd.merge(
                        input_chunk,
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
        temporal=True,
        **kwargs,
    ):
        """
        Returns the train test split of sequences and targets, with the option to convert to torch tensors.
        For time series, use temporal=True to keep chronological order (last test_size fraction as test).
        :params sequences: ndarray, sequences of the data
        :params targets: ndarray, target values
        :params torch_tensor: bool, whether to convert the numpy arrays to torch tensors
        :params shuffle: bool, whether to shuffle the data before splitting
            false by default, due to the nature of time series data
        :params temporal: bool, if True use last test_size fraction as test (no shuffle)
        """
        if temporal:
            n = len(sequences)
            test_n = max(1, int(n * test_size))
            train_val_n = n - test_n
            val_ratio = kwargs.pop("val_ratio", 0.15)
            val_n = max(1, int(train_val_n * val_ratio)) if val_ratio > 0 else 0
            train_n = train_val_n - val_n
            x_test = sequences[train_val_n:]
            y_test = targets[train_val_n:]
            x_train = sequences[:train_n]
            y_train = targets[:train_n]
            x_val = sequences[train_n:train_val_n] if val_n > 0 else sequences[:1]
            y_val = targets[train_n:train_val_n] if val_n > 0 else targets[:1]
            if torch_tensor:
                x_train = torch.tensor(x_train, dtype=torch.float32)
                x_test = torch.tensor(x_test, dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.float32)
                y_test = torch.tensor(y_test, dtype=torch.float32)
                x_val = torch.tensor(x_val, dtype=torch.float32)
                y_val = torch.tensor(y_val, dtype=torch.float32)
            return x_train, x_test, y_train, y_test, x_val, y_val

        if torch_tensor:
            sequences = torch.tensor(sequences, dtype=torch.float32)
            targets = torch.tensor(targets, dtype=torch.float32)

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

    def categorical(self, quantiles: int | list = 4, keep_node: bool = False):
        """
        Applies quantile cut to the data and returns the one hot encoded
        dataframe plus the nnormalized features
        it also returns the binned features
        When keep_node=True and data has node (from grid), node is retained for graph features in to_sequences.
        """
        data, bin_cols = self.cut(self.data, quantiles=quantiles)
        one_hot = self.one_hot(data)
        # one_hot already has node from get_dummies; drop from other to avoid duplicate columns
        drop_cols = ["node"] if not keep_node and "node" in data.columns else []
        other = data.drop(columns=drop_cols + ["node"], errors="ignore")
        concat = pd.concat((one_hot, other), axis=1)
        nobins = list(set(concat.columns) - set(bin_cols))
        result = concat[nobins]
        if not keep_node and "node" in result.columns:
            result = result.drop(columns=["node"], errors="ignore")
        return result, concat[bin_cols]
