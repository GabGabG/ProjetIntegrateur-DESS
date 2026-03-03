import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset


class SpeckleDataset(Dataset):

    def __init__(self, data_root: str, metadata_df: pd.DataFrame, chunk_size: int = np.inf,
                 load_all_in_memory: bool = False):
        self.__data_root = data_root
        self.__metadata_df = metadata_df
        self.__chunk_size = chunk_size
        self.__all_data = None
        self.__all_Ts = None
        self.__all_tau_cs = None
        if load_all_in_memory:
            self.__all_data, self.__all_Ts, self.__all_tau_cs = self.__read_all_files()
            self.__all_data = torch.from_numpy(np.ndarray(self.__all_data)).float()
            self.__all_Ts = torch.from_numpy(np.ndarray(self.__all_Ts)).float()
            self.__all_tau_cs = torch.from_numpy(np.ndarray(self.__all_tau_cs)).float()

    def __len__(self) -> int:
        return len(self.__metadata_df)

    def __read_all_files(self) -> tuple[list, list, list]:
        all_data = []
        all_Ts = []
        all_tau_cs = []
        for i in range(len(self)):
            seq, T, tau_c = self.__read_single_file(i)
            all_data.append(seq)
            all_Ts.append(T)
            all_tau_cs.append(tau_c)
        return all_data, all_Ts, all_tau_cs

    def __read_single_file(self, idx: int) -> tuple[np.ndarray, float, float]:
        line = self.__metadata_df.iloc[idx]
        T = line["Integration time"]
        tau_c = line["Correlation time"]
        path = line["File path"]
        path = os.path.join(self.__data_root, path)
        seq = np.load(path)
        return seq, T, tau_c

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.__all_data is not None:
            T = self.__all_Ts[idx]
            tau_c = self.__all_tau_cs[idx]
            seq = self.__all_data[idx]
        else:
            seq, T, tau_c = self.__read_single_file(idx)
            seq = torch.from_numpy(seq).float()
            T = torch.tensor([T], dtype=torch.float32)
            tau_c = torch.tensor(tau_c, dtype=torch.float32)
        l_seq = seq.shape[0]
        if l_seq > self.__chunk_size:
            start = np.random.randint(0, l_seq - self.__chunk_size)
            seq = seq[start:start + self.__chunk_size]
        seq = seq.unsqueeze(1)  # Shape (chunk size, 1, H, W), 1 pour channel
        return (seq, T), tau_c


class MetadataSplitter:

    def __init__(self, full_metadata: pd.DataFrame):
        self.__full_metadata_df = full_metadata

    @classmethod
    def from_csv(cls, csv_path: str, *read_args, **read_kwargs) -> "MetadataSplitter":
        full_metadata = pd.read_csv(csv_path, *read_args, **read_kwargs)
        return cls(full_metadata)

    @property
    def metadata(self) -> pd.DataFrame:
        return self.__full_metadata_df

    def __len__(self) -> int:
        return len(self.__full_metadata_df)

    def __call__(self, train_ratio: float, random_state: int = 42) -> tuple["MetadataSplitter", "MetadataSplitter"]:
        n_total = len(self)
        n_train = int(n_total * train_ratio)
        train_df = self.__full_metadata_df.sample(n_train, replace=False, random_state=random_state)
        val_df = self.__full_metadata_df.drop(train_df.index)
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        train_df = MetadataSplitter(train_df)
        val_df = MetadataSplitter(val_df)
        return train_df, val_df
