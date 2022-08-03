import numpy as np
import torch.utils.data


class ELM_Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        signals: np.ndarray = None, 
        labels: np.ndarray = None, 
        sample_indices: np.ndarray = None, 
        window_start: np.ndarray = None,
        signal_window_size: int = None,
        prediction_horizon: int = None,  # =0 for time-to-ELM regression; >=0 for classification prediction
    ) -> None:
        self.signals = signals
        self.labels = labels
        self.sample_indices = sample_indices
        self.window_start = window_start
        self.signal_window_size = signal_window_size
        self.prediction_horizon = prediction_horizon if prediction_horizon is not None else 0

    def __len__(self):
        return self.sample_indices.size

    def __getitem__(self, idx: int):
        time_idx = self.sample_indices[idx]
        # BES signal window data
        signal_window = self.signals[
            time_idx : time_idx + self.signal_window_size
        ]
        signal_window = signal_window[np.newaxis, ...]
        signal_window = torch.as_tensor(signal_window, dtype=torch.float32)
        # label for signal window
        label = self.labels[
            time_idx
            + self.signal_window_size
            + self.prediction_horizon
            - 1
        ]
        label = torch.as_tensor(label)

        return signal_window, label


def elm_data_loader(
    dataset: ELM_Dataset = None,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 1,
) -> torch.utils.data.DataLoader:
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return data_loader
