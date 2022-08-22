from typing import Tuple
import dataclasses

import numpy as np

from bes_data.sample_data import sample_elm_data_file
try:
    from ..base.train_base import _Trainer_Base
    from ..base.data import ELM_Dataset
except ImportError:
    from bes_ml.base.train_base import _Trainer_Base
    from bes_ml.base.data import ELM_Dataset


@dataclasses.dataclass(eq=False)
class Trainer(_Trainer_Base):
    max_elms: int = None  # limit ELMs
    log_time: bool = False  # if True, use log(time_to_elm_onset)
    inverse_weight_label: bool = False  # must be False if log_time is False

    def __post_init__(self):
        super().__post_init__()

        self.is_regression = True
        self.is_classification = not self.is_regression

        self.make_model_and_set_device()

        self.finish_subclass_initialization()

    def _get_valid_indices(
        self,
        labels: np.ndarray = None,
        signals: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # indices for active elm times in each elm event
        active_elm_indices = np.nonzero(labels == 1)[0]
        active_elm_start_index = active_elm_indices[0]
        # concat on axis 0 (time dimension)
        valid_t0 = np.ones(active_elm_start_index-1, dtype=np.int32)
        valid_t0[-self.signal_window_size + 1:] = 0
        labels = np.arange(active_elm_start_index, 1, -1, dtype=float)
        signals = signals[:active_elm_start_index-1, :, :]
        if self.log_time:
            if np.any(labels == 0):
                assert False
            labels = np.log10(labels, dtype=float)
        return labels, signals, valid_t0

    def _check_for_balanced_data(self, *args, **kwargs):
        pass

    def _make_datasets(self) -> None:
        self.train_dataset = ELM_Dataset(
            *self.train_data[0:4], 
            signal_window_size = self.signal_window_size,
        )
        self.validation_dataset = ELM_Dataset(
            *self.validation_data[0:4], 
            signal_window_size = self.signal_window_size,
        )


if __name__=='__main__':
    model = Trainer(
        # dense_num_kernels=8,
        cnn_layer1_num_kernels = 8,
        cnn_layer2_num_kernels = 8,
        batch_size=64,
        n_epochs=2,
        minibatch_interval=50,
        fraction_validation=0.2,
        fraction_test=0.2,
        log_time=True,
        inverse_weight_label=True,
    )
    model.train()
