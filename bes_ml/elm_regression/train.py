from typing import Tuple
import dataclasses

import numpy as np
import torch

try:
    from ..base.elm_data import _ELM_Data_Base
    from ..base.models import _Multi_Features_Model_Dataclass
    from ..base.train_base import _Base_Trainer
except ImportError:
    from bes_ml.base.elm_data import _ELM_Data_Base
    from bes_ml.base.models import _Multi_Features_Model_Dataclass
    from bes_ml.base.train_base import _Base_Trainer


@dataclasses.dataclass(eq=False)
class Trainer(
    _ELM_Data_Base,  # ELM data
    _Multi_Features_Model_Dataclass,  # NN model
    _Base_Trainer,  # training and output
):
    # parameters for ELM regression task
    log_time: bool = False  # if True, use label = log(time_to_elm_onset)
    inverse_weight_label: bool = False  # if True, weight losses by 1/label
    normalize_labels: bool = False  # if True, normalize labels to min/max = +/- 1

    def __post_init__(self):

        self.is_regression = True
        self.is_classification = not self.is_regression

        self.raw_label_minmax = None

        super().__post_init__()  # _Base_Trainer.__post_init__()

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
        labels = np.arange(active_elm_start_index, 1, -1, dtype=np.float32)
        signals = signals[:active_elm_start_index-1, :, :]
        if self.log_time:
            assert np.all(labels > 0)
            labels = np.log10(labels)
        return labels, signals, valid_t0

    def _apply_loss_weight(self, losses: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.inverse_weight_label:
            return torch.div(losses, labels)
        else:
            return losses

    def _apply_label_normalization(self, labels: torch.Tensor = None) -> torch.Tensor:
        if self.normalize_labels:
            if self.raw_label_minmax is None:
                self.raw_label_minmax = [labels.min().item(), labels.max().item()]
                self.results['raw_label_minmax'] = self.raw_label_minmax
            self.logger.info(f"  Normalizing labels to min/max = -/+ 1 with raw min/max {self.raw_label_minmax[0]:.4e} {self.raw_label_minmax[1]:.4e}")
            label_range = self.raw_label_minmax[1] - self.raw_label_minmax[0]
            labels = ((labels - self.raw_label_minmax[0]) / label_range - 0.5) * 2
        return labels



if __name__=='__main__':
    model = Trainer(
        dense_num_kernels=8,
        n_epochs=2,
        fraction_validation=0.2,
        fraction_test=0.2,
        normalize_labels=True,
        normalize_signals=True,
    )
    model.train()
