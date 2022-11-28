import dataclasses
import numpy as np
import torch
import torch.distributed

try:
    from ..base.train_base import Trainer_Base
    from ..base.elm_data import ELM_Data
except ImportError:
    from bes_ml.base.train_base import Trainer_Base
    from bes_ml.base.elm_data import ELM_Data


@dataclasses.dataclass(eq=False)
class Trainer(
    ELM_Data,  # ELM data
    Trainer_Base,  # training and output
):
    log_time: bool = False  # if True, use label = log(time_to_elm_onset)
    inverse_weight_label: bool = False  # if True, weight losses by 1/label
    normalize_labels: bool = True  # if True, normalize labels to max/min = +/- 1
    pre_elm_size: int = None  # maximum pre-ELM window in time frames

    def __post_init__(self):

        self.is_regression = True
        self.is_classification = not self.is_regression

        if self.inverse_weight_label and self.normalize_labels:
            assert False, "Invalid options"

        super().__post_init__()  # Trainer_Base.__post_init__()

    def _get_valid_indices(
        self,
        labels: np.ndarray = None,
        signals: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        active_elm_indices = np.nonzero(labels == 1)[0]
        active_elm_start_index = active_elm_indices[0]  # first active ELM index
        n_pre_elm_phase = active_elm_start_index  # length of pre-ELM phase
        assert labels[n_pre_elm_phase-1] == 0  # last pre-ELM element
        assert labels[n_pre_elm_phase] == 1  # first active ELM element
        valid_t0 = np.ones(n_pre_elm_phase, dtype=np.int32)  # size = n_pre_elm_phase
        assert valid_t0.size == n_pre_elm_phase  # valid_t0 is length of pre-ELM phase
        last_signal_window_start_index = n_pre_elm_phase - self.signal_window_size - 1
        valid_t0[last_signal_window_start_index+1:] = 0
        assert valid_t0[last_signal_window_start_index] == 1  # last signal window start with pre-ELM label
        assert valid_t0[last_signal_window_start_index+1] == 0  # first invalid signal window start with active ELM label
        if self.pre_elm_size:
            first_signal_window_start_index = last_signal_window_start_index - self.pre_elm_size + 1
            if first_signal_window_start_index < 0:
                first_signal_window_start_index = 0
            valid_t0[0:first_signal_window_start_index + 1 - 1] = 0
            # valid_t0 = valid_t0[first_signal_window_start_index:]
            assert valid_t0[first_signal_window_start_index] == 1
            assert valid_t0[first_signal_window_start_index - 1] == 0
            n_valid_t0 = np.min([last_signal_window_start_index+1, self.pre_elm_size])
            assert np.count_nonzero(valid_t0) == n_valid_t0
        labels = np.arange(n_pre_elm_phase, 0, -1, dtype=self.label_type)
        assert labels.size == n_pre_elm_phase
        assert labels.min() == 1
        assert labels.max() == n_pre_elm_phase
        assert np.all(labels>0)
        signals = signals[0:n_pre_elm_phase, :, :]
        assert signals.shape[0] == labels.size
        assert signals.shape[0] == valid_t0.size
        if self.log_time:
            labels = np.log10(labels)
        return labels, signals, valid_t0

    def _apply_label_weights(
            self,
            losses: torch.Tensor,
            labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.inverse_weight_label:
            return torch.div(losses, labels)
        else:
            return losses

    def _apply_label_normalization(
        self, 
        labels: np.ndarray = None,
        valid_indices: np.ndarray = None,
    ) -> np.ndarray:
        raw_label_min = labels[valid_indices+self.signal_window_size].min()
        raw_label_max = labels[valid_indices+self.signal_window_size].max()
        if self.is_ddp:
            self.logger.info(f"  Before reduce min {raw_label_min:.4f} max {raw_label_max:.4f}")
            tmp = torch.tensor(raw_label_min, dtype=torch.float)
            torch.distributed.all_reduce(
                tmp,
                op=torch.distributed.ReduceOp.MIN,
            )
            raw_label_min = tmp.numpy()
            tmp = torch.tensor(raw_label_max, dtype=torch.float)
            torch.distributed.all_reduce(
                tmp,
                op=torch.distributed.ReduceOp.MAX,
            )
            raw_label_max = tmp.numpy()
            self.logger.info(f"  After reduce min {raw_label_min:.4f} max {raw_label_max:.4f}")
        self.results['raw_label_min'] = raw_label_min
        self.results['raw_label_max'] = raw_label_max
        self.logger.info(f"  Raw label min/max: {raw_label_min:.4e}, {raw_label_max:.4e}")
        if self.normalize_labels:
            self.logger.info(f"  Normalizing labels to min/max = -/+ 1")
            label_range = raw_label_max - raw_label_min
            labels = ((labels - raw_label_min) / label_range - 0.5) * 2
        return labels


if __name__=='__main__':
    Trainer(
        dense_num_kernels=8,
        # max_elms=5,
        batch_size=16,
        n_epochs=2,
        fraction_test=0,
        # pre_elm_size=2000,
        minibatch_print_interval=100,
        do_train=True,
    )
