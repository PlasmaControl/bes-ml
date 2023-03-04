import dataclasses
import numpy as np
import torch

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
        assert labels[active_elm_start_index-1] == 0  # last pre-ELM element
        assert labels[active_elm_start_index] == 1  # first active ELM element
        valid_t0 = np.zeros(labels.size, dtype=np.int32)  # size = n_pre_elm_phase
        # assert valid_t0.size == active_elm_start_index  # valid_t0 is length of pre-ELM phase
        last_signal_window_start_index = active_elm_start_index - self.signal_window_size - 1
        valid_t0[:last_signal_window_start_index+1] = 1
        assert valid_t0[last_signal_window_start_index] == 1  # last signal window start with pre-ELM label
        assert valid_t0[last_signal_window_start_index+1] == 0  # first invalid signal window start with active ELM label
        if self.pre_elm_size:
            first_signal_window_start_index = last_signal_window_start_index - self.pre_elm_size + 1
            if first_signal_window_start_index > 0:
                valid_t0[:first_signal_window_start_index] = 0
                assert valid_t0[first_signal_window_start_index - 1] == 0
                assert valid_t0[first_signal_window_start_index] == 1
        # labels = np.arange(active_elm_start_index, 0, -1, dtype=self.label_type)
        labels = np.zeros(labels.size, dtype=self.label_type)
        labels[0:active_elm_start_index] = np.arange(active_elm_start_index, 0, -1, dtype=self.label_type)
        labels[active_elm_start_index:] = np.nan
        # assert labels.size == active_elm_start_index
        assert np.nanmin(labels) == 1
        assert np.nanmax(labels) == active_elm_start_index
        assert np.all(labels[np.isfinite(labels)]>0)
        valid_labels = labels[valid_t0==1]
        assert np.all(np.isfinite(valid_labels))
        assert np.all(valid_labels>0)
        # signals = signals[0:active_elm_start_index, :, :]
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

    # def _apply_label_normalization(
    #     self, 
    #     labels: np.ndarray = None,
    #     valid_indices: np.ndarray = None,
    # ) -> np.ndarray:
    #     raw_label_min = labels[valid_indices+self.signal_window_size].min()
    #     raw_label_max = labels[valid_indices+self.signal_window_size].max()
    #     self.results['raw_label_min'] = raw_label_min.item()
    #     self.results['raw_label_max'] = raw_label_max.item()
    #     self.logger.info(f"  Raw label min/max: {raw_label_min:.4e}, {raw_label_max:.4e}")
    #     if self.normalize_labels:
    #         self.logger.info(f"  Normalizing labels to min/max = -/+ 1")
    #         label_range = raw_label_max - raw_label_min
    #         labels = ((labels - raw_label_min) / label_range - 0.5) * 2
    #     return labels


if __name__=='__main__':
    Trainer(
        # data_location= '/global/homes/d/drsmith/ml/scratch/data/labeled_elm_events.hdf5',
        # max_elms=5,
        signal_window_size=128,
        batch_size=128,
        fraction_test=0,
        n_epochs=1,
        fir_num_kernels=8,
        fir_cutoffs=[
            [0.06, 0.16],
            [0.25, 0.35],
        ],
        dense_num_kernels=8,
        # standardize_fft=False,
        # standardize_signals=False,
        # clip_sigma=None,
        # normalize_labels=False,
        debug=True,
        fft_num_kernels=8,
        fft_subwindows=2,
        fft_nbins=2,
        # memory_diagnostics=True,
        # minibatch_print_interval=50,
        do_train=True,
    )
