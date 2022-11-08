from typing import Tuple
import dataclasses
import os
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp

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
    normalize_labels: bool = True  # if True, normalize labels to max/min = +/- 1
    pre_elm_size: int = None  # maximum pre-ELM window in time frames

    def __post_init__(self):

        self.is_regression = True
        self.is_classification = not self.is_regression

        if self.inverse_weight_label and self.normalize_labels:
            assert False, "Invalid options"

        self.raw_label_minmax = None

        super().__post_init__()  # _Base_Trainer.__post_init__()

    def _get_valid_indices(
        self,
        labels: np.ndarray = None,
        signals: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        labels = np.arange(n_pre_elm_phase, 0, -1, dtype=np.float32)
        assert labels.size == n_pre_elm_phase
        assert labels.min() == 1
        assert labels.max() == n_pre_elm_phase
        assert np.all(labels>0)
        signals = signals[0:n_pre_elm_phase, :, :]
        assert signals.shape[0] == labels.size
        assert signals.shape[0] == valid_t0.size
        if self.log_time:
            labels = np.log10(labels)
            assert labels.min() == 0
        return labels, signals, valid_t0

    def _apply_loss_weight(
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
        labels: torch.Tensor = None,
        valid_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.normalize_labels:
            if self.raw_label_minmax is None:
                initialize = True
                self.raw_label_minmax = [
                    labels[valid_indices+self.signal_window_size].min().item(), 
                    labels[valid_indices+self.signal_window_size].max().item(),
                ]
                self.results['raw_label_minmax'] = self.raw_label_minmax
                if self.log_time:
                    assert self.raw_label_minmax[0] == 0
                else:
                    assert self.raw_label_minmax[0] == 1
            else:
                initialize = False
            self.logger.info(
                f"  Normalizing labels[valid_t0] to min/max = -/+ 1 " +
                f"with raw min/max {self.raw_label_minmax[0]:.4e} {self.raw_label_minmax[1]:.4e}"
            )
            label_range = self.raw_label_minmax[1] - self.raw_label_minmax[0]
            labels = ((labels - self.raw_label_minmax[0]) / label_range - 0.5) * 2
            if initialize:
                assert labels[valid_indices+self.signal_window_size].min() == -1
                assert labels[valid_indices+self.signal_window_size].max() == 1
        return labels


def main(rank: int = None, world_size: int = None, **kwargs):
    Trainer(
        dense_num_kernels=8,
        # max_elms=5,
        n_epochs=10,
        fraction_test=0,
        fraction_validation=0,
        seed = 0,
        bad_elm_indices_csv=True,  # read bad ELMs from CSV in bes_data.elm_data_tools
        # pre_elm_size=2000,
        # device='cpu',
        local_rank=rank,
        world_size=world_size,
        do_train=True,
        log_all_ranks=True,
        logger_hash=str(int(datetime.now().timestamp())),
        **kwargs,
    )


def main_mp_spawn(world_size: int = None):
    if 'MASTER_ADDR' not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if 'MASTER_PORT' not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if world_size is None:
        world_size = int(os.environ.get('$SLURM_NTASKS', 1))
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)


if __name__=='__main__':
    main()
    # main_mp_spawn(world_size=4)