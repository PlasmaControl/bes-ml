import os
import socket
from pathlib import Path
from datetime import timedelta
import dataclasses

import numpy as np
import torch
import torch.distributed

try:
    from ..base.train_base import Trainer_Base
    from .confinement_mode_data import Confinement_Mode_Data
except ImportError:
    from bes_ml.base.train_base import Trainer_Base
    from bes_ml.confinement_classification.confinement_mode_data import Confinement_Mode_Data

@dataclasses.dataclass(eq=False)
class Trainer(
    Confinement_Mode_Data,  # confinement mode data
    Trainer_Base,  # training and output
):
    log_time: bool = False  # if True, use label = log(time_to_elm_onset)
    inverse_weight_label: bool = False  # if True, weight losses by 1/label
    normalize_labels: bool = False  # if True, normalize labels to max/min = +/- 1

    def __post_init__(self):
        self.mlp_output_size = 4
        self.is_classification = True
        self.is_regression = not self.is_classification

        if self.inverse_weight_label and self.normalize_labels:
            assert False, "Invalid options"

        super().__post_init__()  # Trainer_Base.__post_init__()

    def _get_valid_indices(
        self,
        labels: np.ndarray = None,
        signals: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid_t0 = np.zeros(labels.size, dtype=np.int32)  # size = n_pre_elm_phase
        last_signal_window_start_index = labels.size - self.signal_window_size - 1
        valid_t0[:last_signal_window_start_index+1] = 1
        assert valid_t0[last_signal_window_start_index] == 1  # last signal window start with pre-ELM label
        assert valid_t0[last_signal_window_start_index+1] == 0  # first invalid signal window start with active ELM label
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

if __name__=='__main__':

    WORLD_SIZE = int(os.environ.get('SLURM_NTASKS'))
    WORLD_RANK = int(os.environ.get('SLURM_PROCID'))
    LOCAL_RANK = int(os.environ.get('SLURM_LOCALID'))
    UNIQUE_IDENTIFIER = os.environ.get('UNIQUE_IDENTIFIER')

    assert LOCAL_RANK <= WORLD_RANK
    assert WORLD_RANK < WORLD_SIZE

    torch.distributed.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        world_size=WORLD_SIZE,
        rank=WORLD_RANK,
    )

    Trainer(
        # num_workers=1,
        # pin_memory=False,
        max_events = 100,
        data_location = '/global/homes/k/kevinsg/m3586/kgill/bes-ml/bes_data/sample_data/kgill_data/balanced_6x8/confinement_data.hdf5',
        signal_window_size=128,
        batch_size=256,
        fraction_test=0,
        fraction_validation=0.5,
        n_epochs=2,
        do_train=True,
        dense_num_kernels=8,
        fft_num_kernels=8,
        fft_subwindows=2,
        fft_nbins=2,
        logger_hash = UNIQUE_IDENTIFIER,
        world_size = WORLD_SIZE,
        world_rank = WORLD_RANK,
        local_rank = LOCAL_RANK,
        # log_all_ranks = True,
    )