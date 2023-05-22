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
        self.mlp_output_size = 3
        # self.threshold = 0.5
        self.is_classification = True
        self.is_regression = not self.is_classification

        if self.inverse_weight_label and self.normalize_labels:
            assert False, "Invalid options"

        super().__post_init__()  # Trainer_Base.__post_init__()

    def _get_valid_indices(
        self,
        labels: np.ndarray = None,
        # signals: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid_t0 = np.zeros(labels.size, dtype=np.int32)  # size = n_pre_elm_phase
        last_signal_window_start_index = labels.size - self.signal_window_size - 1
        valid_t0[:last_signal_window_start_index+1] = 1
        assert valid_t0[last_signal_window_start_index] == 1  # last signal window start with pre-ELM label
        assert valid_t0[last_signal_window_start_index+1] == 0  # first invalid signal window start with active ELM label
        # assert signals.shape[0] == labels.size
        # assert signals.shape[0] == valid_t0.size
        if self.log_time:
            labels = np.log10(labels)
        return labels, valid_t0

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
        # num_workers=0,
        # pin_memory=False,
        # max_events = 10,
        data_location = '/global/homes/k/kevinsg/m3586/kgill/bes-ml/bes_data/sample_data/kgill_data/6x8_confinement_data_0c.hdf5',
        signal_window_size=1024,
        batch_size=256,
        # seed=1,
        fraction_test=0.17,
        # fraction_validation=0.3,
        n_epochs=400,
        do_train=True,
        # cnn_layer1_num_kernels=20,
        # cnn_layer1_kernel_spatial_size=3,
        # cnn_layer1_kernel_time_size=5,
        # cnn_layer1_maxpool_spatial_size=1,
        # cnn_layer1_maxpool_time_size=4,
        # cnn_layer2_num_kernels=20,
        # cnn_layer2_kernel_spatial_size=2,
        # cnn_layer2_kernel_time_size=5,
        # cnn_layer2_maxpool_spatial_size=1,
        # cnn_layer2_maxpool_time_size=4,
        fft_num_kernels=10,
        fft_subwindows=2,
        fft_nbins=2,
        fft_maxpool_freq_size=4,
        fft_maxpool_spatial_size=1,
        mlp_hidden_layers=(60,60),
        logger_hash = UNIQUE_IDENTIFIER,
        world_size = WORLD_SIZE,
        world_rank = WORLD_RANK,
        local_rank = LOCAL_RANK,
        # memory_diagnostics=True,
        # log_all_ranks = True,
        # weight_decay=0.0,
        # dropout_rate=0.2,
        learning_rate=0.0001,
        # clamp_signals=2.0,
        clip_signals=2.0,
        # optimizer_type='adam',
    )