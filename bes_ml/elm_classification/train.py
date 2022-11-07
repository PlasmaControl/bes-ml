from typing import Tuple
import dataclasses
import os

import numpy as np
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
    prediction_horizon: int = 200  # prediction horizon in time samples
    threshold: float = 0.5  # threshold for binary classification
    oversample_active_elm: bool = False  # if True, oversample active ELMs to balance data
    one_hot_encoding: bool = False  # if True, use 2-output, one-hot encoding

    def __post_init__(self):
        self.is_classification = True
        self.is_regression = not self.is_classification

        if self.one_hot_encoding:
            self.mlp_output_size = 2

        super().__post_init__()  # _Base_Trainer.__post_init__()

    def _get_valid_indices(
        self,
        labels: np.ndarray = None,
        signals: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # indices for active elm times in each elm event
        active_elm_indices = np.nonzero(labels == 1)[0]
        active_elm_start_index = active_elm_indices[0]
        # `t0` is first index (or earliest time, or trailing time point) for signal window
        # `valid_t0` denotes valid `t0` time points for signal window
        # initialize to zeros
        valid_t0 = np.zeros(labels.shape, dtype=np.int32)
        # largest `t0` index with signal window in pre-ELM period
        largest_t0_index_for_pre_elm_period = active_elm_start_index - self.signal_window_size
        if largest_t0_index_for_pre_elm_period < 0:
            # insufficient pre-elm period for signal window size
            return None
        assert labels[largest_t0_index_for_pre_elm_period + (self.signal_window_size-1)    ] == 0
        assert labels[largest_t0_index_for_pre_elm_period + (self.signal_window_size-1) + 1] == 1
        # `t0` time points up to `largest_t0` are valid
        valid_t0[0:largest_t0_index_for_pre_elm_period+1] = 1
        assert valid_t0[largest_t0_index_for_pre_elm_period    ] == 1
        assert valid_t0[largest_t0_index_for_pre_elm_period + 1] == 0
        # labels after ELM onset should be active ELM, even if in post-ELM period
        last_label_for_active_elm_in_pre_elm_signal = (
            largest_t0_index_for_pre_elm_period
            + (self.signal_window_size - 1)
            + self.prediction_horizon
        )
        labels[ active_elm_start_index : last_label_for_active_elm_in_pre_elm_signal+1 ] = 1
        assert labels[last_label_for_active_elm_in_pre_elm_signal] == 1
        return labels, signals, valid_t0

    def _check_for_balanced_data(
        self, 
        packaged_labels: np.ndarray = None,
        packaged_valid_t0_indices: np.ndarray = None,
        oversample_active_elm: bool = False,
    ) -> np.ndarray:
        packaged_label_indices_for_valid_t0 = (
            packaged_valid_t0_indices 
            + (self.signal_window_size-1)
            + self.prediction_horizon
            )
        packaged_labels_for_valid_t0 = packaged_labels[packaged_label_indices_for_valid_t0]
        n_active_elm = np.count_nonzero(packaged_labels_for_valid_t0)
        n_inactive_elm = np.count_nonzero(packaged_labels_for_valid_t0-1)
        active_elm_fraction = n_active_elm/(n_active_elm+n_inactive_elm)
        self.logger.info(f"  Count of inactive ELM labels: {n_inactive_elm}")
        self.logger.info(f"  Count of active ELM labels: {n_active_elm}")
        self.logger.info(f"  % active: {active_elm_fraction*1e2:.1f} %")
        min_active_elm_fraction = 0.25
        if oversample_active_elm and active_elm_fraction < min_active_elm_fraction:
            oversample_factor = int(min_active_elm_fraction * n_inactive_elm / (n_active_elm*(1-min_active_elm_fraction)))+1
            self.logger.info(f"  Oversample active ELM factor: {oversample_factor}")
            assert oversample_factor >= 1
            packaged_active_elm_label_indices_for_valid_t0 = packaged_label_indices_for_valid_t0[
                packaged_labels[packaged_label_indices_for_valid_t0] == 1
            ]
            packaged_active_elm_valid_t0_indices = (
                packaged_active_elm_label_indices_for_valid_t0
                - (self.signal_window_size-1)
                - self.prediction_horizon
            )
            for i in np.arange(oversample_factor-1):
                packaged_valid_t0_indices = np.append(
                    packaged_valid_t0_indices,
                    packaged_active_elm_valid_t0_indices,
                )
            packaged_label_indices_for_valid_t0 = (
                packaged_valid_t0_indices
                + (self.signal_window_size-1)
                + self.prediction_horizon
                )
            packaged_labels_for_valid_t0 = packaged_labels[packaged_label_indices_for_valid_t0]
            n_active_elm = np.count_nonzero(packaged_labels_for_valid_t0)
            n_inactive_elm = np.count_nonzero(packaged_labels_for_valid_t0-1)
            active_elm_fraction = n_active_elm/(n_active_elm+n_inactive_elm)
            self.logger.info(f"  New count of inactive ELM labels: {n_inactive_elm}")
            self.logger.info(f"  New count of active ELM labels: {n_active_elm}")
            self.logger.info(f"  New % active: {active_elm_fraction*1e2:.1f} %")
        return packaged_valid_t0_indices


def main(rank: int = None, world_size: int = None):
    Trainer(
        dense_num_kernels=8,
        max_elms=5,
        # n_epochs=20,
        fraction_test=0,
        # fraction_validation=0,
        seed = 0,
        bad_elm_indices_csv=True,  # read bad ELMs from CSV in bes_data.elm_data_tools
        # ELM classification parameters,
        prediction_horizon=100,
        oversample_active_elm=True,
        one_hot_encoding=True,
        do_train=True,
        world_rank=rank,
        world_size=world_size,
        logger_name=__name__+str(np.random.randint(1e12)),
    )


def ddp_main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)


if __name__=='__main__':
    # main()
    ddp_main()