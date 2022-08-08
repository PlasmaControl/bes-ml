from typing import Tuple, Union
from pathlib import Path

import numpy as np

from bes_data.sample_data import sample_elm_data_file
try:
    from ..base.train_base import _Trainer_Base
except ImportError:
    from bes_ml.base.train_base import _Trainer_Base


class Trainer(_Trainer_Base):

    # __init__ must have exact copy of all kwargs from parent class
    def __init__(
        self,
        # subclass parameters
        max_elms: int = None,  # limit ELMs
        log_time: bool = False,  # if True, use log(time_to_elm_onset)
        inverse_weight_label: bool = False,  # must be False if log_time is False
        # parent class `_Trainer` parameters
        data_location: Union[Path, str] = sample_elm_data_file,  # path to data file
        output_dir: Union[Path,str] = 'run_dir',  # path to output dir.
        results_file: str = 'results.yaml',  # output training results
        log_file: str = 'log.txt',  # output log file
        inputs_file: str = 'inputs.yaml',  # save inputs to yaml
        test_data_file: str = 'test_data.pkl',  # if None, do not save test data (can be large)
        checkpoint_file: str = 'checkpoint.pytorch',  # pytorch save file; if None, do not save
        export_onnx: bool = False,  # export ONNX format
        device: str = 'auto',  # auto (default), cpu, cuda, or cuda:X
        num_workers: int = 0,  # number of subprocess workers for pytorch dataloader
        n_epochs: int = 2,  # training epochs
        batch_size: int = 64,  # power of 2, like 16-128
        minibatch_interval: int = 2000,  # print minibatch info
        signal_window_size: int = 128,  # power of 2, like 32-512
        fraction_validation: float = 0.1,  # fraction of dataset for validation
        fraction_test: float = 0.15,  # fraction of dataset for testing
        optimizer_type: str = 'adam',  # adam (default) or sgd
        sgd_momentum: float = 0.0,  # momentum for SGD optimizer
        sgd_dampening: float = 0.0,  # dampening for SGD optimizer
        learning_rate: float = 1e-3,  # optimizer learning rate
        weight_decay: float = 5e-3,  # optimizer L2 regularization factor
        batches_per_print: int = 5000,  # train/validation batches per print update
        # model parameters
        # inputs for `Multi_Features_Model` class`
        mlp_layer1_size: int = 32,  # multi-layer perceptron (mlp)
        mlp_layer2_size: int = 16,
        mlp_output_size: int = 1,
        negative_slope: float = 1e-3,  # relu negatuve slope
        dropout_rate: float = 0.1,
        # inputs for `*Features` classes
        spatial_maxpool_size: int = 1,  # 1 (default, no spatial maxpool), 2, or 4
        time_interval: int = 1,  # time domain slice interval (i.e. time[::interval])
        subwindow_size: int = -1,  # power of 2, or -1 (default) for full signal window
        dense_num_kernels: int = 8,
        cnn_layer1_num_kernels: int = 0,
        cnn_layer1_kernel_time_size: int = 7,
        cnn_layer1_kernel_spatial_size: int = 3,
        cnn_layer1_maxpool_time_size: int = 2,
        cnn_layer1_maxpool_spatial_size: int = 2,
        cnn_layer2_num_kernels: int = 0,
        cnn_layer2_kernel_time_size: int = 7,
        cnn_layer2_kernel_spatial_size: int = 3,
        cnn_layer2_maxpool_time_size: int = 2,
        cnn_layer2_maxpool_spatial_size: int = 1,
        fft_num_kernels: int = 0,
        fft_nbins: int = 4,
        dct_num_kernels: int = 0,
        dct_nbins: int = 2,
        dwt_num_kernels: int = 0,
        dwt_wavelet: str = 'db4',
        dwt_level: int = -1,
    ) -> None:

        # validate inputs
        self._validate_subclass_inputs()

        # construct inputs for parent class
        locals_copy = locals().copy()
        kwargs_for_parent_class = self._create_parent_class_inputs(locals_copy=locals_copy)

        # init parent class
        super().__init__(**kwargs_for_parent_class)

        # save and print inputs
        self._print_inputs(
            locals_copy=locals_copy, 
            logger=self.logger,
        )
        self._save_inputs_to_yaml(
            locals_copy=locals_copy,
            filename=self.output_dir/self.inputs_file,
        )

        self.is_regression = True
        self._set_regression_or_classification_defaults()

        # subclass attributes
        self.max_elms = max_elms
        self.log_time = log_time
        self.inverse_weight_label = inverse_weight_label

        self.make_model_and_device()

        self._finish_subclass_initialization()

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

    def train(self):
        self.results['scores_label'] = 'R2'
        super().train()


if __name__=='__main__':
    model = Trainer(
        batch_size=32, 
        minibatch_interval=50, 
        fraction_validation=0.2,
        fraction_test=0.2,
        log_time=True,
        inverse_weight_label=True,
    )
    model.train()