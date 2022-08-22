import inspect
from typing import Union
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, BatchSampler

from bes_data.sample_data import sample_data_dir
from bes_ml.base.data import MultiSourceDataset

try:
    from ..base.train_base import _Trainer_Base
    from ...bes_data.confinement_data_tools.dataset import TurbulenceDataset
    from bes_ml.base.sampler import RandomBatchSampler
except ImportError:
    from bes_ml.base.train_base import _Trainer_Base
    from bes_data.confinement_data_tools.dataset import TurbulenceDataset
    from bes_ml.base.sampler import RandomBatchSampler


class Trainer(_Trainer_Base):

    # __init__ must have exact copy of all kwargs from parent class
    def __init__(
        self,
        # subclass parameters
        # max_elms: int = None,  # limit ELMs
        # log_time: bool = False,  # if True, use log(time_to_elm_onset)
        # inverse_weight_label: bool = False,  # must be False if log_time is False
        # parent class `_Trainer` parameters
        data_location: str = sample_data_dir / 'turbulence_data', #location of stored data
        output_dir: Union[Path,str] = 'run_dir',  # path to output dir.
        results_file: str = 'results.yaml',  # output training results
        log_file: str = 'log.txt',  # output log file
        inputs_file: str = 'inputs.yaml',  # save inputs to yaml
        test_data_file: str = 'test_data.pkl',  # if None, do not save test data (can be large)
        checkpoint_file: str = 'checkpoint.pytorch',  # pytorch save file; if None, do not save
        # data parameters for velocimetry specific.
        dataset_to_ram: bool = True, # Load datasets to ram
        export_onnx: bool = False,  # export ONNX format
        device: str = 'auto',  # auto (default), cpu, cuda, or cuda:X
        num_workers: int = 1,  # number of subprocess workers for pytorch dataloader
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
        mlp_output_size: int = 4,
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

        self.is_regression = False
        self._set_regression_or_classification_defaults()

        # subclass attributes
        # self.max_elms = max_elms
        # self.log_time = log_time
        # self.inverse_weight_label = inverse_weight_label
        self.dataset_to_ram = dataset_to_ram
        self.turbulence_dataset = None

        self.make_model_and_set_device()

        self.finish_subclass_initialization()

    def train(self) -> None:
        self.results['scores_label'] = 'ROC'
        super().train()

    def _make_datasets(self) -> None:
        kwargs_for_data_class = self._create_data_class_inputs(self.__dict__)
        self.turbulence_dataset = TurbulenceDataset(**kwargs_for_data_class)
        train_set, valid_set = self.turbulence_dataset.train_test_split(self.fraction_validation, seed=42)
        if self.dataset_to_ram:
            # Load datasets into ram
            train_set.load_datasets()
            valid_set.load_datasets()

        self.train_dataset = train_set
        self.validation_dataset = valid_set

    def _make_data_loaders(self) -> None:
        self.train_data_loader = DataLoader(self.train_dataset,
                                            batch_size=None,  # must be disabled when using samplers
                                            sampler=BatchSampler(RandomBatchSampler(self.train_dataset,
                                                                                    self.batch_size,
                                                                                    self.signal_window_size),
                                                                 batch_size=self.batch_size,
                                                                 drop_last=True)
                                      )

        self.validation_data_loader = DataLoader(self.validation_dataset,
                                                 batch_size=None,  # must be disabled when using samplers
                                                 sampler=BatchSampler(RandomBatchSampler(self.validation_dataset,
                                                                                         self.batch_size,
                                                                                         self.signal_window_size),
                                                                      batch_size=self.batch_size,
                                                                      drop_last=True)
                                      )


    def _create_data_class_inputs(self, locals_copy: dict = None) -> dict:
        assert self.__class__ is not _Trainer_Base
        kwargs_for_data_class = {}
        for cls in [TurbulenceDataset, MultiSourceDataset]:
            class_parameters = inspect.signature(cls).parameters
            for parameter_name in class_parameters:
                if parameter_name in locals_copy:
                    kwargs_for_data_class[parameter_name] = locals_copy[parameter_name]
        return kwargs_for_data_class

    def _get_valid_indices(
        self,
        labels: np.ndarray = None,
        signals: np.ndarray = None,
    ) -> None:
        pass

    def _check_for_balanced_data(self, *args, **kwargs):
        pass

    def _get_data(self) -> None:
        pass

    def _save_test_data(self) -> None:
        #TODO: Implement test data saving for confinement classification
        pass
        # if self.turbulence_dataset is not None:
        #     self.turbulence_dataset.save_test_data()


if __name__=='__main__':
    model = Trainer(
        batch_size=32,
        minibatch_interval=50,
        # max_elms=5,
        fraction_validation=0.2,
        fraction_test=0.2,
        # log_time=True,
        # inverse_weight_label=True,
    )
    model.train()