import inspect
from pathlib import Path
from typing import Union
import dataclasses

from torch.utils.data import DataLoader, BatchSampler

from bes_data.sample_data import sample_data_dir
from bes_data.velocimetry_data_tools.dataset import VelocimetryDataset
try:
    from ..base.data import MultiSourceDataset
    from ..base.train_base import _Trainer_Base
    from ..base.sampler import RandomBatchSampler
except ImportError:
    from bes_ml.base.data import MultiSourceDataset
    from bes_ml.base.train_base import _Trainer_Base
    from bes_ml.base.sampler import RandomBatchSampler


@dataclasses.dataclass(eq=False)
class Trainer(_Trainer_Base):
    data_location: Union[Path,str] = sample_data_dir / 'velocimetry_data' #location of stored data
    dataset_to_ram: bool = True # Load datasets to ram
    sinterp: int = 1 # interpolation factor of BES signals

    # __init__ must have exact copy of all kwargs from parent class
    def __post_init__(self):

        self.mlp_output_size = 128
        assert isinstance(self.sinterp, int), f"sinterp must be type int. Got type {type(self.sinterp)}"
        super().__post_init__()

        self.is_regression = True
        self.is_classification = not self.is_regression
        self.velocimetry_dataset = None
        self.make_model_and_set_device()
        self.finish_subclass_initialization()

    def _get_valid_indices(self, *args, **kwargs) -> None:
        pass

    def _check_for_balanced_data(self, *args, **kwargs):
        pass

    def _make_datasets(self) -> None:
        kwargs_for_data_class = self._create_data_class_inputs(self.__dict__)
        self.velocimetry_dataset = VelocimetryDataset(**kwargs_for_data_class)
        train_set, valid_set, test_set = self.velocimetry_dataset.train_test_split(seed=42)

        if self.dataset_to_ram:
            # Load datasets into ram
            train_set.load_datasets()
            valid_set.load_datasets()

        self.train_dataset = train_set
        self.validation_dataset = valid_set
        self.test_dataset = test_set

    def _make_data_loaders(self) -> None:
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=None,  # must be disabled when using samplers
            sampler=BatchSampler(
                RandomBatchSampler(
                    self.train_dataset,
                    self.batch_size,
                    self.signal_window_size
                ),
                batch_size=self.batch_size,
                drop_last=True,
            )
        )
        self.validation_data_loader = DataLoader(
            self.validation_dataset,
            batch_size=None,  # must be disabled when using samplers
            sampler=BatchSampler(
                RandomBatchSampler(
                    self.validation_dataset,
                    self.batch_size,
                    self.signal_window_size
                ),
                batch_size=self.batch_size,
                drop_last=True,
            )
        )

    def _create_data_class_inputs(self, locals_copy: dict = None) -> dict:
        assert self.__class__ is not _Trainer_Base
        kwargs_for_data_class = {}
        for cls in [VelocimetryDataset, MultiSourceDataset]:
            class_parameters = inspect.signature(cls).parameters
            for parameter_name in class_parameters:
                if parameter_name in locals_copy:
                    kwargs_for_data_class[parameter_name] = locals_copy[parameter_name]
        return kwargs_for_data_class

    def _get_data(self) -> None:
        pass

    def _save_test_data(self) -> None:
        if self.test_dataset is not None:
            self.test_dataset.load_datasets()
            self.test_dataset.save(self.output_dir / self.test_data_file)


if __name__=='__main__':
    n_epochs = 50
    t_size = 2
    s_size = 3
    sws = 2
    dense_num = 0
    cnn_l1_num = 512
    model = Trainer(
        data_location='/home/jazimmerman/PycharmProjects/bes-edgeml-models/bes-edgeml-work/velocimetry/data/',
        output_dir=f'/home/jazimmerman/PycharmProjects/bes-edgeml-models/bes-edgeml-work/velocimetry/'
                   f'cnn{cnn_l1_num}_spatial{s_size}_time{t_size}_dense{dense_num}_sws{sws}_epochs{n_epochs}',
        sinterp=5,
        fraction_test=0.1,
        fraction_validation=0.1,
        minibatch_interval=1000,
        dense_num_kernels=dense_num,
        batch_size=64,
        signal_window_size=sws,
        cnn_layer1_kernel_time_size=t_size,
        cnn_layer1_kernel_spatial_size=s_size,
        cnn_layer1_maxpool_time_size=1,
        cnn_layer2_kernel_time_size=t_size,
        cnn_layer2_kernel_spatial_size=s_size,
        cnn_layer2_maxpool_time_size=1,
        cnn_layer1_num_kernels=cnn_l1_num,
        cnn_layer2_num_kernels=cnn_l1_num // 2,
        mlp_layer1_size=(cnn_l1_num + dense_num) // 4,
        mlp_layer2_size=(((cnn_l1_num + dense_num) // 4) + 128) // 2, # halfway between mlp_layer1 and mlp_output
        n_epochs=n_epochs,
        optimizer_type='adam',
        weight_decay=0.0
    )
    model.train()
