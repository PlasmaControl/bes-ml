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

    # __init__ must have exact copy of all kwargs from parent class
    def __post_init__(self):

        self.mlp_output_size = 128

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
    model = Trainer(
        dense_num_kernels=8,
        batch_size=64,
        n_epochs=2,
        minibatch_interval=50,
        fraction_validation=0.2,
        fraction_test=0.2,
    )
    model.train()
