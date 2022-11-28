import inspect
from typing import Union
from pathlib import Path
import dataclasses

import numpy as np
from torch.utils.data import DataLoader, BatchSampler

from bes_data.sample_data import sample_data_dir
from bes_data.confinement_data_tools.dataset import ConfinementDataset

try:
    from ..base.train_base import Trainer_Base
    from ..base.sampler import RandomBatchSampler
    from ..base.models import Multi_Features_Model_Dataclass
    from ..base.multisource_data import MultiSourceDataset, _MultiSource_Data_Base
except ImportError:
    from bes_ml.base.train_base import _Base_Trainer
    from bes_ml.base.sampler import RandomBatchSampler
    from bes_ml.base.models import _Multi_Features_Model_Dataclass
    from bes_ml.base.multisource_data import MultiSourceDataset, _MultiSource_Data_Base


@dataclasses.dataclass(eq=False)
class Trainer(
    _MultiSource_Data_Base,  # multi-source data
    Multi_Features_Model_Dataclass,  # NN model
    Trainer_Base,  # training and output
):
    dataset_to_ram: bool = True # Load datasets to ram
    data_location: str = sample_data_dir / 'turbulence_data' #location of stored data

    # __init__ must have exact copy of all kwargs from parent class
    def __post_init__(self) -> None:

        self.mlp_output_size = 4

        self.is_regression = False
        self.is_classification = not self.is_regression
        self.turbulence_dataset = None

        super().__post_init__()

        self._make_model()
        self.finish_subclass_initialization()


    def _make_datasets(self) -> None:
        kwargs_for_data_class = self._create_data_class_inputs(self.__dict__)
        self.confinement_dataset = ConfinementDataset(**kwargs_for_data_class)
        train_set, valid_set, test_set = self.confinement_dataset.train_test_split()
        if self.dataset_to_ram:
            # Load datasets into ram
            train_set.load_datasets()
            valid_set.load_datasets()

        self.train_dataset = train_set
        self.validation_dataset = valid_set
        self.test_dataset = test_set
        return

    def _make_data_loaders(self) -> None:
        self.train_loader = DataLoader(self.train_dataset,
                                            batch_size=None,  # must be disabled when using samplers
                                            sampler=BatchSampler(RandomBatchSampler(self.train_dataset,
                                                                                    self.batch_size,
                                                                                    self.signal_window_size),
                                                                 batch_size=self.batch_size,
                                                                 drop_last=True)
                                      )

        self.valid_loader = DataLoader(self.validation_dataset,
                                                 batch_size=None,  # must be disabled when using samplers
                                                 sampler=BatchSampler(RandomBatchSampler(self.validation_dataset,
                                                                                         self.batch_size,
                                                                                         self.signal_window_size),
                                                                      batch_size=self.batch_size,
                                                                      drop_last=True)
                                      )


    def _create_data_class_inputs(self, locals_copy: dict = None) -> dict:
        assert self.__class__ is not Trainer_Base
        kwargs_for_data_class = {}
        for cls in [ConfinementDataset, MultiSourceDataset]:
            class_parameters = inspect.signature(cls).parameters
            for parameter_name in class_parameters:
                if parameter_name in locals_copy:
                    kwargs_for_data_class[parameter_name] = locals_copy[parameter_name]
        return kwargs_for_data_class

    def _get_valid_indices(self, *args, **kwargs) -> None:
        pass

    def _check_for_balanced_data(self, *args, **kwargs):
        pass

    def _get_data(self) -> None:
        pass

    def _save_test_data(self) -> None:
        if self.test_dataset is not None:
            self.test_dataset.load_datasets()
            self.test_dataset.save(self.output_dir / self.test_data_file)

if __name__=='__main__':
    model = Trainer(
        batch_size=32,
        dense_num_kernels=8,
        fraction_validation=0.2,
        fraction_test=0.2,
    )
    model.train()
