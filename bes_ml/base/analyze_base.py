from pathlib import Path
from typing import Union
import pickle
import inspect

import yaml
import torch

try:
    from ..base.models import Multi_Features_Model
    from .data import ELM_Dataset, elm_data_loader
except ImportError:
    from bes_ml.base.models import Multi_Features_Model
    from bes_ml.base.data import ELM_Dataset, elm_data_loader


class _Analyzer_Base(object):

    def __init__(
        self,
        directory: Union[str,Path] = 'run_dir',
        inputs_file: Union[str,Path] = 'inputs.yaml',
        device: str = 'auto',  # auto (default), cpu, cuda, or cuda:X
    ) -> None:
        # run directory and inputs file
        self.directory = Path(directory)
        assert self.directory.exists(), \
            f"Directory {self.directory} does not exist."
        self.inputs_file = self.directory / inputs_file
        assert self.inputs_file.exists(), \
            f"{self.inputs_file} does not exist."
        # read inputs and print
        self.inputs = {}
        with self.inputs_file.open('r') as inputs_file:
            inputs_dict = yaml.safe_load(inputs_file)
            self.inputs.update(inputs_dict)
        print("Inputs from inputs file")
        for key in self.inputs:
            print(f"  {key}: {self.inputs[key]}")
        # setup device
        self.device = device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        # instantiate model and send to device
        self.model = Multi_Features_Model(**self.inputs)
        self.model = self.model.to(self.device)
        # restore test data
        test_data_file = self.directory / self.inputs['test_data_file']
        assert test_data_file.exists(), \
            f"{test_data_file} does not exist."
        with test_data_file.open('rb') as file:
            self.test_data = pickle.load(file)
        self.test_dataset = ELM_Dataset(
            signals=self.test_data['signals'],
            labels=self.test_data['labels'],
            sample_indices=self.test_data['sample_indices'],
            window_start=self.test_data['window_start'],
            signal_window_size=self.inputs['signal_window_size'],
            prediction_horizon=self.inputs['prediction_horizon'],
        )
        self.test_data_loader = elm_data_loader(
            dataset=self.test_dataset,
        )

    def run_inference(
        self,
    ) -> None:
        pass

    def _validate_subclass_inputs(self) -> None:
        """
        Ensure subclass call signature contains all parameters in
        parent class signature and model class signature
        """
        parent_class = _Analyzer_Base
        assert self.__class__ is not parent_class
        subclass_parameters = inspect.signature(self.__class__).parameters
        class_parameters = inspect.signature(parent_class).parameters
        for param_name in class_parameters:
            if param_name in ['model_kwargs', 'logger', 'kwargs']:
                continue
            assert param_name in subclass_parameters, \
                [f"Subclass {self.__class__.__name__} "
                    f"missing parameter {param_name} from class {parent_class.__name__}."]

