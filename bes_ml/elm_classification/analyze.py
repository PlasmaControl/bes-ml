from pathlib import Path
from typing import Union
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import yaml

try:
    from .train import ELM_Classification_Trainer
except ImportError:
    from bes_ml.elm_classification.train import ELM_Classification_Trainer


class Analyze(object):

    def __init__(
        self,
        directory: Union[str,Path] ='./run_dir',
    ) -> None:
        self.directory = Path(directory)
        assert self.directory.exists()
        self.inputs = {}
        self._read_inputs()
        model = ELM_Classification_Trainer(**self.inputs)

    def _read_inputs(self) -> None:
        with (self.directory/'model_inputs.yaml').open('r') as inputs_file:
            self.inputs.update(yaml.safe_load(inputs_file))
        with (self.directory/'trainer_inputs.yaml').open('r') as inputs_file:
            tmp = yaml.safe_load(inputs_file)
            tmp.pop('model_kwargs')
            self.inputs.update(tmp)


if __name__=='__main__':
    analyze = Analyze()