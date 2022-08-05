from pathlib import Path
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import yaml

try:
    from ..base.analyze_base import _Analyzer_Base
except ImportError:
    from bes_ml.base.analyze_base import _Analyzer_Base


class Analyzer(_Analyzer_Base):

    def __init__(
        self,
        output_dir: Union[str,Path] = 'run_dir',
        inputs_file: Union[str,Path] = 'inputs.yaml',
        device: str = 'auto',  # auto (default), cpu, cuda, or cuda:X
    ) -> None:
        self._validate_subclass_inputs()
        super().__init__(
            output_dir=output_dir,
            inputs_file=inputs_file,
            device=device,
        )
        
        self.is_regression = False
        self._set_regression_or_classification_defaults()

        self._restore_test_data()

        self.roc_scores = None

        self.is_regression = False
        self._set_regression_or_classification_defaults()

        self._load_test_data()

    def _load_training_results(self):
        super()._load_training_results()
        self.roc_scores = np.array(self.results['roc_scores'])


if __name__=='__main__':
    analyzer = Analyzer()
    analyzer.plot_training(save=True)
    analyzer.run_inference()
    analyzer.plot_inference(save=True)
    analyzer.show()