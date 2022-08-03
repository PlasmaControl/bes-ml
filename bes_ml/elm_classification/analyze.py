from pathlib import Path
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

try:
    from ..base.analyze_base import _Analyzer_Base
except ImportError:
    from bes_ml.base.analyze_base import _Analyzer_Base


class ELM_Classification_Analyzer(_Analyzer_Base):

    def __init__(
        self,
        directory: Union[str,Path] = 'run_dir',
        inputs_file: Union[str,Path] = 'inputs.yaml',
        device: str = 'auto',  # auto (default), cpu, cuda, or cuda:X
    ) -> None:
        self._validate_subclass_inputs()
        super().__init__(
            directory=directory,
            inputs_file=inputs_file,
            device=device,
        )


if __name__=='__main__':
    analyzer = ELM_Classification_Analyzer(
        directory='run_dir',
    )
    analyzer.run_inference()