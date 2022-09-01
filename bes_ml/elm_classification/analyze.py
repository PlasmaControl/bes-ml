import dataclasses

import numpy as np

try:
    from ..base.analyze_base import _Analyzer_Base
except ImportError:
    from bes_ml.base.analyze_base import _Analyzer_Base


@dataclasses.dataclass
class Analyzer(_Analyzer_Base):

    def __post_init__(self):
        super().__post_init__()
        
        self.is_regression = False
        self.is_classification = not self.is_regression

        self.roc_scores = None

    def _load_training_results(self):
        super()._load_training_results()
        self.roc_scores = np.array(self.results['roc_scores'])


if __name__=='__main__':
    analyzer = Analyzer()
    analyzer.plot_training(save=True)
    analyzer.plot_inference(save=True)
    analyzer.show()