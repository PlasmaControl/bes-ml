import dataclasses

try:
    from ..base.analyze_base import _Analyzer_Base
except ImportError:
    from bes_ml.base.analyze_base import _Analyzer_Base


@dataclasses.dataclass
class Analyzer(_Analyzer_Base):

    def __post_init__(self):
        super().__post_init__()
        
        self.is_regression = True
        self.is_classification = not self.is_regression


if __name__=='__main__':
    analyzer = Analyzer()
    analyzer.plot_training(save=True)
    analyzer.run_inference()
    analyzer.plot_inference(save=True)
    analyzer.show()