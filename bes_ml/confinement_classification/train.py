try:
    from ..base.train_base import _Trainer_Base
except ImportError:
    from bes_ml.base.train_base import _Trainer_Base


class Confinement_Classification_Model(_Trainer_Base):

    def __init__(self) -> None:
        pass


if __name__=='__main__':
    Confinement_Classification_Model()