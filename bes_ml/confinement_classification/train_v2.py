import dataclasses

try:
    from ..base.train_base import Trainer_Base
    from .confinement_data_v2 import Confinement_Data_v2
except ImportError:
    from bes_ml.base.train_base import Trainer_Base
    from bes_ml.confinement_classification.confinement_data_v2 import Confinement_Data_v2


@dataclasses.dataclass(eq=False)
class Trainer(
    Confinement_Data_v2,  # confinement data
    Trainer_Base,  # training and output
):

    def __post_init__(self) -> None:
        self.mlp_output_size = 4
        self.is_classification = True
        self.is_regression = not self.is_classification
        super().__post_init__()  # Trainer_Base.__post_init__()


if __name__=='__main__':
    Trainer(
        dense_num_kernels=8,
        fraction_test=0.1,
        dataset_to_ram=True,
        do_train=True,
    )
