from line_profiler import LineProfiler

from bes_ml.elm_regression import Trainer
from bes_ml.base.train_base import _Trainer_Base
from bes_ml.base.models import Multi_Features_Model, Dense_Features, _Base_Features
from bes_ml.base.data import ELM_Dataset


def profile_func():
    model = Trainer(
        dense_num_kernels=64,
        n_epochs=6,
        fraction_validation=0.0,
        fraction_test=0.0,
        log_time=True,
        inverse_weight_label=True,
    )
    model.train()


if __name__=='__main__':

    lp = LineProfiler(
        _Trainer_Base._single_epoch_loop,
        ELM_Dataset.__getitem__,
        _Trainer_Base.train,
        Multi_Features_Model.forward,
        Dense_Features.forward,
        # _Trainer_Base.__post_init__,
        # Trainer.__post_init__,
        # _Trainer_Base._get_data,
        # _Trainer_Base._preprocess_data,
        # _Trainer_Base.finish_subclass_initialization,
    )
    lp.run("profile_func()")
    with open("line_profile.txt", 'w') as f:
        lp.print_stats(f)
