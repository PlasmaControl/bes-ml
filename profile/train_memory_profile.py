from memory_profiler import LineProfiler, memory_usage, show_results

from bes_ml.elm_regression import Trainer
from bes_ml.base.train_base import Trainer_Base
from bes_ml.base.models import Multi_Features_Model, Dense_Features, _Base_Features


def profile_func():
    model = Trainer(
        dense_num_kernels=64,
        n_epochs=1,
        fraction_validation=0.0,
        fraction_test=0.0,
        log_time=True,
        inverse_weight_label=True,
    )
    model.train()


if __name__=='__main__':

    functions = [
        Trainer_Base.train,
        Trainer_Base._single_epoch_loop,
        Trainer_Base.__post_init__,
        Trainer_Base.finish_subclass_initialization,
        Trainer_Base._get_data,
        Trainer_Base._preprocess_data,
        Trainer_Base._make_data_loaders,
        Trainer.__post_init__,
        Trainer._get_valid_indices,
        Trainer._make_datasets,
        Multi_Features_Model.forward,
        Dense_Features.forward,
        _Base_Features._flatten_activation_dropout,
        _Base_Features._time_interval_and_pooling,
    ]

    lp = LineProfiler()
    for f in functions:
        lp.add_function(f)
    wrapped_function = lp(profile_func)
    wrapped_function()
    with open("memory_profile.out", 'w') as f:
        show_results(lp, stream=f)
