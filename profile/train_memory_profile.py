from memory_profiler import LineProfiler, memory_usage, show_results

from bes_ml.elm_regression import Trainer
from bes_ml.base.train_base import _Trainer_Base
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
        _Trainer_Base.train,
        _Trainer_Base._single_epoch_loop,
        _Trainer_Base.__post_init__,
        _Trainer_Base.finish_subclass_initialization,
        _Trainer_Base._get_data,
        _Trainer_Base._preprocess_data,
        _Trainer_Base._make_data_loaders,
        Trainer.__post_init__,
        Trainer._get_valid_indices,
        Trainer._make_datasets,
        Multi_Features_Model.forward,
        Dense_Features.forward,
        _Base_Features._dropout_relu_flatten,
        _Base_Features._time_interval_and_maxpool,
    ]

    lp = LineProfiler()
    for f in functions:
        lp.add_function(f)
    wrapped_function = lp(profile_func)
    wrapped_function()
    with open("memory_profile.out", 'w') as f:
        show_results(lp, stream=f)
