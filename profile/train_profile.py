from contextlib import redirect_stdout
import cProfile
import pstats
from pstats import SortKey

from bes_ml.elm_regression import Trainer


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
    profile_data_file = "profile.out"

    # run profiler
    cProfile.run("profile_func()", profile_data_file)

    # analyze results
    with open('profile.txt', 'w') as f, redirect_stdout(f):
        p = pstats.Stats(profile_data_file)
        filter_string = 'bes_ml'
        p.sort_stats(SortKey.CUMULATIVE).print_stats(filter_string, 10)
        p.sort_stats(SortKey.TIME).print_stats(filter_string, 10)
