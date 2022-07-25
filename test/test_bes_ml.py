import sys
import shutil
import pytest
import bes_ml.elm_regression.train


RUN_DIR = 'run_dir'
DEFAULT_INPUT_ARGS = {
    'max_elms': 5,
    'n_epochs': 2,
    'fraction_validation': 0.2,
    'fraction_test': 0.2,
}


def test_elm_regression_dense_features():
    input_args = DEFAULT_INPUT_ARGS.copy()
    model = bes_ml.elm_regression.train.ELM_Regression_Trainer(
        output_dir=RUN_DIR+'/dense',
        dense_num_kernels=8,
        **input_args,
    )
    model.train()

def test_elm_regression_cnn_features():
    input_args = DEFAULT_INPUT_ARGS.copy()
    model = bes_ml.elm_regression.train.ELM_Regression_Trainer(
        output_dir=RUN_DIR+'/cnn',
        dense_num_kernels = 0,
        cnn_layer1_num_kernels = 8,
        cnn_layer2_num_kernels = 8,
        **input_args,
    )
    model.train()

if __name__=="__main__":
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    sys.exit(pytest.main(['--verbose', '--exitfirst', f'{__file__}']))