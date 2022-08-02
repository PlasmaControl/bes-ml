import sys
import shutil

import pytest

from bes_ml import elm_classification
from bes_ml import elm_regression


RUN_DIR = 'run_dir'
DEFAULT_INPUT_ARGS = {
    'max_elms': 5,
    'n_epochs': 2,
    'fraction_validation': 0.2,
    'fraction_test': 0.2,
}


def test_elm_classification_dense_features():
    input_args = DEFAULT_INPUT_ARGS.copy()
    model = elm_classification.train.ELM_Classification_Trainer(
        output_dir=RUN_DIR+'/elm_classification_dense',
        dense_num_kernels=8,
        **input_args,
    )
    model.train()

def test_elm_classification_cnn_features():
    input_args = DEFAULT_INPUT_ARGS.copy()
    model = elm_classification.train.ELM_Classification_Trainer(
        output_dir=RUN_DIR+'/elm_classification_cnn',
        dense_num_kernels = 0,
        cnn_layer1_num_kernels = 8,
        cnn_layer2_num_kernels = 8,
        **input_args,
    )
    model.train()

def test_elm_regression_dense_features():
    input_args = DEFAULT_INPUT_ARGS.copy()
    model = elm_regression.train.ELM_Regression_Trainer(
        output_dir=RUN_DIR+'/elm_regression_dense',
        dense_num_kernels=8,
        **input_args,
    )
    model.train()

def test_elm_regression_cnn_features():
    input_args = DEFAULT_INPUT_ARGS.copy()
    model = elm_regression.train.ELM_Regression_Trainer(
        output_dir=RUN_DIR+'/elm_regression_cnn',
        dense_num_kernels = 0,
        cnn_layer1_num_kernels = 8,
        cnn_layer2_num_kernels = 8,
        **input_args,
    )
    model.train()


if __name__=="__main__":
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    sys.exit(pytest.main(['--verbose', '--exitfirst', "--ignore-glob='*archive*'"]))