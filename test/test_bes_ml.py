import sys
import shutil
from pathlib import Path

import pytest

from bes_ml import elm_classification
from bes_ml import elm_regression
from bes_ml import velocimetry
from bes_ml import confinement_classification


RUN_DIR = Path('run_dir')
DEFAULT_INPUT_ARGS = {
    'n_epochs': 2,
    'fraction_validation': 0.2,
    'fraction_test': 0.4,
}


def test_elm_classification_dense_features():
    input_args = DEFAULT_INPUT_ARGS.copy()
    output_dir = RUN_DIR / 'elm_classification_dense'
    model = elm_classification.Trainer(
        output_dir=output_dir,
        max_elms=5,
        dense_num_kernels=8,
        **input_args,
    )
    model.train()
    analyzer = elm_classification.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_elm_classification_cnn_features():
    input_args = DEFAULT_INPUT_ARGS.copy()
    output_dir = RUN_DIR / 'elm_classification_cnn'
    model = elm_classification.Trainer(
        output_dir=output_dir,
        max_elms=5,
        dense_num_kernels = 0,
        cnn_layer1_num_kernels = 8,
        cnn_layer2_num_kernels = 8,
        **input_args,
    )
    model.train()
    analyzer = elm_classification.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_elm_regression_dense_features():
    input_args = DEFAULT_INPUT_ARGS.copy()
    output_dir = RUN_DIR / 'elm_regression_dense'
    model = elm_regression.Trainer(
        output_dir=output_dir,
        max_elms=5,
        dense_num_kernels=8,
        **input_args,
    )
    model.train()
    analyzer = elm_regression.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_elm_regression_cnn_features():
    input_args = DEFAULT_INPUT_ARGS.copy()
    output_dir = RUN_DIR / 'elm_regression_cnn'
    model = elm_regression.Trainer(
        output_dir=output_dir,
        max_elms=5,
        dense_num_kernels = 0,
        cnn_layer1_num_kernels = 8,
        cnn_layer2_num_kernels = 8,
        **input_args,
    )
    model.train()
    analyzer = elm_regression.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_velocimetry_training():
    input_args = DEFAULT_INPUT_ARGS.copy()
    output_dir = RUN_DIR / 'velocimetry_test'
    model = velocimetry.Trainer(output_dir=output_dir,
                                **input_args)
    model.train()

def test_confinement_training():
    input_args = DEFAULT_INPUT_ARGS.copy()
    output_dir = RUN_DIR / 'turbulence_test'
    model = confinement_classification.Trainer(output_dir=output_dir,
                                **input_args)
    model.train()


def _common_analysis(analyzer):
    analyzer.plot_training(save=True)
    analyzer.run_inference()
    analyzer.plot_inference(save=True)
    assert (analyzer.output_dir/'training.pdf').exists()
    assert (analyzer.output_dir/'inference.pdf').exists()

if __name__=="__main__":
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    sys.exit(pytest.main(['--verbose', '--exitfirst', "--ignore-glob='*archive*'"]))