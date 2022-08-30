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
        cnn_layer1_num_kernels = 8,
        cnn_layer2_num_kernels = 8,
        **input_args,
    )
    model.train()
    analyzer = elm_classification.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_elm_classification_fft_features():
    input_args = DEFAULT_INPUT_ARGS.copy()
    output_dir = RUN_DIR / 'elm_classification_fft'
    model = elm_classification.Trainer(
        output_dir=output_dir,
        max_elms=5,
        fft_num_kernels=8,
        **input_args,
    )
    model.train()
    analyzer = elm_classification.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_elm_classification_dwt_features():
    input_args = DEFAULT_INPUT_ARGS.copy()
    output_dir = RUN_DIR / 'elm_classification_dwt'
    model = elm_classification.Trainer(
        output_dir=output_dir,
        max_elms=5,
        dwt_num_kernels=8,
        **input_args,
    )
    model.train()
    analyzer = elm_classification.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

# def test_elm_classification_dct_features():
#     input_args = DEFAULT_INPUT_ARGS.copy()
#     output_dir = RUN_DIR / 'elm_classification_dct'
#     model = elm_classification.Trainer(
#         output_dir=output_dir,
#         max_elms=5,
#         dct_num_kernels=8,
#         **input_args,
#     )
#     model.train()
#     analyzer = elm_classification.Analyzer(
#         output_dir=output_dir,
#     )
#     _common_analysis(analyzer)

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
        cnn_layer1_num_kernels = 8,
        cnn_layer2_num_kernels = 8,
        **input_args,
    )
    model.train()
    analyzer = elm_regression.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_velocimetry():
    input_args = DEFAULT_INPUT_ARGS.copy()
    output_dir = RUN_DIR / 'velocimetry_test'
    model = velocimetry.Trainer(
        output_dir=output_dir,
        dense_num_kernels=8,
        **input_args,
    )
    model.train()
    analyzer = velocimetry.Analyzer(output_dir=output_dir)
    _common_analysis(analyzer)

def test_confinement():
    input_args = DEFAULT_INPUT_ARGS.copy()
    output_dir = RUN_DIR / 'turbulence_test'
    model = confinement_classification.Trainer(
        output_dir=output_dir,
        dense_num_kernels=8,
        **input_args,
    )
    model.train()
    analyzer = confinement_classification.Analyzer(
        output_dir=output_dir
    )
    _common_analysis(analyzer)
def _common_analysis(analyzer):
    analyzer.plot_training(save=True)
    analyzer.run_inference()
    analyzer.plot_inference(save=True)
    assert (analyzer.output_dir/'training.pdf').exists()
    assert (analyzer.output_dir/'inference.pdf').exists()

if __name__=="__main__":
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    sys.exit(pytest.main(['--verbose', '--exitfirst', "--ignore-glob='*archive*'"]))