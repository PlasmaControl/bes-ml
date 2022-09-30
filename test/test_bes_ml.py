import sys
import shutil
from pathlib import Path

import pytest

from bes_ml import elm_classification
from bes_ml import elm_regression
# from bes_ml import velocimetry
# from bes_ml import confinement_classification
from bes_ml.base.analyze_base import _Analyzer_Base


RUN_DIR = Path('run_dir')

def test_elm_classification_dense_features():
    output_dir = RUN_DIR / 'elm_classification_dense'
    model = elm_classification.Trainer(
        output_dir=output_dir,
        max_elms=5,
        dense_num_kernels=8,
    )
    model.train()
    analyzer = elm_classification.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_elm_classification_options():
    output_dir = RUN_DIR / 'elm_classification_options'
    model = elm_classification.Trainer(
        output_dir=output_dir,
        max_elms=5,
        dense_num_kernels=8,
        oversample_active_elm=True,
        one_hot_encoding=True,
    )
    model.train()
    analyzer = elm_classification.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_elm_classification_model_options():
    output_dir = RUN_DIR / 'elm_classification_mlp_multilayer'
    model = elm_classification.Trainer(
        output_dir=output_dir,
        max_elms=5,
        dense_num_kernels=8,
        mlp_hidden_layers=[16,14,12,10],
        activation_name='SiLU',
    )
    model.train()
    analyzer = elm_classification.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_elm_classification_zero_validation_test():
    output_dir = RUN_DIR / 'elm_classification_zero_validation_test'
    model = elm_classification.Trainer(
        output_dir=output_dir,
        max_elms=5,
        dense_num_kernels=8,
        fraction_test=0,
        fraction_validation=0,
    )
    model.train()
    analyzer = elm_classification.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_elm_classification_cnn_features():
    output_dir = RUN_DIR / 'elm_classification_cnn'
    model = elm_classification.Trainer(
        output_dir=output_dir,
        max_elms=5,
        cnn_layer1_num_kernels = 8,
        cnn_layer2_num_kernels = 8,
    )
    model.train()
    analyzer = elm_classification.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_elm_classification_fft_features():
    output_dir = RUN_DIR / 'elm_classification_fft'
    model = elm_classification.Trainer(
        output_dir=output_dir,
        max_elms=5,
        fft_num_kernels=8,
    )
    model.train()
    analyzer = elm_classification.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_elm_classification_dwt_features():
    output_dir = RUN_DIR / 'elm_classification_dwt'
    model = elm_classification.Trainer(
        output_dir=output_dir,
        max_elms=5,
        dwt_num_kernels=8,
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
    output_dir = RUN_DIR / 'elm_regression_dense'
    model = elm_regression.Trainer(
        output_dir=output_dir,
        max_elms=5,
        dense_num_kernels=8,
    )
    model.train()
    analyzer = elm_regression.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_elm_regression_logtime():
    output_dir = RUN_DIR / 'elm_regression_options'
    model = elm_regression.Trainer(
        output_dir=output_dir,
        max_elms=5,
        dense_num_kernels=8,
        log_time=True,
        pre_elm_size=200,
    )
    model.train()
    analyzer = elm_regression.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_elm_regression_weight():
    output_dir = RUN_DIR / 'elm_regression_options'
    model = elm_regression.Trainer(
        output_dir=output_dir,
        max_elms=5,
        dense_num_kernels=8,
        inverse_weight_label=True,
        normalize_labels=False,
        pre_elm_size=200,
    )
    model.train()
    analyzer = elm_regression.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

def test_elm_regression_cnn_features():
    output_dir = RUN_DIR / 'elm_regression_cnn'
    model = elm_regression.Trainer(
        output_dir=output_dir,
        max_elms=5,
        cnn_layer1_num_kernels = 8,
        cnn_layer2_num_kernels = 8,
    )
    model.train()
    analyzer = elm_regression.Analyzer(
        output_dir=output_dir,
    )
    _common_analysis(analyzer)

# def test_velocimetry():
#     input_args = DEFAULT_INPUT_ARGS.copy()
#     output_dir = RUN_DIR / 'velocimetry_test'
#     model = velocimetry.Trainer(
#         output_dir=output_dir,
#         dense_num_kernels=8,
#         **input_args,
#     )
#     model.train()
#     analyzer = velocimetry.Analyzer(output_dir=output_dir)
#     _common_analysis(analyzer)

# def test_confinement():
#     input_args = DEFAULT_INPUT_ARGS.copy()
#     output_dir = RUN_DIR / 'turbulence_test'
#     model = confinement_classification.Trainer(
#         output_dir=output_dir,
#         dense_num_kernels=8,
#         **input_args,
#     )
#     model.train()
#     analyzer = confinement_classification.Analyzer(
#         output_dir=output_dir
#     )
#     _common_analysis(analyzer)

def _common_analysis(analyzer: _Analyzer_Base):
    analyzer.plot_training(save=True)
    assert (analyzer.output_dir/'training.pdf').exists()
    analyzer.plot_inference(save=True)
    if analyzer.test_data is not None:
        assert (analyzer.output_dir/'inference.pdf').exists()


if __name__=="__main__":
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    sys.exit(pytest.main(['--verbose', '--exitfirst', "--ignore-glob='*archive*'"]))