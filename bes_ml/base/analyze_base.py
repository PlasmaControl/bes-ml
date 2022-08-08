from pathlib import Path
from typing import Union, Sequence
import pickle
import inspect
import shutil
import subprocess

import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import torch

try:
    from .models import Multi_Features_Model
    from .data import ELM_Dataset, elm_data_loader
except ImportError:
    from bes_ml.base.models import Multi_Features_Model
    from bes_ml.base.data import ELM_Dataset, elm_data_loader


class _Analyzer_Base(object):

    def __init__(
        self,
        output_dir: Union[str,Path] = 'run_dir',
        inputs_file: Union[str,Path] = 'inputs.yaml',
        device: str = 'auto',  # auto (default), cpu, cuda, or cuda:X
    ) -> None:
        # run directory and inputs file
        self.output_dir = Path(output_dir)
        assert self.output_dir.exists(), \
            f"Directory {self.output_dir} does not exist."
        self.inputs_file = self.output_dir / inputs_file
        assert self.inputs_file.exists(), \
            f"{self.inputs_file} does not exist."
        # read inputs and print
        self.inputs = {}
        with self.inputs_file.open('r') as inputs_file:
            inputs_dict = yaml.safe_load(inputs_file)
            self.inputs.update(inputs_dict)
        print("Inputs from inputs file")
        for key in self.inputs:
            print(f"  {key}: {self.inputs[key]}")
        if 'prediction_horizon' not in self.inputs:
            self.inputs['prediction_horizon'] = 0
        # setup device
        self.device = device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        # instantiate model, send to device, and load model parameters
        self.model = Multi_Features_Model(**self.inputs)
        self.model = self.model.to(self.device)
        self._load_model_parameters()
        self.model.eval()

        self.all_predictions = None
        self.all_labels = None
        self.all_signals = None

        self.results = None
        self.scores = None
        self.train_loss = None
        self.valid_loss = None

        self.is_regression = None

    def _load_test_data(self):
        # restore test data
        test_data_file = self.output_dir / self.inputs['test_data_file']
        assert test_data_file.exists(), \
            f"{test_data_file} does not exist."
        with test_data_file.open('rb') as file:
            self.test_data = pickle.load(file)

    def _load_model_parameters(
        self,
    ) -> None:
        checkpoint_file = self.output_dir / self.inputs['checkpoint_file']
        assert checkpoint_file.exists(), f"{checkpoint_file} does not exist"
        model_state_dict = torch.load(
            checkpoint_file, 
            map_location=self.device,
        )
        self.model.load_state_dict(model_state_dict)

    def _set_regression_or_classification_defaults(self) -> None:
        """
        Set defaults for regression or classification tasks.
        """
        # self.threshold = None  # set with kwarg
        if self.is_regression:
            # regression model (e.g. time to ELM onset)
            self.loss_function = torch.nn.MSELoss(reduction="none")
            self.score_function = metrics.r2_score
        else:
            # classification model (e.g. active ELM prediction for `prediction_horizon` horizon
            self.loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")
            self.score_function = metrics.f1_score

    def run_inference(
        self,
    ) -> None:
        n_elms = len(self.test_data['elm_indices'])
        self.all_predictions = []
        self.all_labels = []
        self.all_signals = []
        # losses = np.array(0)
        with torch.no_grad():
            # loop over ELMs in test data
            print('Running inference on test data')
            for i_elm in range(n_elms):
                print(f'  ELM {i_elm+1} of {n_elms}')
                i_start = self.test_data['window_start'][i_elm]
                i_stop = (
                    self.test_data['window_start'][i_elm+1]-1
                    if i_elm < n_elms-1
                    else self.test_data['labels'].size-1
                )
                elm_labels = self.test_data['labels'][i_start:i_stop+1]
                elm_signals = self.test_data['signals'][i_start:i_stop+1, ...]
                elm_sample_indices = self.test_data['sample_indices'][
                    np.logical_and(
                        self.test_data['sample_indices'] >= i_start,
                        self.test_data['sample_indices'] <= i_stop,
                    )
                ] - i_start
                elm_test_dataset = ELM_Dataset(
                    signals=elm_signals,
                    labels=elm_labels,
                    sample_indices=elm_sample_indices,
                    signal_window_size=self.inputs['signal_window_size'],
                    prediction_horizon=0 if self.is_regression else self.inputs['prediction_horizon'],
                )
                elm_test_data_loader = elm_data_loader(
                    dataset=elm_test_dataset,
                    batch_size=128,
                    shuffle=False,
                    num_workers=self.inputs['num_workers'],
                    pin_memory=False,
                    drop_last=False,
                )
                # loop over batches for single ELM
                elm_predictions = np.empty(0, dtype=np.float32)
                for i_batch, (batch_signals, batch_labels) in enumerate(elm_test_data_loader):
                    batch_signals = batch_signals.to(self.device)
                    batch_predictions = self.model(batch_signals)
                    if not self.is_regression:
                        # if evaluation/inference mode and classificaiton model,
                        # apply sigmoid to transform [-inf,inf] logit -> [0,1] probability
                        batch_predictions = batch_predictions.sigmoid()
                    elm_predictions = np.append(elm_predictions, batch_predictions.cpu().numpy())
                self.all_labels.append(elm_labels)
                self.all_predictions.append(elm_predictions)
                self.all_signals.append(elm_signals[:,2,3])
        print('Inference complete')

    def _load_training_results(self):
        results_file = self.output_dir / self.inputs['results_file']
        assert results_file.exists, f"{results_file} does not exist"
        with results_file.open('r') as f:
            self.results = yaml.safe_load(f)
        self.scores = np.array(self.results['scores'])
        self.train_loss = np.array(self.results['train_loss'])
        self.valid_loss = np.array(self.results['valid_loss'])
        self.scores_label = self.results['scores_label']

    def plot_training(
        self,
        save: bool = False,  # save PDF
    ):
        self._load_training_results()
        n_epochs = self.scores.size
        epochs = np.arange(n_epochs) + 1
        _, axes = plt.subplots(ncols=2, nrows=1, figsize=(8,3))
        plt.suptitle(f"{self.output_dir.resolve()}")
        plt.sca(axes.flat[0])
        plt.plot(epochs, self.train_loss, label='Training loss')
        plt.plot(epochs, self.valid_loss, label='Valid. loss')
        plt.title('Training/validation loss')
        plt.ylabel('Loss')
        plt.sca(axes.flat[1])
        plt.plot(epochs, self.scores, label=self.scores_label)
        if not self.is_regression and hasattr(self, 'roc_scores'):
            plt.plot(epochs, self.roc_scores, label='ROC-AUC')
        plt.title('Validation scores')
        plt.ylabel('Score')
        for axis in axes.flat:
            plt.sca(axis)
            plt.xlabel('Epoch')
            plt.xlim([0,None])
            plt.legend()
        plt.tight_layout()
        if save:
            filepath = self.output_dir / "training.pdf"
            print(f'Saving training plot: {filepath}')
            plt.savefig(filepath, format='pdf', transparent=True)
        return

    def plot_inference(
        self,
        save: bool = False,  # save PDFs
    ):
        assert None not in [self.all_labels, self.all_predictions, self.all_signals], \
            print("Nothing to plot; run inference first")
        n_elms = self.test_data['elm_indices'].size
        assert len(self.all_labels) == n_elms and \
            len(self.all_predictions) == n_elms and \
            len(self.all_signals) == n_elms
        prediction_offset = (
            self.inputs['signal_window_size'] - 1
            + self.inputs['prediction_horizon']
        )
        i_page = 1
        for i_elm in range(n_elms):
            elm_index = self.test_data['elm_indices'][i_elm]
            labels = self.all_labels[i_elm]
            predictions = self.all_predictions[i_elm]
            signals = self.all_signals[i_elm]
            elm_time = np.arange(labels.size)
            if i_elm % 6 == 0:
                _, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
            plt.suptitle(f"{self.output_dir.resolve()}")
            plt.sca(axes.flat[i_elm % 6])
            if not self.is_regression:
                signals = signals / np.max(signals)
            plt.plot(elm_time, signals, label="BES")
            plt.plot(elm_time, labels, label="Label")
            plt.plot(
                elm_time[prediction_offset:prediction_offset+predictions.size], 
                predictions, 
                label="Prediction",
            )
            plt.xlabel("Time (micro-s)")
            plt.ylabel("Signal | label")
            plt.legend(fontsize='small')
            plt.title(f'Test ELM index {elm_index}')
            if i_elm % 6 == 5 or i_elm == n_elms-1:
                plt.tight_layout()
                if save:
                    filepath = self.output_dir / f'inference_{i_page:02d}.pdf'
                    print(f'Saving inference file: {filepath}')
                    plt.savefig(filepath, format='pdf', transparent=True)
                    i_page += 1
        if save:
            inputs = sorted(self.output_dir.glob('inference_*.pdf'))
            output = self.output_dir/'inference.pdf'
            self._merge_pdfs(
                inputs=inputs, 
                output=output,
                delete_inputs=True,
            )

    @staticmethod
    def _merge_pdfs(
        inputs: Union[Sequence,list] = None,
        output: Union[str,Path] = None,
        delete_inputs: bool = False,
    ):
        inputs = [Path(input) for input in inputs]
        output = Path(output)
        gs_cmd = shutil.which('gs')
        assert gs_cmd is not None, \
            "`gs` command (ghostscript) not found; available in conda-forge"
        print(f"Merging PDFs into file: {output.as_posix()}")
        cmd = [
            gs_cmd,
            '-q',
            '-dBATCH',
            '-dNOPAUSE',
            '-sDEVICE=pdfwrite',
            '-dPDFSETTINGS=/prepress',
            '-dCompatibilityLevel=1.4',
        ]
        cmd.append(f"-sOutputFile={output.as_posix()}")
        for pdf_file in inputs:
            cmd.append(f"{pdf_file.as_posix()}")
        result = subprocess.run(cmd, check=True)
        assert result.returncode == 0 and output.exists()
        if delete_inputs is True:
            for pdf_file in inputs:
                pdf_file.unlink()

    def _validate_subclass_inputs(self) -> None:
        """
        Ensure subclass call signature contains all parameters in
        parent class signature and model class signature
        """
        parent_class = _Analyzer_Base
        assert self.__class__ is not parent_class
        subclass_parameters = inspect.signature(self.__class__).parameters
        class_parameters = inspect.signature(parent_class).parameters
        for param_name in class_parameters:
            if param_name in ['model_kwargs', 'logger', 'kwargs']:
                continue
            assert param_name in subclass_parameters, \
                [f"Subclass {self.__class__.__name__} "
                    f"missing parameter {param_name} from class {parent_class.__name__}."]

    @staticmethod
    def show(*args, **kwargs):
        """
        Wrapper for `plt.show()`
        """
        plt.show(*args, **kwargs)