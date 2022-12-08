from pathlib import Path
from typing import Union
import pickle
import dataclasses

import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.cuda
import torch.utils.data

try:
    from .models import Multi_Features_Model  #, _Multi_Features_Model_Dataclass
    from .elm_data import ELM_Dataset
    from .utilities import merge_pdfs
except ImportError:
    from bes_ml.base.models import Multi_Features_Model  #, _Multi_Features_Model_Dataclass
    from bes_ml.base.elm_data import ELM_Dataset
    from bes_ml.base.utilities import merge_pdfs


@dataclasses.dataclass
class Analyzer_Base(
    # _Multi_Features_Model_Dataclass,
):
    output_dir: Union[str,Path] = 'run_dir'
    inputs_file: Union[str,Path] = 'inputs.yaml'
    device: str = 'auto'  # auto (default), cpu, cuda, or cuda:X
    verbose: bool = True

    def __post_init__(self):
        self.output_dir = Path(self.output_dir).resolve()
        assert self.output_dir.exists(), f"{self.output_dir} does not exist."

        self.inputs_file = self.output_dir / self.inputs_file
        assert self.inputs_file.exists(), f"{self.inputs_file} does not exist."

        # read inputs and print
        self.inputs = {}
        with self.inputs_file.open('r') as inputs_file:
            self.inputs.update(yaml.safe_load(inputs_file))
        if self.verbose:
            print(f"Inputs from inputs file {self.inputs_file}")
            for key in self.inputs:
                print(f"  {key}: {self.inputs[key]}")

        # setup device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        if self.verbose:
            print(f"Using device: {self.device}")

        # instantiate model, send to device, and load model parameters
        fields = list(dataclasses.fields(Multi_Features_Model))
        for i_field, field_name in enumerate([field.name for field in fields]):
            if field_name == 'logger':
                fields.pop(i_field)
        model_kwargs = {field.name: self.inputs[field.name] for field in fields if field.init}
        self.model = Multi_Features_Model(**model_kwargs)
        self.model = self.model.to(self.device)
        checkpoint_file = self.output_dir / self.inputs['checkpoint_file']
        assert checkpoint_file.exists(), f"{checkpoint_file} does not exist"
        model_state_dict = torch.load(
            checkpoint_file,
            map_location=self.device,
        )
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

        self.test_data = None
        if self.inputs['fraction_test'] > 0.0:
            self._load_test_data()

        self.all_predictions = None
        self.all_labels = None
        self.all_signals = None

        self.results = None
        self.train_score = None
        self.train_loss = None
        self.valid_loss = None

        self.is_regression = None
        self.is_classification = None

    def _load_test_data(self) -> None:
        test_data_file = self.output_dir / self.inputs['test_data_file']
        assert test_data_file.exists(), f"{test_data_file} does not exist."
        with test_data_file.open('rb') as file:
            self.test_data = pickle.load(file)

    def run_inference(
        self,
        max_elms: int = None,
    ) -> None:
        if self.test_data is None:
            print("Skipping inference, no test data")
            return
        n_elms = len(self.test_data['elm_indices'])
        self.all_predictions = []
        self.all_labels = []
        self.all_signals = []
        with torch.no_grad():
            # loop over ELMs in test data
            print('Running inference on test data')
            for i_elm in range(n_elms):
                if max_elms and i_elm>= max_elms:
                    break
                print(f'  ELM {i_elm+1} of {n_elms}')
                i_start = self.test_data['window_start'][i_elm]
                i_stop = (
                    self.test_data['window_start'][i_elm+1]-1
                    if i_elm < n_elms-1
                    else self.test_data['labels'].size-1
                )
                elm_labels = self.test_data['labels'][i_start:i_stop+1]
                elm_signals = self.test_data['signals'][i_start:i_stop+1, ...]
                # elm_sample_indices = self.test_data['sample_indices'][
                #     np.logical_and(
                #         self.test_data['sample_indices'] >= i_start,
                #         self.test_data['sample_indices'] <= i_stop,
                #     )
                # ] - i_start
                # elm_sample_indices = self.test_data['sample_indices'][i_start:i_stop+1] - self.test_data['sample_indices'][i_start]
                elm_sample_indices = np.arange(elm_labels.size - self.inputs['signal_window_size'], dtype=int)
                # print(elm_labels.size, elm_sample_indices.size+self.inputs['signal_window_size'])
                elm_test_dataset = ELM_Dataset(
                    signals=elm_signals,
                    labels=elm_labels,
                    sample_indices=elm_sample_indices,
                    signal_window_size=self.inputs['signal_window_size'],
                    prediction_horizon=0 if self.is_regression else self.inputs['prediction_horizon'],
                )
                elm_test_data_loader = torch.utils.data.DataLoader(
                    dataset=elm_test_dataset,
                    batch_size=128,
                    shuffle=False,
                    num_workers=2 if torch.cuda.is_available() else 0,
                    pin_memory=False,
                    drop_last=False,
                )
                # loop over batches for single ELM
                elm_predictions = np.empty(0, dtype=np.float32)
                count_predictions = 0
                for batch_signals, _ in elm_test_data_loader:
                    batch_signals = batch_signals.to(self.device)
                    batch_predictions = self.model(batch_signals)
                    count_predictions += batch_predictions.numel()
                    if self.is_classification:
                        # if evaluation/inference mode and classificaiton model,
                        # apply sigmoid to transform [-inf,inf] logit -> [0,1] probability
                        batch_predictions = batch_predictions.sigmoid()
                    elm_predictions = np.append(elm_predictions, batch_predictions.cpu().numpy())
                self.all_labels.append(elm_labels)
                self.all_predictions.append(elm_predictions)
                self.all_signals.append(elm_signals[:,2,3])
                # finite_labels = np.count_nonzero(np.isfinite(elm_labels))
                # pred_sws = elm_predictions.size + self.inputs['signal_window_size']
                # print(f"  ELM {i_elm+1}: finite labels {finite_labels} predictions+sws {pred_sws}")
                # print(f"    Count pred {count_predictions} len(dataset) {len(elm_test_dataset)} len(sample_indices) {len(elm_sample_indices)}")
        print('Inference complete')

    def _load_training_results(self) -> None:
        results_file = self.output_dir / self.inputs['results_file']
        assert results_file.exists, f"{results_file} does not exist"
        with results_file.open('r') as f:
            self.results = yaml.safe_load(f)
        self.train_loss = self.results.get('train_loss')
        self.loss_function_name = self.results.get('loss_function_name')
        self.train_score = self.results.get('train_score')
        self.score_function_name = self.results.get('score_function_name')
        self.lr = self.results.get('lr')
        self.epoch_time = self.results.get('epoch_time')
        self.train_roc = self.results.get('train_roc', None)
        self.valid_loss = self.results.get('valid_loss', None)
        self.valid_score = self.results.get('valid_score', None)
        self.valid_roc = self.results.get('valid_roc', None)

    def plot_training(
        self,
        save: bool = False,  # save PDF
    ) -> None:
        self._load_training_results()
        n_epochs = len(self.train_loss)
        epochs = np.arange(n_epochs) + 1
        _, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,6))
        plt.suptitle(f"{self.output_dir}")
        # loss
        plt.sca(axes.flat[0])
        plt.plot(epochs, self.train_loss, label='Train loss', c='C0')
        if self.valid_loss:
            plt.plot(epochs, self.valid_loss, label='Valid. loss', c='C1')
        plt.yscale('log')
        plt.title(f'{self.loss_function_name} loss')
        plt.ylabel('Loss')
        plt.legend()
        # scores
        plt.sca(axes.flat[1])
        plt.plot(epochs, self.train_score, label=f'Train {self.score_function_name}', c='C0')
        if self.valid_score:
            plt.plot(epochs, self.valid_score, label=f'Valid. {self.score_function_name}', c='C1')
        if self.train_roc:
            plt.plot(epochs, self.train_roc, label='Train ROC', c='C0', ls='--')
        if self.valid_roc:
            plt.plot(epochs, self.valid_roc, label='Valid. ROC', c='C1', ls='--')
        plt.title(f'Scores')
        plt.ylabel('Score')
        plt.legend()
        # lr
        plt.sca(axes.flat[2])
        plt.semilogy(epochs, self.lr)
        plt.title('Learning rate')
        plt.ylabel('Learning rate')
        # epoch time
        plt.sca(axes.flat[3])
        plt.plot(epochs, self.epoch_time)
        plt.title('Epoch time')
        plt.ylabel('Epoch time (s)')
        for axis in axes.flat:
            plt.sca(axis)
            plt.xlabel('Epoch')
            plt.xlim([0,None])
        plt.tight_layout()
        if save:
            filepath = self.output_dir / "training.pdf"
            if self.verbose:
                print(f'Saving training plot: {filepath}')
            plt.savefig(filepath, format='pdf', transparent=True)
        return

    def plot_inference(
        self,
        max_elms: int = None,
        save: bool = False,  # save PDFs
    ) -> None:
        if self.test_data is None:
            print("Skipping inference, no test data")
            return
        if None in [self.all_labels, self.all_predictions, self.all_signals]:
            self.run_inference(max_elms=max_elms)
        prediction_offset = self.inputs['signal_window_size'] - 1
        if self.is_classification:
            prediction_offset += self.inputs['prediction_horizon']
        i_page = 1
        n_elms = len(self.all_labels)
        _, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
        plt.suptitle(f"{self.output_dir.resolve()}")
        axes_twinx = [axis.twinx() for axis in axes.flat]
        for i_elm in range(n_elms):
            elm_index = self.test_data['elm_indices'][i_elm]
            labels = self.all_labels[i_elm]
            predictions = self.all_predictions[i_elm]
            signals = self.all_signals[i_elm]
            elm_time = np.arange(labels.size)
            # finite_labels = np.count_nonzero(np.isfinite(labels))
            # pred_sws = predictions.size + prediction_offset
            # print(f"  ELM {i_elm+1}: finite labels {finite_labels} predictions+sws {pred_sws}")
            if i_elm % 6 == 0:
                for i_axis in range(axes.size):
                    axes.flat[i_axis].clear()
                    axes_twinx[i_axis].clear()
            plt.sca(axes.flat[i_elm % 6])
            plt.plot(elm_time, labels, label="Label", color='C0')
            plt.plot(
                elm_time[prediction_offset:prediction_offset+predictions.size],
                predictions,
                label="Prediction",
                color='C1',
            )
            plt.xlabel("Time (micro-s)")
            plt.ylabel("Label | Prediction")
            plt.ylim(-1.1,1.1)
            plt.legend(fontsize='small', loc='upper left')
            plt.title(f'Test ELM index {elm_index}')
            twinx = axes_twinx[i_elm%6]
            twinx.plot(elm_time, signals, label="BES", color='C2')
            twinx.set_ylabel('Scaled signal')
            twinx.legend(fontsize='small', loc='upper right')
            if i_elm % 6 == 5 or i_elm == n_elms-1:
                plt.tight_layout()
                if save:
                    filepath = self.output_dir / f'inference_{i_page:02d}.pdf'
                    if self.verbose:
                        print(f'Saving inference file: {filepath}')
                    plt.savefig(filepath, format='pdf', transparent=True)
                    i_page += 1
        if save:
            inputs = sorted(self.output_dir.glob('inference_*.pdf'))
            output = self.output_dir/'test_inference.pdf'
            merge_pdfs(
                inputs=inputs,
                output=output,
                delete_inputs=True,
                verbose=True,
            )

    @staticmethod
    def show(*args, **kwargs) -> None:
        """
        Wrapper for `plt.show()`
        """
        plt.show(*args, **kwargs)