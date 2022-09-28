from pathlib import Path
from typing import Union
import pickle
import dataclasses

import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import h5py

try:
    from .models import Multi_Features_Model  #, _Multi_Features_Model_Dataclass
    from .elm_data import ELM_Dataset
    from .utilities import merge_pdfs
except ImportError:
    from bes_ml.base.models import Multi_Features_Model  #, _Multi_Features_Model_Dataclass
    from bes_ml.base.elm_data import ELM_Dataset
    from bes_ml.base.utilities import merge_pdfs


@dataclasses.dataclass
class _Analyzer_Base(
    # _Multi_Features_Model_Dataclass,
):
    output_dir: Union[str,Path] = 'run_dir'
    inputs_file: Union[str,Path] = 'inputs.yaml'
    device: str = 'auto'  # auto (default), cpu, cuda, or cuda:X

    def __post_init__(self):
        self.output_dir = Path(self.output_dir).resolve()
        assert self.output_dir.exists(), f"{self.output_dir} does not exist."

        self.inputs_file = self.output_dir / self.inputs_file
        assert self.inputs_file.exists(), f"{self.inputs_file} does not exist."

        # read inputs and print
        self.inputs = {}
        with self.inputs_file.open('r') as inputs_file:
            self.inputs.update(yaml.safe_load(inputs_file))
        print(f"Inputs from inputs file {self.inputs_file}")
        for key in self.inputs:
            print(f"  {key}: {self.inputs[key]}")

        # setup device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        print(f"Using device: {self.device}")

        # instantiate model, send to device, and load model parameters
        fields = list(dataclasses.fields(Multi_Features_Model))
        for i_field, field_name in enumerate([field.name for field in fields]):
            if field_name == 'logger':
                fields.pop(i_field)
        model_kwargs = {field.name: self.inputs[field.name] for field in fields}
        self.model = Multi_Features_Model(**model_kwargs)
        self.model = self.model.to(self.device)
        self._load_model_parameters()
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
        # restore test data
        # data_partition_file = self.output_dir / 'data_partition.yaml'
        # with data_partition_file.open('r') as file:
        #     data_partition = yaml.safe_load(file)
        # test_elms = data_partition['test_elms']
        # with h5py.File(self.inputs['data_location'], 'r') as h5_file:
        #     for elm_index in test_elms:
        #         elm_key = f"{elm_index:05d}"
        #         elm_event = h5_file[elm_key]
        #         signals = np.array(elm_event["signals"], dtype=np.float32)  # (64, <time>)
        #         signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)  # reshape to (<time>, 8, 8)
        #         labels = np.array(elm_event["labels"])
        #         # labels, signals, valid_t0 = self._get_valid_indices(labels, signals)
        #         # assert labels.size == valid_t0.size
        # return
        test_data_file = self.output_dir / self.inputs['test_data_file']
        assert test_data_file.exists(), f"{test_data_file} does not exist."
        with test_data_file.open('rb') as file:
            self.test_data = pickle.load(file)
        return

    def _load_model_parameters(self) -> None:
        checkpoint_file = self.output_dir / self.inputs['checkpoint_file']
        assert checkpoint_file.exists(), f"{checkpoint_file} does not exist"
        model_state_dict = torch.load(
            checkpoint_file,
            map_location=self.device,
        )
        self.model.load_state_dict(model_state_dict)

    def run_inference(self) -> None:
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
                elm_test_data_loader = torch.utils.data.DataLoader(
                    dataset=elm_test_dataset,
                    batch_size=128,
                    shuffle=False,
                    num_workers=self.inputs['num_workers'],
                    pin_memory=False,
                    drop_last=False,
                )
                # loop over batches for single ELM
                elm_predictions = np.empty(0, dtype=np.float32)
                for batch_signals, _ in elm_test_data_loader:
                    batch_signals = batch_signals.to(self.device)
                    batch_predictions = self.model(batch_signals)
                    if self.is_classification:
                        # if evaluation/inference mode and classificaiton model,
                        # apply sigmoid to transform [-inf,inf] logit -> [0,1] probability
                        batch_predictions = batch_predictions.sigmoid()
                    elm_predictions = np.append(elm_predictions, batch_predictions.cpu().numpy())
                self.all_labels.append(elm_labels)
                self.all_predictions.append(elm_predictions)
                self.all_signals.append(elm_signals[:,2,3])
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
            print(f'Saving training plot: {filepath}')
            plt.savefig(filepath, format='pdf', transparent=True)
        return

    def plot_inference(
        self,
        save: bool = False,  # save PDFs
    ) -> None:
        if self.test_data is None:
            print("Skipping inference, no test data")
            return
        if None in [self.all_labels, self.all_predictions, self.all_signals]:
            self.run_inference()
        n_elms = self.test_data['elm_indices'].size
        assert len(self.all_labels) == n_elms and \
            len(self.all_predictions) == n_elms and \
            len(self.all_signals) == n_elms
        prediction_offset = self.inputs['signal_window_size'] - 1
        if self.is_classification:
            prediction_offset += self.inputs['prediction_horizon']
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
            if self.is_classification:
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
            merge_pdfs(
                inputs=inputs,
                output=output,
                delete_inputs=True,
            )

    @staticmethod
    def show(*args, **kwargs) -> None:
        """
        Wrapper for `plt.show()`
        """
        plt.show(*args, **kwargs)