from pathlib import Path
from random import sample
from typing import Union
import pickle
import inspect
import time

import yaml
import numpy as np
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
        # setup device
        self.device = device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        # instantiate model, send to device, and load model parameters
        self.model = Multi_Features_Model(**self.inputs)
        self.model = self.model.to(self.device)
        self._restore_model_parameters()
        self.model.eval()

        self.is_regression = None

    def _restore_test_data(self):
        # restore test data
        test_data_file = self.output_dir / self.inputs['test_data_file']
        assert test_data_file.exists(), \
            f"{test_data_file} does not exist."
        with test_data_file.open('rb') as file:
            self.test_data = pickle.load(file)
        # self.test_dataset = ELM_Dataset(
        #     signals=self.test_data['signals'],
        #     labels=self.test_data['labels'],
        #     sample_indices=self.test_data['sample_indices'],
        #     window_start=self.test_data['window_start'],
        #     signal_window_size=self.inputs['signal_window_size'],
        #     prediction_horizon=0 if self.is_regression else self.inputs['prediction_horizon'],
        # )
        # self.test_data_loader = elm_data_loader(
        #     dataset=self.test_dataset,
        # )


    def _restore_model_parameters(
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
        # self.oversample_active_elm = None  # set by kwarg
        # self.prediction_horizon = None  # set with kwarg
        # self.threshold = None  # set with kwarg
        # self.inverse_weight_label = None  # not applicable for classification
        # self.log_time = None  # not applicable for classification
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
        # losses = np.array(0)
        self.all_predictions = []
        self.all_labels = []
        self.all_signals = []
        with torch.no_grad():
            # loop over ELMs in test data
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
                # print(i_start, i_stop, elm_labels.shape, elm_signals.shape, elm_sample_indices.shape)
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
                # elm_labels = np.empty(0, dtype=int)
                elm_predictions = np.empty(0)
                # elm_signals = np.empty([0,self.inputs['signal_window_size']])
                # n_batch = len(elm_test_data_loader)
                for i_batch, (batch_signals, batch_labels) in enumerate(elm_test_data_loader):
                    # if (i_batch+1)%self.inputs['minibatch_interval'] == 0:
                    #     t_start_minibatch = time.time()
                    batch_signals = batch_signals.to(self.device)
                    batch_predictions = self.model(batch_signals)
                    if not self.is_regression:
                        # if evaluation/inference mode and classificaiton model,
                        # apply sigmoid to transform [-inf,inf] logit -> [0,1] probability
                        batch_predictions = batch_predictions.sigmoid()
                    # elm_labels = np.append(elm_labels, batch_labels.cpu().numpy().astype(int))
                    elm_predictions = np.append(elm_predictions, batch_predictions.cpu().numpy())
                    # elm_signals = np.append(elm_signals, batch_signals[:,0,:,2,3], axis=0)
                    # if (i_batch+1)%self.inputs['minibatch_interval'] == 0:
                    #     info =  f"  Inference batch {i_batch+1:05d}/{n_batch} (single ELM)  "
                    #     info += f"minibatch time {time.time()-t_start_minibatch:.3f} s"
                    #     print(info)
                # print(elm_labels.shape, elm_predictions.shape, elm_signals.shape)
                self.all_labels.append(elm_labels)
                self.all_predictions.append(elm_predictions)
                self.all_signals.append(elm_signals[:,2,3])
        print('Inference complete')

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

