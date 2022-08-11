# python library imports
import contextlib
import inspect
import io
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Union, Iterable, Tuple

# 3rd-party package imports
import h5py
import numpy as np
import torch
import torch.utils.data
import torchinfo
import yaml
from sklearn import metrics

# repo import
from bes_data.sample_data import sample_elm_data_file

try:
    from .models import Multi_Features_Model
    from .data import ELM_Dataset, elm_data_loader
except ImportError:
    from bes_ml.base.models import Multi_Features_Model
    from bes_ml.base.data import ELM_Dataset, elm_data_loader

class _Trainer_Base(object):

    def __init__(
        self,
        data_location: Union[Path, str] = sample_elm_data_file,  # path to data
        output_dir: Union[Path,str] = 'run_dir',  # path to output dir.
        results_file: str = 'results.yaml',  # output training results
        log_file: str = 'log.txt',  # output log file
        inputs_file: str = 'inputs.yaml',  # save inputs to yaml
        test_data_file: str = 'test_data.pkl',  # if None, do not save test data (can be large)
        checkpoint_file: str = 'checkpoint.pytorch',  # pytorch save file; if None, do not save
        export_onnx: bool = False,  # export ONNX format
        device: str = 'auto',  # auto (default), cpu, cuda, or cuda:X
        num_workers: int = 0,  # number of subprocess workers for pytorch dataloader
        n_epochs: int = 2,  # training epochs
        batch_size: int = 64,  # power of 2, like 16-128
        minibatch_interval: int = 2000,  # print minibatch info
        signal_window_size: int = 128,  # power of 2, like 32-512
        fraction_validation: float = 0.1,  # fraction of dataset for validation
        fraction_test: float = 0.15,  # fraction of dataset for testing
        optimizer_type: str = 'adam',  # adam (default) or sgd
        sgd_momentum: float = 0.0,  # momentum for SGD optimizer
        sgd_dampening: float = 0.0,  # dampening for SGD optimizer
        learning_rate: float = 1e-3,  # optimizer learning rate
        weight_decay: float = 5e-3,  # optimizer L2 regularization factor
        batches_per_print: int = 5000,  # train/validation batches per print update
        # kwargs for model
        **model_kwargs,
    ) -> None:

        # input data file and output directory
        data_location = Path(data_location)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        self.data_location = data_location
        self.output_dir = output_dir
        self.results_file = results_file
        self.log_file = log_file
        self.inputs_file = inputs_file
        self.test_data_file = test_data_file
        self.checkpoint_file = checkpoint_file
        self.export_onnx = export_onnx
        self.device = device
        self.num_workers = num_workers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.minibatch_interval = minibatch_interval
        self.signal_window_size = signal_window_size
        self.fraction_validation = fraction_validation
        self.fraction_test = fraction_test
        self.optimizer_type = optimizer_type
        self.sgd_momentum = sgd_momentum
        self.sgd_dampening = sgd_dampening
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batches_per_print = batches_per_print
        self.model_kwargs = model_kwargs

        self.is_regression = None  # set in subclass
        self.task = None # set in velocimetry and turbulence subclasses

        # create logger (logs to file and terminal)
        self.logger = None
        self._create_logger()

    def _validate_subclass_inputs(self) -> None:
        """
        Ensure subclass call signature contains all parameters in
        parent class signature and model class signature
        """
        assert self.__class__ is not _Trainer_Base
        subclass_parameters = inspect.signature(self.__class__).parameters
        for cls in [_Trainer_Base, Multi_Features_Model]:
            class_parameters = inspect.signature(cls).parameters
            for param_name in class_parameters:
                if param_name in ['model_kwargs', 'logger', 'kwargs']:
                    continue
                assert param_name in subclass_parameters, \
                    [f"Subclass {self.__class__.__name__} "
                     f"missing parameter {param_name} from class {cls.__name__}."]

    def _create_parent_class_inputs(self, locals_copy: dict = None) -> dict:
        assert self.__class__ is not _Trainer_Base
        kwargs_for_parent_class = {}
        for cls in [_Trainer_Base, Multi_Features_Model]:
            class_parameters = inspect.signature(cls).parameters
            for parameter_name in class_parameters:
                if parameter_name in locals_copy:
                    kwargs_for_parent_class[parameter_name] = locals_copy[parameter_name]
        return kwargs_for_parent_class

    def _print_inputs(
        self,
        locals_copy: dict = None,
        logger: logging.Logger = None,
    ) -> None:
        # print kwargs from __init__
        logger.info(f"Class `{self.__class__.__name__}` parameters:")
        class_parameters = inspect.signature(self.__class__).parameters
        for p_name in class_parameters:
            if p_name == 'logger': continue
            local_value = locals_copy[p_name]
            default_value = class_parameters[p_name].default
            if isinstance(local_value, dict):
                local_value = default_value = '<dict>'
            if local_value == default_value:
                logger.info(f"  {p_name:24s}:  {local_value}")
            else:
                logger.info(f"  {p_name:24s}:  {local_value}  (default {default_value})")

    def _save_inputs_to_yaml(
        self, 
        locals_copy: dict = None,
        filename: Union[str,Path] = None,
    ) -> None:
        """
        Save locals from __init__() call to yaml.
        """
        filename = Path(filename)
        parameters = inspect.signature(self.__class__).parameters
        inputs = {}
        for p_name in parameters:
            if p_name == 'logger': continue
            value = locals_copy[p_name]
            inputs[p_name] = value if not isinstance(value, Path) else value.as_posix()
        with filename.open('w') as parameters_file:
            yaml.safe_dump(
                inputs,
                parameters_file,
                default_flow_style=False,
                sort_keys=False,
            )

    def _create_logger(self) -> None:
        """
        Use python's logging to allow simultaneous print to console and log file.
        """
        self.logger = logging.getLogger(name=__name__)
        self.logger.setLevel(logging.INFO)

        # logs to log file
        log_file = self.output_dir / self.log_file
        f_handler = logging.FileHandler(log_file.as_posix(), mode="w")
        # create formatters and add it to the handlers
        f_format = logging.Formatter("%(asctime)s:  %(message)s")
        f_handler.setFormatter(f_format)
        # add handlers to the logger
        self.logger.addHandler(f_handler)

        # logs to console
        s_handler = logging.StreamHandler()
        self.logger.addHandler(s_handler)

    def _set_regression_or_classification_defaults(self) -> None:
        """
        Set defaults for regression or classification tasks.
        """
        self.oversample_active_elm = None  # set by kwarg
        self.prediction_horizon = None  # set with kwarg
        self.threshold = None  # set with kwarg
        self.inverse_weight_label = None  # not applicable for classification
        self.log_time = None  # not applicable for classification
        if self.is_regression:
            # regression model (e.g. time to ELM onset)
            self.loss_function = torch.nn.MSELoss(reduction="none")
            self.score_function = metrics.r2_score
        else:
            # classification model (e.g. active ELM prediction for `prediction_horizon` horizon
            self.loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")
            self.score_function = metrics.f1_score

    def make_model_and_device(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        self.logger.info(f"Device: {self.device}")

        # make model
        self.model = Multi_Features_Model(
            logger=self.logger,
            signal_window_size=self.signal_window_size,
            **self.model_kwargs,
        )

        self.model = self.model.to(self.device)

        self.input_size = None
        self._print_model_summary()

    def _finish_subclass_initialization(self) -> None:
        """
        Finalize subclass initialization.
        """

        if self.log_time is False and self.inverse_weight_label is True:
            self.inverse_weight_label = False
            self.logger.info("WARNING: setting `inverse_weight_time` to False; required for log_time==False")

        self.optimizer = None
        self.scheduler = None
        self._make_optimizer_scheduler_loss()

        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self._get_data()
        
        self.train_dataset = None
        self.validation_dataset = None
        self._make_datasets()

        self.train_data_loader = None
        self.validation_data_loader = None
        self._make_data_loaders()

        if self.test_data_file and self.fraction_test>0.0:
            self._save_test_data()

        self.results = {}

    def _get_data(self) -> None:
        self.data_location = self.data_location.resolve()
        assert self.data_location.exists(), f"{self.data_location} does not exist"
        self.logger.info(f"Data file: {self.data_location}")

        with h5py.File(self.data_location, "r") as data_file:
            elm_indices = np.array(
                [int(key) for key in data_file], 
                dtype=np.int32,
            )
            time_frames = sum([data_file[key]['time'].shape[0] for key in data_file])
        self.logger.info(f"Events in data file: {elm_indices.size}")
        self.logger.info(f"Total time frames: {time_frames}")

        np.random.shuffle(elm_indices)
        if self.max_elms:
            elm_indices = elm_indices[:self.max_elms]
            self.logger.info(f"Limiting data to {self.max_elms} ELM events")

        n_validation_elms = int(self.fraction_validation * elm_indices.size)
        n_test_elms = int(self.fraction_test * elm_indices.size)

        test_elms, validation_elms, training_elms = np.split(
            elm_indices,
            [n_test_elms, n_test_elms+n_validation_elms]
        )

        self.logger.info(f"Training ELM events: {training_elms.size}")
        self.train_data = self._preprocess_data(
            training_elms,
            shuffle_indices=True,
            oversample_active_elm=self.oversample_active_elm,
        )

        self.logger.info(f"Validation ELM events: {validation_elms.size}")
        self.validation_data = self._preprocess_data(
            validation_elms,
            shuffle_indices=False,
            oversample_active_elm=False,
        )

        if self.fraction_test > 0.0:
            self.logger.info(f"Test ELM events: {test_elms.size}")
            self.test_data = self._preprocess_data(
                test_elms,
                shuffle_indices=False,
                oversample_active_elm=False,
            )
        else:
            self.logger.info("Skipping test data")
            self.test_data = None

    def _save_test_data(self) -> None:
        """
        Save test data to pickle file.
        """
        test_data_file = self.output_dir / self.test_data_file
        self.logger.info(f"Test data file: {test_data_file}")
        with test_data_file.open('wb') as file:
            pickle.dump(
                {
                    "signals": self.test_data[0],
                    "labels": self.test_data[1],
                    "sample_indices": self.test_data[2],
                    "window_start": self.test_data[3],
                    "elm_indices": self.test_data[4],
                },
                file,
            )
        self.logger.info(f"  File size: {test_data_file.stat().st_size/1e6:.1f} MB")

    def _preprocess_data(
        self,
        elm_indices: Iterable = None,
        shuffle_indices: bool = False,
        oversample_active_elm: bool = False,
    ) -> None:
        packaged_signals = None
        packaged_window_start = None
        packaged_valid_t0 = []
        packaged_labels = []
        with h5py.File(self.data_location, 'r') as h5_file:
            for elm_index in elm_indices:
                elm_key = f"{elm_index:05d}"
                elm_event = h5_file[elm_key]
                signals = np.array(elm_event["signals"], dtype=np.float32)
                # transpose so time dim. first
                signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)
                if self.is_regression:
                    label_type = np.float32
                else:
                    label_type = np.int8
                try:
                    labels = np.array(elm_event["labels"], dtype=label_type)
                except KeyError:
                    labels = np.array(elm_event["manual_labels"], dtype=label_type)
                labels, signals, valid_t0 = self._get_valid_indices(labels, signals)
                if packaged_signals is None:
                    packaged_window_start = np.array([0])
                    packaged_valid_t0 = valid_t0
                    packaged_signals = signals
                    packaged_labels = labels
                else:
                    last_index = packaged_labels.size - 1
                    packaged_window_start = np.append(
                        packaged_window_start, 
                        last_index + 1
                    )
                    packaged_valid_t0 = np.concatenate([packaged_valid_t0, valid_t0])
                    packaged_signals = np.concatenate([packaged_signals, signals], axis=0)
                    packaged_labels = np.concatenate([packaged_labels, labels], axis=0)                

        # valid indices for data sampling
        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype="int")
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]

        if not self.is_regression:
            packaged_valid_t0_indices = self._check_for_balanced_data(
                packaged_labels=packaged_labels,
                packaged_valid_t0_indices=packaged_valid_t0_indices,
                oversample_active_elm=oversample_active_elm,
            )

        if shuffle_indices:
            np.random.shuffle(packaged_valid_t0_indices)

        self.logger.info( "  Data tensors -> signals, labels, sample_indices, window_start_indices:")
        for tensor in [
            packaged_signals,
            packaged_labels,
            packaged_valid_t0_indices,
            packaged_window_start,
        ]:
            tmp = f"  shape {tensor.shape}, dtype {tensor.dtype},"
            tmp += f" min {np.min(tensor):.3f}, max {np.max(tensor):.3f}"
            if hasattr(tensor, "device"):
                tmp += f" device {tensor.device[-5:]}"
            self.logger.info(tmp)

        return (
            packaged_signals, 
            packaged_labels, 
            packaged_valid_t0_indices, 
            packaged_window_start, 
            elm_indices,
        )

    def _check_for_balanced_data(self) -> None:
        # must implement in subclass
        raise NotImplementedError

    def _make_datasets(self) -> None:
        self.train_dataset = ELM_Dataset(
            *self.train_data[0:4], 
            signal_window_size = self.signal_window_size,
            prediction_horizon = self.prediction_horizon,
        )
        self.validation_dataset = ELM_Dataset(
            *self.validation_data[0:4], 
            signal_window_size = self.signal_window_size,
            prediction_horizon = self.prediction_horizon,
        )

    def _get_valid_indices(self) -> None:
        # must implement in subclass
        raise NotImplementedError

    def _make_data_loaders(self) -> None:
        self.train_data_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        self.validation_data_loader = torch.utils.data.DataLoader(
                self.validation_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )

    def _print_model_summary(self) -> None:
        self.logger.info("MODEL SUMMARY")

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.input_size = (
            self.batch_size,
            1,
            self.signal_window_size,
            8,
            8,
        )
        x = torch.rand(*self.input_size)
        x = x.to(self.device)
        tmp_io = io.StringIO()
        sys.stdout = tmp_io
        print()
        torchinfo.summary(self.model, input_size=self.input_size, device=self.device)
        sys.stdout = sys.__stdout__
        self.logger.info(tmp_io.getvalue())
        self.logger.info(f"Model contains {n_params} trainable parameters")
        self.logger.info(f'Batched input size: {x.shape}')
        self.logger.info(f"Batched output size: {self.model(x).shape}")

    def _make_optimizer_scheduler_loss(self) -> None:
        if self.optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay,
                momentum=self.sgd_momentum,
                dampening=self.sgd_dampening,
            )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True,
        )

    def train(self) -> None:
        assert 'scores_label' in self.results, \
            f"Subclass must define self.results['scores_label']"
        best_score = -np.inf
        self.results['train_loss'] = []
        self.results['valid_loss'] = []
        self.results['scores'] = []
        checkpoint_file = self.output_dir / self.checkpoint_file

        if not self.is_regression:
            self.results['roc_scores'] = []

        # send model to device
        self.model = self.model.to(self.device)

        self.logger.info(f"Batches per epoch {len(self.train_data_loader)}")
        self.logger.info(f"Begin training loop over {self.n_epochs} epochs")
        t_start_training = time.time()
        # loop over epochs
        for i_epoch in range(self.n_epochs):
            t_start_epoch = time.time()

            self.logger.info(f"Ep {i_epoch+1:03d}: begin")
            
            # train_loss = self._train_epoch()
            train_loss = self._single_epoch_loop(
                is_train=True,
                data_loader=self.train_data_loader,
            )
            if self.is_regression:
                train_loss = np.sqrt(train_loss)

            self.results['train_loss'].append(train_loss.item())

            # valid_loss, predictions, true_labels = self._validation_epoch()
            valid_loss, predictions, true_labels = self._single_epoch_loop(
                is_train=False,
                data_loader=self.validation_data_loader,
            )
            if self.is_regression:
                # regression loss is MSE, so take sqrt to get units of time
                valid_loss = np.sqrt(valid_loss)

            self.results['valid_loss'].append(valid_loss.item())

            # apply learning rate scheduler
            self.scheduler.step(valid_loss)

            if self.is_regression:
                score = self.score_function(true_labels, predictions)
            else:
                prediction_labels = (predictions > self.threshold).astype(int)
                score = self.score_function(
                    true_labels,
                    prediction_labels,
                )

            self.results['scores'].append(score.item())

            if not self.is_regression:
                # ROC-AUC score for classification
                roc_score = metrics.roc_auc_score(
                    true_labels,
                    predictions,
                )
                self.results['roc_scores'].append(roc_score.item())

            with (self.output_dir/self.results_file).open('w') as results_file:
                yaml.dump(
                    self.results,
                    results_file,
                    default_flow_style=False,
                )

            # best score and save model
            if score > best_score:
                best_score = score
                self.logger.info(f"Ep {i_epoch+1:03d}: Best score {best_score:.3f}, saving model...")
                self.logger.info(f"Saving model to: {checkpoint_file.as_posix()}")
                torch.save(self.model.state_dict(), checkpoint_file.as_posix())
                self.logger.info(f"  File size: {checkpoint_file.stat().st_size/1e3:.1f} kB")                
                if self.export_onnx:
                    onnx_file = self.output_dir / 'checkpoint.onnx'
                    self.logger.info(f"Saving to ONNX: {onnx_file.as_posix()}")
                    torch.onnx.export(
                        self.model, 
                        torch.rand(*self.input_size)[0].unsqueeze(0),
                        onnx_file.as_posix(),
                        input_names=['signal_window'],
                        output_names=['micro_prediction'],
                        verbose=True,
                        opset_version=11
                    )
                    self.logger.info(f"  File size: {onnx_file.stat().st_size/1e3:.1f} kB")                

            prediction_labels =  f"Ep {i_epoch+1:03d}: "
            prediction_labels += f"train loss {train_loss:.3f}  "
            prediction_labels += f"val loss {valid_loss:.3f}  "
            prediction_labels += f"score {score:.3f}  "
            if not self.is_regression:
                prediction_labels += f"roc {roc_score:.3f}  "
            prediction_labels += f"ep time {time.time()-t_start_epoch:.1f} s "
            prediction_labels += f"(total time {time.time()-t_start_training:.1f} s)"
            self.logger.info(prediction_labels)

        self.logger.info(f"End training loop")
        self.logger.info(f"Elapsed time {time.time()-t_start_training:.1f} s")

    def _single_epoch_loop(
        self,
        is_train: bool = True,  # True for train, False for evaluation/inference
        data_loader: torch.utils.data.DataLoader = None,  # train or validation data loader
    ) -> Union[np.ndarray, Tuple]:
        losses = np.array(0)
        all_predictions = []
        all_labels = []
        if is_train:
            self.model.train()
            context = contextlib.nullcontext()
            mode = 'Train'
        else:
            self.model.eval()
            context = torch.no_grad()
            mode = 'Valid'
        with context:
            for i_batch, (signal_windows, labels) in enumerate(data_loader):
                if (i_batch+1) % self.minibatch_interval == 0:
                    t_start_minibatch = time.time()
                if is_train:
                    # reset grads
                    self.optimizer.zero_grad()
                signal_windows = signal_windows.to(self.device)
                labels = labels.to(self.device)
                predictions = self.model(signal_windows)
                if not is_train and not self.is_regression:
                    # if evaluation/inference mode and classificaiton model,
                    # apply sigmoid to get [0,1] probability
                    predictions = predictions.sigmoid()
                loss = self.loss_function(
                    predictions.squeeze(),
                    labels.type_as(predictions),
                )
                if self.is_regression and self.inverse_weight_label:
                    loss = torch.div(loss, labels)
                loss = loss.mean()  # batch loss
                losses = np.append(losses, loss.detach().cpu().numpy())  # track batch losses
                if is_train:
                    # backpropagate and take optimizer step
                    loss.backward()
                    self.optimizer.step()
                else:
                    all_labels.append(labels.cpu().numpy())
                    all_predictions.append(predictions.cpu().numpy())
                if (i_batch+1)%self.minibatch_interval == 0:
                    tmp =  f"  {mode} batch {i_batch+1:05d}/{len(self.train_data_loader)}  "
                    tmp += f"batch loss {loss:.3f} (avg loss {losses.mean():.3f})  "
                    tmp += f"minibatch time {time.time()-t_start_minibatch:.3f} s"
                    self.logger.info(tmp)
        if is_train:
            return_value = losses.mean()
        else:
            all_labels = np.concatenate(all_labels)
            all_predictions = np.concatenate(all_predictions)
            return_value = (
                losses.mean(),
                all_predictions,
                all_labels,
            )
        return return_value


if __name__=='__main__':
    m = _Trainer_Base()
