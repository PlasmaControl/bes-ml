# python library imports
import contextlib
import io
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Union, Iterable, Tuple
import dataclasses

# 3rd-party imports
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchinfo
import yaml
from sklearn import metrics

try:
    import optuna  # available on conda-forge
except ImportError:
    optuna = None

# repo import
from bes_data.sample_data import sample_elm_data_file
try:
    from .models import Multi_Features_Model, _Multi_Features_Model_Dataclass
    from .utilities import merge_pdfs
except ImportError:
    from bes_ml.base.models import Multi_Features_Model, _Multi_Features_Model_Dataclass
    from bes_ml.base.utilities import merge_pdfs


@dataclasses.dataclass
class _Trainer_Base(_Multi_Features_Model_Dataclass):
    data_location: Union[Path, str] = sample_elm_data_file  # path to data; dir or file depending on task
    output_dir: Union[Path,str] = Path('run_dir')  # path to output dir.
    results_file: str = 'results.yaml'  # output training results
    log_file: str = 'log.txt'  # output log file
    inputs_file: str = 'inputs.yaml'  # save inputs to yaml
    test_data_file: str = 'test_data.pkl'  # if None, do not save test data (can be large)
    checkpoint_file: str = 'checkpoint.pytorch'  # pytorch save file; if None, do not save
    export_onnx: bool = False  # export ONNX format
    device: str = 'auto'  # auto (default), cpu, cuda, or cuda:X
    num_workers: int = 0  # number of subprocess workers for pytorch dataloader
    n_epochs: int = 2  # training epochs
    batch_size: int = 64  # power of 2, like 16-128
    minibatch_interval: int = 2000  # print minibatch info
    signal_window_size: int = 128  # power of 2, like 32-512
    fraction_validation: float = 0.1  # fraction of dataset for validation
    fraction_test: float = 0.15  # fraction of dataset for testing
    optimizer_type: str = 'adam'  # adam (default) or sgd
    sgd_momentum: float = 0.0  # momentum for SGD optimizer
    sgd_dampening: float = 0.0  # dampening for SGD optimizer
    learning_rate: float = 1e-3  # optimizer learning rate
    weight_decay: float = 5e-3  # optimizer L2 regularization factor
    batches_per_print: int = 5000  # train/validation batches per print update
    logger: logging.Logger = None
    seed: int = None  # RNG seed for deterministic, reproducable shuffling (ELMs, sample indices, etc.)
    trial = None  # optuna trial

    def __post_init__(self):
        self.data_location = Path(self.data_location)
        assert self.data_location.exists(), f"{self.data_location} does not exist"

        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # subclass must set either is_regression or is_classification to True
        self.is_regression = False
        self.is_classification = False

        self.rng_generator = np.random.default_rng(seed=self.seed)

        self._create_logger()
        self._print_inputs()
        self._save_inputs_to_yaml()

    def _create_logger(self):
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

    def _print_inputs(self):
        cls = self.__class__
        self.logger.info(f"Class `{cls.__name__}` parameters:")
        cls_fields_tuple = dataclasses.fields(cls)
        self_fields_dict = dataclasses.asdict(self)
        assert set([field.name for field in cls_fields_tuple]) == set(self_fields_dict.keys())
        for field in cls_fields_tuple:
            if field.name == 'logger': continue
            if self_fields_dict[field.name] == field.default:
                tmp = f"  {field.name}: {self_fields_dict[field.name]}"
            else:
                tmp = f"  {field.name}: {self_fields_dict[field.name]}  (default {field.default})"
            self.logger.info(tmp)

    def _save_inputs_to_yaml(self):
        filename = Path(self.output_dir / self.inputs_file)
        self_fields_dict = dataclasses.asdict(self)
        self_fields_dict.pop('logger')
        for key in self_fields_dict:
            if isinstance(self_fields_dict[key], Path):
                self_fields_dict[key] = self_fields_dict[key].as_posix()
        with filename.open('w') as parameters_file:
            yaml.safe_dump(
                self_fields_dict,
                parameters_file,
                default_flow_style=False,
                sort_keys=False,
            )

    def make_model_and_set_device(self):

        # make model
        model_kwargs = {
            field.name: getattr(self, field.name) 
            for field in dataclasses.fields(Multi_Features_Model)
        }
        self.model = Multi_Features_Model(**model_kwargs)

        # setup device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        self.logger.info(f"Device: {self.device}")

        # send model to device
        self.model = self.model.to(self.device)

        # print model summary
        self.input_shape = (self.batch_size, 1, self.signal_window_size, 8 * self.sinterp, 8 * self.sinterp)
        self._print_model_summary()

    def _print_model_summary(self):
        self.logger.info("MODEL SUMMARY")

        # catpure torchinfo.summary() output
        tmp_io = io.StringIO()
        sys.stdout = tmp_io
        print()
        torchinfo.summary(self.model, input_size=self.input_shape, device=self.device)
        sys.stdout = sys.__stdout__
        # print model summary
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        x = torch.rand(*self.input_shape)
        x = x.to(self.device)
        self.logger.info(tmp_io.getvalue())
        self.logger.info(f"Model contains {n_params} trainable parameters")
        self.logger.info(f'Batched input size: {x.shape}')
        self.logger.info(f"Batched output size: {self.model(x).shape}")

    def finish_subclass_initialization(self) -> None:
        # subclass must set is_regression XOR is_classification
        assert self.is_regression ^ self.is_classification  # XOR
        if self.is_regression:
            # regression model (e.g. time to ELM onset)
            self.loss_function = torch.nn.MSELoss(reduction="none")
            self.score_function = metrics.r2_score
            self.score_function_name = 'R2'
        elif self.is_classification:
            if self.model.mlp_output_size == 1:
                # binary classification (e.g. active ELM?)
                self.loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")
            else:
                # multi-class classification (e.g. confinement mode)
                self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")
            self.score_function = metrics.f1_score
            self.score_function_name = 'F1'
        assert (
            self.loss_function and 
            self.score_function and 
            self.score_function_name
        )

        self.optimizer = None
        self.lr_scheduler = None
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
        assert self.train_data_loader

        if self.test_data_file and self.fraction_test>0.0:
            self._save_test_data()

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
        assert test_data_file.exists(), f"{test_data_file} does not exist"
        self.logger.info(f"  File size: {test_data_file.stat().st_size/1e6:.1f} MB")

    def _get_data(self) -> None:
        self.data_location = self.data_location.resolve()
        assert self.data_location.exists(), f"{self.data_location} does not exist"
        self.logger.info(f"Data file: {self.data_location}")

        with h5py.File(self.data_location, "r") as data_file:
            elm_indices = np.array(
                [int(key) for key in data_file], 
                dtype=int,
            )
            time_frames = sum([data_file[key]['time'].shape[0] for key in data_file])
        self.logger.info(f"Events in data file: {elm_indices.size}")
        self.logger.info(f"Total time frames: {time_frames}")

        self.rng_generator.shuffle(elm_indices)

        # TODO: remove `max_elms` from base class
        if hasattr(self, 'max_elms') and self.max_elms:
            elm_indices = elm_indices[:self.max_elms]
            self.logger.info(f"Limiting data to {self.max_elms} ELM events")

        n_validation_elms = int(self.fraction_validation * elm_indices.size)
        n_test_elms = int(self.fraction_test * elm_indices.size)

        test_elms, validation_elms, training_elms = np.split(
            elm_indices,
            [n_test_elms, n_test_elms+n_validation_elms]
        )

        if not hasattr(self, 'oversample_active_elm'):  # TODO: remove `oversample_active_elm` from base class
            self.oversample_active_elm = False

        self.logger.info(f"Training ELM events: {training_elms.size}")
        self.train_data = self._preprocess_data(
            elm_indices=training_elms,
            shuffle_indices=True,
            oversample_active_elm=self.oversample_active_elm,
        )

        if self.fraction_validation > 0.0:
            self.logger.info(f"Validation ELM events: {validation_elms.size}")
            self.validation_data = self._preprocess_data(
                elm_indices=validation_elms,
                shuffle_indices=False,
                oversample_active_elm=False,
                save_filename='validation_elms',
            )
        else:
            self.logger.info("Skipping validation data")

        if self.fraction_test > 0.0:
            self.logger.info(f"Test ELM events: {test_elms.size}")
            self.test_data = self._preprocess_data(
                elm_indices=test_elms,
                shuffle_indices=False,
                oversample_active_elm=False,
                save_filename='test_elms',
            )
        else:
            self.logger.info("Skipping test data")

    def _preprocess_data(
        self,
        elm_indices: Iterable = None,
        shuffle_indices: bool = False,
        oversample_active_elm: bool = False,
        save_filename: str = '',
    ) -> None:
        packaged_signals = None
        packaged_window_start = None
        packaged_valid_t0 = []
        packaged_labels = []
        if self.is_regression:
            label_type = np.float32
        elif self.is_classification:
            label_type = np.int8
        assert label_type
        if save_filename:
            plt.ioff()
            _, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9))
            self.logger.info(f"  Plotting valid indices: {save_filename}.pdf")
            i_page = 1
        with h5py.File(self.data_location, 'r') as h5_file:
            for i_elm, elm_index in enumerate(elm_indices):
                if save_filename and i_elm%12==0:
                    for axis in axes.flat:
                        plt.sca(axis)
                        plt.cla()
                elm_key = f"{elm_index:05d}"
                elm_event = h5_file[elm_key]
                signals = np.array(elm_event["signals"], dtype=np.float32)  # (64, <time>)
                signals = np.transpose(signals, (1, 0)).reshape(-1, 8, 8)  # reshape to (<time>, 8, 8)
                try:
                    labels = np.array(elm_event["labels"], dtype=label_type)
                except KeyError:
                    labels = np.array(elm_event["manual_labels"], dtype=label_type)
                labels, signals, valid_t0 = self._get_valid_indices(labels, signals)
                if save_filename:
                    plt.sca(axes.flat[i_elm%12])
                    plt.plot(signals[:,2,3]/10, label='BES 20')
                    plt.plot(signals[:,2,5]/10, label='BES 22')
                    plt.plot(labels, label='Label')
                    plt.title(f"ELM index {elm_key}")
                    plt.legend(fontsize='x-small')
                    plt.xlabel('Time (mu-s)')
                    if i_elm%12==11 or i_elm==elm_indices.size-1:
                        plt.tight_layout()
                        output_file = self.output_dir/(save_filename + f"_{i_page:02d}.pdf")
                        plt.savefig(
                            output_file, 
                            format="pdf", 
                            transparent=True,
                        )
                        i_page += 1
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

        if save_filename:
            plt.close()
            pdf_files = sorted(self.output_dir.glob(f'{save_filename}_*.pdf'))
            output = self.output_dir / f'{save_filename}.pdf'
            merge_pdfs(pdf_files, output, delete_inputs=True)
        
        # valid indices for data sampling
        packaged_valid_t0_indices = np.arange(packaged_valid_t0.size, dtype="int")
        packaged_valid_t0_indices = packaged_valid_t0_indices[packaged_valid_t0 == 1]

        if self.is_classification:
            packaged_valid_t0_indices = self._check_for_balanced_data(
                packaged_labels=packaged_labels,
                packaged_valid_t0_indices=packaged_valid_t0_indices,
                oversample_active_elm=oversample_active_elm,
            )

        if shuffle_indices:
            self.rng_generator.shuffle(packaged_valid_t0_indices)

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
        # must implement in subclass
        raise NotImplementedError

    def _get_valid_indices(self) -> None:
        # must implement in subclass
        raise NotImplementedError

    def _make_data_loaders(self) -> None:
        self.train_data_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True if self.seed is None else False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        if self.fraction_validation > 0.0:
            self.validation_data_loader = torch.utils.data.DataLoader(
                    self.validation_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=True,
                )

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
        assert self.optimizer
        self.logger.info(
            f"Optimizer {self.optimizer_type.upper()} " +
            f"with learning rate {self.learning_rate:.1e} " +
            f"and weight decay {self.weight_decay:.1e}"
        )

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True,
        )

    def train(self) -> dict:
        best_score = -np.inf
        self.results = {
            'train_loss': [],
            'valid_loss': [],
            'scores': [],
            'scores_label': self.score_function_name,
        }
        checkpoint_file = self.output_dir / self.checkpoint_file

        if self.is_classification:
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

            score = None
            valid_loss = None
            if self.validation_data_loader is not None:
                valid_loss, predictions, true_labels = self._single_epoch_loop(
                    is_train=False,
                    data_loader=self.validation_data_loader,
                )
                if self.is_regression:
                    # regression loss is MSE, so take sqrt to get units of time
                    valid_loss = np.sqrt(valid_loss)

                self.results['valid_loss'].append(valid_loss.item())

                # apply learning rate scheduler
                self.lr_scheduler.step(valid_loss)

                if self.is_regression:
                    score = self.score_function(true_labels, predictions)
                elif self.is_classification:
                    if self.model.mlp_output_size == 1:
                        assert hasattr(self, 'threshold')
                        prediction_labels = (predictions > self.threshold).astype(int)
                        score = self.score_function(
                            true_labels,
                            prediction_labels,
                        )
                    else:
                        prediction_labels = predictions.argmax(axis=1)
                        score = self.score_function(
                            true_labels,
                            prediction_labels,
                            average='weighted'
                        )
                assert score is not None
                self.results['scores'].append(score.item())

                # ROC-AUC score for classification
                if self.is_classification:
                    if self.model.mlp_output_size == 1:
                        roc_score = metrics.roc_auc_score(
                            true_labels,
                            predictions,
                        )
                    else:
                        one_hot = np.zeros_like(predictions)
                        for i, j in zip(one_hot, true_labels):
                            i[j] = 1
                        try:
                            roc_score = metrics.roc_auc_score(
                                one_hot,
                                predictions,
                                multi_class='ovo', 
                                average='macro', 
                                labels=[0, 1, 2, 3],
                            )
                        except:
                            roc_score = np.float32(0)
                    self.results['roc_scores'].append(roc_score.item())

            with (self.output_dir/self.results_file).open('w') as results_file:
                yaml.dump(
                    self.results,
                    results_file,
                    default_flow_style=False,
                )

            # best score and save model
            if score is None or score > best_score:
                if score is not None:
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
                        torch.rand(*self.input_shape)[0].unsqueeze(0),
                        onnx_file.as_posix(),
                        input_names=['signal_window'],
                        output_names=['micro_prediction'],
                        verbose=True,
                        opset_version=11
                    )
                    self.logger.info(f"  File size: {onnx_file.stat().st_size/1e3:.1f} kB")

            # report epoch result to optuna
            if optuna is not None and self.trial is not None:
                assert score is not None
                self.trial.report(score, i_epoch)
                # save outputs as lists in trial user attributes
                for key in self.results:
                    self.trial.set_user_attr(key, self.results[key])
                if self.trial.should_prune():
                    self.logger.info("==> Pruning trial with Optuna")
                    for handler in self.logger.handlers[:]:
                        handler.close()
                        self.logger.removeHandler(handler)
                    optuna.TrialPruned()

            prediction_labels =  f"Ep {i_epoch+1:03d}: "
            prediction_labels += f"train loss {train_loss:.3f}  "
            if score is not None and valid_loss is not None:
                prediction_labels += f"val loss {valid_loss:.3f}  "
                prediction_labels += f"{self.score_function_name} {score:.3f}  "
                if self.is_classification:
                    prediction_labels += f"ROC {roc_score:.3f}  "
            prediction_labels += f"ep time {time.time()-t_start_epoch:.1f} s "
            prediction_labels += f"(total time {time.time()-t_start_training:.1f} s)"
            self.logger.info(prediction_labels)

        self.logger.info(f"End training loop")
        self.logger.info(f"Elapsed time {time.time()-t_start_training:.1f} s")

        return self.results.copy()

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
                if not is_train and self.is_classification and self.model.mlp_output_size == 1:
                    # if evaluation/inference mode and classificaiton model,
                    # apply sigmoid to get [0,1] probability
                    predictions = predictions.sigmoid()
                    labels = labels.type_as(predictions)
                elif self.is_classification and self.model.mlp_output_size > 1:
                    # torch.nn.CrossEntropyLoss needs labels to be long and predictions not sigmoid
                    labels = labels.type(torch.long)
                else:
                    # Set only label type, leave predictions not sigmoid
                    labels = labels.type_as(predictions)
                loss = self.loss_function(
                    predictions.squeeze(),
                    labels,
                )
                if self.is_regression and hasattr(self, 'inverse_weight_loss') and self.inverse_weight_label:
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
    m = _Trainer_Base(dense_num_kernels=8)
