# python library imports
import contextlib
import logging
import time
from pathlib import Path
from typing import Union, Tuple, Any
import dataclasses
import time

# 3rd-party imports
import numpy as np
import torch
import torch.utils.data
import yaml
from sklearn import metrics

try:
    import optuna  # conda-forge
except ImportError:
    optuna = None

# repo import
try:
    from .models import Multi_Features_Model
except ImportError:
    from bes_ml.base.models import Multi_Features_Model


@dataclasses.dataclass(eq=False)
class _Base_Trainer_Dataclass:
    # output parameters
    output_dir: Path = Path('run_dir')  # path to output dir.
    results_file: str = 'results.yaml'  # output training results
    log_file: str = 'log.txt'  # output log file
    inputs_file: str = 'inputs.yaml'  # save inputs to yaml
    checkpoint_file: str = 'checkpoint.pytorch'  # pytorch save file
    save_onnx_model: bool = False  # export ONNX format
    onnx_checkpoint_file: str = 'checkpoint.onnx'  # onnx save file
    logger: logging.Logger = None
    terminal_output: bool = True  # terminal output if True
    # training parameters
    device: str = 'auto'  # auto (default), cpu, cuda, or cuda:X
    all_data_to_device: bool = False  # if True, send full dataset to device; if False (default) only send batches to device
    n_epochs: int = 2  # training epochs
    minibatch_print_interval: int = 2000  # print minibatch info
    optimizer_type: str = 'sgd'  # adam (default) or sgd
    sgd_momentum: float = 0.0  # momentum for SGD optimizer, 0-1
    sgd_dampening: float = 0.0  # dampening for SGD optimizer, 0-1
    learning_rate: float = 1e-3  # optimizer learning rate
    lr_scheduler_patience: int = 10  # epochs to wait before triggering lr scheduler
    lr_scheduler_factor: float = 0.5  # reduction factor for lr scheduler
    lr_scheduler_threshold: float = 1e-3  # threshold for *relative* decrease in loss to *not* trigger LR scheduler
    weight_decay: float = 1e-3  # optimizer L2 regularization factor
    # optuna integration
    optuna_trial: Any = None  # optuna trial
    # global parameters
    is_regression: bool = dataclasses.field(default=None, init=False)
    is_classification: bool = dataclasses.field(default=None, init=False)
    model: Multi_Features_Model = dataclasses.field(default=None, init=False)
    train_data_loader: torch.utils.data.DataLoader = dataclasses.field(default=None, init=False)
    validation_data_loader: torch.utils.data.DataLoader = dataclasses.field(default=None, init=False)
    results: dict = dataclasses.field(default=None, init=False)


@dataclasses.dataclass(eq=False)
class _Base_Trainer(_Base_Trainer_Dataclass):
    """Base class for model trainer"""

    def __post_init__(self):
        self.output_dir = Path(self.output_dir).resolve()
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self._create_logger()
        self._print_inputs()
        self._save_inputs_to_yaml()

        self.results = {
            'train_loss': [],
            'train_score': [],
            'epoch_time': [],
            'lr': [],
        }

        # subclass must set is_regression XOR is_classification
        assert self.is_regression ^ self.is_classification  # XOR

        self._make_model()
        self._set_device()
        self.model = self.model.to(self.device)
        self._make_optimizer_scheduler()
        self._prepare_data()

        # validate data loaders
        for data_loader in [self.train_data_loader, self.validation_data_loader]:
            assert isinstance(data_loader, torch.utils.data.DataLoader) or data_loader is None


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
        if self.terminal_output:
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
        for skip_key in ['logger', 'optuna_trial']:
            self_fields_dict.pop(skip_key)
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

    def _make_model(self) -> None:
        model_kwargs = {
            field.name: getattr(self, field.name) 
            for field in dataclasses.fields(Multi_Features_Model)
        }
        self.model = Multi_Features_Model(**model_kwargs)
        self.model.print_model_summary()

    def _set_device(self) -> None:
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        self.logger.info(f"Device: {self.device}")

    def _make_optimizer_scheduler(self) -> None:
        assert self.optimizer_type in ['adam', 'sgd']
        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.weight_decay,
                momentum=self.sgd_momentum,
                dampening=self.sgd_dampening,
            )
        self.logger.info(
            f"Optimizer {self.optimizer_type.upper()} " +
            f"with learning rate {self.learning_rate:.1e} " +
            f"and weight decay {self.weight_decay:.1e}"
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=self.lr_scheduler_factor,
            threshold=self.lr_scheduler_threshold,
            patience=self.lr_scheduler_patience,
            mode='min',
            verbose=True,
        )

    def _prepare_data(self) -> None:
        # implement in subclass
        # must generate self.train_data_loader and self.validation_data_loader
        raise NotImplementedError

    def train(self) -> dict:

        if self.is_regression:
            self.results['loss_function_name'] = 'MSELoss'
            self.loss_function = torch.nn.MSELoss(reduction="none")
        elif self.is_classification:
            if self.model.mlp_output_size == 1:
                assert hasattr(self, 'threshold')  # binary classification must specify threshold
                self.results['loss_function_name'] = 'BCEWithLogitsLoss'
                # labels are binary [0,1]; predictions are logits [-inf,inf]
                self.loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")
            else:
                self.results['loss_function_name'] = 'CrossEntropyLoss'
                # labels are true class C; predictions are scores for all classes
                self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")
        best_score = -np.inf
        best_epoch = 0

        self.logger.info(f"Training batches per epoch {len(self.train_data_loader)}")
        if self.validation_data_loader:
            self.logger.info(f"Validation batches per epoch {len(self.validation_data_loader)}")
        self.logger.info(f"Begin training loop over {self.n_epochs} epochs")
        t_start_training = time.time()
        valid_loss = valid_score = valid_roc = None
        # loop over epochs
        for i_epoch in range(self.n_epochs):
            t_start_epoch = time.time()
            self.logger.info(f"Ep {i_epoch+1:03d}: begin")
            self.results['lr'].append(self.optimizer.param_groups[0]['lr'])
            for is_train, data_loader in zip(
                [True, False],
                [self.train_data_loader, self.validation_data_loader],
            ):
                if not data_loader:
                    continue  # skip if validation data is empty
                # calculate loss and predictions
                loss, predictions, labels = self._single_epoch_loop(
                    is_train=is_train,
                    data_loader=data_loader,
                )
                # F1/R2 score
                if self.is_regression:
                    if 'score_function_name' not in self.results: 
                        self.results['score_function_name'] = 'R2'
                    score = metrics.r2_score(labels, predictions)
                elif self.is_classification:
                    if 'score_function_name' not in self.results: 
                        self.results['score_function_name'] = 'F1'
                    if self.model.mlp_output_size == 1:
                        modified_predictions = (predictions > self.threshold).astype(int)
                        score = metrics.f1_score(labels, modified_predictions, average='binary')
                    else:
                        modified_predictions = predictions.argmax(axis=1)  # select class with highest score
                        score = metrics.f1_score(labels, modified_predictions, average='weighted')
                # ROC score if classification
                if self.is_classification:
                    if self.model.mlp_output_size == 1:
                        roc = metrics.roc_auc_score(labels, predictions)
                    else:
                        # TODO: clarify this block
                        one_hot_labels = np.zeros_like(predictions)
                        for i, j in zip(one_hot_labels, labels):
                            i[j] = 1
                        roc = metrics.roc_auc_score(
                            one_hot_labels,
                            predictions,
                            multi_class='ovo', 
                            average='macro', 
                            labels=[0, 1, 2, 3],
                        )
                if is_train:
                    self.results['train_loss'].append(loss := (train_loss := loss.item()))
                    self.results['train_score'].append(score := (train_score := score.item()))
                    if self.is_classification:
                        if 'train_roc' not in self.results: self.results['train_roc'] = []
                        self.results['train_roc'].append(train_roc := roc.item())
                else:
                    if 'valid_loss' not in self.results: self.results['valid_loss'] = []
                    self.results['valid_loss'].append(loss := (valid_loss := loss.item()))
                    if 'valid_score' not in self.results: self.results['valid_score'] = []
                    self.results['valid_score'].append(score := (valid_score := score.item()))
                    if self.is_classification:
                        if 'valid_roc' not in self.results: self.results['valid_roc'] = []
                        self.results['valid_roc'].append(valid_roc := roc.item())

            # step LR scheduler
            self.lr_scheduler.step(loss)

            # log training time
            self.results['epoch_time'].append(time.time()-t_start_epoch)

            # save results to yaml
            with (self.output_dir/self.results_file).open('w') as results_file:
                yaml.dump(
                    self.results,
                    results_file,
                    default_flow_style=False,
                    sort_keys=False,
                )

            # best score and save model
            if score > best_score:
                best_score = score
                best_epoch = i_epoch
                self.logger.info(f"  Best score: {best_score:.3f}")
                self.model.save_pytorch_model(filename=self.output_dir/self.checkpoint_file)
                if self.save_onnx_model:
                    self.model.save_onnx_model(filename=self.output_dir/self.onnx_checkpoint_file)

            # print epoch summary
            status =  f"Ep {i_epoch+1:03d}: "
            status += f"train loss {train_loss:.3f}  "
            status += f"train {self.results['score_function_name']} {train_score:.3f}  "
            if self.is_classification:
                status += f"train ROC {train_roc:.3f}  "
            if valid_loss:
                status += f"val loss {valid_loss:.3f}  "
                status += f"val {self.results['score_function_name']} {valid_score:.3f}  "
                if self.is_classification:
                    status += f"val ROC {valid_roc:.3f}  "
            status += f"ep time {time.time()-t_start_epoch:.1f} s "
            status += f"(total time {time.time()-t_start_training:.1f} s)"
            self.logger.info(status)

            # optuna integration; report epoch result to optuna
            do_optuna_prune = False
            if optuna is not None and self.optuna_trial is not None:
                # save results dict in trial user attributes
                for key in self.results:
                    self.optuna_trial.set_user_attr(key, self.results[key])
                if self.optuna_trial.user_attrs['maximize_score']:
                    report_value = score
                else:
                    report_value = loss
                assert np.isfinite(report_value)
                report_successful = False
                tries = 0
                while report_successful is False and tries < 5:
                    try:
                        self.optuna_trial.report(report_value, i_epoch)
                    except:
                        self.logger.info("Failed optuna report, trying again...")
                        time.sleep(3)
                        tries += 1
                    else:
                        report_successful = True
                if report_successful is False:
                    self.logger.info("==> Failed Optuna report, exiting training loop")
                    break
                if self.optuna_trial.should_prune():
                    do_optuna_prune = True
                    self.logger.info("==> Pruning trial with Optuna")
                    break  # exit epoch training loop

            # break loop if score stops improving
            if (i_epoch > 40) and (i_epoch > best_epoch+20) and (score < 0.9 * best_score):
                self.logger.info("==> Score is < 90% best score; breaking")
                break

        self.logger.info(f"End training loop")
        self.logger.info(f"Elapsed time {time.time()-t_start_training:.1f} s")

        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

        if do_optuna_prune:
            optuna.TrialPruned()

        return self.results

    def _single_epoch_loop(
        self,
        is_train: bool = True,  # True for train, False for evaluation/inference
        data_loader: torch.utils.data.DataLoader = None,  # train or validation data loader
    ) -> Union[np.ndarray, Tuple]:
        batch_losses = []
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
                if not self.all_data_to_device:
                    # send batches to device
                    signal_windows = signal_windows.to(self.device)
                    labels = labels.to(self.device)
                if i_batch % self.minibatch_print_interval == 0:
                    t_start_minibatch = time.time()
                if is_train:
                    self.optimizer.zero_grad()
                # predictions are floats: regression scalars or classification logits
                predictions = self.model(signal_windows)
                if self.is_regression:
                    labels = labels.type_as(predictions)
                elif self.is_classification:
                    if self.model.mlp_output_size == 1:
                        labels = labels.type_as(predictions)
                    else:
                        labels = labels.type(torch.int64)
                sample_losses = self.loss_function(
                    predictions.squeeze(),
                    labels,
                )
                sample_losses = self._apply_loss_weight(sample_losses, labels)
                batch_loss = sample_losses.mean()  # batch loss
                if is_train:
                    batch_loss.backward()
                    self.optimizer.step()

                batch_losses.append(batch_loss.item())  # accumulate batch losses
                all_labels.append(labels.cpu().numpy())
                all_predictions.append(predictions.detach().cpu().numpy())

                # minibatch status
                if (i_batch+1) % self.minibatch_print_interval == 0:
                    epoch_loss = np.mean(batch_losses)
                    status =  f"  {mode} batch {i_batch+1:05d}/{len(data_loader)}  "
                    status += f"batch loss {batch_loss:.3f} (avg loss {epoch_loss:.3f})  "
                    status += f"minibatch time {time.time()-t_start_minibatch:.3f} s"
                    self.logger.info(status)

        epoch_loss = np.mean(batch_losses)
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        return epoch_loss, all_predictions, all_labels

    def _apply_loss_weight(self, losses: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return losses


if __name__=='__main__':
    m = _Base_Trainer(dense_num_kernels=8)
