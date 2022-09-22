# python library imports
import contextlib
import logging
import time
from pathlib import Path
from typing import Union, Tuple, Any
import dataclasses

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
    # training parameters
    device: str = 'auto'  # auto (default), cpu, cuda, or cuda:X
    n_epochs: int = 2  # training epochs
    minibatch_print_interval: int = 2000  # print minibatch info
    optimizer_type: str = 'adam'  # adam (default) or sgd
    sgd_momentum: float = 0.0  # momentum for SGD optimizer
    sgd_dampening: float = 0.0  # dampening for SGD optimizer
    learning_rate: float = 1e-3  # optimizer learning rate
    lr_scheduler_patience: int = 4  # epochs to wait before triggering lr scheduler
    weight_decay: float = 5e-3  # optimizer L2 regularization factor
    # optuna integration
    optuna_trial: Any = None  # optuna trial
    # global parameters
    is_regression: bool = dataclasses.field(default=None, init=False)
    is_classification: bool = dataclasses.field(default=None, init=False)
    model: Multi_Features_Model = dataclasses.field(default=None, init=False)
    train_data_loader: torch.utils.data.DataLoader = dataclasses.field(default=None, init=False)
    validation_data_loader: torch.utils.data.DataLoader = dataclasses.field(default=None, init=False)
    results: dict = dataclasses.field(default=None, init=False)
    label_type: np.dtype = dataclasses.field(default=None, init=False)


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
            'valid_loss': [],
            'loss_function_name': '',
            'scores': [],
            'score_function_name': '',
            'training_time': [],
            'lr': [],
        }

        # subclass must set is_regression XOR is_classification
        assert self.is_regression ^ self.is_classification  # XOR

        self._make_model()
        self._set_device()
        self.model = self.model.to(self.device)
        self._set_regression_classification()
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

    def _set_regression_classification(self) -> None:

        if self.is_regression:
            # regression model (e.g. time to ELM onset, velocimetry surrogate model)
            self.label_type = np.float32
            self.score_function_name = 'R2'
            self.loss_function_name = 'MSELoss'
        elif self.is_classification:
            self.label_type = np.int8
            self.score_function_name = 'F1'
            if self.model.mlp_output_size == 1:
                # binary classification (e.g. active/inactive ELM)
                self.loss_function_name = 'BCEWithLogitsLoss'
                assert hasattr(self, 'threshold')  # binary classification must specify threshold
            else:
                # multi-class classification (e.g. confinement mode)
                self.loss_function_name = 'CrossEntropyLoss'

        self.loss_function = getattr(torch.nn, self.loss_function_name)(reduction="none")

        self.results['score_function_name'] = self.score_function_name
        self.results['loss_function_name'] = self.loss_function_name

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
            factor=0.5,
            threshold=1e-3,
            patience=self.lr_scheduler_patience,
            verbose=True,
        )

    def _prepare_data(self) -> None:
        # implement in subclass
        # must generate self.train_data_loader and self.validation_data_loader
        raise NotImplementedError

    def train(self) -> dict:
        best_score = -np.inf

        # TODO: move to ELM classification
        if self.is_classification:
            self.results['roc_scores'] = []

        self.logger.info(f"Batches per epoch {len(self.train_data_loader)}")
        self.logger.info(f"Begin training loop over {self.n_epochs} epochs")
        t_start_training = time.time()
        # loop over epochs
        for i_epoch in range(self.n_epochs):
            t_start_epoch = time.time()
            self.logger.info(f"Ep {i_epoch+1:03d}: begin")
            train_loss = self._single_epoch_loop(
                is_train=True,
                data_loader=self.train_data_loader,
            )
            self.results['train_loss'].append(train_loss.item())
            self.results['lr'].append(self.optimizer.param_groups[0]['lr'])

            if self.validation_data_loader is None:
                score = None
                valid_loss = None
            else:
                valid_loss, predictions, true_labels = self._single_epoch_loop(
                    is_train=False,
                    data_loader=self.validation_data_loader,
                )
                self.results['valid_loss'].append(valid_loss.item())  # log validation score
                self.lr_scheduler.step(valid_loss)  # appy learning rate scheduler
                if self.is_regression:
                    score = metrics.r2_score(true_labels, predictions)
                elif self.is_classification:
                    if self.model.mlp_output_size == 1:
                        modified_predictions = (predictions > self.threshold).astype(int)
                        score = metrics.f1_score(true_labels, modified_predictions, average='binary')
                    else:
                        modified_predictions = predictions.argmax(axis=1)
                        score = metrics.f1_score(true_labels, modified_predictions, average='weighted')
                self.results['scores'].append(score.item())

                # ROC-AUC score for classification
                if self.is_classification:
                    if self.model.mlp_output_size == 1:
                        roc_score = metrics.roc_auc_score(true_labels, predictions)
                    else:
                        # TODO: clarify
                        one_hot = np.zeros_like(predictions)
                        for i, j in zip(one_hot, true_labels):
                            i[j] = 1
                        roc_score = metrics.roc_auc_score(
                            one_hot,
                            predictions,
                            multi_class='ovo', 
                            average='macro', 
                            labels=[0, 1, 2, 3],
                        )
                    self.results['roc_scores'].append(roc_score.item())

            # log training time
            self.results['training_time'].append(time.time()-t_start_training)

            # save results to yaml
            with (self.output_dir/self.results_file).open('w') as results_file:
                yaml.dump(
                    self.results,
                    results_file,
                    default_flow_style=False,
                    sort_keys=False,
                )

            # best score and save model
            if score is None or score > best_score:
                if score is not None:
                    best_score = score
                    self.logger.info(f"  Best {self.score_function_name}: {best_score:.3f}")
                self.model.save_pytorch_model(filename=self.output_dir/self.checkpoint_file)
                if self.save_onnx_model:
                    self.model.save_onnx_model(filename=self.output_dir/self.onnx_checkpoint_file)

            # print epoch summary
            modified_predictions =  f"Ep {i_epoch+1:03d}: "
            modified_predictions += f"train loss {train_loss:.3f}  "
            if score is not None and valid_loss is not None:
                modified_predictions += f"val loss {valid_loss:.3f}  "
                modified_predictions += f"{self.score_function_name} {score:.3f}  "
                if self.is_classification:
                    modified_predictions += f"ROC {roc_score:.3f}  "
            modified_predictions += f"ep time {time.time()-t_start_epoch:.1f} s "
            modified_predictions += f"(total time {time.time()-t_start_training:.1f} s)"
            self.logger.info(modified_predictions)

            # optuna integration; report epoch result to optuna
            do_prune = False
            if optuna is not None and self.optuna_trial is not None:
                maximize_score = self.optuna_trial.user_attrs['maximize_score']
                if maximize_score is True:
                    assert np.isfinite(score)
                    self.optuna_trial.report(score, i_epoch)
                else:
                    if not np.isfinite(train_loss) or np.isnan(train_loss):
                        train_loss = 1
                    self.optuna_trial.report(train_loss, i_epoch)
                # save results dict in trial user attributes
                for key in self.results:
                    self.optuna_trial.set_user_attr(key, self.results[key])
                if self.optuna_trial.should_prune():
                    do_prune = True
                    self.logger.info("==> Pruning trial with Optuna")
                    break  # exit epoch training loop

        self.logger.info(f"End training loop")
        self.logger.info(f"Elapsed time {time.time()-t_start_training:.1f} s")

        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

        if do_prune:
            optuna.TrialPruned()

        return self.results.copy()

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
                if (i_batch+1) % self.minibatch_print_interval == 0:
                    t_start_minibatch = time.time()
                if is_train:
                    self.optimizer.zero_grad()
                predictions = self.model(signal_windows)
                if self.is_classification and self.mlp_output_size > 1:
                    labels = labels.type(torch.long)  # must be torch.long for CrossEntropy() loss (why?)
                else:
                    labels = labels.type_as(predictions)
                sample_losses = self.loss_function(
                    predictions.squeeze(),
                    labels,
                )
                sample_losses = self._apply_loss_weight(sample_losses, labels)
                batch_loss = sample_losses.mean()  # batch loss
                batch_losses.append(batch_loss.item())  # accumulate batch losses for this epoch
                if is_train:
                    # for training, backpropagate and step optimizer
                    batch_loss.backward()
                    self.optimizer.step()
                else:
                    # for validation, accumulate labels and predictions
                    all_labels.append(labels.cpu().numpy())
                    all_predictions.append(predictions.cpu().numpy())

                # minibatch status
                if (i_batch+1)%self.minibatch_print_interval == 0:
                    epoch_loss = np.mean(batch_losses)
                    status =  f"  {mode} batch {i_batch+1:05d}/{len(self.train_data_loader)}  "
                    status += f"batch loss {batch_loss:.3f} (avg loss {epoch_loss:.3f})  "
                    status += f"minibatch time {time.time()-t_start_minibatch:.3f} s"
                    self.logger.info(status)
        epoch_loss = np.mean(batch_losses)
        if is_train:
            return_value = epoch_loss
        else:
            all_labels = np.concatenate(all_labels)
            all_predictions = np.concatenate(all_predictions)
            return_value = (
                epoch_loss,
                all_predictions,
                all_labels,
            )
        return return_value

    def _apply_loss_weight(self, losses: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return losses


if __name__=='__main__':
    m = _Base_Trainer(dense_num_kernels=8)
