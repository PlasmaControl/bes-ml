# python library imports
import contextlib
import logging
from pathlib import Path
from typing import Union, Tuple, Any
import dataclasses
import time
import os

# 3rd-party imports
import yaml
import numpy as np
from sklearn import metrics
import torch
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
    output_dir: Path | str = Path('run_dir')  # path to output dir.
    results_file: str = 'results.yaml'  # output training results
    log_file: str = 'log.txt'  # output log file
    inputs_file: str = 'inputs.yaml'  # save inputs to yaml
    checkpoint_file: str = 'checkpoint.pytorch'  # pytorch save file
    save_onnx_model: bool = False  # export ONNX format
    onnx_checkpoint_file: str = 'checkpoint.onnx'  # onnx save file
    logger: logging.Logger = None
    logger_hash: str = None
    log_all_ranks: bool = False
    terminal_output: bool = True  # terminal output if True
    # training parameters
    device: str | torch.device = 'auto'  # auto (default), cpu, cuda, or cuda:X
    local_rank: int = None
    world_rank: int = None
    world_size: int = None
    # ddp: bool = None
    n_epochs: int = 2  # training epochs
    minibatch_print_interval: int = 2000  # print minibatch info
    optimizer_type: str = 'sgd'  # adam (default) or sgd
    sgd_momentum: float = 0.0  # momentum for SGD optimizer, 0-1
    sgd_dampening: float = 0.0  # dampening for SGD optimizer, 0-1
    sgd_nesterov: bool = False  # True for Nesterov momentum forumula
    learning_rate: float = 1e-3  # optimizer learning rate
    lr_scheduler_patience: int = 20  # epochs to wait before triggering lr scheduler
    lr_scheduler_factor: float = 0.5  # reduction factor for lr scheduler
    lr_scheduler_threshold: float = 1e-3  # threshold for *relative* decrease in loss to *not* trigger LR scheduler
    low_score_patience: int = 20  # epochs to wait before aborting due to low score
    weight_decay: float = 1e-3  # optimizer L2 regularization factor
    do_train: bool = False  # if True, start training at end of init
    # optuna integration
    optuna_trial: Any = None  # optuna trial
    # non-init attributes visible to subclasses
    is_regression: bool = dataclasses.field(default=None, init=False)
    is_classification: bool = dataclasses.field(default=None, init=False)
    is_ddp: bool = dataclasses.field(default=None, init=False)
    is_main_process: bool = dataclasses.field(default=None, init=False)
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

        if self.world_size is None and self.local_rank is None:
            self.world_size = int(os.environ.get('SLURM_NTASKS', 1))
            self.local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        self.world_rank = int(os.environ.get('SLURM_PROCID', self.local_rank))
        if self.world_rank < self.local_rank:
            self.world_rank = self.local_rank

        self.is_ddp = self.world_size > 1
        self.is_main_process = self.world_rank == 0

        if self.is_ddp:
            dist.init_process_group(
                'nccl' if torch.cuda.is_available() else 'gloo',
                rank=self.world_rank,
                world_size=self.world_size,
            )

        print(f"World size {self.world_size}  world rank {self.world_rank}  local rank {self.local_rank}")

        self._create_logger()

        self._barrier()

        if self.is_main_process:
            self.logger.info(f"In main process with world rank {self.world_rank}")
            self.logger.info(f"Using DDP: {self.is_ddp}")
            self._print_inputs()
            self._save_inputs_to_yaml()

        self._barrier()

        # subclass must set is_regression XOR is_classification
        assert self.is_regression ^ self.is_classification  # XOR

        self.results = {}

        self._make_model()
        self._setup_device()
        self._make_optimizer_scheduler()
        self._prepare_data()

        # validate data loaders
        for data_loader in [self.train_data_loader, self.validation_data_loader]:
            assert isinstance(data_loader, torch.utils.data.DataLoader) or data_loader is None

        if self.do_train:
            self.train()

    def _barrier(self):
        if self.is_ddp:
            dist.barrier()

    def _create_logger(self):
        """
        Use python's logging to allow simultaneous print to console and log file.
        """
        if self.logger_hash is None:
            self.logger_hash = str(np.random.randint(1e12))
        self.logger = logging.getLogger(name=f'logger_{self.logger_hash}')
        self.logger.setLevel(logging.INFO)

        if self.is_main_process or self.log_all_ranks:
            # logs to log file
            f_handler = logging.FileHandler(
                self.output_dir / f'log_{self.logger_hash}.txt',
                # mode="w",
            )
            self.logger.addHandler(f_handler)
            if self.terminal_output:
                # logs to console
                self.logger.addHandler(logging.StreamHandler())
            self.logger.info(f"Logging for world rank {self.world_rank}")
        else:
            # null logger if not main process
            self.logger.addHandler(logging.NullHandler())

    def _print_inputs(self):
        cls = self.__class__
        self.logger.info(f"Class `{cls.__name__}` parameters:")
        cls_fields = sorted(dataclasses.fields(cls), key=lambda field: field.name)
        self_fields_dict = dataclasses.asdict(self)
        assert set([field.name for field in cls_fields]) == set(self_fields_dict.keys())
        for field in cls_fields:
            if field.name == 'logger':
                continue
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
            )

    def _make_model(self) -> None:
        model_kwargs = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(Multi_Features_Model)
        }
        self.model = Multi_Features_Model(**model_kwargs)
        self._barrier()
        if self.is_main_process:
            self.model.print_model_summary()
        self._barrier()

    def _setup_device(self) -> None:
        if self.device == 'auto':
            self.device = f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Device {self.device}  world size {self.world_size}  " +
                         f"world rank {self.world_rank}  local rank {self.local_rank}")
        self._barrier()
        self.device = torch.device(self.device)

        self.model = self.model.to(self.device)
        if self.is_ddp:
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.world_rank] if self.device.type == 'cuda' else None,
                output_device=self.world_rank if self.device.type == 'cuda' else None,
            )
            self._model_alias = self.ddp_model
        else:
            self._model_alias = self.model
        self._barrier()

    def _make_optimizer_scheduler(self) -> None:
        assert self.optimizer_type in ['adam', 'sgd']
        if self.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self._model_alias.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self._model_alias.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.sgd_momentum,
                dampening=self.sgd_dampening,
                nesterov=self.sgd_nesterov,
            )
        # if self.is_main_process:
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
        self._barrier()

    def _prepare_data(self) -> None:
        # implement in subclass
        # must generate self.train_data_loader and self.validation_data_loader
        raise NotImplementedError

    def train(self) -> dict:

        self.results['completed_epochs'] = 0
        self.results['train_loss'] = []
        self.results['train_score'] = []
        self.results['epoch_time'] = []
        self.results['lr'] = []

        if self.is_regression:
            self.results['score_function_name'] = 'R2'
            self.results['loss_function_name'] = 'MSELoss'
            self.loss_function = torch.nn.MSELoss(reduction="none")
        elif self.is_classification:
            self.results['score_function_name'] = 'F1'
            self.results['train_roc'] = []
            if self.model.mlp_output_size == 1:
                assert hasattr(self, 'threshold')  # binary classification must specify threshold
                self.results['loss_function_name'] = 'BCEWithLogitsLoss'
                # labels are binary [0,1]; predictions are logits [-inf,inf]
                self.loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")
            else:
                self.results['loss_function_name'] = 'CrossEntropyLoss'
                # labels are true class C; predictions are scores for all classes
                self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")

        if self.is_main_process:
            self.logger.info(f"Training batches per epoch {len(self.train_data_loader)}")
        if self.validation_data_loader:
            self.results['valid_loss'] = []
            self.results['valid_score'] = []
            if self.is_classification:
                self.results['valid_roc'] = []
            if self.is_main_process:
                self.logger.info(f"Validation batches per epoch {len(self.validation_data_loader)}")

        best_score = -np.inf
        best_epoch = 0
        do_optuna_prune = False

        # loop over epochs
        if self.is_main_process:
            self.logger.info(f"Begin training loop over {self.n_epochs} epochs")
        t_start_training = time.time()
        for i_epoch in range(self.n_epochs):
            t_start_epoch = time.time()
            if self.is_main_process:
                self.logger.info(f"Ep {i_epoch + 1:03d}: begin")
            self.results['lr'].append(self.optimizer.param_groups[0]['lr'])
            for is_train, data_loader in zip(
                    [True, False],
                    [self.train_data_loader, self.validation_data_loader],
            ):
                if data_loader is None:
                    continue  # skip if validation data is empty
                # loss and predictions
                loss, predictions, labels = self._single_epoch_loop(
                    is_train=is_train,
                    data_loader=data_loader,
                )
                # scores
                if self.is_regression:
                    score = metrics.r2_score(labels, predictions)
                elif self.is_classification:
                    if self.model.mlp_output_size == 1:
                        modified_predictions = (predictions > self.threshold).astype(int)
                        score = metrics.f1_score(labels, modified_predictions, average='binary')
                        roc = metrics.roc_auc_score(labels, predictions)
                    else:
                        modified_predictions = predictions.argmax(axis=1)  # select class with highest score
                        score = metrics.f1_score(labels, modified_predictions, average='weighted')
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
                        self.results['train_roc'].append(train_roc := roc.item())
                else:
                    self.results['valid_loss'].append(loss := (valid_loss := loss.item()))
                    self.results['valid_score'].append(score := (valid_score := score.item()))
                    if self.is_classification:
                        self.results['valid_roc'].append(valid_roc := roc.item())

            # step LR scheduler
            self.lr_scheduler.step(loss)

            # log training time
            self.results['epoch_time'].append(time.time() - t_start_epoch)
            self.results['completed_epochs'] += 1

            if not self.is_main_process:
                # skip remainder if not main process
                continue

            # record layer statistics
            for module_name, module in self._model_alias.named_modules():
                n_params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
                if n_params == 0:
                    continue
                for attr_name in ['weight', 'bias']:
                    attr = getattr(module, attr_name)
                    if attr is None or attr.numel() < 3:
                        continue
                    mean = attr.mean()
                    stdev = torch.sqrt(torch.mean((attr - mean) ** 2))
                    scores = (attr - mean) / stdev
                    skew = torch.mean(scores ** 3)
                    kurt = torch.mean(scores ** 4) - 3
                    module_attr_name = f"{module_name}.{attr_name}"
                    if module_attr_name not in self.results:
                        self.results[module_attr_name] = {
                            'size': attr.numel(),
                            'shape': list(attr.size()),
                            'mean': [],
                            'stdev': [],
                            'skew': [],
                            'kurt': [],
                        }
                    for stat_value, stat_name in zip(
                            [mean, stdev, skew, kurt],
                            ['mean', 'stdev', 'skew', 'kurt'],
                    ):
                        self.results[module_attr_name][stat_name].append(stat_value.item())

            # save results to yaml
            with (self.output_dir / self.results_file).open('w') as results_file:
                yaml.dump(
                    self.results,
                    results_file,
                    default_flow_style=False,
                    sort_keys=False,
                )

            # print epoch summary
            status = f"Ep {i_epoch + 1:03d}: "
            status += f"train loss {train_loss:.4f}  "
            status += f"train {self.results['score_function_name']} {train_score:.4f}  "
            if self.is_classification:
                status += f"train ROC {train_roc:.3f}  "
            if 'valid_loss' in locals():
                status += f"val loss {valid_loss:.4f}  "
                status += f"val {self.results['score_function_name']} {valid_score:.4f}  "
                if self.is_classification:
                    status += f"val ROC {valid_roc:.3f}  "
            status += f"ep time {time.time() - t_start_epoch:.1f} s "
            status += f"(total time {time.time() - t_start_training:.1f} s)"
            self.logger.info(status)

            # best score and save model
            if score > best_score:
                best_score = score
                best_epoch = i_epoch
                self.logger.info(f"  Best score: {best_score:.4f}")
                self.model.save_pytorch_model(filename=self.output_dir / self.checkpoint_file)
                if self.save_onnx_model:
                    self.model.save_onnx_model(filename=self.output_dir / self.onnx_checkpoint_file)

            # optuna integration; report epoch result to optuna
            if optuna is not None and self.optuna_trial is not None:
                # save results dict in trial user attributes
                for stat_name in self.results:
                    self.optuna_trial.set_user_attr(stat_name, self.results[stat_name])
                report_value = score if self.optuna_trial.user_attrs['maximize_score'] else loss
                assert np.isfinite(report_value)
                report_successful = False
                tries = 1
                while report_successful is False and tries <= 10:
                    try:
                        self.optuna_trial.report(report_value, i_epoch)
                    except:
                        self.logger.info("Failed optuna report, trying again...")
                        tries += 1
                        time.sleep(2)
                    else:
                        report_successful = True
                # break loop if report fails
                if report_successful is False:
                    self.logger.info("==> Failed Optuna report, exiting training loop")
                    break
                # break loop if pruning
                if self.optuna_trial.should_prune():
                    do_optuna_prune = True
                    self.logger.info("==> Pruning trial with Optuna")
                    break  # exit epoch training loop

            # break loop if score stops improving
            if (i_epoch > 20) and (i_epoch > best_epoch + self.low_score_patience) and (score < 0.95 * best_score):
                self.logger.info("==> Score is < 95% best score; breaking")
                break

        self.logger.info(f"End training loop")
        self.logger.info(f"Elapsed time {time.time() - t_start_training:.1f} s")

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
            self._model_alias.train()
        else:
            self._model_alias.eval()
        mode = 'Train' if is_train else 'Valid'
        context = contextlib.nullcontext() if is_train else torch.no_grad()
        with context:
            for i_batch, (signal_windows, labels) in enumerate(data_loader):
                if self.is_ddp and torch.cuda.is_available():
                    signal_windows = signal_windows.to(self.world_rank, non_blocking=True)
                    labels = labels.to(self.world_rank, non_blocking=True)
                else:
                    signal_windows = signal_windows.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                if i_batch % self.minibatch_print_interval == 0:
                    t_start_minibatch = time.time()
                if is_train:
                    self.optimizer.zero_grad()
                # predictions are floats: regression scalars or classification logits
                predictions = self._model_alias(signal_windows)
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
                if (i_batch + 1) % self.minibatch_print_interval == 0 and self.is_main_process:
                    status = f"  {mode} batch {i_batch + 1:05d}/{len(data_loader):05d}  "
                    status += f"batch loss {batch_loss:.3f} "
                    status += f"(ep loss {np.mean(batch_losses):.3f})  "
                    status += f"minibatch time {time.time() - t_start_minibatch:.3f} s"
                    self.logger.info(status)

        epoch_loss = np.mean(batch_losses)
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        return epoch_loss, all_predictions, all_labels

    def _apply_loss_weight(
            self,
            losses: torch.Tensor,
            labels: torch.Tensor,
    ) -> torch.Tensor:
        return losses


if __name__ == '__main__':
    m = _Base_Trainer(dense_num_kernels=8)
