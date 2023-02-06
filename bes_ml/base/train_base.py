# python library imports
from __future__ import annotations

import contextlib
import logging
from pathlib import Path
import typing
import dataclasses
from datetime import datetime
import time

# 3rd-party imports
import yaml
import numpy as np
from sklearn import metrics
import torch
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed
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
class Trainer_Base_Dataclass:
    # output parameters
    output_dir: Path | str = Path('run_dir')  # path to output dir.
    results_file: str = 'results.yaml'  # output training results
    inputs_file: str = 'inputs.yaml'  # save inputs to yaml
    save_onnx_model: bool = False  # export ONNX format
    logger_hash: str | int = None
    terminal_output: bool = True  # terminal output if True
    # Pytorch DDP (multi-GPU training)
    world_size: int = 1
    world_rank: int = 0
    local_rank: int = 0
    log_all_ranks: bool = False
    # training parameters
    device: str | torch.device = 'auto'  # auto (default), cpu, cuda, or cuda:X
    n_epochs: int = 1  # training epochs
    optimizer_type: str = 'sgd'  # adam (default) or sgd
    sgd_momentum: float = 0.0  # momentum for SGD optimizer, 0-1
    sgd_dampening: float = 0.0  # dampening for SGD optimizer, 0-1
    sgd_nesterov: bool = False  # True for Nesterov momentum forumula
    learning_rate: float = 1e-3  # optimizer learning rate
    lr_scheduler_patience: int = 20  # epochs to wait before triggering lr scheduler
    lr_scheduler_factor: float = 0.5  # reduction factor for lr scheduler
    lr_scheduler_threshold: float = 1e-3  # threshold for *relative* decrease in loss to *not* trigger LR scheduler
    low_score_patience: int = 30  # epochs to wait before aborting due to low score
    low_score_threshold: float = 0.95  # abort if score drops below threshold for number of patience epochs
    weight_decay: float = 1e-3  # optimizer L2 regularization factor
    minibatch_print_interval: int = 5000
    do_train: bool = False  # if True, start training at end of init
    maximum_parameters: int = None
    # optuna integration
    optuna_trial: typing.Any = None  # optuna trial
    # non-init attributes visible to subclasses
    logger: logging.Logger = dataclasses.field(default=None, init=False)
    is_regression: bool = dataclasses.field(default=None, init=False)
    is_classification: bool = dataclasses.field(default=None, init=False)
    is_ddp: bool = dataclasses.field(default=None, init=False)
    is_main_process: bool = dataclasses.field(default=None, init=False)
    model: Multi_Features_Model = dataclasses.field(default=None, init=False)
    train_loader: torch.utils.data.DataLoader = dataclasses.field(default=None, init=False)
    valid_loader: torch.utils.data.DataLoader = dataclasses.field(default=None, init=False)
    train_sampler: torch.utils.data.distributed.DistributedSampler = dataclasses.field(default=None, init=False)
    valid_sampler: torch.utils.data.distributed.DistributedSampler = dataclasses.field(default=None, init=False)
    results: dict = dataclasses.field(default=None, init=False)
    _ddp_barrier: callable = dataclasses.field(default=None, init=False)


@dataclasses.dataclass(eq=False)
class Trainer_Base(Trainer_Base_Dataclass):
    """Base class for model trainer"""

    def __post_init__(self):

        t_start_setup = time.time()

        self.output_dir = Path(self.output_dir).resolve()
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # multi-GPU/multi-node with python DistributedDataParallel (ddp)
        assert self.local_rank <= self.world_rank
        assert self.world_rank < self.world_size
        self.is_ddp = self.world_size > 1
        self.is_main_process = self.world_rank == 0

        self._create_logger()

        self.logger.info(f"Using DDP?: {self.is_ddp}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA local device count: {torch.cuda.device_count()}")
        else:
            self.logger.info(f"CUDA devices not available")
        self.logger.info(f"world_size/world_rank/local_rank: {self.world_size}/{self.world_rank}/{self.local_rank}")
        self._print_inputs()
        self._save_inputs_to_yaml()

        # subclass must set is_regression XOR is_classification
        assert self.is_regression ^ self.is_classification  # XOR

        self.results = {}

        self._make_model()
        self._setup_device()
        self._make_optimizer_scheduler()
        self._prepare_data()  # implement in data class; e.g. ELMs, confinement mode, velocimetry

        # validate data loaders
        for data_loader in [self.train_loader, self.valid_loader]:
            if data_loader:
                assert isinstance(data_loader, torch.utils.data.DataLoader)

        self.logger.info(f"Setup time {time.time() - t_start_setup:.1f} s")

        if self.do_train:
            self.train()

    def _ddp_barrier(self):
        if self.is_ddp:
            torch.distributed.barrier()

    def _create_logger(self):
        """
        Use python's logging to allow simultaneous print to console and log file.
        """
        self._ddp_barrier()
        if self.is_ddp:
            assert self.logger_hash
        if not self.logger_hash:
            self.logger_hash = int(datetime.now().timestamp())
        self.logger = logging.getLogger(name=f"{__name__}_{self.logger_hash}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.NullHandler())

        if self.is_main_process or self.log_all_ranks:
            log_file = self.output_dir / f'log.txt'
            log_file.unlink(missing_ok=True)
            if self.is_ddp:
                formatter = logging.Formatter(f"Rank {self.world_rank}: %(message)s")
            else:
                formatter = logging.Formatter(f"%(message)s")
            # logs to log file
            f_handler = logging.FileHandler(log_file)
            f_handler.setFormatter(formatter)
            self.logger.addHandler(f_handler)
            if self.terminal_output:
                # logs to console
                s_handler = logging.StreamHandler()
                s_handler.setFormatter(formatter)
                self.logger.addHandler(s_handler)
            self.logger.info(f"Logging for world rank {self.world_rank}")

    def _print_inputs(self):
        if self.is_main_process:
            cls = self.__class__
            self.logger.info(f"Class {cls.__name__} parameters:")
            cls_fields = sorted(dataclasses.fields(cls), key=lambda field: field.name)
            self_fields_dict = dataclasses.asdict(self)
            assert set([field.name for field in cls_fields]) == set(self_fields_dict.keys())
            for field in cls_fields:
                if field.name in ['logger', '_ddp_barrier']:
                    continue
                if self_fields_dict[field.name] == field.default:
                    field_str = f"  {field.name}: {self_fields_dict[field.name]}"
                else:
                    field_str = f"  {field.name}: {self_fields_dict[field.name]}  (default {field.default})"
                self.logger.info(field_str)

    def _save_inputs_to_yaml(self):
        if self.is_main_process:
            filename = Path(self.output_dir / self.inputs_file)
            self_fields_dict = dataclasses.asdict(self)
            for skip_key in ['logger', 'optuna_trial', '_ddp_barrier']:
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
        self._ddp_barrier()
        model_kwargs = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(Multi_Features_Model)
            if field.init
        }
        self.model = Multi_Features_Model(**model_kwargs)
        if self.is_main_process:
            self.model.print_model_summary()
        if self.maximum_parameters:
            total_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            assert total_parameters <= self.maximum_parameters, f'Model is too large with {total_parameters} parameters'

    def _setup_device(self) -> None:
        self._ddp_barrier()
        if self.device == 'auto':
            self.device = f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"world_size/world_rank/local_rank {self.world_size}/{self.world_rank}/{self.local_rank}  device {self.device}")
        self.device = torch.device(self.device)

        self.model = self.model.to(self.device)
        if self.is_ddp:
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.local_rank] if self.device.type == 'cuda' else None,
                output_device=self.local_rank if self.device.type == 'cuda' else None,
            )
            self._model_alias = self.ddp_model
        else:
            self._model_alias = self.model

    def _make_optimizer_scheduler(self) -> None:
        self._ddp_barrier()
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
        self.logger.info(f"Optimizer {self.optimizer_type.upper()} lr {self.learning_rate:.1e}")
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=self.lr_scheduler_factor,
            threshold=self.lr_scheduler_threshold,
            patience=self.lr_scheduler_patience,
            mode='min',
            verbose=True,
        )
        self.logger.info(f"LR scheduler with patience {self.lr_scheduler_patience}")

    def _prepare_data(self) -> None:
        # subclass must generate self.train_loader and self.validation_loader
        raise NotImplementedError

    def _setup_train(self) -> None:

        self.results['completed_epochs'] = 0
        self.results['train_loss'] = []
        self.results['train_score'] = []
        self.results['epoch_time'] = []
        self.results['lr'] = []
        self.results['trainable_parameters'] = self.model.trainable_parameters
        self.results['feature_count'] = self.model.feature_count

        self.logger.info(f"Training batches per epoch {len(self.train_loader)}")
        if self.valid_loader:
            self.results['valid_loss'] = []
            self.results['valid_score'] = []
            if self.is_classification:
                self.results['valid_roc'] = []
            self.logger.info(f"Validation batches per epoch {len(self.valid_loader)}")

        if self.is_regression:
            self.results['score_function_name'] = 'R2'
            self.results['loss_function_name'] = 'MSELoss'
            self.loss_function = torch.nn.MSELoss()
        elif self.is_classification:
            self.results['score_function_name'] = 'F1'
            self.results['train_roc'] = []
            if self.model.mlp_output_size == 1:
                assert hasattr(self, 'threshold')  # binary classification must specify threshold
                self.results['loss_function_name'] = 'BCEWithLogitsLoss'
                self.loss_function = torch.nn.BCEWithLogitsLoss()
            else:
                self.results['loss_function_name'] = 'CrossEntropyLoss'
                self.loss_function = torch.nn.CrossEntropyLoss()

    def train(self) -> dict:

        self._ddp_barrier()
        self._setup_train()

        best_score = -np.inf
        best_epoch = 0
        do_optuna_prune = False

        # loop over epochs
        self.logger.info(f"Begin training loop over {self.n_epochs} epochs")
        t_start_training = time.time()
        for i_epoch in range(self.n_epochs):
            t_start_epoch = time.time()
            self.logger.info(f"Ep {i_epoch + 1:03d}: begin")
            self.results['lr'].append(self.optimizer.param_groups[0]['lr'])
            for is_train, data_loader, sampler in zip(
                [True, False],
                [self.train_loader, self.valid_loader],
                [self.train_sampler, self.valid_sampler],
            ):
                # self._ddp_barrier()
                if data_loader is None:
                    continue  # skip if validation data is empty
                if sampler is not None:
                    sampler.set_epoch(i_epoch)
                # loss and predictions
                loss, predictions, labels = self._single_epoch_loop(
                    is_train=is_train,
                    data_loader=data_loader,
                )
                # scores
                if self.is_regression:
                    score = metrics.r2_score(labels, predictions).item()
                elif self.is_classification:
                    if self.model.mlp_output_size == 1:
                        modified_predictions = (predictions > self.threshold).astype(int)
                        score = metrics.f1_score(labels, modified_predictions, average='binary').item()
                        roc = metrics.roc_auc_score(labels, predictions).item()
                    else:
                        modified_predictions = predictions.argmax(axis=1)  # select class with highest score
                        score = metrics.f1_score(labels, modified_predictions, average='weighted').item()
                        one_hot_labels = np.zeros_like(predictions)
                        for i, j in zip(one_hot_labels, labels):
                            i[j] = 1
                        roc = metrics.roc_auc_score(
                            one_hot_labels,
                            predictions,
                            multi_class='ovo',
                            average='macro',
                            labels=[0, 1, 2, 3],
                        ).item()

                if self.is_ddp:
                    # self._ddp_barrier()
                    tmp = torch.tensor([score], device=self.device) / self.world_size
                    torch.distributed.all_reduce(tmp)
                    score = tmp.item()
                    if self.is_classification:
                        tmp = torch.tensor([roc], device=self.device) / self.world_size
                        torch.distributed.all_reduce(tmp)
                        roc = tmp.item()

                if is_train:
                    train_loss = loss.item()
                    train_score = score
                    self.results['train_loss'].append(train_loss)
                    self.results['train_score'].append(train_score)
                    if self.is_classification:
                        train_roc = roc
                        self.results['train_roc'].append(train_roc)
                else:
                    valid_loss = loss.item()
                    valid_score = score
                    self.results['valid_loss'].append(valid_loss)
                    self.results['valid_score'].append(valid_score)
                    if self.is_classification:
                        valid_roc = roc
                        self.results['valid_roc'].append(valid_roc)

                # end train/validation block

            # step LR scheduler
            self.lr_scheduler.step(loss)

            # log training time
            self.results['epoch_time'].append(time.time() - t_start_epoch)
            self.results['completed_epochs'] += 1

            # record layer statistics
            for module_name, module in self.model.named_modules():
                n_params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
                if n_params == 0:
                    continue
                for attr_name in ['weight', 'bias']:
                    attr = getattr(module, attr_name)
                    if attr is None or attr.numel() < 3:
                        continue
                    mean = attr.mean().item()
                    stdev = torch.sqrt(torch.mean((attr - mean) ** 2)).item()
                    scores = (attr - mean) / stdev
                    skew = torch.mean(scores ** 3).item()
                    kurt = (torch.mean(scores ** 4) - 3).item()
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
                        self.results[module_attr_name][stat_name].append(stat_value)

            training_time = time.time() - t_start_training
            self.results['training_time'] = training_time

            if self.is_main_process:
                # save results to yaml
                with (self.output_dir / self.results_file).open('w') as results_file:
                    yaml.dump(
                        self.results,
                        results_file,
                        default_flow_style=False,
                        sort_keys=False,
                    )

            # epoch summary
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
            self.logger.info(status)

            # best score and save model
            if score > best_score:
                best_score = score
                best_epoch = i_epoch
                self.logger.info(f"  Best score: {best_score:.4f}")
                if self.is_main_process:
                    self.model.save_pytorch_model(filename=self.output_dir / 'checkpoint.pytorch')
                    if self.save_onnx_model:
                        self.model.save_onnx_model(filename=self.output_dir / 'checkpoint.onnx')

            # optuna integration; report epoch result to optuna
            if optuna is not None and self.optuna_trial is not None:
                self._ddp_barrier()
                assert np.isfinite(score)
                report_successful = False
                report_attempts = 1
                while report_successful is False and report_attempts<=10:
                    try:
                        self.optuna_trial.report(score, i_epoch)
                        report_successful = True
                    except:
                        report_attempts += 1
                        time.sleep(2)
                    else:
                        report_successful = True
                # break epoch loop if report fails
                if report_successful is False:
                    self.logger.info("==> Failed Optuna report, exiting training loop")
                    break
                # break epoch loop if pruning
                if self.optuna_trial.should_prune():
                    do_optuna_prune = True
                    self.logger.info("==> Pruning trial with Optuna")
                    break  # exit epoch training loop

            # break epoch loop if score stops improving
            if (i_epoch > 50) and \
                (i_epoch > best_epoch + self.low_score_patience) and \
                (score < self.low_score_threshold * best_score):
                self.logger.info(f"==> Score is < {self.low_score_threshold} best score; breaking")
                break

            # end epoch block

        self.logger.info(f"End training loop")
        self.logger.info(f"Training time {training_time/60:.1f} min")

        if hasattr(self.model, 'fft_features') and self.model.fft_features.fft_calc_histogram:
            fft_features = self.model.fft_features
            self.logger.info(f"  FFT min/max:{fft_features.fft_min:.4f}, {fft_features.fft_max:.4f}")
            bin_center = fft_features.bin_edges[:-1] + (fft_features.bin_edges[1] - fft_features.bin_edges[0]) / 2
            mean = np.sum(fft_features.cummulative_hist * bin_center) / np.sum(fft_features.cummulative_hist)
            stdev = np.sqrt(np.sum(fft_features.cummulative_hist * (bin_center - mean) ** 2) / np.sum(fft_features.cummulative_hist))
            self.logger.info(f"  FFT mean {mean:.4f}  stdev {stdev:.4f}")

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
    ) -> np.ndarray|tuple:
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
                    signal_windows = signal_windows.to(self.local_rank, non_blocking=True)
                    labels = labels.to(self.local_rank, non_blocking=True)
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
                batch_loss = self.loss_function(
                    predictions.squeeze(),
                    labels
                )
                full_batch_loss = batch_loss.detach().clone() / self.world_size
                if self.is_ddp:
                    torch.distributed.all_reduce(full_batch_loss)

                if is_train:
                    batch_loss.backward()
                    self.optimizer.step()

                batch_losses.append(full_batch_loss.item())  # accumulate batch losses
                all_labels.append(labels.detach().cpu().numpy())
                all_predictions.append(predictions.detach().cpu().numpy())

                # minibatch status
                if (i_batch + 1) % self.minibatch_print_interval == 0:
                    status = f"  {mode} batch {i_batch + 1:05d}/{len(data_loader):05d}  "
                    status += f"batch loss {full_batch_loss:.3f} "
                    status += f"(ep loss {np.mean(batch_losses):.3f})  "
                    status += f"minibatch time {time.time() - t_start_minibatch:.3f} s"
                    self.logger.info(status)

                # end loop over batches

        epoch_loss = np.mean(batch_losses)
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        return epoch_loss, all_predictions, all_labels

    def _apply_label_weights(
            self,
            losses: torch.Tensor,
            labels: torch.Tensor,
    ) -> torch.Tensor:
        # override in subclass to allow for sample weights
        return losses


if __name__ == '__main__':
    m = Trainer_Base(dense_num_kernels=8)
