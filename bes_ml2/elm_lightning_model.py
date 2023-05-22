from __future__ import annotations
import dataclasses
import os
import time
from collections import namedtuple

import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt

import torch
import torch.nn
import torch.utils.data

from lightning.pytorch import loggers
import torchmetrics

try:
    from . import elm_torch_model
except:
    from bes_ml2 import elm_torch_model


class BCEWithLogit(torchmetrics.Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = True
    bce: torch.Tensor
    counts: torch.Tensor

    def __init__(self):
        super().__init__()
        self.add_state("bce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("counts", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, input: torch.Tensor, target: torch.Tensor):
        self.bce += torch.nn.functional.binary_cross_entropy_with_logits(
            input=input, 
            target=target.type_as(input),
            reduction='sum',
        )
        self.counts += target.numel()

    def compute(self):
        return self.bce / self.counts


@dataclasses.dataclass(eq=False)
class Lightning_Model(
    elm_torch_model.Torch_CNN_Mixin,
    elm_torch_model.Torch_MLP_Mixin,
):
    lr: float = 1e-3
    lr_scheduler_patience: int = 20
    lr_scheduler_threshold: float = 1e-3
    weight_decay: float = 1e-6
    monitor_metric: str = 'sum_loss/val'
    log_dir: str = dataclasses.field(default='.', init=False)
    # the following must be listed in `_frontend_names`
    reconstruction_decoder: bool = True
    classifier_mlp: bool = True
    time_to_elm_mlp: bool = True
    _frontend_names = ['reconstruction_decoder', 'classifier_mlp', 'time_to_elm_mlp']
    
    def __post_init__(self):
        super().__post_init__()
        self.save_hyperparameters()

        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

        # CNN encoder `backend` to featurize input data
        self.cnn_encoder, cnn_features, cnn_output_shape = self.make_cnn()

        # `frontends` for regression, classification, and self-supervised learning
        self.frontends = torch.nn.ModuleDict()
        self.frontends_active = {}
        for frontend_key in self._frontend_names:
            if getattr(self, frontend_key) is True:
                self.frontends_active[frontend_key] = True
                if 'mlp' in frontend_key:
                    new_module = self.make_mlp(mlp_in_features=cnn_features)
                    self.frontends.update({frontend_key: new_module})
                    if 'time_to_elm' in frontend_key:
                        setattr(self, f"{frontend_key}_mse_loss", torchmetrics.MeanSquaredError())
                        setattr(self, f"{frontend_key}_r2_score", torchmetrics.R2Score())
                    elif 'classifier' in frontend_key:
                        setattr(self, f"{frontend_key}_bce_loss", BCEWithLogit())
                        setattr(self, f"{frontend_key}_f1_score", torchmetrics.F1Score(task='binary'))
                    else:
                        raise KeyError
                elif 'decoder' in frontend_key:
                    new_module = self.make_cnn_decoder(input_data_shape=cnn_output_shape)
                    self.frontends.update({frontend_key: new_module})
                    if 'reconstruction' in frontend_key:
                        setattr(self, f"{frontend_key}_mse_loss", torchmetrics.MeanSquaredError())
                    else:
                        raise KeyError
                else:
                    raise KeyError
            
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters {total_parameters:,}")
        cnn_parameters = sum(p.numel() for p in self.cnn_encoder.parameters() if p.requires_grad)
        print(f"  CNN encoder parameters {cnn_parameters:,}")

        for key, frontend_key in self.frontends.items():
            n_parameters = sum(p.numel() for p in frontend_key.parameters() if p.requires_grad)
            print(f"  Frontend `{key}` parameters: {n_parameters:,}")

        self.example_input_array = torch.zeros(
            (4, 1, self.signal_window_size, 8, 8), 
            dtype=torch.float32,
        )

        self.initialize_layers()

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=0.5,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_threshold,
            min_lr=2e-5,
            mode='min' if 'loss' in self.monitor_metric else 'max',
            verbose=True,
        )
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'monitor': self.monitor_metric,
        }

    def forward(self, signals: torch.Tensor) -> dict[str, torch.Tensor]:
        results = {}
        features = self.cnn_encoder(signals)
        for frontend_key, frontend in self.frontends.items():
            if self.frontends_active[frontend_key]:
                results[frontend_key] = frontend(features)
        return results

    def update_step(self, batch, batch_idx) -> torch.Tensor:
        signals, labels, class_labels = batch
        results = self(signals)
        sum_loss = None
        for frontend_key, frontend_is_active in self.frontends_active.items():
            if frontend_is_active is False:
                continue
            frontend_result = results[frontend_key]
            if 'time_to_elm' in frontend_key:
                metric_suffices = ['mse_loss', 'r2_score']
            elif 'reconstruction' in frontend_key:
                metric_suffices = ['mse_loss']
            elif 'classifier' in frontend_key:
                metric_suffices = ['bce_loss', 'f1_score']
            else:
                raise ValueError
            for metric_suffix in metric_suffices:
                metric_name = f"{frontend_key}_{metric_suffix}"
                metric: torchmetrics.Metric = getattr(self, metric_name)
                if 'time_to_elm' in metric_name:
                    target = labels
                elif 'reconstruction' in metric_name:
                    target = signals
                elif 'classifier' in metric_name:
                    target = class_labels
                else:
                    raise ValueError
                metric_value = metric(frontend_result, target)
                if 'loss' in metric_name:
                    sum_loss = metric_value if sum_loss is None else sum_loss + metric_value
        return sum_loss
    
    def compute_log_reset(self, stage: str):
        sum_loss = None
        for frontend_key, frontend_is_active in self.frontends_active.items():
            if frontend_is_active is False:
                continue
            if 'time_to_elm' in frontend_key:
                metric_suffices = ['mse_loss', 'r2_score']
            elif 'reconstruction' in frontend_key:
                metric_suffices = ['mse_loss']
            elif 'classifier' in frontend_key:
                metric_suffices = ['bce_loss', 'f1_score']
            else:
                raise ValueError
            for metric_suffix in metric_suffices:
                metric_name = f"{frontend_key}_{metric_suffix}"
                metric: torchmetrics.Metric = getattr(self, metric_name)
                metric_value = metric.compute()
                self.log(f"{metric_name}/{stage}", metric_value, sync_dist=True)
                if 'loss' in metric_name:
                    sum_loss = metric_value if sum_loss is None else sum_loss + metric_value
                metric.reset()
        self.log(f"sum_loss/{stage}", sum_loss, sync_dist=True)

    def on_fit_start(self) -> None:
        self.t_fit_start = time.time()

    def on_train_epoch_start(self):
        self.t_train_epoch_start = time.time()
        print(f"Epoch {self.current_epoch} start")

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self.update_step(batch, batch_idx)

    def on_train_batch_end(self, *args, **kwargs):
        self.compute_log_reset(stage='train')

    def on_train_epoch_end(self) -> None:
        for name, param in self.named_parameters():
            if 'weight' not in name:
                continue
            values = param.data.detach()
            mean = torch.mean(values).item()
            std = torch.std(values).item()
            z_scores = (values-mean)/std
            skew = torch.mean(z_scores**3).item()
            kurt = torch.mean(z_scores**4).item()
            self.log(f"param_mean/{name}", mean, sync_dist=True)
            self.log(f"param_std/{name}", std, sync_dist=True)
            self.log(f"param_skew/{name}", skew, sync_dist=True)
            self.log(f"param_kurt/{name}", kurt, sync_dist=True)
        print(f"Epoch {self.current_epoch} elapsed train time: {(time.time()-self.t_train_epoch_start)/60:0.1f} min")

    def on_validation_epoch_start(self) -> None:
        self.t_val_epoch_start = time.time()

    def validation_step(self, batch, batch_idx) -> None:
        self.update_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        self.compute_log_reset(stage='val')
        print(f"Epoch {self.current_epoch} elapsed valid. time: {(time.time()-self.t_val_epoch_start)/60:0.1f} min")

    def on_fit_end(self) -> None:
        print(f"Fit elapsed time {(time.time()-self.t_fit_start)/60:0.1f} min")

    def on_test_start(self) -> None:
        self.t_test_start = time.time()

    def test_step(self, batch, batch_idx) -> None:
        self.update_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self.compute_log_reset(stage='test')
        print(f"Test elapsed time {(time.time()-self.t_test_start)/60:0.1f} min")

    def on_predict_start(self) -> None:
        self.predict_outputs: list[list] = []
        self.t_predict_start = time.time()

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        signals, labels, class_labels = batch
        results = self(signals)
        if batch_idx == 0:
            self.predict_outputs.append([])
            assert dataloader_idx == len(self.predict_outputs)-1
        prediction_outputs = {
            'labels': labels,
            'signals': signals,
            'class_labels': class_labels,
        }
        prediction_outputs.update(results)
        self.predict_outputs[-1].append(prediction_outputs)
    
    def on_predict_epoch_end(self) -> None:
        if self.global_rank != 0 or 'time_to_elm_regression' not in self.predict_outputs[0][0]:
            return
        i_page = 1
        for i_elm, result in enumerate(self.predict_outputs):
            labels: torch.Tensor = torch.concat([batch['labels'] for batch in result]).squeeze().numpy(force=True)
            predictions: torch.Tensor = torch.concat([batch['time_to_elm_regression'] for batch in result]).squeeze().numpy(force=True)
            signals: torch.Tensor = torch.concat([batch['signals'] for batch in result]).squeeze().numpy(force=True)
            assert labels.shape[0] == predictions.shape[0] and labels.shape[0] == signals.shape[0]
            signal = signals[:, -1, 2, 3].squeeze()
            pre_elm_size = np.count_nonzero(np.isfinite(labels))
            time = (np.arange(len(labels)) - pre_elm_size) / 1e3
            if i_elm % 6 == 0:
                plt.close('all')
                fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))
                plt.suptitle(f"Inference on ELMs in test dataset (page {i_page})")
            plt.sca(axes.flat[i_elm%6])
            plt.plot(time, labels, label='Label')
            plt.plot(time, predictions, label='Prediction')
            plt.ylabel("Label | Prediction")
            plt.xlabel('Time to ELM (ms)')
            plt.legend(fontsize='small', loc='upper right')
            twinx: matplotlib.axes.Axes = axes.flat[i_elm%6].twinx()
            twinx.plot(time, signal, label='Signal', color='C2')
            twinx.set_ylabel('Scaled signal')
            twinx.legend(fontsize='small', loc='lower right')
            if i_elm % 6 == 5 or i_elm == len(self.predict_outputs)-1:
                plt.tight_layout()
                filename = f'inference_{i_page:02d}'
                filepath = os.path.join(self.log_dir, filename)
                print(f"Saving figures {filepath}{{.pdf,.png}}")
                plt.savefig(filepath+'.pdf', format='pdf', transparent=True)
                plt.savefig(filepath+'.png', format='png', transparent=True)
                for logger in self.loggers:
                    if isinstance(logger, loggers.TensorBoardLogger):
                        logger.experiment.add_figure(f"inference/{filename}", fig, close=False)
                    elif isinstance(logger, loggers.WandbLogger):
                        logger.log_image(key='inference', images=[filepath+'.png'])
                i_page += 1
                plt.close(fig)

    def on_predict_end(self) -> None:
        print(f"Predict elapsed time {(time.time()-self.t_predict_start)/60:0.1f} min")
