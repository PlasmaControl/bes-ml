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

from lightning.pytorch import LightningModule, loggers
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


class SumLoss(torchmetrics.Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = True

    def __init__(self, classification_bce=True, regression_mse=True, reconstruction_mse=True):
        super().__init__()
        self.add_state("counts", default=torch.tensor(0), dist_reduce_fx="sum")
        if classification_bce:
            self.add_state("classification_bce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        if regression_mse:
            self.add_state("regression_mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        if reconstruction_mse:
            self.add_state("reconstruction_mse", default=torch.tensor(0.0), dist_reduce_fx="sum")

@dataclasses.dataclass(eq=False)
class Lightning_Model(
    elm_torch_model.Torch_CNN_Mixin,
    elm_torch_model.Torch_MLP_Mixin,
):
    lr: float = 1e-3
    lr_scheduler_patience: int = 20
    lr_scheduler_threshold: float = 1e-3
    weight_decay: float = 1e-6
    monitor_metric: str = 'loss/sum/val'
    log_dir: str = dataclasses.field(default='.', init=False)
    # the following must be listed in `_frontend_names`
    reconstruction_decoder: bool = True
    time_to_elm_mlp: bool = True
    classifier_mlp: bool = True
    _frontend_names = ['reconstruction_decoder', 'time_to_elm_mlp', 'classifier_mlp']
    
    def __post_init__(self):
        # super().__init__()
        super().__post_init__()
        self.save_hyperparameters()

        # CNN encoder `backend` to featurize input data
        self.cnn_encoder, cnn_features, cnn_output_shape = self.make_cnn()

        # `frontends` for regression, classification, and self-supervised learning
        self.frontends = torch.nn.ModuleDict()
        self.frontends_active = {}
        for frontend_key in self._frontend_names:
            if getattr(self, frontend_key) is True:
                self.frontends_active[frontend_key] = True
                if 'mlp' in frontend_key:
                    self.frontends.update({frontend_key: self.make_mlp(mlp_in_features=cnn_features)})
                elif 'decoder' in frontend_key:
                    self.frontends.update({frontend_key: self.make_cnn_decoder(input_data_shape=cnn_output_shape)})
                else:
                    raise ValueError
            
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters {total_parameters:,}")
        cnn_parameters = sum(p.numel() for p in self.cnn_encoder.parameters() if p.requires_grad)
        print(f"  CNN encoder parameters {cnn_parameters:,}")

        for key, frontend_key in self.frontends.items():
            n_parameters = sum(p.numel() for p in frontend_key.parameters() if p.requires_grad)
            print(f"  Frontend `{key}` parameters: {n_parameters:,}")

        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

        self.example_input_array = torch.zeros(
            (2, 1, self.signal_window_size, 8, 8), 
            dtype=torch.float32,
        )
        self.regression_mse_loss = torchmetrics.MeanSquaredError()
        self.regression_r2_score = torchmetrics.R2Score()
        self.reconstruction_mse_loss = torchmetrics.MeanSquaredError()
        self.classification_bce_loss = BCEWithLogit()
        self.classification_f1_score = torchmetrics.F1Score(task='binary')

        # self.sum_loss = None
        # if self.frontends_active['time_to_elm_regression']:
        #     self.sum_loss = (
        #         self.regression_mse_loss if self.sum_loss is None 
        #         else self.sum_loss + self.regression_mse_loss
        #     )
        # if self.frontends_active['autoencoder_reconstruction']:
        #     self.sum_loss = (
        #         self.reconstruction_mse_loss if self.sum_loss is None 
        #         else self.sum_loss + self.reconstruction_mse_loss
        #     )
        # if self.frontends_active['active_elm_classification']:
        #     self.sum_loss = (
        #         self.classification_bce_loss if self.sum_loss is None 
        #         else self.sum_loss + self.classification_bce_loss
        #     )
        
        self.initialize_layers()

    def forward(self, signals: torch.Tensor) -> dict[torch.Tensor]:
        results = {}
        features = self.cnn_encoder(signals)
        for key, frontend in self.frontends.items():
            if self.frontends_active[key]:
                results[key] = frontend(features)
        return results

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        signals, labels, class_labels = batch
        results = self(signals)
        sum_loss = None
        sum_metric = None
        for frontend_key in self._frontend_names:
            if self.frontends_active[frontend_key] is False:
                continue
            frontend_result = results[frontend_key]
            loss = None
            if 'time_to_elm' in frontend_key:
                loss = self.regression_mse_loss(frontend_result, labels)
                self.log("loss/regression_mse/train", self.regression_mse_loss)
                loss_compute = self.regression_mse_loss.compute()
                self.regression_r2_score.update(frontend_result, labels)
                self.log("score/regression_r2/train", self.regression_r2_score)
            elif 'reconstruction' in frontend_key:
                loss = self.reconstruction_mse_loss(frontend_result, signals)
                self.log("loss/reconstruction_mse/train", self.reconstruction_mse_loss)
                loss_compute = self.reconstruction_mse_loss.compute()
            elif 'classifier' in frontend_key:
                loss = self.classification_bce_loss(frontend_result, class_labels)
                self.log("loss/classification_bce/train", self.classification_bce_loss)
                loss_compute = self.classification_bce_loss.compute()
                self.classification_f1_score.update(frontend_result, class_labels)
                self.log("score/classification_f1/train", self.classification_f1_score)
            else:
                raise ValueError
            sum_loss = loss if sum_loss is None else sum_loss + loss
            sum_metric = loss_compute if sum_metric is None else sum_metric + loss_compute
        self.log("loss/sum/train", sum_metric)
        return sum_loss

    def validation_step(self, batch, batch_idx) -> None:
        signals, labels, class_labels = batch
        results = self(signals)
        sum_metric = None
        for frontend_key in self._frontend_names:
            if self.frontends_active[frontend_key] is False:
                continue
            frontend_result = results[frontend_key]
            if 'time_to_elm' in frontend_key:
                self.regression_mse_loss.update(frontend_result, labels)
                self.log("loss/regression_mse/val", self.regression_mse_loss)
                loss_compute = self.regression_mse_loss.compute()
                self.regression_r2_score.update(frontend_result, labels)
                self.log("score/regression_r2/val", self.regression_r2_score)
            elif 'reconstruction' in frontend_key:
                self.reconstruction_mse_loss.update(frontend_result, signals)
                self.log("loss/reconstruction_mse/val", self.reconstruction_mse_loss)
                loss_compute = self.reconstruction_mse_loss.compute()
            elif 'classifier' in frontend_key:
                self.classification_bce_loss.update(frontend_result, class_labels)
                self.log("loss/classification_bce/val", self.classification_bce_loss)
                loss_compute = self.classification_bce_loss.compute()
                self.classification_f1_score.update(frontend_result, class_labels)
                self.log("score/classification_f1/val", self.classification_f1_score)
            else:
                raise ValueError
            sum_metric = loss_compute if sum_metric is None else sum_metric + loss_compute
        self.log("loss/sum/val", sum_metric)

    def test_step(self, batch, batch_idx) -> None:
        signals, labels, class_labels = batch
        results = self(signals)
        sum_metric = None
        for frontend_key in self._frontend_names:
            if self.frontends_active[frontend_key] is False:
                continue
            frontend_result = results[frontend_key]
            if 'time_to_elm' in frontend_key:
                self.regression_mse_loss.update(frontend_result, labels)
                self.log("loss/regression_mse/test", self.regression_mse_loss)
                loss_compute = self.regression_mse_loss.compute()
                self.regression_r2_score.update(frontend_result, labels)
                self.log("score/regression_r2/test", self.regression_r2_score)
            elif 'reconstruction' in frontend_key:
                self.reconstruction_mse_loss.update(frontend_result, signals)
                self.log("loss/reconstruction_mse/test", self.reconstruction_mse_loss)
                loss_compute = self.reconstruction_mse_loss.compute()
            elif 'classifier' in frontend_key:
                self.classification_bce_loss.update(frontend_result, class_labels)
                self.log("loss/classification_bce/test", self.classification_bce_loss)
                loss_compute = self.classification_bce_loss.compute()
                self.classification_f1_score.update(frontend_result, class_labels)
                self.log("score/classification_f1/test", self.classification_f1_score)
            else:
                raise ValueError
            sum_metric = loss_compute if sum_metric is None else sum_metric + loss_compute
        self.log("loss/sum/test", sum_metric)

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        signals, labels, class_labels = batch
        results = self(signals)
        if batch_idx == 0:
            self.predict_outputs.append([])
            assert dataloader_idx == len(self.predict_outputs)-1
        outputs = {
            'labels': labels,
            'signals': signals,
            'class_labels': class_labels,
        }
        outputs.update(results)
        self.predict_outputs[-1].append(outputs)
    
    def on_train_epoch_start(self):
        self.t_train_epoch_start = time.time()
        if self.global_rank != 0:
            return
        print(f"Epoch {self.current_epoch} start")
        for name, param in self.named_parameters():
            if 'weight' in name:
                values = param.data.detach()
                mean = torch.mean(values).item()
                std = torch.std(values).item()
                z_scores = (values-mean)/std
                skew = torch.mean(z_scores**3).item()
                kurt = torch.mean(z_scores**4).item()
                self.log(f"param_mean/{name}", mean, rank_zero_only=True, sync_dist=True)
                self.log(f"param_std/{name}", std, rank_zero_only=True, sync_dist=True)
                self.log(f"param_skew/{name}", skew, rank_zero_only=True, sync_dist=True)
                self.log(f"param_kurt/{name}", kurt, rank_zero_only=True, sync_dist=True)

    def on_train_epoch_end(self) -> None:
        print(f"Epoch {self.current_epoch} elapsed train time: {(time.time()-self.t_train_epoch_start)/60:0.1f} min")

    def on_validation_epoch_start(self) -> None:
        self.t_val_epoch_start = time.time()

    def on_validation_epoch_end(self) -> None:
        print(f"Epoch {self.current_epoch} elapsed valid. time: {(time.time()-self.t_val_epoch_start)/60:0.1f} min")

    def on_fit_start(self) -> None:
        self.t_fit_start = time.time()

    def on_fit_end(self) -> None:
        print(f"Fit elapsed time {(time.time()-self.t_fit_start)/60:0.1f} min")

    def on_test_start(self) -> None:
        self.t_test_start = time.time()

    def on_test_end(self) -> None:
        print(f"Test elapsed time {(time.time()-self.t_test_start)/60:0.1f} min")

    def on_predict_start(self) -> None:
        self.predict_outputs: list[list] = []
        self.t_predict_start = time.time()

    def on_predict_end(self) -> None:
        print(f"Predict elapsed time {(time.time()-self.t_predict_start)/60:0.1f} min")

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
