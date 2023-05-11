import dataclasses
import os
import time

import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt

import torch
import torch.nn
import torch.utils.data

from lightning.pytorch import LightningModule, loggers
import torchmetrics


@dataclasses.dataclass(eq=False)
class Lightning_Model(LightningModule):
    lr: float = 1e-3
    lr_scheduler_patience: int = 20
    lr_scheduler_threshold: float = 1e-3
    weight_decay: float = 1e-6
    monitor_metric: str = 'loss/sum/val'
    log_dir: str = dataclasses.field(default='.', init=False)
    
    def __post_init__(self):
        super().__init__()
        self.torch_model = None

    def set_torch_model(self, torch_model: torch.nn.Module):
        assert hasattr(torch_model, 'signal_window_size')
        hp_fields = dataclasses.asdict(self) | dataclasses.asdict(torch_model)
        hp_fields['torch_model_name'] = torch_model.__class__.__name__
        for field in ['lr_scheduler_patience','lr_scheduler_threshold']:
            hp_fields.pop(field)
        self.save_hyperparameters(hp_fields)

        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

        self.torch_model = torch_model
        self.signal_window_size = self.torch_model.signal_window_size

        self.example_input_array = torch.zeros(
            (2, 1, self.signal_window_size, 8, 8), 
            dtype=torch.float32,
        )
        self.regression_mse_loss = torchmetrics.MeanSquaredError()
        self.regression_r2_score = torchmetrics.R2Score()
        self.reconstruction_mse_loss = torchmetrics.MeanSquaredError()
        self.classification_bce_loss = torchmetrics.Accuracy(task='binary')
        self.classification_f1_score = torchmetrics.F1Score(task='binary')

    def forward(self, signals) -> torch.Tensor:
        return self.torch_model(signals)

    def training_step(self, batch, batch_idx) -> list[torch.Tensor]:
        signals, labels, class_labels = batch
        results = self(signals)
        losses = []
        if self.torch_model.cnn_regression_active:
            predictions = results.pop(0)
            regression_mse_loss = self.regression_mse_loss(predictions, labels)
            self.log("loss/regression_mse/train", self.regression_mse_loss)
            losses.append(regression_mse_loss)
        if self.torch_model.autoencoder_active:
            predictions = results.pop(0)
            reconstruction_mse_loss = self.reconstruction_mse_loss(predictions, signals)
            self.log("loss/reconstruction_mse/train", self.reconstruction_mse_loss)
            losses.append(reconstruction_mse_loss)
        if self.torch_model.cnn_classification_active:
            class_predictions = results.pop(0)
            classification_bce_loss = self.classification_bce_loss(
                class_predictions,
                class_labels,
            )
            self.log("loss/classification_bce/train", self.reconstruction_mse_loss)
            losses.append(classification_bce_loss)
        loss = sum(losses)
        self.log("loss/sum/train", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        signals, labels, class_labels = batch
        results = self(signals)
        losses = []
        if self.torch_model.cnn_regression_active:
            predictions = results.pop(0)
            regression_mse_loss = self.regression_mse_loss(predictions, labels)
            self.log("loss/regression_mse/val", self.regression_mse_loss)
            losses.append(regression_mse_loss)
            self.regression_r2_score(predictions, labels)
            self.log("score/regression_r2/val", self.regression_r2_score)
        if self.torch_model.autoencoder_active:
            predictions = results.pop(0)
            reconstruction_mse_loss = self.reconstruction_mse_loss(predictions, signals)
            self.log("loss/reconstruction_mse/val", self.reconstruction_mse_loss)
            losses.append(reconstruction_mse_loss)
        if self.torch_model.cnn_classification_active:
            class_predictions = results.pop(0)
            classification_bce_loss = self.classification_bce_loss(
                class_predictions,
                class_labels,
            )
            self.log("loss/classification_bce/val", self.reconstruction_mse_loss)
            losses.append(classification_bce_loss)
            self.classification_f1_score(
                class_predictions,
                class_labels,
            )
            self.log("score/classification_f1/val", self.classification_f1_score)
        loss = sum(losses)
        self.log("loss/sum/val", loss)

    def test_step(self, batch, batch_idx) -> None:
        signals, labels, class_labels = batch
        results = self(signals)
        losses = []
        if self.torch_model.cnn_regression_active:
            predictions = results.pop(0)
            regression_mse_loss = self.regression_mse_loss(predictions, labels)
            self.log("loss/regression_mse/test", self.regression_mse_loss)
            losses.append(regression_mse_loss)
            self.regression_r2_score(predictions, labels)
            self.log("score/regression_r2/test", self.regression_r2_score)
        if self.torch_model.autoencoder_active:
            predictions = results.pop(0)
            reconstruction_mse_loss = self.reconstruction_mse_loss(predictions, signals)
            self.log("loss/reconstruction_mse/test", self.reconstruction_mse_loss)
            losses.append(reconstruction_mse_loss)
        if self.torch_model.cnn_classification_active:
            class_predictions = results.pop(0)
            classification_bce_loss = self.classification_bce_loss(
                class_predictions,
                class_labels,
            )
            self.log("loss/classification_bce/test", self.reconstruction_mse_loss)
            losses.append(classification_bce_loss)
            self.classification_f1_score(
                class_predictions,
                class_labels,
            )
            self.log("score/classification_f1/test", self.classification_f1_score)
        loss = sum(losses)
        self.log("loss/sum/test", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        signals, labels, class_labels = batch
        results = self(signals)
        predictions = results.pop(0) if self.torch_model.cnn_regression_active else None
        reconstruction = results.pop(0) if self.torch_model.autoencoder_active else None
        class_predictions = results.pop(0) if self.torch_model.cnn_classification_active else None
        if batch_idx == 0:
            self.predict_outputs.append([])
            assert dataloader_idx == len(self.predict_outputs)-1
        self.predict_outputs[-1].append({
            'labels': labels,
            'predictions': predictions,
            'signals': signals,
            'reconstruction': reconstruction,
            'class_predictions': class_predictions,
        })
    
    def on_train_epoch_start(self):
        self.t_train_epoch_start = time.time()
        if self.global_rank != 0:
            return
        print(f"Epoch {self.current_epoch} start")
        for name, param in self.torch_model.named_parameters():
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
        if self.global_rank != 0:
            return
        i_page = 1
        for i_elm, result in enumerate(self.predict_outputs):
            labels = torch.concat([batch['labels'] for batch in result]).squeeze().numpy(force=True)
            predictions = torch.concat([batch['predictions'] for batch in result]).squeeze().numpy(force=True)
            signals = torch.concat([batch['signals'] for batch in result]).squeeze().numpy(force=True)
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
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=0.5,
                patience=self.lr_scheduler_patience,
                threshold=self.lr_scheduler_threshold,
                min_lr=1e-5,
                mode='min' if 'loss' in self.monitor_metric else 'max',
            ),
            'monitor': self.monitor_metric,
        }
