import dataclasses
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from torch.utils.data import DataLoader, BatchSampler

from bes_ml.base.utilities import merge_pdfs

try:
    from ..base.analyze_base import _Analyzer_Base
    from ..base.sampler import RandomBatchSampler, SequentialBatchSampler
    from ...bes_data.velocimetry_data_tools.dataset import VelocimetryDataset
except ImportError:
    from bes_ml.base.analyze_base import _Analyzer_Base
    from bes_ml.base.sampler import RandomBatchSampler, SequentialBatchSampler
    from bes_data.velocimetry_data_tools.dataset import VelocimetryDataset

@dataclasses.dataclass
class Analyzer(_Analyzer_Base):

    def __post_init__(self) -> None:
        super().__post_init__()

        self.is_regression = True
        self.is_classification = not self.is_regression

    def run_inference(
            self,
    ) -> None:

        self.all_predictions = []
        self.all_labels = []
        self.all_signals = []

        with torch.no_grad():
            print('Running inference on test data')

            signals = self.test_data['signals']
            labels = self.test_data['labels']

            velocimetry_test_dataset = VelocimetryDataset(
                data_location=self.inputs['data_location'],
                output_dir=self.inputs['output_dir'],
                signal_window_size=self.inputs['signal_window_size'],
                batch_size=self.inputs['batch_size'],
                dataset_to_ram=self.inputs['dataset_to_ram'],
                state='test'
            )

            velocimetry_test_dataset.load_datasets(signals=signals, labels=labels)

            test_data_loader = DataLoader(velocimetry_test_dataset,
                                          batch_size=None,  # must be disabled when using samplers
                                          sampler=BatchSampler(SequentialBatchSampler(velocimetry_test_dataset,
                                                                                  self.inputs['batch_size'],
                                                                                  self.inputs['signal_window_size']),
                                                               batch_size=self.inputs['batch_size'],
                                                               drop_last=True
                                                               )
                                          )

            predictions = np.empty((len(test_data_loader) * self.inputs['batch_size'], 128), dtype=np.float32)
            for i_batch, (batch_signals, batch_labels) in enumerate(test_data_loader):
                batch_signals = batch_signals.to(self.device)
                batch_predictions = self.model(batch_signals)
                predictions[i_batch * self.inputs['batch_size']:(i_batch + 1) * self.inputs['batch_size']] = \
                    batch_predictions.cpu().numpy()

            self.all_labels.append(labels)
            self.all_predictions.append(predictions)
            self.all_signals.append(signals)
        print('Inference complete')

    def plot_inference(self,
                       idx: int = None,
                       save: bool = False,
                       ):
        if None in [self.all_labels, self.all_predictions, self.all_signals]:
            self.run_inference()

        plt.rcParams['axes.labelsize'] = 24
        plt.rcParams['axes.titlesize'] = 18
        _, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 6))

        axes = axes.flat
        plt.suptitle(f"{self.output_dir} | Test data (full)")
        signals = self.all_signals[0].reshape((-1, 8, 8))

        vR_labels = self.all_labels[0][:, :64].reshape((-1, 8, 8))
        vR_predictions = self.all_predictions[0][:, :64].reshape((-1, 8, 8))

        vZ_labels = self.all_labels[0][:, 64:].reshape((-1, 8, 8))
        vZ_predictions = self.all_predictions[0][:, 64:].reshape((-1, 8, 8))

        # Pad labels and predictions to account for signal window
        l_diff = len(signals) - len(vZ_predictions)
        # vZ_labels = np.pad(vZ_labels, ((l_diff, 0), (0, 0), (0, 0)))
        # vR_labels = np.pad(vR_labels, ((l_diff, 0), (0, 0), (0, 0)))
        vZ_predictions = np.pad(vZ_predictions, ((l_diff, 0), (0, 0), (0, 0)))
        vR_predictions = np.pad(vR_predictions, ((l_diff, 0), (0, 0), (0, 0)))

        print('vZ Range ', vZ_predictions.min(), vZ_predictions.max())
        print('vR Range ', vR_predictions.min(), vR_predictions.max())

        elm_time = np.arange(len(signals))
        # plot signal, labels, and prediction
        axes[0].plot(elm_time, signals[:, 2, 6],
                     label="BES ch 22",
                     c='g',
                     alpha=0.5,
                     zorder=0.0)
        # add axis labels for prediction
        ax1 = axes[0].twinx()
        ax1.plot(elm_time, vZ_labels[:, 2, 6],
                 label="Ground truth (ch 22)",
                 c='orange',
                 alpha=0.5,
                 zorder=1.0)

        ax1.plot(elm_time, mean_squared_error(vZ_labels.reshape((-1, 64)).T, vZ_predictions.reshape((-1, 64)).T,
                                              multioutput='raw_values',
                                              squared=False),
                 label="RMS Error",
                 c='b',
                 alpha=0.5,
                 zorder=2.0)

        axes[0].set_title("vZ")
        axes[0].set_ylabel("Signal")
        axes[0].set_xlabel("Time ($\mu$s)")
        ax1.set_ylabel("Velocity")
        axes[0].legend(loc=2)
        ax1.legend(loc=1)

        axes[1].plot(elm_time, signals[:, 2, 6],
                     label="BES ch 22",
                     c='g',
                     alpha=0.5,
                     zorder=0.0)
        # add axis labels for prediction
        ax2 = axes[1].twinx()
        ax2.plot(elm_time,
                 vR_labels[:, 2, 6],
                 label="Ground truth (ch 22)",
                 c='orange',
                 alpha=0.5,
                 zorder=1.0)

        ax2.plot(elm_time,
                 mean_squared_error(vZ_labels.reshape((-1, 64)).T, vZ_predictions.reshape((-1, 64)).T,
                                    multioutput='raw_values',
                                    squared=False),
                 label="RMS Error",
                 c='b',
                 alpha=0.5,
                 zorder=2.0)

        axes[1].set_title(f'vR')
        axes[1].set_xlabel("Time ($\mu$s)")
        axes[1].set_ylabel("Signal")
        ax2.set_ylabel("Velocity")
        ax2.legend(loc=1)
        axes[1].legend(loc=2)

        # Plot vector field of velocimetry predictions
        time_slice = idx if idx else len(signals) // 2
        # Normalize BES array
        tx = signals[time_slice]
        tx[:4] = tx[:4] / 10
        tx[4:] = tx[4:] / 5

        Y, X = np.mgrid[0:8, 0:8]
        axes[2].imshow(tx, interpolation='hanning', cmap='gist_stern', alpha=0.8,
                       norm=TwoSlopeNorm(vcenter=.1))
        axes[2].quiver(X, Y, vR_labels[time_slice], vZ_labels[time_slice])
        axes[2].set_xlabel('R')
        axes[2].set_ylabel('Z')
        axes[2].set_title(f'ODP Calculated Velocity Field (t={time_slice} $\mu$s)')

        axes[3].imshow(tx, interpolation='hanning', cmap='gist_stern', alpha=0.8,
                       norm=TwoSlopeNorm(vcenter=.1))
        axes[3].quiver(X, Y, vR_predictions[time_slice], vZ_predictions[time_slice])
        axes[3].set_xlabel('R')
        axes[3].set_ylabel('Z')
        axes[3].set_title(f'ML Calculated Velocity Field (t={time_slice} $\mu$s)')

        axes[0].axvline(time_slice, alpha=0.5, c='r')
        axes[0].text(time_slice, 0.9, f'{time_slice} $\mu$s')
        axes[1].axvline(time_slice, alpha=0.5, c='r')
        axes[1].text(time_slice, 0.9, f'{time_slice} $\mu$s')

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'inference_idx_{idx}.pdf'
            print(f'Saving inference file: {filepath}')
            plt.savefig(filepath, format='pdf', transparent=True)

            inputs = sorted(self.output_dir.glob('inference_*.pdf'))
            output = self.output_dir / 'inference.pdf'
            merge_pdfs(
                inputs=inputs,
                output=output,
                delete_inputs=True,
            )


if __name__ == '__main__':
    analyzer = Analyzer()
    analyzer.plot_training(save=True)
    analyzer.plot_inference(save=True)
    analyzer.show()
