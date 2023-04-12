import dataclasses
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from torch.utils.data import DataLoader, BatchSampler

from bes_ml.base.utilities import merge_pdfs

try:
    from ..base.analyze_base import Analyzer_Base
    from ..base.sampler import RandomBatchSampler, SequentialBatchSampler
    from ...bes_data.confinement_data_tools.dataset import ConfinementDataset
    from .confinement_data_v2 import ConfinementDataset
    from .confinement_mode_data import Confinement_Mode_Dataset
except ImportError:
    from bes_ml.base.analyze_base import Analyzer_Base
    from bes_ml.base.sampler import RandomBatchSampler, SequentialBatchSampler
    from bes_ml.confinement_classification.confinement_data_v2 import ConfinementDataset
    from bes_ml.confinement_classification.confinement_mode_data import Confinement_Mode_Dataset

# plt.style.use('seaborn')

@dataclasses.dataclass
class Analyzer(Analyzer_Base):

    def __post_init__(self) -> None:
        super().__post_init__()

        self.is_regression = False
        self.is_classification = not self.is_regression

    def run_inference(
            self,
            max_confinement_modes: int = None,
    ) -> None:

        n_confinement_modes = len(self.test_data['indices'])
        self.all_predictions = []
        self.all_labels = []
        self.all_signals = []
        self.all_signals_1 = []
        with torch.no_grad():
            # loop over confinement modes in test data
            print('Running inference on test data')
            for i_confinement_mode in range(n_confinement_modes):
                if max_confinement_modes and i_confinement_mode >= max_confinement_modes:
                    break
                print(f'  Confinement mode {i_confinement_mode+1} of {n_confinement_modes}')
                i_start = self.test_data['window_start'][i_confinement_mode]
                i_stop = (
                    self.test_data['window_start'][i_confinement_mode+1]-1
                    if i_confinement_mode < n_confinement_modes-1
                    else self.test_data['labels'].size-1
                )
                confinement_mode_signals = self.test_data['signals'][i_start:i_stop+1, ...]
                confinement_mode_labels = self.test_data['labels'][i_start:i_stop+1]
                confinement_mode_sample_indices = np.arange(confinement_mode_labels.size - self.inputs['signal_window_size'], dtype=int)
                # confinement_mode_sample_indices = self.test_data['sample_indices']

                confinement_mode_test_dataset = Confinement_Mode_Dataset(
                        signals=confinement_mode_signals,
                        labels=confinement_mode_labels,
                        sample_indices=confinement_mode_sample_indices,
                        signal_window_size=self.inputs['signal_window_size'],
                        prediction_horizon=self.prediction_horizon if hasattr(self, 'prediction_horizon') else 0,
                    )
                test_data_loader = torch.utils.data.DataLoader(
                    dataset=confinement_mode_test_dataset,
                    batch_size=self.inputs['batch_size'],
                    shuffle=False,
                    num_workers=2 if torch.cuda.is_available() else 0,
                    pin_memory=False,
                    drop_last=True,
                    )
                confinement_mode_predictions = np.empty((len(test_data_loader) * self.inputs['batch_size'], self.inputs['mlp_output_size']), dtype=np.float32)
                for i_batch, (batch_signals, batch_labels) in enumerate(test_data_loader):
                    batch_signals = batch_signals.to(self.device)
                    batch_predictions = self.model(batch_signals)
                    confinement_mode_predictions[i_batch * self.inputs['batch_size']:(i_batch + 1) * self.inputs['batch_size']] = \
                        batch_predictions.cpu().numpy()
                self.all_labels.append(confinement_mode_labels[:confinement_mode_predictions.shape[0]])
                self.all_predictions.append(confinement_mode_predictions)
                self.all_signals.append(confinement_mode_signals[:,2,3])
                self.all_signals_1.append(confinement_mode_signals)
                # either truncate labels or pad predictions
                # tmp = np.zeros((confinement_mode_labels.shape[0], 4))
                # tmp[:confinement_mode_predictions.shape[0], :] = confinement_mode_predictions
                # self.all_predictions.append(tmp)

        print('Inference complete')

    def plot_inference(self,
                       idx: int = None,
                       save: bool = False,
                       class_labels: list = ['L-mode', 'H-mode', 'QH-mode', 'WP QH-mode']):

        if None in [self.all_labels, self.all_predictions, self.all_signals]:
            self.run_inference()

        labels = np.concatenate([i for i in self.all_labels], axis=0)
        signals = np.concatenate([i for i in self.all_signals], axis=0)
        preds = np.concatenate([i for i in self.all_predictions], axis=0)

        if self.inputs['mlp_output_size'] == 1:
            preds[np.where(preds<0.5)[0]] = 0
            preds[np.where(preds>=0.5)[0]] = 1
        else:
            preds = preds.argmax(axis=1)
        
        # Pad labels and predictions to account for signal window
        # l_diff = len(signals) - len(preds)
        # preds = np.pad(preds, ((l_diff, 0), (0, 0)))

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
        cm = confusion_matrix(labels, preds, normalize='true')
        cm = np.round(cm, 2)

        self.plot_confusion_matrix(cm,
                                   classes=class_labels,
                                   ax=ax1)

        cr = classification_report(labels, preds, target_names=class_labels, zero_division=0, labels=range(len(class_labels)))

        ax1.text(-1.3, 0.5,
                 f'{cr}\n'
                 f'ROC: {np.max(self.train_score):0.2f} (epoch {np.argmax(self.train_score)})    '
                 f'Best Loss: {np.min(self.valid_loss):0.2f} (epoch {np.argmin(self.valid_loss)})',
                 transform=ax1.transAxes,
                 ha='right', va='center', ma='left',
                 bbox=dict(boxstyle="square", fc='w', lw=2))

        ax2.plot(self.valid_loss, label='valid. loss')
        ax2.plot(self.train_loss, label='training loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Performance Metrics by Epoch')
        ax2.legend()

        ax3 = ax2.twinx()
        ax3.plot(self.train_score, label='ROC-AUC score', color='r')
        ax3.set_ylabel('ROC-AUC Score')
        ax3.legend()

        fig.suptitle(f'Summary results of {type(self.model).__name__}')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary.png')
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

    def plot_prediction(self,
                        save_filename: str=None,
                        class_labels: list=['L-mode', 'H-mode', 'QH-mode', 'WP QH-mode']):

        if None in [self.all_labels, self.all_predictions, self.all_signals]:
            self.run_inference()

        y_dim = 9 * int(len(self.all_signals) / 2)
        fig = plt.figure(figsize=(20, y_dim))  # Create a figure with custom size
        # Create a gridspec with len(self.all_signals) rows and 2 columns, where the second column takes up less space
        gs = gridspec.GridSpec(len(self.all_signals), 2, width_ratios=[9, 1])

        for idx, signal in enumerate(self.all_signals):
            shot = str(self.test_data['indices'][idx])[:6]
            time = str(self.test_data['indices'][idx])[6:]
            labels = self.all_labels[idx]
            preds = self.all_predictions[idx]
            
            if self.inputs['mlp_output_size'] == 1:
                preds[np.where(preds<0.5)[0]] = 0
                preds[np.where(preds>=0.5)[0]] = 1
            else:
                preds = preds.argmax(axis=1)

            ax1 = plt.subplot(gs[idx, 0])  # Create a subplot in the first column
            ax2 = plt.subplot(gs[idx, 1])  # Create a subplot in the second column

            ax1.plot(signal, label='BES 20', color='black', zorder=0)
            
            # colors = ['gold', 'deeppink', 'mediumpurple', 'springgreen']
            # colors = ['#0000ff', '#b56ae8', '#68afb1', '#f3ff2c']
            # colors = ['#005a74', '#68afb1', '#ed717e', '#800040']
            colors = ['#ed717e', '#b56ae8', '#68afb1', '#f3ff2c']

            for i in range(self.inputs['mlp_output_size']):
                indices = np.where(preds == i)[0]
                frac = indices.shape[0]/preds.shape[0]
                if len(indices) > 0:
                    # ax1.fill_between(range(len(signal)), 2, 3, where=np.isin(range(len(signal)), indices), color=colors[i], alpha=0.2, label=f'predicted', zorder=1)
                    # ax1.scatter(indices, [3]*len(indices), color=colors[i], alpha=0.2, marker='|', label=f'predicted')
                    ymin = i + 2
                    ymax = i + 3
                    ax1.vlines(indices, ymin=ymin, ymax=ymax, color=colors[i], alpha=frac)

            for i in range(self.inputs['mlp_output_size']):
                indices = np.where(labels == i)[0]
                if len(indices) > 0:
                    ax1.fill_between(range(len(signal)), -4, 0, where=np.isin(range(len(signal)), indices), color=colors[i], alpha=0.4, label=f'ground truth', zorder=1)
                    # ax1.scatter(indices, [-4]*len(indices), color=colors[i], alpha=0.2, marker='|', label=f'ground truth')

            if self.inputs['mlp_output_size']==1:
                pie_values = [np.where(preds==0)[0].shape[0], np.where(preds==1)[0].shape[0]]
            elif self.inputs['mlp_output_size']==3:
                pie_values = [np.where(preds==0)[0].shape[0], np.where(preds==1)[0].shape[0], np.where(preds==2)[0].shape[0]]
            else:
                pie_values = [np.where(preds==0)[0].shape[0], np.where(preds==1)[0].shape[0], np.where(preds==2)[0].shape[0], np.where(preds==3)[0].shape[0]]

            explode = np.zeros(len(class_labels), dtype=np.float32)
            explode[labels[0]] = 0.2
            pie_labels = ['']*len(class_labels)
            pie_labels[labels[0]] = 'correct'
            ax2.pie(pie_values, labels=pie_labels, explode=explode, colors=colors, autopct='%1.1f%%', startangle=90)

            legend_handles = [ax1.scatter([], [], color=colors[idx], label=f'{label}') for idx, label in enumerate(class_labels)]
            ax1.set_ylabel('BES 20 Signal')
            ax1.set_ylim(-5,6)
            ax1.legend(handles=legend_handles, prop={'size':15}, loc='lower left', markerscale=4)
            ax1.set_title(f"Shot {shot} at time {time}")
            ax1.set_xlabel('Time (mu-s)')
            if idx==len(self.all_signals)-1:
                plt.tight_layout()
                output_file = self.output_dir/(save_filename)
                plt.savefig(
                    output_file, 
                    format=save_filename[-3:], 
                    transparent=True,
                )

        print("finished plotting inference on test data")


    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              ax=None,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if ax is None:
            ax = plt.gca()
        img = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        plt.colorbar(img, ax=ax)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks, classes, rotation=45)
        ax.set_yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

        return ax

    def bes_plot(self,
                save_filename: str=None,
                ):

        if None in [self.all_labels, self.all_predictions, self.all_signals]:
            self.run_inference()

        signal_grid = self.all_signals_1[0][1]

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

        # fig, ax = plt.figure(frameon=False)
        for ax in axs:
            ax.imshow(signal_grid, interpolation='bilinear', cmap=plt.get_cmap('cubehelix'))
        output_file = self.output_dir/(save_filename)
        plt.tight_layout()
        plt.savefig(output_file, format='pdf')
        # breakpoint()
        



if __name__ == '__main__':
    class_labels = ['H-mode', 'QH-mode', 'WP QH-mode']
    analyzer = Analyzer()
    analyzer.run_inference()
    analyzer.plot_prediction(save_filename='plot_prediction.jpg', class_labels=class_labels)
    analyzer.plot_training(save=True)
    analyzer.plot_inference(save=True, class_labels=class_labels)
    # analyzer.bes_plot(save_filename='test.pdf')
    analyzer.show()
