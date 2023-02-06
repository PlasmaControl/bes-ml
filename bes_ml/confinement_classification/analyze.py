import dataclasses
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, BatchSampler

from bes_ml.base.utilities import merge_pdfs

try:
    from ..base.analyze_base import Analyzer_Base
    from ..base.sampler import RandomBatchSampler, SequentialBatchSampler
    from ...bes_data.confinement_data_tools.dataset import ConfinementDataset
    from .confinement_data_v2 import ConfinementDataset
except ImportError:
    from bes_ml.base.analyze_base import Analyzer_Base
    from bes_ml.base.sampler import RandomBatchSampler, SequentialBatchSampler
    from bes_ml.confinement_classification.confinement_data_v2 import ConfinementDataset

@dataclasses.dataclass
class Analyzer(Analyzer_Base):

    def __post_init__(self) -> None:
        super().__post_init__()

        self.is_regression = False
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
            time = self.test_data['time']

            signals = np.array(self.test_data['signals'])
            labels = np.array(self.test_data['labels'])
            # one_hot_labels = np.zeros((len(self.test_data['labels']), 4))
            # for i in range(len(labels)):
            #     one_hot_labels[i][labels[i]] = 1
            # labels = np.array(one_hot_labels)
            # breakpoint()

            confinement_test_dataset = ConfinementDataset(
                data_location=self.inputs['data_location'],
                signal_window_size=self.inputs['signal_window_size'],
                batch_size=self.inputs['batch_size'],
                dataset_to_ram=self.inputs['dataset_to_ram'],
                state='test',
            )

            confinement_test_dataset.load_datasets(signals=signals, labels=labels, time=time)

            test_data_loader = DataLoader(
                confinement_test_dataset,
                batch_size=None,  # must be disabled when using samplers
                sampler=BatchSampler(
                    RandomBatchSampler(
                        confinement_test_dataset,
                        self.inputs['batch_size'],
                        self.inputs['signal_window_size'],
                    ),
                    batch_size=self.inputs['batch_size'],
                    drop_last=True,
                )
            )

            predictions = np.empty((len(test_data_loader) * self.inputs['batch_size'], 4), dtype=np.float32)
            # predictions = []
            
            for i_batch, (batch_signals, batch_labels) in enumerate(test_data_loader):
                batch_signals = batch_signals.to(self.device)
                batch_predictions = self.model(batch_signals)
                predictions[i_batch * self.inputs['batch_size']:(i_batch + 1) * self.inputs['batch_size']] = \
                    batch_predictions.cpu().numpy()
                # for file_signals, file_labels in zip(batch_signals, batch_labels):
                #     curr_predictions = self.model(file_signals)
                #     predictions.append(curr_predictions.cpu().numpy())
                
            self.all_predictions.append(predictions)
            # self.all_labels.append(labels)
            # self.all_signals.append(signals)

            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    self.all_labels.append(labels[i][j])
                    # self.all_predictions.append(predictions[i][j])
                    self.all_signals.append(signals[i][j])

        print('Inference complete')

    def plot_inference(self,
                       idx: int = None,
                       save: bool = False,
                       ):

        if None in [self.all_labels, self.all_predictions, self.all_signals]:
            self.run_inference()

        signals = self.all_signals
        labels = self.all_labels
        preds = self.all_predictions[0]
        
        # preds_final = []
        # for pred in preds:
        #     pred_labels = np.argmax(pred, axis=1)
        #     label_counts = Counter(pred_labels)
        #     preds_final.append(label_counts.most_common(1)[0][0])

        # Pad labels and predictions to account for signal window
        l_diff = len(signals) - len(preds)
        preds = np.pad(preds, ((l_diff, 0), (0, 0)))
        # preds_final = preds_final.extend([0] * l_diff)

        class_labels = ['L-Mode',
                        'H-Mode',
                        'QH-Mode',
                        'WP QH-Mode'
                        ]

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
        cm = confusion_matrix(labels, preds.argmax(axis=1))
        
        self.plot_confusion_matrix(cm,
                                   classes=class_labels,
                                   ax=ax1)

        cr = classification_report(labels, preds.argmax(axis=1), target_names=class_labels, zero_division=0, labels=[0,1,2,3])

        ax1.text(-1.3, 0.5,
                 f'{cr}\n'
                 f'ROC: {np.max(self.train_score):0.2f} (epoch {np.argmax(self.train_score)})    '
                 f'Best Loss: {np.max(self.valid_loss):0.2f} (epoch {np.argmax(self.valid_loss)})',
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


if __name__ == '__main__':
    analyzer = Analyzer()
    analyzer.run_inference()
    analyzer.plot_training(save=True)
    analyzer.plot_inference(save=True)
    analyzer.show()
