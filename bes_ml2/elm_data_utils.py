import os

import numpy as np
import matplotlib.pyplot as plt

try:
    from . import elm_datamodule
except:
    from bes_ml2 import elm_datamodule


def plot_stats(
    max_elms = None,
    data_file = None,
    save = False, 
    figure_dir = '.', 
    block_show = True,
    mask_sigma_outliers = 8,
):
    datamodule = elm_datamodule.ELM_Datamodule(
        data_file=data_file,
        max_elms=max_elms,
        fraction_test=1.,
        mask_sigma_outliers=mask_sigma_outliers,
    )
    
    datamodule.setup(stage='predict')

    dataloaders = datamodule.predict_dataloader()
    n_elms = len(dataloaders)
    i_page = 1
    fig, axes = plt.subplots(ncols=5, nrows=4, figsize=(14, 8.5))
    axes = axes.flatten()
    n_axes = axes.size
    channels = np.arange(1,65)
    for i_elm, dataloader in enumerate(dataloaders):
        dataset: elm_datamodule.ELM_Predict_Dataset = dataloader.dataset
        elm_index = dataset.elm_index
        shot = dataset.shot
        pre_elm_size = dataset.active_elm_start_index-1
        stats = dataset.pre_elm_stats()
        if i_elm%n_axes == 0:
            plt.suptitle(f"Channel-wise ELM stats (page {i_page})")
            for ax in axes:
                ax.clear()
        plt.sca(axes[i_elm%n_axes])
        for stat_name in ['min','max','mean','std']:
            plt.plot(channels, stats[stat_name], label=stat_name)
        plt.axhline(0, linestyle='--')
        plt.title(f"ELM index {elm_index} Shot {shot}", fontsize='medium')
        plt.xlabel('Channel')
        plt.ylim(-8,8)
        if i_elm%n_axes==0:
            plt.legend(loc='lower right', fontsize='small')
        if i_elm%n_axes==n_axes-1 or i_elm==n_elms-1:
            plt.tight_layout()
            if save:
                filepath = os.path.join(figure_dir, f'elm_stats_{i_page:03d}.pdf')
                print(f"Saving figure {filepath}")
                plt.savefig(filepath, format='pdf', transparent=True)
            i_page += 1
            plt.show(block=block_show)
    plt.close('all')


if __name__=='__main__':
    plot_stats(
        # data_file='/global/homes/d/drsmith/ml/scratch/data/labeled_elm_events.hdf5',
        save=True, 
        # block_show=False,
        max_elms=100,
        mask_sigma_outliers=4,
    )
    