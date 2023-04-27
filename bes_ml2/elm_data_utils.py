import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import matplotlib.pyplot as plt

try:
    from . import elm_datamodule
except:
    from bes_ml2 import elm_datamodule


def plot_stats(
    max_elms = None,
    data_file = None,
    figure_dir = '.', 
    block_show = True,
    mask_sigma_outliers = 8,
    save = True, 
    merge = True,
):
    datamodule = elm_datamodule.ELM_Datamodule(
        data_file=data_file,
        max_elms=max_elms,
        fraction_test=1.,
        mask_sigma_outliers=mask_sigma_outliers,
        max_predict_elms=None,
    )
    
    datamodule.setup(stage='predict')

    dataloaders = datamodule.predict_dataloader()
    n_elms = len(dataloaders)
    i_page = 1
    fig, axes = plt.subplots(ncols=5, nrows=4, figsize=(14, 8.5))
    axes = axes.flatten()
    n_elms_per_page = axes.size // 2
    channels = np.arange(1,65)
    for i_elm, dataloader in enumerate(dataloaders):
        dataset: elm_datamodule.ELM_Predict_Dataset = dataloader.dataset
        elm_index = dataset.elm_index
        shot = dataset.shot
        pre_elm_size = dataset.active_elm_start_index-1
        stats = dataset.pre_elm_stats()
        i_elm_on_page = i_elm%n_elms_per_page
        if i_elm_on_page == 0:
            plt.suptitle(f"Channel-wise ELM stats (page {i_page})")
            for ax in axes:
                ax.clear()
        # plot stats
        plt.sca(axes[i_elm_on_page + 5*(i_elm_on_page//5)])
        for key in stats:
            plt.plot(channels, stats[key].flatten(), label=key)
        plt.axhline(0, linestyle='--', color='k')
        plt.axhline(datamodule.max_abs_valid_signal, linestyle='--', color='k')
        plt.title(f"ELM index {elm_index} Shot {shot}", fontsize='medium')
        plt.xlabel('Channel')
        plt.ylim(-1,1.3*datamodule.max_abs_valid_signal)
        if i_elm_on_page==0:
            plt.legend(loc='lower right', fontsize='small')
        # plot time-series signals
        plt.sca(axes[i_elm_on_page + 5*(i_elm_on_page//5) + 5])
        max_abs_channel = np.unravel_index(np.argmax(stats['maxabs']), stats['maxabs'].shape)
        max_std_channel = np.unravel_index(np.argmax(stats['std']), stats['std'].shape)
        interval = np.amax([pre_elm_size//500,1])
        time_axis = (np.arange(-pre_elm_size,0)/1e3)[::interval]
        plt.plot(time_axis, dataset.signals[0, 0:pre_elm_size:interval, max_abs_channel[0], max_abs_channel[1]])
        if not np.array_equal(max_abs_channel, max_std_channel):
            plt.plot(time_axis, dataset.signals[0, 0:pre_elm_size:interval, max_std_channel[0], max_std_channel[1]])
        plt.title(f"ELM index {elm_index} Shot {shot}", fontsize='medium')
        plt.xlabel('Time-to-ELM (ms)')
        plt.ylabel('Scaled BES signals')
        plt.axhline(datamodule.max_abs_valid_signal, linestyle='--', color='k')
        plt.axhline(-datamodule.max_abs_valid_signal, linestyle='--', color='k')
        plt.axhline(0, linestyle='--', color='k')
        plt.ylim(np.array([-1.3,1.3])*datamodule.max_abs_valid_signal)
        if i_elm_on_page==n_elms_per_page-1 or i_elm==n_elms-1:
            plt.tight_layout()
            if save:
                filepath = os.path.join(figure_dir, f'elm_stats_{i_page:03d}.pdf')
                print(f"Saving figure {filepath}")
                plt.savefig(filepath, format='pdf', transparent=True)
            i_page += 1
            plt.show(block=block_show)
    plt.close('all')

    if merge:
        inputs = sorted(Path(figure_dir).glob('elm_stats_*.pdf'))
        assert len(inputs) > 0 and inputs[0].exists()
        output = Path(figure_dir) / 'elm_stats.pdf'
        output.unlink(missing_ok=True)
        gs_cmd = shutil.which('gs')
        assert gs_cmd is not None, \
            "`gs` command (ghostscript) not found; available in conda-forge"
        cmd = [
            gs_cmd,
            '-q',
            '-dBATCH',
            '-dNOPAUSE',
            '-sDEVICE=pdfwrite',
            '-dPDFSETTINGS=/prepress',
            '-dCompatibilityLevel=1.4',
            f"-sOutputFile={output.as_posix()}",
        ]
        cmd.extend([f"{pdf_file.as_posix()}" for pdf_file in inputs])
        print(f"Merging files into {output}")
        result = subprocess.run(cmd, check=True)
        assert result.returncode == 0 and output.exists()
        for pdf_file in inputs:
            pdf_file.unlink(missing_ok=True)


if __name__=='__main__':
    plot_stats(
        # data_file='/global/homes/d/drsmith/ml/scratch/data/labeled_elm_events.hdf5',
        block_show=False,
        max_elms=500,
        mask_sigma_outliers=8,
        # save=False,
        merge=False,
    )
    