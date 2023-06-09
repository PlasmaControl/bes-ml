from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import matplotlib.pyplot as plt
import h5py

try:
    from . import elm_datamodule
except:
    from bes_ml2 import elm_datamodule


def delete_elms(
    data_file: str|Path = None,
    elm_indices: list = None,
) -> None:
    data_file = Path(data_file)
    assert data_file.exists()
    with h5py.File(data_file, 'r+') as h5_file:
        print(f"Initial ELM count in H5 file: {len(h5_file)}")
        for elm_index in elm_indices:
            elm_key = f"{elm_index:05d}"
            if elm_key in h5_file:
                print(f"  Deleting group `{elm_key}`")
                del h5_file[elm_key]
                h5_file.flush()
            else:
                print(f"  Group `{elm_key}` not in H5 file")
        print(f"Final ELM count in H5 file: {len(h5_file)}")
    return


def make_new_data_file(
    data_file: str|Path = None,
    new_file_name: str = './test_data_50.hdf5',
    n_elms: int = 50,
):
    data_file = Path(data_file)
    assert data_file.exists()
    with h5py.File(data_file, 'r+') as h5_file:
        print(f"Initial ELM count in H5 file: {len(h5_file)}")
        keys = None


def plot_stats(
    max_elms: int = None,
    data_file: str = None,
    figure_dir: str = '.', 
    block_show: bool = True,
    mask_sigma_outliers: float = None,
    limit_preelm_max_stdev: float = None,
    limit_preelm_max_abs: float = None,
    save: bool = True, 
    merge: bool = True,
    bad_elm_indices_csv: bool = True,
    bad_elm_indices: list = None,
    skip_elm_plots: bool = False,
):
    t_start = time.time()

    datamodule = elm_datamodule.ELM_Datamodule(
        data_file=data_file,
        max_elms=max_elms,
        mask_sigma_outliers=mask_sigma_outliers,
        limit_preelm_max_stdev=limit_preelm_max_stdev,
        limit_preelm_max_abs=limit_preelm_max_abs,
        bad_elm_indices_csv=bad_elm_indices_csv,
        bad_elm_indices=bad_elm_indices,
        fraction_validation=0.,
        fraction_test=1.,
        max_predict_elms=None,
    )
    
    datamodule.setup(stage='predict')
    dataloaders = datamodule.predict_dataloader()
    n_elms = len(dataloaders)
    i_page = 1
    all_channel_stats = {
        'max_std': [],
        'min_maxabs': [],
        'max_maxabs': [],
        'channels_above_sigma': [],
    }
    if skip_elm_plots:
        merge = False
    for i_elm, dataloader in enumerate(dataloaders):
        dataset: elm_datamodule.ELM_Predict_Dataset = dataloader.dataset
        channel_wise_stats = dataset.pre_elm_stats()
        all_channel_stats['max_std'].append(np.amax(channel_wise_stats['std']))
        all_channel_stats['min_maxabs'].append(np.amin(channel_wise_stats['maxabs']))
        all_channel_stats['max_maxabs'].append(np.amax(channel_wise_stats['maxabs']))
        all_channel_stats['channels_above_sigma'].append(
            np.count_nonzero(channel_wise_stats['maxabs']>datamodule.max_abs_valid_signal)
        )
        elm_index = dataset.elm_index
        shot = dataset.shot
        pre_elm_size = dataset.active_elm_start_index-1
        if skip_elm_plots:
            continue
        if i_elm==0:
            _, axes = plt.subplots(ncols=5, nrows=4, figsize=(14, 8.5))
            axes = axes.flatten()
            n_elms_per_page = axes.size // 2
        i_elm_on_page = i_elm%n_elms_per_page
        if i_elm_on_page == 0:
            plt.suptitle(f"Channel-wise pre-ELM stats (page {i_page})")
            for ax in axes:
                ax.clear()
        plt.sca(axes[i_elm_on_page + 5*(i_elm_on_page//5)])
        for key in channel_wise_stats:
            plt.plot(np.arange(1,65), np.abs(channel_wise_stats[key].flatten()), label=key)
            if key == 'std':
                plt.annotate(
                    f"max std {channel_wise_stats[key].max():.1f}",
                    xy=[0.65, 0.85],
                    xycoords='axes fraction')
                plt.axhline(channel_wise_stats[key].max(), linestyle='--', color='C2', linewidth=0.75)
        plt.axhline(datamodule.max_abs_valid_signal, linestyle='--', color='k', linewidth=0.75)
        plt.title(f"ELM index {elm_index} Shot {shot}", fontsize='medium', color='k')
        plt.xlabel('Channel', fontsize='medium')
        plt.xticks(fontsize='medium')
        plt.yticks(fontsize='medium')
        plt.yscale('log')
        plt.ylim(0.03, 1.4*datamodule.max_abs_valid_signal)
        if i_elm_on_page==0:
            plt.legend(loc='lower right', fontsize='small')
        # plot time-series signals
        plt.sca(axes[i_elm_on_page + 5*(i_elm_on_page//5) + 5])
        max_abs_channel = np.unravel_index(np.argmax(channel_wise_stats['maxabs']), channel_wise_stats['maxabs'].shape)
        max_std_channel = np.unravel_index(np.argmax(channel_wise_stats['std']), channel_wise_stats['std'].shape)
        interval = np.amax([pre_elm_size//1000,1])
        time_axis = (np.arange(-pre_elm_size,0)/1e3)[::interval]
        plt.plot(
            time_axis, 
            dataset.signals[0, 0:pre_elm_size:interval, max_std_channel[0], max_std_channel[1]],
            label=f"Ch. {max_std_channel[1]+1 + 8*max_std_channel[0]}",
            alpha=0.8
        )
        if not np.array_equal(max_abs_channel, max_std_channel):
            plt.plot(
                time_axis, 
                dataset.signals[0, 0:pre_elm_size:interval, max_abs_channel[0], max_abs_channel[1]],
                label=f"Ch. {max_abs_channel[1]+1 + 8*max_abs_channel[0]}",
                alpha=0.8,
            )
        plt.legend(fontsize='small')
        plt.title(f"ELM index {elm_index} Shot {shot}", fontsize='medium',color='k')
        plt.xlabel('Time-to-ELM (ms)', fontsize='medium')
        plt.ylabel('Scaled BES signals', fontsize='medium')
        plt.xticks(fontsize='medium')
        plt.yticks(fontsize='medium')
        plt.axhline(datamodule.max_abs_valid_signal, linestyle='--', color='k', linewidth=0.5)
        plt.axhline(-datamodule.max_abs_valid_signal, linestyle='--', color='k', linewidth=0.5)
        plt.axhline(0, linestyle='--', color='k', linewidth=0.5)
        plt.ylim(np.array([-1.3,1.3])*datamodule.max_abs_valid_signal)
        if i_elm_on_page==n_elms_per_page-1 or i_elm==n_elms-1:
            plt.tight_layout()
            if save:
                filepath = os.path.join(figure_dir, f'elm_stats_{i_page:03d}.pdf')
                print(f"Saving figure {filepath}")
                plt.savefig(filepath, format='pdf', transparent=True)
            i_page += 1
            plt.show(block=block_show)


    _, axes = plt.subplots(nrows=2, ncols=2)
    plt.suptitle('Pre-ELM channel-wise stats (processed signals)')
    axes = axes.flatten()
    for i_axis, key in enumerate(all_channel_stats):
        plt.sca(axes[i_axis])
        plt.hist(all_channel_stats[key], bins=31)
        plt.ylabel('# ELMs')
        plt.yscale('log')
        plt.ylim(bottom=0.8)
        plt.xlabel(key)
    plt.tight_layout()
    if save:
        filepath = os.path.join(figure_dir, f'elm_processed_stats_summary.pdf')
        print(f"Saving figure {filepath}")
        plt.savefig(filepath, format='pdf', transparent=True)
    plt.show(block=block_show)

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

    print(f"Elapsed time {(time.time()-t_start)/60:.1f} m")

    return


if __name__=='__main__':
    plot_stats(
        data_file='/global/homes/d/drsmith/ml/scratch/data/labeled_elm_events.hdf5',
        # max_elms=50,
        bad_elm_indices_csv=False,
        block_show=False,
        mask_sigma_outliers=8,
        limit_preelm_max_stdev=0.25,
        # skip_elm_plots=True,
    )
    