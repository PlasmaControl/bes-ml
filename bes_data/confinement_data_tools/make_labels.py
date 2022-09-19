import logging
import re
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd

import argparse


def make_parser():
    parser = argparse.ArgumentParser(description='Make labels')

    parser.add_argument('--data_dir',
                        type=str,
                        default=str(Path(__file__).parents[1] / 'sample_data'),
                        help="Home directory of project. Should contain 'confinement_database.xlsx' and 'turbulence_data/'"
                        )
    parser.add_argument('--df_name',
                        type=str,
                        default='confinement_database.xlsx',
                        help='Name of the confinement regime database file'
                        )
    parser.add_argument('--shot_dir',
                        type=str,
                        default='confinement_data',
                        help='Path to directory containing datasets (rel. to base_dir or specify whole path.)'
                        )
    parser.add_argument('--labeled_dir',
                        type=str,
                        default='labeled_datasets',
                        help='Path to labeled datasets (rel. to data_dir or specify whole path.)'
                        )

    return parser
def make_labels(data_dir: str = None,
                df_name: str = 'confinement_database.xlsx',
                shot_dir: Union[str,Path] = 'confinement_data',
                labeled_dir: Union[str,Path] = 'labeled_datasets',
                logger: logging.Logger = None):
    """
    Function to create labeled datasets for turbulence regime classification.
    Shot data is sourced from base_dir / turbulence_data.
    Resulting labeled datasets are stored as HDF5 files in base_dir / data / labeled_datasets.
    :param data_dir: Home directory of project. Should contain 'confinement_database.xlsx' and 'turbulence_data/'
    :param df_name: Name of the confinement regime database file.
    :param shot_dir: Path to datasets (rel. to base_dir or specify whole path.)
    :param labeled_dir: Path to labeled datasets (rel. to data_dir or specify whole path.)
    :param logger: Logger
    :return: None
    """

    # Pathify all directories
    if data_dir:
        data_dir = Path(data_dir).resolve()
    else:
        data_dir = (Path(__file__).parents[1] / 'sample_data/').resolve()

    if Path(shot_dir).exists():
        shot_dir = Path(shot_dir)
    else:
        shot_dir = Path(data_dir) / shot_dir

    if Path(labeled_dir).exists():
        labeled_dir = Path(labeled_dir)
    else:
        labeled_dir = Path(shot_dir) / labeled_dir

    if logger is None:
        logger = logging.getLogger('make_labels')

    labeled_dir.mkdir(exist_ok=True)
    labeled_shots = {}
    for file in (labeled_dir.iterdir()):
        try:
            shot_num = re.findall(r'_(\d+).+.hdf5', str(file))[0]
        except IndexError:
            continue
        if shot_num not in labeled_shots.keys():
            labeled_shots[shot_num] = file

    # Find unlabeled datasets (shots not in base_dir/data/labeled_datasets)
    shots = {}
    for file in (shot_dir.iterdir()):
        try:
            shot_num = re.findall(r'_(\d+).hdf5', str(file))[0]
        except IndexError:
            continue
        if shot_num not in labeled_shots.keys():
            shots[shot_num] = file

    if len(shots) == 0:
        logger.info('No new labels to make')
        return
    logger.info(f'Making labels for shots {[sn for sn in shots.keys()]}')

    # Read labeled df.
    label_df = pd.read_excel(data_dir / df_name).fillna(0)
    for shot_num, file in shots.items():
        shot_df = label_df.loc[label_df['shot'] == float(shot_num)]
        if len(shot_df) == 0:
            print(f'{shot_num} not in confinement database.')
            continue
        else:
            print(f'Processing shot {shot_num}')

        with h5py.File(file, 'a') as shot_data:
            try:
                labels = np.array(shot_data['labels']).tolist()
            except KeyError:
                time = np.array(shot_data['time'])
                signals = np.array(shot_data['signals'])
                labels = np.zeros_like(time)

                for i, row in shot_df.iterrows():
                    tstart = row['tstart (ms)']
                    tstop = row['tstop (ms)']
                    label = row[[col for col in row.index if 'mode' in col]].values.argmax() + 1
                    labels[np.nonzero((time > tstart) & (time < tstop))] = label

                signals = signals[:, np.nonzero(labels)[0]]
                time = time[np.nonzero(labels)[0]]
                labels = labels[np.nonzero(labels)[0]] - 1

        sname = f'bes_signals_{shot_num}_labeled.hdf5'
        with h5py.File(labeled_dir / sname, 'w') as sd:
            sd.create_dataset('labels', data=labels)
            sd.create_dataset('signals', data=signals)
            sd.create_dataset('time', data=time)

    return

if __name__ == '__main__':

    parser = make_parser()
    args = parser.parse_args()

    make_labels(data_dir=args.data_dir,
                df_name=args.df_name,
                shot_dir=args.shot_dir,
                labeled_dir=args.labeled_dir)
