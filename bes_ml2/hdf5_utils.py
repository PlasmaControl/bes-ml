from __future__ import annotations
from pathlib import Path

import numpy as np
import h5py


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


if __name__=='__main__':
    delete_elms(
        data_file='/global/homes/d/drsmith/ml/scratch/data/labeled_elm_events.hdf5',
        elm_indices=[77, 217, 219, 221, 223, 239, 247, 262, 315, 319],
    )