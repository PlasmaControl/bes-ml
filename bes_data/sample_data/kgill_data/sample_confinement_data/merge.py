import h5py
import os
import numpy as np

from bes_data.sample_data import sample_elm_data_file

with h5py.File("confinement_data.hdf5", "w") as f_dst:
    h5files = [f for f in os.listdir() if f.startswith("bes_signals")]
    print(h5files[0][12:18] + h5files[0][20:-5])
    for i, filename in enumerate(h5files):
        with h5py.File(filename) as file:
            group = f_dst.create_group(h5files[i][12:18] + h5files[i][20:-5])
            group.create_dataset('signals', data = file['signals'][:48,:])
            group.create_dataset('labels', data = np.where(file['labels'])[1])
            # group.create_dataset('time', data = file['time'])

# filename = "confinement_data.hdf5"

# with h5py.File(filename, "r") as f:
#     print("Keys: %s" % f.keys())

#     print(f['172211at1500']['signals'], f['172211at1500']['labels'], f['172211at1500']['time'])
