from __future__ import annotations

import os
import numpy as np
import h5py

data_folder = '/global/homes/k/kevinsg/m3586/kgill/bes-ml/bes_data/sample_data/kgill_data/sample_confinement_data/'
keys=[]
for idx, file in enumerate(os.listdir(data_folder)):
    print(idx, file)
    keys.append(idx)

indices = np.array([int(key) for key in keys], dtype=int)
rng_generator = np.random.default_rng()
rng_generator.shuffle(indices)

print(indices)

data_files = os.listdir(data_folder)

# s = data_folder + '/' + data_files[9]

for train_idx in indices:
    s = data_folder + '/' + data_files[train_idx]
    with h5py.File(s, "r") as f:
        print(f["signals"])