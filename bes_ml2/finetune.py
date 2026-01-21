import sys
import os
import time
from pathlib import Path

import numpy as np

from bes_ml2.train_velo import BES_Trainer
from bes_ml2 import velocimetry_datamodule
from bes_ml2 import elm_lightning_model

import json, numpy as np, torch
from pathlib import Path
from collections.abc import Sequence

def _first_pred_loader(dm):
    loaders = dm.predict_dataloader()
    return loaders[0] if isinstance(loaders, Sequence) else loaders

def _find_model_tensor(batch):
    # Recursively find the first reasonably-shaped float tensor
    def walk(x):
        if torch.is_tensor(x) and x.ndim >= 3:
            return x
        if isinstance(x, dict):
            for v in x.values():
                t = walk(v)
                if t is not None: return t
        if isinstance(x, (list, tuple)):
            for v in x:
                t = walk(v)
                if t is not None: return t
        return None
    return walk(batch)

def save_sample_inputs_from_dm(datamodule, out_base_path: Path, W_model: int | None = None, max_batch_to_save: int = 4):
    loader = _first_pred_loader(datamodule)
    batch = next(iter(loader))  # first batch from predict
    x_batch = _find_model_tensor(batch)
    if x_batch is None:
        raise RuntimeError("Could not locate a tensor input in the first predict batch.")

    # Detach & put on CPU
    x_batch = x_batch.detach().cpu().float()
    x_small = x_batch[:max_batch_to_save]
    x_example = x_batch[:1].clone()

    base = out_base_path  # e.g., /.../epoch=4-step=57120_sample_inputs
    base.parent.mkdir(parents=True, exist_ok=True)

    # 1) Torch: easy for PyTorch users
    torch.save(
        {"x_example": x_example, "x_batch": x_small},
        base.with_suffix(".pt")
    )

    # 2) NumPy: portable for anyone
    np.savez_compressed(
        str(base.with_suffix(".npz")),
        x_example=x_example.numpy(),
        x_batch=x_small.numpy(),
    )

    # 3) Raw batch (exactly what the DataLoader yielded) for auditability
    torch.save({"raw_first_batch": batch}, base.parent / (base.name + "_raw_batch.pt"))

    # 4) Human-friendly metadata
    meta = {
        "example_shape": list(x_example.shape),
        "batch_shape": list(x_small.shape),
        "dtype": "float32",
        "notes": "Shapes are exactly as produced by your predict dataloader. "
                 "Typical is (B, 1, W, R, C).",
    }
    if W_model is not None:
        meta["signal_window_size_W"] = int(W_model)
    (base.with_suffix(".json")).write_text(json.dumps(meta, indent=2))

    print(f"[samples] Saved: {base.with_suffix('.npz')}")
    print(f"[samples] Saved: {base.with_suffix('.pt')}")
    print(f"[samples] Saved: {(base.parent / (base.name + '_raw_batch.pt'))}")
    print(f"[samples] Saved: {base.with_suffix('.json')}")


logger_hash = int(os.getenv('UNIQUE_IDENTIFIER', 0))
world_size = int(os.getenv('SLURM_NTASKS', 1))
world_rank = int(os.getenv('SLURM_PROCID', 0))
local_rank = int(os.getenv('SLURM_LOCALID', 0))
node_rank = int(os.getenv('SLURM_NODEID', 0))
print(f'World rank {world_rank} of {world_size} (local rank {local_rank} on node {node_rank})')

is_global_zero = (world_rank == 0)

# Quiet non-rank0 stdout (keeps logs readable)
if not is_global_zero:
    f = open(os.devnull, 'w')
    sys.stdout = f

def _get_int(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            try:
                return int(getattr(obj, n))
            except Exception:
                pass
    return default

try:
    t_start = time.time()

    # -------------------------
    # 0) Point to pretrained ckpt
    # -------------------------
    pretrain_ckpt = "/pscratch/sd/k/kevinsg/bes_ml_jobs/exp_gill01/47160693/checkpoints/epoch=12-step=581815.ckpt"
    if not pretrain_ckpt:
        raise RuntimeError("PRETRAIN_CKPT env var is not set (path to .ckpt).")

    pretrain_ckpt = str(Path(pretrain_ckpt).expanduser().resolve())
    if not Path(pretrain_ckpt).exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrain_ckpt}")

    # -------------------------
    # 1) Load pretrained model (weights + hparams)
    # -------------------------
    lightning_model = elm_lightning_model.Lightning_Model.load_from_checkpoint(
        checkpoint_path=pretrain_ckpt,
        map_location="cpu",   # Lightning will move to GPU
        strict=True,
    )

    # OPTIONAL: override fine-tune LR without changing architecture
    FT_ENCODER_LR = 1e-3
    FT_DECODER_LR = 1e-3
    if hasattr(lightning_model, "hparams"):
        try:
            lightning_model.hparams.encoder_lr = FT_ENCODER_LR
            lightning_model.hparams.decoder_lr = FT_DECODER_LR
        except Exception:
            pass
    if hasattr(lightning_model, "encoder_lr"): lightning_model.encoder_lr = FT_ENCODER_LR
    if hasattr(lightning_model, "decoder_lr"): lightning_model.decoder_lr = FT_DECODER_LR

    # Pull key shape knobs from checkpoint so you cannot accidentally mismatch
    W_model   = _get_int(lightning_model.hparams, ["signal_window_size"], default=_get_int(lightning_model, ["signal_window_size"], 48))
    n_rows_m  = _get_int(lightning_model.hparams, ["n_rows"], default=_get_int(lightning_model, ["n_rows"], None))
    n_cols_m  = _get_int(lightning_model.hparams, ["n_cols"], default=_get_int(lightning_model, ["n_cols"], None))

    # -------------------------
    # 2) block selection knobs (MUST match what the pretrained model expects)
    # -------------------------
    block_cols  = [1, 3, 5, 7]
    row_stride  = 1
    row_offset  = 4

    R_sel = len(np.arange(8)[row_offset::row_stride])
    C_sel = len(block_cols)

    if (n_rows_m is not None) and (R_sel != n_rows_m):
        raise RuntimeError(f"n_rows mismatch: checkpoint has {n_rows_m}, but your selection gives {R_sel}")
    if (n_cols_m is not None) and (C_sel != n_cols_m):
        raise RuntimeError(f"n_cols mismatch: checkpoint has {n_cols_m}, but your selection gives {C_sel}")
    if W_model != int(W_model):
        raise RuntimeError("Bad signal_window_size from checkpoint?")

    # -------------------------
    # 3) Choose fine-tune + predict shots 
    # -------------------------
    FT_TRAIN_SHOTS = ["205996"]
    FT_VAL_SHOTS = ["205996"]
    FT_TEST_SHOTS = ["205996"]
    PREDICT_SHOTS = ["205982", "205996"]

    good_times_psi_93 = {
        # 145384: [()], # this means use all times
        # 145387: [()],
        # 145388: [()],
        145391: [(1900, 4200)],
        # 145410: [()],
        # 145419: [()],
        # 145420: [()],
        # 145422: [()],
        # 145425: [()],
        # 145427: [()],
        # 157303: [()],
        # 157322: [()],
        # 157323: [()],
        # 157372: [()],
        # 157373: [()],
        # 157374: [()],
        157375: [(1900, 3600), (4000, 5500)],
        # 157376: [()],
        157377: [(1800, 5000), (5400, 6000)],
        # 158076: [()],
        159443: [(1900, 5600)],
        189189: [(2000, 4700)],
        # 189191: [()],
        189199: [(1800, 3000), (4200, 4700)],
        # 200021: [()],
        # 200632: [()],
        # 200634: [()],
        200637: [(1100, 3800)],
        200638: [(800, 3500)],
        200639: [(800, 4600)],
        200643: [(800, 4600)],
        # 203152: [()],
        # 203416: [()],
        203417: [(2250, 2700), (2900, 3300), (3600, 4100)],
        # 203418: [()],
        203419: [(2250, 3500), (3750, 4000)],
        # 203420: [()],
        203423: [(2300, 3850)],
        203469: [(600, 4600)],
        203470: [(500, 3900)],
        203471: [(500, 3800)],
        # 203475: [()],
        203483: [(800, 4300)],
        203484: [(800, 4200)],
        203485: [(800, 2100), (3200, 4500)],
        # 203659: [()],
        203660: [(1100, 2900)],
        # 203662: [()],
        203663: [(1100, 3900)],
        203664: [(1000, 4000)],
        # 203665: [()],
        # 203667: [()],
        # 203671: [()],
        # 203672: [()],
        203946: [(4000, 5400)],
        204286: [(1800, 4600)],
        204287: [(1500, 4000)],
        # 204288: [()],
        204289: [(1100, 4300)],
        204290: [(1100, 4100)],
        204291: [(1100, 3700)],
        # 204292: [()],
        # 204293: [()],
        204294: [(1100, 4100)],
        # 204295: [()],
        # 204296: [()],
        # 204297: [()],
        204299: [(1500, 4500)],
        # 204301: [()],
        204302: [(1800, 4100)],
        204303: [(1600, 4400)],
        # 204837: [()],
        205867: [(2200, 4400)],
    }

    # -------------------------
    # 5) Build datamodule for fine-tune + predict
    # -------------------------
    datamodule = velocimetry_datamodule.Velocimetry_Datamodule(
        data_file="/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/jan15_psi_interp.hdf5",
        signal_window_size=W_model,
        batch_size=1024,
        num_workers=4,
        seed=0,
        world_size=world_size,
        standardize_signals=False,
        split_method="shot",
        train_shots=FT_TRAIN_SHOTS,
        validation_shots=FT_VAL_SHOTS,
        test_shots=FT_TEST_SHOTS,
        predict_shots=PREDICT_SHOTS,
        split_train_data_per_gpu=False,   
        do_flip_augmentation=True,
        shot_time_windows=good_times_psi_93,
        block_cols=block_cols,
        row_stride=row_stride,
        row_offset=row_offset,
        target_sampling_hz=1_000_000.0,
        label_target_psi=0.92,
        label_tolerance_ms=0.6,
        window_hop=1,
        predict_window_stride=48,
        n_rows=R_sel,
        n_cols=C_sel,
    )

    # -------------------------
    # 6) Trainer wrapper (reuse BES_Trainer)
    # -------------------------
    trial_name = f"ft_{logger_hash}"

    trainer = BES_Trainer(
        lightning_model=lightning_model,
        datamodule=datamodule,
        experiment_dir="./exp_gill01",
        trial_name=trial_name,
        wandb_log=True,   
        log_freq=100,
    )

    # Fine-tune settings
    FT_MAX_EPOCHS = 5

    trainer.run_all(
        max_epochs=FT_MAX_EPOCHS,
        early_stopping_min_delta=2e-3,
        early_stopping_patience=50,
        skip_test=False,         # set False if you want test after fine-tune
        skip_predict=False,     # we DO want predictions
        float_precision=32,
    )
    # Derive an output base next to the checkpoint you're already using
    checkpoint = Path(trainer.best_model_path)
    samples_base = checkpoint.parent / (checkpoint.stem + "_sample_inputs")

    # Save a tiny batch and a single-example tensor
    save_sample_inputs_from_dm(datamodule, samples_base, W_model=W_model, max_batch_to_save=4)

    if is_global_zero:
        print(f"Loaded ckpt: {pretrain_ckpt}")
        print(f"Fine-tuned best ckpt: {trainer.best_model_path}")
        print(f"Fine-tuned last ckpt: {trainer.last_model_path}")
        print(f"Python elapsed time {(time.time()-t_start)/60:.1f} min")

except:
    if not is_global_zero:
        f.close()
        sys.stdout = sys.__stdout__
    raise
finally:
    if not is_global_zero:
        f.close()
        sys.stdout = sys.__stdout__
