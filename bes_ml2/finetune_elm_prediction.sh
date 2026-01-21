#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=gpu
#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4

#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --qos=premium
###SBATCH --array=0

# -------------------------
# USER KNOBS 
# -------------------------
PRETRAIN_CKPT="/pscratch/sd/k/kevinsg/bes_ml_jobs/exp_gill01/47858859/checkpoints/epoch=7-step=118952.ckpt"
FT_MAX_EPOCHS=80
FT_ENCODER_LR="1e-3"
FT_DECODER_LR="1e-3"
# -------------------------

echo Python executable: $(which python)
echo
echo Job name: $SLURM_JOB_NAME
echo QOS: $SLURM_JOB_QOS
echo Account: $SLURM_JOB_ACCOUNT
echo Submit dir: $SLURM_SUBMIT_DIR
echo
echo Job array ID: $SLURM_ARRAY_JOB_ID
echo Job ID: $SLURM_JOBID
echo Job array task: $SLURM_ARRAY_TASK_ID
echo Job array task count: $SLURM_ARRAY_TASK_COUNT
echo
echo Nodes: $SLURM_NNODES
echo Head node: $SLURMD_NODENAME
echo hostname $(hostname)
echo Nodelist: $SLURM_NODELIST
echo Tasks per node: $SLURM_NTASKS_PER_NODE
echo GPUs per node: $SLURM_GPUS_PER_NODE

if [[ -n $SLURM_ARRAY_JOB_ID ]]; then
    export UNIQUE_IDENTIFIER=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
else
    export UNIQUE_IDENTIFIER=$SLURM_JOBID
fi
echo UNIQUE_IDENTIFIER: $UNIQUE_IDENTIFIER

JOB_DIR=/pscratch/sd/k/kevinsg/bes_ml_jobs/
mkdir --parents $JOB_DIR || exit
cd $JOB_DIR || exit
echo Job directory: $PWD

export WANDB__SERVICE_WAIT=300

PYTHON_SCRIPT=$(cat << END

import sys
import os
import time
from pathlib import Path

import numpy as np

from bes_ml2.train_velo import BES_Trainer
from bes_ml2 import velocimetry_datamodule
from bes_ml2 import elm_prediction_datamodule
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
    pretrain_ckpt = os.getenv("PRETRAIN_CKPT", "").strip()
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
        map_location="cpu",
        strict=True,
        finetune_last_linear_only=True,
        velocimetry_output_index=1,
        velocimetry_log_label="pELM",
    )

    lightning_model.weight_decay = 0.0
    if hasattr(lightning_model, "hparams"):
        lightning_model.hparams.weight_decay = 0.0

    mlp = lightning_model.frontends["velocimetry_mlp"]
    last_linear = None
    for m in mlp.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m

    b = last_linear.bias.detach().cpu().numpy()
    w_norms = last_linear.weight.detach().cpu().norm(dim=1).numpy()
    print("last bias:", b)
    print("row weight norms:", w_norms)

    trainable = [(n, tuple(p.shape)) for n, p in lightning_model.named_parameters() if p.requires_grad]
    print("Trainable params:")
    for n, sh in trainable:
        print(" ", n, sh)

    # OPTIONAL: override fine-tune LR without changing architecture
    FT_ENCODER_LR = float(os.getenv("FT_ENCODER_LR", "1e-4"))
    FT_DECODER_LR = float(os.getenv("FT_DECODER_LR", "1e-4"))
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
    # 5) Build datamodule for fine-tune + predict
    # -------------------------
    datamodule = elm_prediction_datamodule.ELM_Prediction_Datamodule(
            data_file="/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/step_6_labeled_elm_events.hdf5",
            signal_window_size=48,
            target_sampling_hz=1_000_000.0,
            block_cols=block_cols,
            row_stride=row_stride,
            row_offset=row_offset,
            split_method="event",
            fraction_validation=0.15,
            fraction_test=0.15,
            split_train_data_per_gpu=True,
            standardize_signals=False,
            world_size=world_size,
            num_workers=4,
            batch_size=1024,
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
    FT_MAX_EPOCHS = int(os.getenv("FT_MAX_EPOCHS", "10"))

    trainer.run_all(
        max_epochs=FT_MAX_EPOCHS,
        early_stopping_min_delta=2e-3,
        early_stopping_patience=50,
        skip_test=False,         # set False if you want test after fine-tune
        skip_predict=False,     # we DO want predictions
        float_precision=32,
    )

    mlp = lightning_model.frontends["velocimetry_mlp"]
    last_linear = None
    for m in mlp.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m

    b = last_linear.bias.detach().cpu().numpy()
    w_norms = last_linear.weight.detach().cpu().norm(dim=1).numpy()
    print("last bias:", b)
    print("row weight norms:", w_norms)


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

END
)

echo Script:
echo "${PYTHON_SCRIPT}"

START_TIME=$(date +%s)
srun --export=ALL,\
PRETRAIN_CKPT="$PRETRAIN_CKPT",\
FT_MAX_EPOCHS="$FT_MAX_EPOCHS",\
FT_ENCODER_LR="$FT_ENCODER_LR",\
FT_DECODER_LR="$FT_DECODER_LR" \
python -c "${PYTHON_SCRIPT}"
EXIT_CODE=$?
END_TIME=$(date +%s)
echo Slurm elapsed time $(( (END_TIME - START_TIME)/60 )) min $(( (END_TIME - START_TIME)%60 )) s

exit $EXIT_CODE
