from __future__ import annotations
import sys
import os
from pathlib import Path
import concurrent.futures
import time
from datetime import timedelta
from typing import Callable
import multiprocessing as mp
import warnings

import numpy as np
import torch
import torch.distributed
import optuna

warnings.filterwarnings(action='ignore', module='optuna')


def fail_stale_trials(
        db_name: str,
):
    db_file = Path(db_name) / f'{db_name}.db'
    db_file.parent.mkdir(parents=True, exist_ok=True)
    assert db_file.exists()
    db_url = f'sqlite:///{db_file.as_posix()}'

    # connect to optuna database
    success = False
    attempts = 0
    while success is False:
        try:
            storage = optuna.storages.RDBStorage(
                url=db_url,
            )
            success = True
        except:
            attempts += 1
            time.sleep(2)
            if attempts >= 10:
                assert False, "Failed DB connection"

    # create/load optuna study
    study = optuna.load_study(
        study_name='study',
        storage=storage,
    )

    stale_trials = storage.get_all_trials(
        study_id=study._study_id,
        deepcopy=False,
        states=(optuna.trial.TrialState.RUNNING,),
    )

    for stale_trial in stale_trials:
        print(f'Setting trial {stale_trial.number} with state {stale_trial.state} to FAIL')
        status = storage.set_trial_state_values(
            trial_id=stale_trial._trial_id,
            state=optuna.trial.TrialState.FAIL,
        )
        print(f'Success?: {status}')


def run_optuna(
        trial_generator: Callable,
        trainer_class: Callable,
        db_name: str = None,
        db_url: str = None,
        study_dir: str|Path = None,
        study_name: str = None,
        n_gpus: int = None,
        n_workers_per_device: int = 1,
        n_trials_per_worker: int = 1000,
        analyzer_class: Callable = None,
        sampler_startup_trials: int = 100000,  # random startup trials before activating sampler
        pruner_startup_trials: int = 100000,  # startup trials before pruning
        pruner_warmup_epochs: int = 10,  # initial epochs before pruning
        pruner_minimum_trials_at_epoch: int = 20,  # minimum trials at each epoch before pruning
        pruner_patience: int = 10,  # epochs to wait for improvement before pruning
        pruner_min_delta: float = 1e-3,
        constant_liar: bool = False,  # if True, add penalty to running trials to avoid redundant sampling
        world_size: int = 1,
        world_rank: int = 0,
        local_rank: int = 0,
        dry_run: bool = False,
        logger_hash: str|int = None,
        stop_after_success: bool = False,
) -> None:

    assert db_name or (db_url and study_name)

    if db_name:
        if study_dir is None:
            study_dir = Path(db_name).resolve()
            study_name = db_name
        else:
            study_dir = Path(study_dir).resolve()
            if not study_name:
                study_name = 'study'
        db_file = study_dir / f'{db_name}.db'
        db_url = f'sqlite:///{db_file.as_posix()}'
    else:
        study_dir = Path(study_name)

    assert db_url and study_name and study_dir

    study_dir.mkdir(exist_ok=True)

    assert world_rank < world_size
    assert local_rank <= world_rank

    print(f"world_size/world_rank/local_rank {world_size}/{world_rank}/{local_rank}")

    if world_rank == 0:
        # connect to optuna database
        success = False
        attempts = 0
        while success is False:
            try:
                storage = optuna.storages.RDBStorage(url=db_url, skip_table_creation=True, skip_compatibility_check=True)
                success = True
            except:
                attempts += 1
                time.sleep(2)
                if attempts >= 40:
                    assert False, f"Failed DB connection with {attempts} attempts on world rank {world_rank}"

        # print(f"Creating/loading study {study_name}")
        optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction='maximize',
        )

    # if world_size > 1:
    #     torch.distributed.barrier()
        # async_handle = torch.distributed.barrier(async_op=True)
        # async_handle.wait(timeout=timedelta(seconds=30))

    # if fail_stale_trials:
    #     # FAIL any zombie trials that are stuck in `RUNNING` state
    #     stale_trials = storage.get_all_trials(
    #         study._study_id,
    #         deepcopy=False,
    #         states=(optuna.trial.TrialState.RUNNING,),
    #     )
    #     for stale_trial in stale_trials:
    #         print(f'Setting trial {stale_trial.number} with state {stale_trial.state} to FAIL')
    #         status = storage.set_trial_state_values(
    #             stale_trial._trial_id,
    #             optuna.trial.TrialState.FAIL,
    #         )
    #         print(f'Success?: {status}')

    # launch workers
    worker_kwargs = {
        'db_url': db_url,
        'study_dir': study_dir.as_posix() if study_dir else None,
        'study_name': study_name,
        'n_trials_per_worker': n_trials_per_worker,
        'trial_generator': trial_generator,
        'trainer_class': trainer_class,
        'analyzer_class': analyzer_class,
        'sampler_startup_trials': sampler_startup_trials,
        'pruner_startup_trials': pruner_startup_trials,
        'pruner_warmup_epochs': pruner_warmup_epochs,
        'pruner_minimum_trials_at_epoch': pruner_minimum_trials_at_epoch,
        'pruner_patience': pruner_patience,
        'pruner_min_delta': pruner_min_delta,
        'constant_liar': constant_liar,
        'local_rank': local_rank,
        'world_rank': world_rank,
        'world_size': world_size,
        'dry_run': dry_run,
        'logger_hash': logger_hash,
        'stop_after_success': stop_after_success,
    }
    if world_size == 1:
        if n_gpus is None:
            n_gpus = torch.cuda.device_count()
        n_workers = n_gpus * n_workers_per_device if n_gpus else n_workers_per_device
    else:
        n_workers = 1
    if n_workers > 1:
        mp_context = mp.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_context,
        ) as executor:
            futures = []
            for i_worker in range(n_workers):
                i_gpu = i_worker % n_gpus if n_gpus else 'cpu'
                print(f'Launching worker {i_worker+1} '
                      f'(of {n_workers}) on gpu/device {i_gpu} '
                      f'to run {n_trials_per_worker} trials')
                future = executor.submit(
                    worker,  # callable that calls study.optimize()
                    i_gpu=i_gpu,
                    **worker_kwargs,
                )
                futures.append(future)
                # time.sleep(2)
            concurrent.futures.wait(futures)
            for i_future, future in enumerate(futures):
                if future.exception() is None:
                    print(f'Future {i_future} returned {future.result()}')
                else:
                    print(f'Future {i_future} exception:')
                    e = future.exception()
                    print(e)
                    print(e.args)
    else:
        worker(**worker_kwargs)


def study_callback(
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
) -> None:
    # stop study process after single successful trial
    if trial.state != optuna.trial.TrialState.FAIL:
        study.stop()


def worker(
        db_url: str,
        trial_generator: Callable,
        trainer_class: Callable,
        study_dir: str,
        study_name: str,
        i_gpu: int|str = 'auto',
        n_trials_per_worker: int = 1000,
        analyzer_class: Callable = None,
        sampler_startup_trials: int = 1000,
        pruner_startup_trials: int = 1000,
        pruner_warmup_epochs: int = 10,
        pruner_minimum_trials_at_epoch: int = 20,
        pruner_patience: int = 10,
        pruner_min_delta: float = 1e-3,
        constant_liar: bool = False,
        local_rank: int = 0,
        world_rank: int = 0,
        world_size: int = 1,
        dry_run: bool = False,
        logger_hash: str|int = None,
        stop_after_success = False,
) -> None:

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=sampler_startup_trials,  # trials with random sampling before enabling sampler
        constant_liar=constant_liar,
    )
    sampler.reseed_rng()

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=pruner_startup_trials,  # trials before enabling pruner
        n_warmup_steps=pruner_warmup_epochs,  # epochs before enabling pruner
        n_min_trials=pruner_minimum_trials_at_epoch,  # min. trials at each epoch to enable pruner
    )
    pruner = optuna.pruners.PatientPruner(
        pruner,
        patience=pruner_patience,
        min_delta=pruner_min_delta,
    )

    def objective_wrapper(trial) -> float:
        if world_size > 1:
            torch.distributed.barrier()
            trial = optuna.integration.TorchDistributedTrial(trial)
            i_gpu = local_rank
        value = objective(
            trial=trial,
            study_dir=study_dir,
            i_gpu=i_gpu,
            trial_generator=trial_generator,
            trainer_class=trainer_class,
            analyzer_class=analyzer_class,
            world_rank=world_rank,
            world_size=world_size,
            local_rank=local_rank,
            dry_run=dry_run,
            logger_hash=logger_hash,
        )
        return value

    if world_rank == 0:
        attempts = 0
        success = False
        while success is False:
            try:
                study = optuna.load_study(
                    study_name=study_name,
                    storage=db_url,
                    sampler=sampler,
                    pruner=pruner,
                )
                success = True
            except:
                attempts += 1
                time.sleep(1)
                if attempts >= 15:
                    assert False, f"Worker: Failed load_study()"
        study.optimize(
            objective_wrapper,
            n_trials=n_trials_per_worker,  # trials for this study.optimize() call
            gc_after_trial=True,
            callbacks=[study_callback] if stop_after_success else None,
        )
    else:
        for _ in range(n_trials_per_worker):
            try:
                objective_wrapper(None)
            except:
                pass

def objective(
        trial: optuna.trial.Trial|optuna.integration.TorchDistributedTrial,
        study_dir: str,
        trial_generator: Callable,
        trainer_class: Callable,
        analyzer_class: Callable = None,
        i_gpu: int | str = 'auto',
        world_size: int = 1,
        world_rank: int = 0,
        local_rank: int = 0,
        dry_run: bool = False,
        logger_hash: int = None,
) -> float:

    study_dir = Path(study_dir)
    assert study_dir.exists()
    trial_dir = study_dir / f'trial_{trial.number:05d}'

    if world_size > 1:
        torch.distributed.barrier()

    with open(os.devnull, 'w') as f:
        sys.stdout = f
        sys.stderr = f
        input_kwargs = trial_generator(trial)
        input_kwargs['output_dir'] = trial_dir.as_posix()
        input_kwargs['device'] = f'cuda:{i_gpu:d}' if isinstance(i_gpu, int) else i_gpu
        try:
            trainer = trainer_class(
                optuna_trial=trial,
                world_size=world_size,
                world_rank=world_rank,
                local_rank=local_rank,
                logger_hash=f"{logger_hash}_{trial.number}",
                **input_kwargs,
            )
            if dry_run is True:
                assert False, "dry_run is True"
            outputs = trainer.train()
        except Exception as e:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            objective_value = np.NAN
            if world_rank == 0:
                print(f"Trial {trial.number}: failed with error {repr(e)}")
        else:
            scores = outputs['valid_score'] if 'valid_score' in outputs else outputs['train_score']
            objective_value = np.max(scores)
            if world_rank == 0:
                if analyzer_class is not None:
                    analyzer = analyzer_class(
                        output_dir=input_kwargs['output_dir'],
                        device=input_kwargs['device'],
                        verbose=False,
                    )
                    analyzer.plot_training(save=True)
                    analyzer.plot_inference(save=True, max_elms=30)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            
    return objective_value
