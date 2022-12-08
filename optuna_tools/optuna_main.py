import sys
import os
from pathlib import Path
import concurrent.futures
import time
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
            time.sleep(1)
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
        # maximize_score: bool = True,  #  True (default) to maximize validation score; False to minimize training loss
        fail_stale_trials: bool = False,  # if True, fail any stale trials
        constant_liar: bool = False,  # if True, add penalty to running trials to avoid redundant sampling
        world_size: int = None,
        world_rank: int = None,
        local_rank: int = None,
        dry_run: bool = False,
        logger_hash: str|int = None,
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
            time.sleep(1)
            if attempts >= 10:
                assert False, "Failed DB connection"

    if world_rank in [None, 0]:
        print(f'Existing studies in storage:')
        for study in optuna.get_all_study_summaries(db_url):
            print(f'  Study {study.study_name} with {study.n_trials} trials')

    if world_rank in [None, 0]:
        print(f"Creating/loading study {study_name}")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize',
    )

    if None not in [world_size, world_rank, local_rank]:
        assert world_size == torch.cuda.device_count()
        assert world_rank < world_size
        assert local_rank <= world_rank

    if fail_stale_trials:
        # FAIL any zombie trials that are stuck in `RUNNING` state
        stale_trials = storage.get_all_trials(
            study._study_id,
            deepcopy=False,
            states=(optuna.trial.TrialState.RUNNING,),
        )
        for stale_trial in stale_trials:
            print(f'Setting trial {stale_trial.number} with state {stale_trial.state} to FAIL')
            status = storage.set_trial_state_values(
                stale_trial._trial_id,
                optuna.trial.TrialState.FAIL,
            )
            print(f'Success?: {status}')

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
        # 'maximize_score': maximize_score,
        'constant_liar': constant_liar,
        'local_rank': local_rank,
        'world_rank': world_rank,
        'world_size': world_size,
        'dry_run': dry_run,
        'logger_hash': logger_hash,
    }
    if local_rank is None:
        if n_gpus is None:
            n_gpus = torch.cuda.device_count()
        n_workers = n_gpus * n_workers_per_device if n_gpus else n_workers_per_device
    else:
        # single-node, multi-GPU training; single worker per node
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
                time.sleep(2)
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
        if world_rank in [None, 0]:
            print(f"Starting worker to run {n_trials_per_worker} trials")
        if local_rank is not None:
            torch.distributed.barrier()
        worker(**worker_kwargs)


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
        # maximize_score: bool = True,
        constant_liar: bool = False,
        local_rank: int = None,
        world_rank: int = None,
        world_size: int = None,
        dry_run: bool = False,
        logger_hash: str|int = None,
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
    )

    def objective_wrapper(trial) -> float:
        if local_rank is not None:
            assert world_rank is not None
            trial = optuna.integration.TorchDistributedTrial(
                trial,
                device=torch.device('cuda', local_rank),
            )
            i_gpu = local_rank
        return objective(
            trial=trial,
            study_dir=study_dir,
            i_gpu=i_gpu if local_rank is None else local_rank,
            trial_generator=trial_generator,
            trainer_class=trainer_class,
            analyzer_class=analyzer_class,
            # maximize_score=maximize_score,
            world_rank=world_rank,
            world_size=world_size,
            local_rank=local_rank,
            dry_run=dry_run,
            logger_hash=logger_hash,
        )

    if local_rank is None or local_rank == 0:
        study = optuna.load_study(
            study_name=study_name,
            storage=db_url,
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(
            objective_wrapper,
            n_trials=n_trials_per_worker,  # trials for this study.optimize() call
            gc_after_trial=True,
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
        # maximize_score: bool = True,
        i_gpu: int | str = 'auto',
        world_size: int = None,
        world_rank: int = None,
        local_rank: int = None,
        dry_run: bool = False,
        logger_hash: int = None,
) -> float:

    study_dir = Path(study_dir)
    assert study_dir.exists()
    trial_dir = study_dir / f'trial_{trial.number:04d}'

    # trial.set_user_attr('maximize_score', maximize_score)

    if world_rank in [None, 0]:
        print(f"Trial {trial.number} starting")

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
                logger_hash=logger_hash,
                **input_kwargs,
            )
            if dry_run is True:
                assert False, "dry_run is True"
            outputs = trainer.train()
        except Exception as e:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            if world_rank in [None, 0]:
                print(f"Trial {trial.number} failed: {repr(e)}")
            objective_value = np.NAN
        else:
            if world_rank in [None, 0]:
                scores = outputs['valid_score'] if 'valid_score' in outputs else outputs['train_score']
                objective_value = np.max(scores)
                if analyzer_class is not None:
                    analyzer = analyzer_class(
                        output_dir=input_kwargs['output_dir'],
                        device=input_kwargs['device'],
                        verbose=False,
                    )
                    analyzer.plot_training(save=True)
                    analyzer.plot_inference(save=True, max_elms=90)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            if world_rank in [None, 0]:
                print(f"Trial {trial.number} finished with final/max score {scores[-1]:.3f}/{objective_value:.3f} and training time {outputs['training_time']/60:.1f} min")

    return objective_value
