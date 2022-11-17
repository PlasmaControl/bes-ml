import sys
import os
from pathlib import Path
import concurrent.futures
import time
from typing import Callable
import multiprocessing as mp

import numpy as np
import torch.cuda
import optuna


def fail_stale_trials(
        db_name: str,
):
    db_file = Path(db_name) / f'{db_name}.db'
    db_file.parent.mkdir(parents=True, exist_ok=True)
    assert db_file.exists()
    db_url = f'sqlite:///{db_file.as_posix()}'

    storage = optuna.storages.RDBStorage(
        url=db_url,
    )

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
        n_epochs: int,
        objective_func: Callable,
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
        maximize_score: bool = True,  #  True (default) to maximize validation score; False to minimize training loss
        fail_stale_trials: bool = False,  # if True, fail any stale trials
        constant_liar: bool = False,  # if True, add penalty to running trials to avoid redundant sampling
) -> None:

    assert db_name or (db_url and study_name)

    if db_name:
        if study_dir is None:
            study_dir = Path(db_name)
            study_name = db_name
        else:
            study_dir = Path(study_dir)
            if not study_name:
                study_name = 'study'
        db_file = study_dir / f'{db_name}.db'
        db_url = f'sqlite:///{db_file.as_posix()}'
    else:
        study_dir = Path(study_name)

    assert db_url and study_name and study_dir

    study_dir.mkdir(exist_ok=True)

    storage = optuna.storages.RDBStorage(url=db_url)
    print(f'Existing studies in storage:')
    for study in optuna.get_all_study_summaries(db_url):
        print(f'  Study {study.study_name} with {study.n_trials} trials')

    print(f"Creating/loading study {study_name}")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize' if maximize_score else 'minimize',
    )

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
    subprocess_kwargs = {
        'db_url': db_url,
        'study_dir': study_dir.as_posix() if study_dir else None,
        'study_name': study_name,
        'n_trials_per_worker': n_trials_per_worker,
        'n_epochs': n_epochs,
        'objective_func': objective_func,
        'trainer_class': trainer_class,
        'analyzer_class': analyzer_class,
        'sampler_startup_trials': sampler_startup_trials,
        'pruner_startup_trials': pruner_startup_trials,
        'pruner_warmup_epochs': pruner_warmup_epochs,
        'pruner_minimum_trials_at_epoch': pruner_minimum_trials_at_epoch,
        'pruner_patience': pruner_patience,
        'maximize_score': maximize_score,
        'constant_liar': constant_liar,
    }
    if n_gpus is None:
        n_gpus = torch.cuda.device_count()
    n_workers = n_gpus * n_workers_per_device if n_gpus else n_workers_per_device
    if n_workers > 1:
        mp_context = mp.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp_context,
        ) as executor:
            futures = []
            for i_worker in range(n_workers):
                i_gpu = i_worker % n_gpus if n_gpus else 'auto'
                print(f'Launching worker {i_worker+1} '
                      f'(of {n_workers}) on gpu/device {i_gpu} '
                      f'and running {n_trials_per_worker} trials')
                future = executor.submit(
                    subprocess_worker,  # callable that calls study.optimize()
                    i_gpu=i_gpu,
                    **subprocess_kwargs,
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
        print("Starting trial")
        subprocess_worker(
            i_gpu=0 if n_gpus else 'auto',
            **subprocess_kwargs,
        )


def subprocess_worker(
    db_url: str,
    n_epochs: int,
    objective_func: Callable,
    trainer_class: Callable,
    study_dir: str,
    study_name: str,
    i_gpu: int | str = 'auto',
    n_trials_per_worker: int = 1000,
    analyzer_class: Callable = None,
    sampler_startup_trials: int = 1000,
    pruner_startup_trials: int = 1000,
    pruner_warmup_epochs: int = 10,
    pruner_minimum_trials_at_epoch: int = 20,
    pruner_patience: int = 10,
    maximize_score: bool = True,
    constant_liar: bool = False,
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

    # load study
    study = optuna.load_study(
        study_name=study_name,
        storage=db_url,
        sampler=sampler,
        pruner=pruner,
    )

    def launch_trial_wrapper(trial) -> float:
        return launch_trial(
            trial=trial,
            study_dir=study_dir,
            i_gpu=i_gpu,
            n_epochs=n_epochs,
            objective_func=objective_func,
            trainer_class=trainer_class,
            analyzer_class=analyzer_class,
            maximize_score=maximize_score,
        )

    # run an optimization process
    study.optimize(
        launch_trial_wrapper,
        n_trials=n_trials_per_worker,  # trials for this study.optimize() call
        gc_after_trial=True,
    )


def launch_trial(
        trial: optuna.trial.Trial | optuna.trial.FrozenTrial,
        n_epochs: int,
        study_dir: str,
        objective_func: Callable,
        trainer_class: Callable,
        analyzer_class: Callable = None,
        maximize_score: bool = True,
        i_gpu: int | str = 'auto',
) -> float:

    study_dir = Path(study_dir)
    assert study_dir.exists()
    trial_dir = study_dir / f'trial_{trial.number:04d}'

    trial.set_user_attr('maximize_score', maximize_score)

    with open(os.devnull, 'w') as f:
        sys.stdout = f
        sys.stderr = f

        input_kwargs = objective_func(trial)
        input_kwargs['n_epochs'] = n_epochs
        input_kwargs['output_dir'] = trial_dir.as_posix()
        device = f'cuda:{i_gpu:d}' if isinstance(i_gpu, int) else i_gpu
        input_kwargs['device'] = device

        try:
            trainer = trainer_class(
                optuna_trial=trial,
                **input_kwargs,
            )
            outputs = trainer.train()
        except Exception as e:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"Trial {trial.number} failed: {repr(e)}")
            result = np.NAN
        else:
            result = outputs['valid_score'][-1] if maximize_score else outputs['train_loss'][-1]
            if analyzer_class is not None:
                analysis = analyzer_class(
                    output_dir=input_kwargs['output_dir'],
                    device=device,
                    verbose=False,
                )
                analysis.plot_training(save=True)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    return result


# def study_example(
#         trial: Union[optuna.trial.Trial, optuna.trial.FrozenTrial],
# ) -> dict:
#     input_kwargs = {
#         'fraction_test': 0.0,
#         'fraction_validation': 0.0,
#         'log_time': False,
#         'inverse_weight_label': False,
#         'learning_rate': 10 ** trial.suggest_int('lr_exp', -6, -2),
#         'cnn_layer1_num_kernels': 10 * trial.suggest_int('cnn_layer1_num_kernels_factor_10', 1, 8),
#         'cnn_layer2_num_kernels': 5 * trial.suggest_int('cnn_layer2_num_kernels_factor_5', 1, 8),
#     }
#     return input_kwargs
#
#
# if __name__ == '__main__':
#
#     run_optuna(
#         db_name=study_example.__name__,
#         n_gpus=2,  # <=2 for head node, <=4 for compute node, ==1 if run_on_cpu
#         n_workers_per_gpu=2,  # max 3 for V100
#         n_trials_per_worker=5,
#         n_epochs=8,
#         objective_func=study_example,
#         trainer_class=elm_regression.Trainer,
#         sampler_startup_trials=60,
#         pruner_startup_trials=10,
#         pruner_warmup_epochs=6,
#         pruner_minimum_trials_at_epoch=20,
#         pruner_patience=4,
#         maximize_score=False,
#         constant_liar=True,
#     )
