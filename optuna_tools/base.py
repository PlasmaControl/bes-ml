import sys
import os
from pathlib import Path
import concurrent.futures
import time
from typing import Callable, Union

import optuna

from bes_ml.elm_regression import Trainer


def run_optuna(
        db_name: str,
        n_gpus: int,
        n_workers_per_gpu: int,
        n_trials_per_worker: int,
        sampler_startup_trials: int,
        pruner_startup_trials: int,
        pruner_warmup_epochs: int,
        pruner_minimum_trials_at_epoch: int,
        pruner_patience: int,
        n_epochs: int,
        objective_func: Callable,
):

    db_file = Path(db_name) / f'{db_name}.db'
    db_file.parent.mkdir(parents=True, exist_ok=True)
    db_url = f'sqlite:///{db_file.as_posix()}'
    if db_file.exists():
        print(f'Studies in storage: {db_url}')
        for study in optuna.get_all_study_summaries(db_url):
            print(f'  Study {study.study_name} with {study.n_trials} trials')

    storage = optuna.storages.RDBStorage(url=db_url)

    study = optuna.create_study(
        study_name='study',
        storage=storage,
        load_if_exists=True,
        direction='maximize',
    )

    # FAIL any zombie trials that are stuck in `RUNNING` state
    stale_trials = storage.get_all_trials(
        study._study_id,
        deepcopy=False,
        states=(optuna.trial.TrialState.RUNNING,),
    )

    for stale_trial in stale_trials:
        print(f'Setting trial {stale_trial.number} with state {stale_trial.state} to FAIL')
        status = storage.set_trial_state(
            stale_trial._trial_id,
            optuna.trial.TrialState.FAIL,
        )
        print(f'Success?: {status}')

    # launch workers
    n_workers = n_gpus * n_workers_per_gpu
    if n_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i_worker in range(n_workers):
                i_gpu = i_worker % n_gpus
                print(f'Launching worker {i_worker+1} (of {n_workers}) on gpu {i_gpu} and running {n_trials_per_worker} trials')
                future = executor.submit(
                    subprocess_worker,  # callable that calls study.optimize()
                    db_url,
                    db_file.parent.as_posix(),
                    n_trials_per_worker,
                    i_gpu,
                    sampler_startup_trials,
                    pruner_startup_trials,
                    pruner_warmup_epochs,
                    pruner_minimum_trials_at_epoch,
                    pruner_patience,
                    n_epochs,
                    objective_func,
                )
                futures.append(future)
                time.sleep(5)
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
            db_url=db_url,
            db_dir=db_file.parent.as_posix(),
            n_trials_per_worker=n_trials_per_worker,
            i_gpu=0,
            sampler_startup_trials=sampler_startup_trials,
            pruner_startup_trials=pruner_startup_trials,
            pruner_warmup_epochs=pruner_warmup_epochs,
            pruner_minimum_trials_at_epoch=pruner_minimum_trials_at_epoch,
            pruner_patience=pruner_patience,
            n_epochs=n_epochs,
            objective_func=objective_func,
        )


def subprocess_worker(
    db_url: str,
    db_dir: str,
    n_trials_per_worker: int,
    i_gpu: int,
    sampler_startup_trials: int,
    pruner_startup_trials: int,
    pruner_warmup_epochs: int,
    pruner_minimum_trials_at_epoch: int,
    pruner_patience: int,
    n_epochs: int,
    objective_func: Callable,
):

    # sampler = optuna.samplers.CmaEsSampler(
    #     n_startup_trials=trials_before_sampler,
    #     independent_sampler=optuna.samplers.TPESampler(),
    #     warn_independent_sampling=True,
    #     restart_strategy='ipop',
    #     consider_pruned_trials=False,
    # )
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=sampler_startup_trials,
        constant_liar=True,
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
        study_name='study',
        storage=db_url,
        sampler=sampler,
        pruner=pruner,
    )

    def launch_trial_wrapper(trial):
        return launch_trial(
            trial=trial,
            db_dir=db_dir,
            i_gpu=i_gpu,
            n_epochs=n_epochs,
            objective_func=objective_func,
        )

    # run an optimization process
    study.optimize(
        launch_trial_wrapper,
        n_trials=n_trials_per_worker,  # trials for this study.optimize() call
        gc_after_trial=True,
        # catch=(AssertionError,),  # fail trials with assertion error and continue study
    )


def launch_trial(
        trial: Union[optuna.trial.Trial, optuna.trial.FrozenTrial],
        db_dir: str,
        i_gpu: int,
        n_epochs: int,
        objective_func: Callable,
):

    db_dir = Path(db_dir)
    assert db_dir.exists()
    trial_dir = db_dir / f'trial_{trial.number:04d}'

    with open(os.devnull, 'w') as f:
        sys.stdout = f
        sys.stderr = f

        input_kwargs = objective_func(trial)
        input_kwargs['n_epochs'] = n_epochs
        input_kwargs['output_dir'] = trial_dir.as_posix()
        input_kwargs['device'] = f'cuda:{i_gpu:d}'

        print(f'Trial {trial.number}')
        for key, value in trial.params.items():
            print(f'  Optuna param: {key}, value: {value}')
        for key, value in input_kwargs.items():
            print(f'  Model input: {key}, value: {value}')

        trainer = Trainer(
            trial=trial,
            **input_kwargs,
        )

        outputs = trainer.train()

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    return outputs['scores'][-1]


def study_test(
        trial: Union[optuna.trial.Trial, optuna.trial.FrozenTrial],
) -> dict:
    input_kwargs = {
        'fraction_test': 0.0,
        'fraction_validation': 0.2,
        'log_time': True,
        'inverse_weight_label': True,
    }

    input_kwargs['cnn_layer1_num_kernels'] = 10 * trial.suggest_int('cnn_layer1_num_kernels_factor_10', 1, 8)
    input_kwargs['cnn_layer2_num_kernels'] = 5 * trial.suggest_int('cnn_layer2_num_kernels_factor_5', 1, 8)
    input_kwargs['learning_rate'] = 10 ** trial.suggest_int('lr_exp', -6, -2)

    return input_kwargs


if __name__=='__main__':

    run_optuna(
        db_name=study_test.__name__,
        n_gpus=2,  # <=2 for head node, <=4 for compute node
        n_workers_per_gpu=3,  # max 3 for V100
        n_trials_per_worker=12,
        sampler_startup_trials=60,
        pruner_startup_trials=10,
        pruner_warmup_epochs=6,
        pruner_minimum_trials_at_epoch=20,
        pruner_patience=4,
        n_epochs=16,
        objective_func=study_test,
    )
