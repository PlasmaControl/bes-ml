from pathlib import Path
import shutil
from typing import Union, Sequence
import subprocess
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

import optuna

# from bes_ml.elm_regression.analyze import Analyzer


NON_RUNNING_STATES = (
    optuna.trial.TrialState.COMPLETE,
    optuna.trial.TrialState.PRUNED,
    optuna.trial.TrialState.FAIL,
)


def open_study(
        study_dir: Union[Path, str] = None,
        study_name: str = None,
        db_url: str = None,
) -> optuna.study.Study:
    study_dir = Path(study_dir).resolve()
    assert study_dir.exists() and study_dir.is_dir()
    print(f"Opening study {study_dir}")

    if study_name is None:
        study_name = study_dir.name

    if db_url is None:
        db_file = study_dir / f"{study_dir.name}.db"
        assert db_file.exists()
        print(f"Opening database {db_file}")
        db_url = f'sqlite:///{db_file.as_posix()}'

    assert study_name and db_url

    study = optuna.load_study(study_name=study_name, storage=db_url)

    return study


def merge_pdfs(
    inputs: Sequence,
    output: Union[str,Path],
    delete_inputs: bool = False,
):
    inputs = [Path(input) for input in inputs]
    output = Path(output)
    output.unlink(missing_ok=True)
    gs_cmd = shutil.which('gs')
    if gs_cmd is None:
        return
    print(f"Merging PDFs into file: {output.as_posix()}")
    cmd = [
        gs_cmd,
        '-q',
        '-dBATCH',
        '-dNOPAUSE',
        '-sDEVICE=pdfwrite',
        '-dPDFSETTINGS=/prepress',
        '-dCompatibilityLevel=1.4',
        f"-sOutputFile={output.as_posix()}",
    ]
    for pdf_file in inputs:
        cmd.append(f"{pdf_file.as_posix()}")
    result = subprocess.run(cmd, check=True)
    assert result.returncode == 0 and output.exists()
    print("Merge finished")
    if delete_inputs is True:
        for pdf_file in inputs:
            pdf_file.unlink()


def plot_study(
        study_dir: Union[Path, str] = None,  # study dir. or db file
        study_name: str = None,
        db_url: str = None,
        save: bool = False,
        metric: str = 'valid_score',
        use_last: bool = True,  # use last metric, not extremum
        quantile: float = None,
        plot_trials: int = 4,
        analyzer: Callable = None,
        max_elms: int = None,
):
    study_dir = Path(study_dir).resolve()
    study = open_study(
        study_dir=study_dir,
        study_name=study_name,
        db_url=db_url,
    )

    completed_trials = study.get_trials(
        states=(
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        ),
    )
    print(f'Completed trials: {len(completed_trials)}')

    trials = []
    for trial in completed_trials:
        trial_dir = study_dir / f"trial_{trial.number:04d}"
        assert trial_dir.exists()
        results_file = trial_dir / 'results.yaml'
        assert results_file.exists()
        with results_file.open() as results_file:
            trial_results = yaml.safe_load(results_file)
        trial_results['number'] = trial.number
        trial_results['trial'] = trial
        if use_last:
            trial_results['value'] = trial_results[metric][-1]
        else:
            trial_results['value'] = np.max(trial_results[metric]) if 'score' in metric else np.min(trial_results[metric])
        trials.append(trial_results.copy())

    # if use_last:
    #     sort_key = lambda results: results[metric][-1]
    # else:
    #     if 'score' in metric:
    #         sort_key = lambda results: np.max(results[metric])
    #     elif 'loss' in metric:
    #         sort_key = lambda results: np.min(results[metric])
    #     else:
    #         raise KeyError

    # sort trials
    trials.sort(
        key=lambda trial: trial['value'], 
        reverse=True if 'score' in metric else False,
    )
    values = np.array([trial['value'] for trial in trials])

    if quantile is not None:
        if 'loss' in metric:
            quantile = 1-quantile
        quantile = np.quantile(values, quantile)
        if 'score' in metric:
            trials = [trial for trial, value in zip(trials, values) if value >= quantile]
        else:
            trials = [trial for trial, value in zip(trials, values) if value <= quantile]
        values = np.array([trial['value'] for trial in trials])

    print("Top trials")
    for i_trial, trial in enumerate(trials[0:10]):
        print(f"  Trial {trial['number']}  {metric} {values[i_trial]:.4g}")

    params = tuple(trials[0]['trial'].params.keys())
    n_params = len(params)
    ncols = 4
    nrows = n_params // ncols if n_params % ncols == 0 else (n_params // ncols) + 1
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3.125, nrows * 2))
    for i_param, param in enumerate(params):
        param_values = np.array([
            (
                trial['trial'].params[param] 
                if param in trial['trial'].distributions else 0
            ) 
            for trial in trials]
        )
        plt.sca(axes.flat[i_param])
        sns.violinplot(
            x=param_values,
            y=values,
            inner='stick',
            scale='count',
            bw=0.2,
        )
        plt.xlabel(param)
    plt.suptitle(f"{study_dir.as_posix()} | {len(completed_trials)} trials | Best {metric} {values[0]:.3f}")
    plt.tight_layout()

    if save:
        filepath = study_dir.parent / (study_dir.stem + '.pdf')
        print(f'Saving file: {filepath.as_posix()}')
        plt.savefig(filepath, transparent=True)

    if plot_trials and analyzer:
        for trial in trials[0:plot_trials]:
            plot_trial(
                analyzer=analyzer,
                study_dir=study_dir,
                trial_number=trial['number'],
                max_elms=max_elms,
                save=True,
            )


def plot_trial(
        analyzer: Callable,
        trial_dir: Path|str = None,
        study_dir: Path|str = None,
        trial_number: int = None,
        device: str = 'auto',
        max_elms: int = None,
        save: bool = False,
) -> None:
    if trial_dir:
        trial_dir = Path(trial_dir)
    elif study_dir is not None and trial_number is not None:
        study_dir = Path(study_dir)
        trial_dir = study_dir / f"trial_{trial_number:04d}"
    assert trial_dir.exists()
    trial_result = analyzer(trial_dir, device=device)
    trial_result.plot_training(save=save)
    if max_elms:
        trial_result.plot_inference(max_elms=max_elms, save=save)


def plot_top_trials(
        study_dir: Union[Path, str] = None,  # study dir. or db file
        study_name: str = None,
        db_url: str = None,
        n_trials = 1,
        device = None,
        max_elms = None,
        use_train_loss = False,
        analyzer: Callable = None,
):
    study_dir = Path(study_dir).resolve()
    study = open_study(
        study_dir=study_dir,
        study_name=study_name,
        db_url=db_url,
    )

    trials = study.get_trials(
        states=(optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,),
    )
    print(f'Completed trials: {len(trials)}')
    if use_train_loss:
        attr_name = 'train_loss'
    else:
        attr_name = 'valid_score'
    values = np.array([np.max(trial.user_attrs[attr_name]) for trial in trials])

    sorted_indices = np.flip(np.argsort(values))
    for i in np.arange(n_trials):
        i_trial = sorted_indices[i]
        trial = trials[i_trial]
        trial_dir = study_dir / f'trial_{trial.number:04d}'
        run = analyzer(trial_dir, device=device)
        run.plot_training_epochs()
        # run.plot_valid_indices_analysis()
        if max_elms:
            run.plot_full_inference(max_elms=max_elms)


def summarize_study(
        study_dir: Union[Path, str] = None,  # study dir. or db file
        study_name: str = None,
        db_url: str = None,
) -> None:
    study_dir = Path(study_dir).resolve()
    study = open_study(
        study_dir=study_dir,
        study_name=study_name,
        db_url=db_url,
    )

    trials = study.get_trials()
    for trial in trials:
        tmp = f"  Trial {trial.number}  state {trial.state}"
        if trial.state is optuna.trial.TrialState.FAIL:
            continue
        if 'train_score' in trial.user_attrs:
            tmp += f"  ep {len(trial.user_attrs['train_score'])}"
            tmp += f"  train_score {trial.user_attrs['train_score'][-1]:.4f}"
            if 'valid_score' in trial.user_attrs:
                tmp += f"  valid_score {trial.user_attrs['valid_score'][-1]:.4f}"
        print(tmp)

    print(f"Total trials: {len(trials)}")

    count = 0
    for state in [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.PRUNED,
    ]:
        n_state = sum([int(trial.state == state) for trial in trials])
        count += n_state
        print(f"{state} trial count: {n_state}")

    assert count == len(trials)

    if sum([int(trial.state == optuna.trial.TrialState.COMPLETE) for trial in trials]) > 20:
        importance = optuna.importance.get_param_importances(study)
        for param_name in importance:
            print(f"  {param_name}  importance {importance[param_name]:0.3f}")


if __name__ == '__main__':
    plt.close('all')

    study_dir = Path.home() / 'edgeml/scratch/study_reg_cnn_v08'

    summarize_study(study_dir)

    plot_study(study_dir, metric='valid_loss', quantile=0.0, use_last=False, save=True)

    # plot_top_trials(study_dir, use_train_loss=True, n_trials=4)

    plt.show()
