from pathlib import Path
import shutil
from typing import Union, Sequence
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import optuna
import seaborn as sns

from bes_ml.elm_regression.analyze import Analyzer


NON_RUNNING_STATES = (
    optuna.trial.TrialState.COMPLETE,
    optuna.trial.TrialState.PRUNED,
    optuna.trial.TrialState.FAIL,
)


def open_study(
        study_dir: Union[Path, str],
) -> optuna.study.Study:
    study_dir = Path(study_dir).resolve()
    assert study_dir.exists() and study_dir.is_dir()
    print(f"Opening study {study_dir}")

    db_file = study_dir / f"{study_dir.name}.db"
    assert db_file.exists()
    print(f"Opening database {db_file}")

    db_url = f'sqlite:///{db_file.as_posix()}'

    study = optuna.load_study('study', db_url)

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
        study_dir: Union[Path, str],  # study dir. or db file
        save: bool = False,
        metric: str = 'train_loss',
        use_last: bool = True,  # use last metric, not extremum
        quantile: float = None,
):
    study_dir = Path(study_dir).resolve()
    study = open_study(study_dir)

    trials = study.get_trials(
        states=(
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        ),
    )
    print(f'Completed trials: {len(trials)}')

    if use_last:
        key = lambda trial: trial.user_attrs[metric][-1]
    else:
        if 'score' in metric:
            key = lambda trial: np.max(trial.user_attrs[metric])
        else:
            key = lambda trial: np.min(trial.user_attrs[metric])

    # sort trials by key
    trials = sorted(trials, key=key)
    if 'score' in metric:
        trials.reverse()

    values = np.array([key(trial) for trial in trials])

    if quantile is not None:
        if 'loss' in metric:
            quantile = 1-quantile
        quantile = np.quantile(values, quantile)
        if 'score' in metric:
            trials = [trial for trial, value in zip(trials, values) if value >= quantile]
        else:
            trials = [trial for trial, value in zip(trials, values) if value <= quantile]
        values = np.array([key(trial) for trial in trials])

    # subset of top trials
    # top_quantile = np.quantile(values, 0.0)
    # trials = [trial for i_trial, trial in enumerate(trials) if values[i_trial] >= top_quantile]
    # values = np.array([values[i_trial] for i_trial in range(len(trials))])

    print("Top trials")
    for i_trial, trial in enumerate(trials[0:10]):
        print(f"  Trial {trial.number}  {metric} {values[i_trial]:.4g}")

    # top_trial = trials[0]
    # top_value = values[0]
    # top_trial_dir = (study_dir / f'trial_{top_trial.number:04d}').resolve()
    # print(f'Top trial {top_trial.number}  Max value: {top_value:.4f}  Dir: {top_trial_dir}')

    params = tuple(trials[-1].params.keys())
    n_params = len(params)

    ncols = 4
    nrows = n_params // ncols if n_params % ncols == 0 else (n_params // ncols) + 1
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3.125, nrows * 2))

    for i_param, param in enumerate(params):
        param_values = np.array([(trial.params[param] if param in trial.distributions else 0) for trial in trials])
        plt.sca(axes.flat[i_param])
        sns.violinplot(
            x=param_values,
            y=values,
            inner='stick',
            scale='count',
            bw=0.2,
        )
        plt.xlabel(param)
    plt.suptitle(f"{study_dir.as_posix()} | {len(trials)} trials | Best {metric} {values[0]:.3f}")
    plt.tight_layout()

    if save:
        filepath = study_dir.parent / (study_dir.stem + '.pdf')
        print(f'Saving file: {filepath.as_posix()}')
        plt.savefig(filepath, transparent=True)


def plot_top_trials(
        study_dir,
        n_trials = 1,
        device = None,
        max_elms = None,
        use_train_loss = False,
):
    study_dir = Path(study_dir).resolve()
    study = open_study(study_dir)

    trials = study.get_trials(
        states=(optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,),
    )
    print(f'Completed trials: {len(trials)}')
    if use_train_loss:
        attr_name = 'train_loss'
    else:
        attr_name = 'scores'
    values = np.array([np.max(trial.user_attrs[attr_name]) for trial in trials])

    sorted_indices = np.flip(np.argsort(values))
    for i in np.arange(n_trials):
        i_trial = sorted_indices[i]
        trial = trials[i_trial]
        trial_dir = study_dir / f'trial_{trial.number:04d}'
        run = Analyzer(trial_dir, device=device)
        run.plot_training_epochs()
        # run.plot_valid_indices_analysis()
        if max_elms:
            run.plot_full_inference(max_elms=max_elms)


def summarize_study(
        study_dir: Union[Path, str],  # study dir. or db file
) -> None:
    study_dir = Path(study_dir).resolve()
    study = open_study(study_dir)

    trials = study.get_trials()
    for trial in trials:
        tmp = f"  Trial {trial.number}  state {trial.state}"
        if trial.state in [
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        ]:
            tmp += f"  ep {len(trial.user_attrs['train_loss'])}"
            tmp += f"  train_loss {trial.user_attrs['train_loss'][-1]:.4f}"
            if trial.user_attrs['valid_score']:
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

    importance = optuna.importance.get_param_importances(
        study,
    )
    for param_name in importance:
        print(f"  {param_name}  importance {importance[param_name]:0.3f}")


if __name__ == '__main__':
    plt.close('all')

    study_dir = Path.home() / 'edgeml/scratch/study_reg_cnn_v03'

    # summarize_study(study_dir)

    plot_study(study_dir, metric='valid_loss', quantile=0.8)

    # plot_top_trials(study_dir, use_train_loss=True, n_trials=4)

    # work_dir = Path.home() / 'edgeml/scratch/work/study_06'
    # assert work_dir.exists()
    #
    # study_dirs = sorted(work_dir.glob('s06_*'))
    # assert study_dirs
    # for study_dir in study_dirs:
    #     if not study_dir.is_dir():
    #         continue
    #     plot_study(study_dir, save=True, plot_top_trials=False)
    # # merge study PDFs
    # inputs = sorted(work_dir.glob('*.pdf'))
    # assert inputs
    # output = work_dir / 'optuna_results.pdf'
    # output.unlink(missing_ok=True)
    # merge_pdfs(inputs, output, delete_inputs=True)

    # studies = (
    #     sorted(work_dir.glob('s06_class_*')) +
    #     sorted(work_dir.glob('s06_logreg_*'))
    # )
    # for study in studies:
    #     if not study.is_dir(): continue
    #     assert study.exists()
    #     try:
    #         plot_top_trials(
    #             study,
    #             n_trials=4,
    #             device='cuda:1',
    #             max_elms=18,
    #         )
    #     except:
    #         pass

    plt.show()
