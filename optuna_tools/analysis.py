from pathlib import Path
import shutil
from typing import Union, Sequence
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import optuna
import seaborn as sns

# from bes_ml.elm_regression.analyze import Analyzer


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
        plot_top_trials=False,
        save=False,
):
    study_dir = Path(study_dir).resolve()
    study = open_study(study_dir)

    trials = study.get_trials(
        states=(
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        ),
    )

    if not trials[0].user_attrs['scores']:
        for i_trial in range(len(trials)):
            trials[i_trial].user_attrs['scores'] = -1 * np.array(trials[i_trial].user_attrs['train_loss'])

    # sort trials by max score during training
    trials = sorted(
        trials,
        key=lambda trial: np.max(trial.user_attrs['scores']),
        reverse=True,
    )
    values = np.array([np.max(trial.user_attrs['scores']) for trial in trials])
    print(f'Completed trials: {len(trials)}')

    # subset of top trials
    top_quantile = np.quantile(values, 0.0)
    top_trials = [trial for i_trial, trial in enumerate(trials) if values[i_trial] >= top_quantile]
    top_values = np.array([values[i_trial] for i_trial in range(len(top_trials))])
    # top_values = np.array([np.max(trial.user_attrs['scores']) for trial in top_trials])
    # print(f'Trials with max score >= {top_quantile:.2f}: {len(top_trials)}')
    #
    # for i_trial, trial in enumerate(top_trials[:10]):
    #     print(f"  Trial {trial.number}  Max value: {top_values[i_trial]:.4f}"
    #           f"  Epochs: {len(trial.user_attrs['scores'])}  State {trial.state}")

    top_trial = top_trials[0]
    top_value = top_values[0]
    top_trial_dir = (study_dir / f'trial_{top_trial.number:04d}').resolve()
    print(f'Top trial {top_trial.number}  Max value: {top_value:.4f}  Dir: {top_trial_dir}')

    params = tuple(trials[-1].params.keys())
    n_params = len(params)

    ncols = 4
    nrows = n_params // ncols if n_params % ncols == 0 else (n_params // ncols) + 1
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3.125, nrows * 2))

    for i_param, param in enumerate(params):
        param_values = np.array([(trial.params[param] if param in trial.distributions else 0) for trial in top_trials])
        plt.sca(axes.flat[i_param])
        sns.violinplot(
            x=param_values,
            y=top_values,
            inner='stick',
            scale='count',
            bw=0.5,
        )
        plt.xlabel(param)
    plt.suptitle(f"{study_dir.as_posix()} | {len(trials)} trials | Max score {top_value:.3f}")
    plt.tight_layout()

    if save:
        filepath = study_dir.parent / (study_dir.stem + '.pdf')
        print(f'Saving file: {filepath.as_posix()}')
        plt.savefig(filepath, transparent=True)

    if plot_top_trials:
        top_trial = top_trials[0]
        top_trial_dir = study_dir / f"trial_{top_trial.number:04d}"
        run = Analysis(run_dir=top_trial_dir, save=save)
        run.plot_training_epochs()
        run.plot_valid_indices_analysis()

def plot_top_trials(
        study_dir,
        n_trials = 1,
        device = None,
        max_elms = None,
):
    study_dir = Path(study_dir).resolve()
    assert study_dir.exists()

    db_file = list(study_dir.glob('*.db'))[0]
    print(f'Opening RDB: {db_file.as_posix()}')

    db_url = f'sqlite:///{db_file.as_posix()}'

    study = optuna.load_study('study', db_url)
    trials = study.get_trials(
        states=(optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,),
    )
    print(f'Completed trials: {len(trials)}')
    values = np.array([np.max(trial.user_attrs['scores']) for trial in trials])

    sorted_indices = np.flip(np.argsort(values))
    for i in np.arange(n_trials):
        i_trial = sorted_indices[i]
        trial = trials[i_trial]
        trial_dir = study_dir / f'trial_{trial.number:04d}'
        run = Analysis(trial_dir, device=device)
        run.plot_training_epochs()
        run.plot_valid_indices_analysis()
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
            tmp += f"  loss {trial.user_attrs['train_loss'][-1]:.4f}"
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
        # params=[
        #     'lr_exp',
        #     'batch_size_pow2',
        #     'weight_decay',
        #     'signal_window_size_pow2',
        #     'mlp_layer1_size_pow2',
        #     'mlp_layer2_size_pow2',
        #     'sgd_momentum_fac',
        # ],
    )
    for param_name in importance:
        print(f"  {param_name}  importance {importance[param_name]:0.3f}")


if __name__ == '__main__':
    plt.close('all')

    study_dir = Path.home() / 'edgeml/scratch/study_logreg_cnn_v01'

    summarize_study(study_dir)

    plot_study(study_dir)

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
