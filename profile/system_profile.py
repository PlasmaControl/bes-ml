import time
from pathlib import Path

from bes_ml.elm_regression import Trainer


def train_job(
        num_workers: int,
        dense_num_kernels: int,
        batch_size: int,
        signal_window_size: int,
        device: str,
        output_dir: Path | str,
) -> tuple[float, float]:
    init_start = time.time()
    model = Trainer(
        # model parameters
        dense_num_kernels=dense_num_kernels,
        signal_window_size=signal_window_size,
        mlp_hidden_layers=[128, 32],
        # ELM dataset parameters
        data_location=Path.home() / 'ml/scratch/data/labeled_elm_events.hdf5',
        batch_size=batch_size,
        max_elms=100,
        bad_elm_indices_csv=True,  # read bad ELMs from CSV in bes_data.elm_data_tools
        # _Base_Trainer parameters
        device=device,
        minibatch_print_interval=1000,
        terminal_output=False,
        # ELM regression parameters,
        fraction_test=0,
        num_workers=num_workers,
        seed=0,
        output_dir=output_dir,
    )
    init_time = time.time() - init_start
    train_start = time.time()
    model.train()
    train_time = time.time() - train_start
    return init_time, train_time


def main_loop():
    run_index = 0
    for num_workers in [0]:
        for batch_size in [16, 128]:
            for signal_window_size in [32, 128]:
                for dense_num_kernels in [32, 256]:
                    kwargs = {
                        'num_workers': num_workers,
                        'batch_size': batch_size,
                        'dense_num_kernels': dense_num_kernels,
                        'signal_window_size': signal_window_size,
                        'output_dir': f"run_dir_{run_index:02d}"
                    }
                    init, train = train_job(
                        device='cuda',
                        **kwargs,
                    )
                    print(f"Run {run_index}", kwargs)
                    print(f"    Init = {init:.2f} s    Train = {train:.2f} s")
                    run_index += 1


if __name__ == '__main__':
    main_loop()
