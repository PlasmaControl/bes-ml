try:
    from . import elm_datamodule
except:
    from bes_ml2 import elm_datamodule


if __name__=='__main__':
    signal_window_size = 256

    """
    Step 1a: Initiate pytorch_lightning.LightningDataModule
    """
    datamodule = elm_datamodule.ELM_Datamodule(
        # data_file='/global/homes/d/drsmith/ml/scratch/data/labeled_elm_events.hdf5',
        signal_window_size=signal_window_size,
        max_elms=100,
        batch_size=64,
        fraction_validation=0.,
        fraction_test=1.,
    )
    datamodule.setup(stage='predict')
    dataloaders = datamodule.predict_dataloader()
    for dataloader in dataloaders:
        dataset = dataloader.dataset
        elm_index = dataset.elm_index
        shot = dataset.shot
        pre_elm_size = dataset.active_elm_start_index-1
        print(f"ELM index {elm_index} shot {shot} pre-ELM size {pre_elm_size}")
        stats = dataset.stats()
        for key in stats:
            assert stats[key].shape[0] == 64
