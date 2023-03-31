# Temporal Fusion Transformer
## Transformer based model for multivariate time series forecasting.

Transformers are all the rage these days and every paper coming out of the ML community is basically "but with transformers!"
So naturally, I went ahead and implemented a transformer for our data. Luckily there is a great library called 
"PyTorch Lightning" that makes it easy to implement and train models in PyTorch. The library is very well documented and
has a lot of great features. I highly recommend it. Using that, someone built a library called "Pytorch Forecasting" that
is less well documented and has a lot of bugs, but it does have a nice interface for building the **temporal fusion transformer**.

The Temporal Fusion Transformer is unique in that it allows for integration of static features as well as multiple time series.
It also has the ability to use future known variables in its predictions.
I think it would work well if we decide to incorporate other sources of data like planned shot profiles into our model.
It will not work on an FPGA. It is highly complex, with LSTM layers and attention layers.
But it might provide a decent baseline for SOTA performance.
I have not yet tested it on the full dataset, but it does work on a small subset of the data.

## Installation
1. Install requirements_tft.txt

## Usage
**Note**: I had to change `_TORCH_GREATER_EQUAL_1_11 = compare_version("torch", operator.ge, "1.11.0")` to `_TORCH_GREATER_EQUAL_1_11 = True`
in the file `venv/lib/python3.10/site-packages/lightning_fabric/utilities/imports.py` to get the imports to work properly.
This might be different for you.

The main configuration variables are in the second cell of the notebook.

The notebook can be run all at once, or you can run each cell individually. The first cell is just imports and the second
cell is the configuration variables. Pytorch-forecasting throws an error if you use a normalizer
for at least one of the time series. So I have set the normalizer to None for all of them.
I think this is a bug in the library, but I haven't done enough testing to be sure.

If you'd like to run it on the full dataset, note that the data is loaded into a Pandas dataframe.
Pytorch-forecasting is limited to in-memory dataframes, so you will need to have enough RAM to load the full dataset or 
you will need to use a smaller subset of the data. From the docs:

```text
Pytorch-forecasting is limited to in-memory dataframes, Large datasets:  Currently the class is limited to in-memory operations (that can be sped up by an existing installation of `numba <https://pypi.org/project/numba/>`_). If you have extremely large data, however, you can pass prefitted encoders and and scalers to it and a subset of sequences to the class to construct a valid dataset (plus, likely the EncoderNormalizer should be used to normalize targets). when fitting a network, you would then to create a custom DataLoader that rotates through the datasets. There is currently no in-built methods to do this.
```

## Up Next
I think the next step is to try a different model architecture called N-HiTS.
It might allow for integration of data with different sampling rates, and is less computationally expensive than the TFT.
I also want to dive into the probabilistic side of things. I think there's some information in the data, related to the
distribution of ELMs in time that may be useful for prediction. With a bayesian approach, we could model the distribution
of ELMs as a wait time problem and predict the parameters of certain wait-time statistical distributions (exponential, Weibull, etc.).
This might also tell us some neat new things about ELM physics like if they're memoryless (time-to-ELM is independent of time-since-ELM)
or if there's a hidden parameter that "decays" over time.