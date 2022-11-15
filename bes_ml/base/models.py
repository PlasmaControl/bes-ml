import io
import logging
from pathlib import Path
import sys
from typing import Iterable
import inspect
import dataclasses

import numpy as np
import torch
import torch.nn as nn
import torchinfo
import pywt
from pytorch_wavelets.dwt.transform1d import DWT1DForward

try:
    from . import dct
except ImportError:
    from bes_ml.base import dct


@dataclasses.dataclass(eq=False)
class _Base_Features_Dataclass:
    signal_window_size: int = 64  # power of 2; ~16-512
    time_slice_interval: int = 1  # power of 2; time domain slice interval (i.e. time[::interval])
    spatial_pool_size: int = 1  # power of 2; spatial pooling size
    time_pool_size: int = 1  # power of 2; time pooling size
    pool_func: str = 'avg'  # `avg` or `max`
    activation_name: str = 'LeakyReLU'  # activation function in torch.nn like `LeakyReLu` or `SiLu`
    leakyrelu_negative_slope: float = 1e-3  # leaky relu negative slope; ~1e-3
    dropout_rate: float = 0.1  # ~0.1
    logger: logging.Logger = None
    debug: bool = False  # do some debugging checks


@dataclasses.dataclass(eq=False)
class _Base_Features(nn.Module, _Base_Features_Dataclass):

    def __post_init__(self):
        super().__init__()  # nn.Module.__init__()

        if self.logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

        # signal window
        assert np.log2(self.signal_window_size).is_integer()  # ensure power of 2

        # time slice interval to simulate lower sample rate
        assert np.log2(self.time_slice_interval).is_integer()  # ensure power of 2
        self.time_points = self.signal_window_size // self.time_slice_interval

        # time/space pooling
        assert np.log2(self.spatial_pool_size).is_integer() and self.spatial_pool_size <= 8
        assert np.log2(self.time_pool_size).is_integer() and self.time_pool_size <= self.time_points
        if self.spatial_pool_size > 1 or self.time_pool_size > 1:
            if self.pool_func.lower().startswith('avg'):
                pool_func = nn.AvgPool3d
            elif self.pool_func.lower().startswith('max'):
                pool_func = nn.MaxPool3d
            else:
                assert False, f"Invalid pool_func: {self.pool_func}"
            self.pooling_layer = pool_func(
                kernel_size=(self.time_pool_size, self.spatial_pool_size, self.spatial_pool_size),
            )
        else:
            self.pooling_layer = None

        self.time_points = self.time_points // self.time_pool_size
        assert np.log2(self.time_points).is_integer()

        # subwindows
        # assert np.log2(self.subwindows).is_integer() and self.subwindow_size < self.time_points
        # self.subwindow_size = self.time_points // self.subwindow_size
        # assert np.log2(self.subwindow_size).is_integer()

        # if self.subwindow_size == -1:
        #     self.subwindow_size = self.time_points
        # assert np.log2(self.subwindow_size).is_integer()  # ensure power of 2
        # assert self.subwindow_size <= self.time_points
        # self.subwindows = self.time_points // self.subwindow_size
        # assert self.subwindows >= 1
        
        # activation function
        self.activation_function = getattr(nn, self.activation_name)
        if self.activation_name == 'LeakyReLu':
            self.activation = self.activation_function(negative_slope=self.leakyrelu_negative_slope)
        else:
            self.activation = self.activation_function()

        # dropout
        self.dropout = nn.Dropout3d(p=self.dropout_rate)

        self.features = None  # set in subclass
        self.conv = None  # set in subclass

        self._input_size_after_timeslice_pooling = (
            self.time_points,
            8 // self.spatial_pool_size,
            8 // self.spatial_pool_size,
        )

    def _time_interval_and_pooling(self, x: torch.Tensor) -> torch.Tensor:
        if self.time_slice_interval > 1:
            x = x[:, :, ::self.time_slice_interval, :, :]
        if self.pooling_layer:
            x = self.pooling_layer(x)
        return x

    def _flatten_activation_dropout(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.activation(self.dropout(x)), 1)


@dataclasses.dataclass(eq=False)
class _Dense_Features_Dataclass(_Base_Features_Dataclass):
    dense_num_kernels: int = 0


@dataclasses.dataclass(eq=False)
class Dense_Features(_Dense_Features_Dataclass, _Base_Features):

    def __post_init__(self):
        super().__post_init__()

        self.features = self.dense_num_kernels

        kernel_size = self._input_size_after_timeslice_pooling

        # self.conv = nn.ModuleList(
        #     [
        #         nn.Conv3d(
        #             in_channels=1,
        #             out_channels=self.dense_num_kernels,
        #             kernel_size=kernel_size,
        #         ) for _ in range(self.subwindows)
        #     ]
        # )
        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=self.dense_num_kernels,
            kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._time_interval_and_pooling(x)
        # if self.subwindows == 1:
        x = self.conv(x)
        # else:
        #     x_new_size = [
        #         x.shape[0],
        #         self.dense_num_kernels,
        #         self.subwindows,
        #         1,
        #         1,
        #     ]
        #     x_new = torch.empty(size=x_new_size, dtype=x.dtype, device=x.device)
        #     for i_bin in range(self.subwindows):
        #         i_start = i_bin * self.subwindow_size
        #         i_stop = (i_bin+1) * self.subwindow_size
        #         # if torch.any(torch.isnan(self.conv[i_bin].weight)) or torch.any(torch.isnan(self.conv[i_bin].bias)):
        #         #     assert False
        #         x_new[:, :, i_bin:i_bin+1, :, :] = self.conv[i_bin](
        #             x[:, :, i_start:i_stop, :, :]
        #         )
        x = self._flatten_activation_dropout(x)
        return x


@dataclasses.dataclass(eq=False)
class _CNN_Features_Dataclass(_Base_Features_Dataclass):
    cnn_layer1_num_kernels: int = 0
    cnn_layer1_kernel_time_size: int = 5  # must be odd
    cnn_layer1_kernel_spatial_size: int = 3
    cnn_layer1_maxpool_time_size: int = 4  # must be power of 2
    cnn_layer1_maxpool_spatial_size: int = 1  # must be power of 2
    cnn_layer2_num_kernels: int = 0
    cnn_layer2_kernel_time_size: int = 5  # must be odd
    cnn_layer2_kernel_spatial_size: int = 3
    cnn_layer2_maxpool_time_size: int = 4  # must be power of 2
    cnn_layer2_maxpool_spatial_size: int = 2  # must be power of 2


@dataclasses.dataclass(eq=False)
class CNN_Features(_CNN_Features_Dataclass, _Base_Features):

    def __post_init__(self):
        super().__post_init__()

        assert (
            np.log2(self.cnn_layer1_maxpool_spatial_size).is_integer() and
            np.log2(self.cnn_layer2_maxpool_spatial_size).is_integer() and
            np.log2(self.cnn_layer1_maxpool_time_size).is_integer() and
            np.log2(self.cnn_layer2_maxpool_time_size).is_integer()
        ), 'Maxpool dims must be power of 2'

        # ensure valid maxpool in time dimension
        assert self.cnn_layer1_maxpool_time_size * self.cnn_layer2_maxpool_time_size <= self.time_points, \
            f"Maxpool time sizes {self.cnn_layer1_maxpool_time_size} and {self.cnn_layer2_maxpool_time_size} not compatible with time points {self.time_points}"

        assert (
            self.cnn_layer1_kernel_time_size % 2 == 1 and
            self.cnn_layer2_kernel_time_size % 2 == 1
            # self.cnn_layer1_kernel_spatial_size % 2 == 1 and
            # self.cnn_layer2_kernel_spatial_size % 2 == 1
        ), 'Kernel time dims must be odd'

        self.logger.info("CNN transformation")
        data_shape = tuple([1]+list(self._input_size_after_timeslice_pooling))
        self.logger.info(f"  Input after pre-pooling, pre-slicing: {data_shape}")

        def test_bad_shape(shape):
            assert np.all(np.array(shape) >= 1), f"Bad shape: {shape}"

        # Conv #1
        self.layer1_conv = nn.Conv3d(
            in_channels=1,
            out_channels=self.cnn_layer1_num_kernels,
            kernel_size=(
                self.cnn_layer1_kernel_time_size,
                self.cnn_layer1_kernel_spatial_size,
                self.cnn_layer1_kernel_spatial_size,
            ),
            padding=((self.cnn_layer1_kernel_time_size-1)//2, 0, 0),  # pad time dimension
        )
        data_shape = [
            self.cnn_layer1_num_kernels,
            data_shape[1],  # time dim unchanged due to padding
            data_shape[2]-(self.cnn_layer1_kernel_spatial_size-1),
            data_shape[3]-(self.cnn_layer1_kernel_spatial_size-1),
        ]
        self.logger.info(f"  After conv #1: {data_shape}")
        assert data_shape[2]-(self.cnn_layer1_kernel_spatial_size-1) > 0, \
            f"Spatial size {data_shape[2]} not compatible with kernel spatial size {self.cnn_layer1_kernel_spatial_size}  (layer 1)"
        test_bad_shape(data_shape)
        assert data_shape[2] % self.cnn_layer1_maxpool_spatial_size == 0, \
            f"Spatial size {data_shape[2]} not compatible with maxpool spatial {self.cnn_layer1_maxpool_spatial_size} (layer 1)"

        # maxpool #1
        self.layer1_maxpool = nn.MaxPool3d(
            kernel_size=(
                self.cnn_layer1_maxpool_time_size,
                self.cnn_layer1_maxpool_spatial_size,
                self.cnn_layer1_maxpool_spatial_size,
            ),
        )
        data_shape = [
            data_shape[0],
            data_shape[1] // self.cnn_layer1_maxpool_time_size,
            data_shape[2] // self.cnn_layer1_maxpool_spatial_size,
            data_shape[3] // self.cnn_layer1_maxpool_spatial_size,
        ]
        self.logger.info(f"  After maxpool #1: {data_shape}")
        test_bad_shape(data_shape)

        # conv #2
        self.layer2_conv = nn.Conv3d(
            in_channels=self.cnn_layer1_num_kernels,
            out_channels=self.cnn_layer2_num_kernels,
            kernel_size=(
                self.cnn_layer2_kernel_time_size,
                self.cnn_layer2_kernel_spatial_size,
                self.cnn_layer2_kernel_spatial_size,
            ),
            padding=((self.cnn_layer2_kernel_time_size-1)//2, 0, 0),
        )
        data_shape = [
            self.cnn_layer2_num_kernels,
            data_shape[1],
            data_shape[2] - (self.cnn_layer2_kernel_spatial_size-1),
            data_shape[3] - (self.cnn_layer2_kernel_spatial_size-1),
        ]
        self.logger.info(f"  After conv #2: {data_shape}")
        assert data_shape[2]-(self.cnn_layer2_kernel_spatial_size-1) > 0, \
            f"Spatial size {data_shape[2]} not compatible with kernel spatial size {self.cnn_layer2_kernel_spatial_size} (layer 2)"
        test_bad_shape(data_shape)
        assert data_shape[2] % self.cnn_layer2_maxpool_spatial_size == 0, \
            f"Spatial size {data_shape[2]} not compatible with maxpool spatial {self.cnn_layer2_maxpool_spatial_size}  (layer 2)"

        # maxpool #2
        self.layer2_maxpool = nn.MaxPool3d(
            kernel_size=(
                self.cnn_layer2_maxpool_time_size,
                self.cnn_layer2_maxpool_spatial_size,
                self.cnn_layer2_maxpool_spatial_size,
            ),
        )
        data_shape = [
            data_shape[0],
            data_shape[1] // self.cnn_layer2_maxpool_time_size,
            data_shape[2] // self.cnn_layer2_maxpool_spatial_size,
            data_shape[3] // self.cnn_layer2_maxpool_spatial_size,
        ]
        self.logger.info(f"  After maxpool #2 (output): {data_shape}")
        test_bad_shape(data_shape)

        self.features = np.prod(data_shape, dtype=int)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._time_interval_and_pooling(x)
        x = self.activation(self.dropout(self.layer1_conv(x)))
        x = self.layer1_maxpool(x)
        x = self.activation(self.dropout(self.layer2_conv(x)))
        x = self.layer2_maxpool(x)
        if self.debug:
            assert torch.all(torch.isfinite(x))
        return torch.flatten(x, 1)


@dataclasses.dataclass(eq=False)
class _FFT_Features_Dataclass(_Base_Features_Dataclass):
    fft_num_kernels: int = 0
    fft_nbins: int = 4
    fft_histogram: bool = False
    fft_mean: float = None
    fft_stdev: float = None
    fft_kernel_time_size: int = 5
    fft_kernel_spatial_size: int = 3
    fft_maxpool_freq_size: int = 2
    fft_maxpool_spatial_size: int = 2


@dataclasses.dataclass(eq=False)
class FFT_Features(_FFT_Features_Dataclass, _Base_Features):

    def __post_init__(self):
        super().__post_init__()

        assert (
            np.log2(self.fft_maxpool_freq_size).is_integer() and
            np.log2(self.fft_maxpool_spatial_size).is_integer()
        ), 'FFT maxpool dims must be power of 2'

        assert(
            self.fft_kernel_time_size%2 == 1 and
            self.fft_kernel_spatial_size%2 == 1
        ), 'FFT kernel dims must be odd'

        assert np.log2(self.fft_nbins).is_integer(), 'FFT nbins must be power of 2'

        self.nfft = self.signal_window_size // self.fft_nbins
        self.nfreqs = self.nfft // 2 + 1
        self.min = np.inf
        self.max = -np.inf
        self.hist_bins = 230
        self.cummulative_hist = np.zeros(self.hist_bins, dtype=int)

        self.logger.info("FFT transformation")

        data_shape = tuple([1]+list(self._input_size_after_timeslice_pooling))
        self.logger.info(f"  Input after pre-pooling, pre-slicing: {data_shape}")

        data_shape = [
            self.fft_num_kernels,
            self.nfreqs-1,
            data_shape[2],
            data_shape[3],
        ]
        self.logger.info(f"  After FFT: {data_shape}")

        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=self.fft_num_kernels,
            kernel_size=(
                self.fft_kernel_time_size,
                self.fft_kernel_spatial_size,
                self.fft_kernel_spatial_size,
            ),
        )
        data_shape = [
            data_shape[0],
            data_shape[1] - (self.fft_kernel_time_size-1),
            data_shape[2]-(self.fft_kernel_spatial_size-1),
            data_shape[3]-(self.fft_kernel_spatial_size-1),
        ]
        self.logger.info(f"  After conv: {data_shape}")

        # maxpool
        self.fft_maxpool = nn.MaxPool3d(
            kernel_size=(
                self.fft_maxpool_freq_size,
                self.fft_maxpool_spatial_size,
                self.fft_maxpool_spatial_size,
            ),
        )
        data_shape = [
            data_shape[0],
            data_shape[1] // self.fft_maxpool_freq_size,
            data_shape[2] // self.fft_maxpool_spatial_size,
            data_shape[3] // self.fft_maxpool_spatial_size,
        ]
        self.logger.info(f"  After maxpool: {data_shape}")

        self.features = np.prod(data_shape, dtype=int)

    def forward(self, x):
        x = self._time_interval_and_pooling(x)
        fft_bins_size = [
            x.shape[0],
            self.fft_nbins,
            self.nfreqs-1,
            x.shape[3],
            x.shape[4],
        ]
        fft_bins = torch.empty(size=fft_bins_size, dtype=x.dtype, device=x.device)
        for i_bin in torch.arange(self.fft_nbins):
            rfft = torch.fft.rfft(
                x[:, :, i_bin * self.nfft:(i_bin+1) * self.nfft, :, :], 
                dim=2,
            )
            fft_bins[:, i_bin: i_bin + 1, :, :, :] = torch.abs(rfft[:, :, 1:, :, :]) ** 2
        fft_sw = torch.mean(fft_bins, dim=1, keepdim=True)
        fft_sw[fft_sw<1e-5] = 1e-5
        fft_sw = torch.log10(fft_sw)
        if self.fft_mean and self.fft_stdev:
            fft_sw = (fft_sw - self.fft_mean) / self.fft_stdev
            fft_sw[fft_sw<-6] = -6
            fft_sw[fft_sw>6] = 6
        if self.fft_histogram:
            self.min = np.min([fft_sw.min().item(), self.min])
            self.max = np.max([fft_sw.max().item(), self.min])
            hist, bin_edges = np.histogram(
                fft_sw.cpu(),
                bins=self.hist_bins,
                range=[-7,7],
            )
            self.cummulative_hist += hist
            self.bin_edges = bin_edges
        fft_sw_features = self.conv(fft_sw)
        output_features = self.activation(self.dropout(fft_sw_features))
        output_features = self.fft_maxpool(output_features)
        if self.debug:
            assert torch.all(torch.isfinite(output_features))
        return torch.flatten(output_features, 1)


@dataclasses.dataclass(eq=False)
class _DCT_Features_Dataclass(_Base_Features_Dataclass):
    dct_num_kernels: int = 0
    dct_nbins: int = 2


@dataclasses.dataclass(eq=False)
class DCT_Features(_DCT_Features_Dataclass, _Base_Features):

    def __post_init__(self):
        super().__post_init__()

        assert np.log2(self.dct_nbins) % 1 == 0  # ensure power of 2

        self.ndct = self.subwindow_size // self.dct_nbins
        # self.nfreqs = self.ndct // 2 + 1
        self.nfreqs = self.ndct

        filter_size = (
            self.nfreqs, 
            8 // self.spatial_pool_size,
            8 // self.spatial_pool_size,
        )

        self.conv = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels=1,
                    out_channels=self.dct_num_kernels,
                    kernel_size=filter_size,
                ) for _ in range(self.subwindows)
            ]
        )

    def forward(self, x):
        # x = x.to(self.device)  # needed for PowerPC architecture
        x = self._time_interval_and_pooling(x)
        dct_features_size = [
            x.shape[0],
            self.subwindows,
            self.dct_num_kernels,
            1,
            1,
            1,
        ]
        dct_features = torch.empty(size=dct_features_size, dtype=x.dtype, device=x.device)
        for i_sw in torch.arange(self.subwindows):
            dct_bins_size = [
                x.shape[0],
                self.dct_nbins,
                self.nfreqs,
                x.shape[3],
                x.shape[4],
            ]
            dct_bins = torch.empty(size=dct_bins_size, dtype=x.dtype, device=x.device)
            x_subwindow = x[:, :, i_sw*self.subwindow_size:(i_sw+1)*self.subwindow_size, :, :]
            for i_bin in torch.arange(self.dct_nbins):
                tmp = dct.dct_3d(
                    x_subwindow[:, :, i_bin * self.ndct:(i_bin+1) * self.ndct, :, :]
                )
                dct_bins[:, i_bin: i_bin + 1, :, :, :] = tmp
            dct_sw = torch.mean(dct_bins, dim=1, keepdim=True)
            dct_sw_features = torch.unsqueeze(self.conv[i_sw](dct_sw), 1)
            dct_features[:, i_sw:i_sw+1, :, :, :, :] = dct_sw_features
        output_features = self._flatten_activation_dropout(dct_features)
        return output_features


@dataclasses.dataclass(eq=False)
class _DWT_Features_Dataclass(_Base_Features_Dataclass):
    dwt_num_kernels: int = 0
    dwt_wavelet: str = 'db4'
    dwt_level: int = -1


@dataclasses.dataclass(eq=False)
class DWT_Features(_DWT_Features_Dataclass, _Base_Features):

    def __post_init__(self):
        super().__post_init__()

        # self.features = self.dwt_num_kernels

        max_level = pywt.dwt_max_level(
            self.subwindow_size, 
            self.dwt_wavelet
        )

        if self.dwt_level == -1:
            self.dwt_level = max_level

        assert self.dwt_level <= max_level

        # DWT and sample calculation to get new time domain size
        self.dwt = DWT1DForward(
            wave=self.dwt_wavelet,
            J=self.dwt_level,
            mode="reflect",
        )
        x_tmp = torch.empty(1, 1, self.subwindow_size)
        x_lo, x_hi = self.dwt(x_tmp)
        self.dwt_output_length = sum(
            [x_lo.shape[2]] + [hi.shape[2] for hi in x_hi]
        )

        filter_size = (
            self.dwt_output_length, 
            8 // self.spatial_pool_size,
            8 // self.spatial_pool_size,
        )

        self.conv = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels=1,
                    out_channels=self.dwt_num_kernels,
                    kernel_size=filter_size,
                ) for _ in range(self.subwindows)
            ]
        )

    def forward(self, x):
        x = self._time_interval_and_pooling(x)
        dwt_features_size = [
            x.shape[0],
            self.subwindows,
            self.dwt_num_kernels,
            1,
            1,
            1,
        ]
        dwt_features = torch.empty(size=dwt_features_size, dtype=x.dtype, device=x.device)
        for i_sw in torch.arange(self.subwindows):
            x_sw = x[:, :, i_sw*self.subwindow_size:(i_sw+1)*self.subwindow_size, :, :]
            dwt_sw_size = [
                x_sw.shape[0],
                x_sw.shape[1],
                self.dwt_output_length,
                x_sw.shape[3],
                x_sw.shape[4],
            ]
            dwt_sw = torch.empty(dwt_sw_size, dtype=x.dtype, device=x.device)
            for i_batch in torch.arange(x.shape[0]):
                x_tmp = (
                    x_sw[i_batch, 0, :, :, :]
                    .permute(1, 2, 0)
                )  # make 3D and move time dim. to last
                x_lo, x_hi = self.dwt(x_tmp)  # multi-level DWT on last dim.
                coeff = [x_lo] + [hi for hi in x_hi]  # make list of coeff.
                dwt_sw[i_batch, 0, :, :, :] = (
                    torch.cat(coeff, dim=2)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )  # concat list in dwt coeff. dim, unpermute, and expand
            dwt_sw_features = self.conv[i_sw](dwt_sw)
            dwt_features[:, i_sw:i_sw+1, :, :, :, :] = \
                torch.unsqueeze(dwt_sw_features, 1)
        output_features = self._flatten_activation_dropout(dwt_features)
        return output_features


@dataclasses.dataclass(eq=False)
class _Multi_Features_Model_Dataclass(
    _Dense_Features_Dataclass,
    _CNN_Features_Dataclass,
    _FFT_Features_Dataclass,
    _DCT_Features_Dataclass,
    _DWT_Features_Dataclass,
):
    mlp_hidden_layers: Iterable = (32, 16)  # size and number of MLP hidden layers
    mlp_output_size: int = 1


@dataclasses.dataclass(eq=False)
class Multi_Features_Model(nn.Module, _Multi_Features_Model_Dataclass):

    def __post_init__(self):
        super().__init__()  # nn.Module.__init__()

        assert (
            self.dense_num_kernels == 0 and
            self.fft_num_kernels == 0 and
            self.dct_num_kernels == 0 and
            self.dwt_num_kernels == 0 and
            (self.cnn_layer1_num_kernels == 0 and self.cnn_layer2_num_kernels == 0)
        ) is False, "All features are inactive"

        if self.logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            # self.logger.addHandler(logging.StreamHandler())
            self.logger.addHandler(logging.NullHandler())

        self_parameters = inspect.signature(self.__class__).parameters

        def get_feature_class_parameters(feature_class) -> dict:
            feature_parameters = inspect.signature(feature_class).parameters
            feature_kwargs = {}
            for param_name in feature_parameters:
                if param_name in self_parameters:
                    feature_kwargs[param_name] = getattr(self, param_name)
            return feature_kwargs

        self.features = []
        if self.dense_num_kernels > 0:
            feature_kwargs = get_feature_class_parameters(Dense_Features)
            self.dense_features = Dense_Features(**feature_kwargs)
            self.features.append(self.dense_features)
        if self.fft_num_kernels > 0:
            feature_kwargs = get_feature_class_parameters(FFT_Features)
            self.fft_features = FFT_Features(**feature_kwargs)
            self.features.append(self.fft_features)
        if self.dct_num_kernels > 0:
            feature_kwargs = get_feature_class_parameters(DCT_Features)
            self.dct_features = DCT_Features(**feature_kwargs)
            self.features.append(self.dct_features)
        if self.dwt_num_kernels > 0:
            feature_kwargs = get_feature_class_parameters(DWT_Features)
            self.dwt_features = DWT_Features(**feature_kwargs)
            self.features.append(self.dwt_features)
        if self.cnn_layer1_num_kernels > 0 and self.cnn_layer2_num_kernels > 0:
            feature_kwargs = get_feature_class_parameters(CNN_Features)
            self.cnn_features = CNN_Features(**feature_kwargs)
            self.features.append(self.cnn_features)
        assert len(self.features) > 0

        self.feature_count = {}
        total_features = 0
        self.logger.info('Features')
        for feature in self.features:
            feature_count = feature.forward(torch.rand([1, 1, feature.time_points, 8, 8])).numel()
            total_features += feature_count
            self.logger.info(f"  {feature.__class__.__name__}: {feature_count}")
            self.feature_count[feature.__class__.__name__] = feature_count
        self.feature_count['total'] = total_features
        self.logger.info(f"  Total features: {total_features}")

        hidden_layers = []
        in_features = total_features
        for i_layer, layer_size in enumerate(self.mlp_hidden_layers):
            layer = nn.Linear(in_features=in_features, out_features=layer_size)
            hidden_layers.append(layer)
            in_features = layer_size
            self.logger.info(f"MLP layer {i_layer+1} size: {layer_size}")
        self.hidden_layers = nn.ModuleList(hidden_layers)

        self.output_layer = nn.Linear(
            in_features=in_features, 
            out_features=self.mlp_output_size, 
        )
        self.logger.info(f"MLP output size: {self.mlp_output_size}")

        self.activation_function = getattr(nn, self.activation_name)
        if self.activation_name == 'LeakyReLu':
            self.activation = self.activation_function(negative_slope=self.leakyrelu_negative_slope)
        else:
            self.activation = self.activation_function()
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.trainable_parameters = {}

    def forward(self, x):
        all_features = [features(x) for features in self.features]
        if self.debug:
            for features in all_features:
                assert torch.all(torch.isfinite(features))
        all_features = torch.cat(all_features, dim=1)
        for hidden_layer in self.hidden_layers:
            all_features = self.activation(self.dropout(hidden_layer(all_features)))
        prediction = self.output_layer(all_features)
        return prediction

    def save_pytorch_model(self, filename: Path | str = None) -> None:
        filename = Path(filename)
        torch.save(self.state_dict(), filename.as_posix())
        self.logger.info(f"  Saved model: {filename}  file size: {filename.stat().st_size/1e3:.1f} kB")

    def save_onnx_model(self, filename: Path | str = None) -> None:
        filename = Path(filename)
        torch.onnx.export(
            model=self, 
            args=torch.rand(1, 1, self.signal_window_size, 8, 8),
            f=filename.as_posix(),
            input_names=['signal_window'],
            output_names=['prediction'],
            opset_version=11,
        )
        self.logger.info(f"  Saved ONNX model: {filename}  file size: {filename.stat().st_size/1e3:.1f} kB")

    def print_model_summary(self) -> None:
        self.logger.info("MODEL SUMMARY")
        data_shape = (1, 1, self.signal_window_size, 8, 8)
        input_data = torch.rand(*data_shape)

        # catpure torchinfo.summary() output
        tmp_io = io.StringIO()
        sys.stdout = tmp_io
        torchinfo.summary(self, input_data=input_data, device=torch.device('cpu'))
        sys.stdout = sys.__stdout__
        self.logger.info('\n'+tmp_io.getvalue())
        # print model summary
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"Model contains {n_params} trainable parameters")
        self.logger.info(f'Single input size: {input_data.shape}')
        output = self(input_data)
        self.logger.info(f"Single output size: {output.shape}")

        self.trainable_parameters = {}
        self.logger.info("Trainable parameters")
        feature_parameters = 0
        for feature in self.features:
            trainable_params = sum(p.numel() for p in feature.parameters() if p.requires_grad)
            feature_parameters += trainable_params
            self.trainable_parameters[feature.__class__.__name__] = trainable_params
            self.logger.info(f"  {feature.__class__.__name__} parameters: {trainable_params}")
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mlp_parameters = total_parameters - feature_parameters
        self.trainable_parameters['mlp'] = mlp_parameters
        self.logger.info(f"  MLP parameters: {mlp_parameters}")
        self.trainable_parameters['total'] = total_parameters
        self.logger.info(f"  Total parameters: {total_parameters}")


if __name__ == '__main__':
    m = Multi_Features_Model(
        dense_num_kernels=8,
        fft_num_kernels=8,
        dwt_num_kernels=8,
        dct_num_kernels=8,
        cnn_layer1_num_kernels=8,
        cnn_layer2_num_kernels=8,
    )
