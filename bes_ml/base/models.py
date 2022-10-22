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
    # subwindows: int = 1  # power of 2; subwindows
    # subwindow_size: int = -1  # power of 2, or -1 (default) for full signal window
    activation_name: str = 'LeakyReLU'  # activation function in torch.nn like `LeakyReLu` or `SiLu`
    leakyrelu_negative_slope: float = 1e-3  # leaky relu negative slope; ~1e-3
    dropout_rate: float = 0.1  # ~0.1
    logger: logging.Logger = None


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
                kernel_size=[self.time_pool_size, self.spatial_pool_size, self.spatial_pool_size],
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

        self.num_kernels = None  # set in subclass
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

        self.num_kernels = self.dense_num_kernels

        kernel_size = self._input_size_after_timeslice_pooling

        # list of conv. filter banks (each with self.num_kernels) with size self.subwindow_bins
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
    cnn_layer1_kernel_spatial_size: int = 3  # must be odd
    cnn_layer1_maxpool_time_size: int = 4  # must be power of 2
    cnn_layer1_maxpool_spatial_size: int = 1  # must be power of 2
    cnn_layer2_num_kernels: int = 0
    cnn_layer2_kernel_time_size: int = 5  # must be odd
    cnn_layer2_kernel_spatial_size: int = 3  # must be odd
    cnn_layer2_maxpool_time_size: int = 4  # must be power of 2
    cnn_layer2_maxpool_spatial_size: int = 2  # must be power of 2


@dataclasses.dataclass(eq=False)
class CNN_Features(_CNN_Features_Dataclass, _Base_Features):

    def __post_init__(self):
        super().__post_init__()

        # CNN only valid with subwindow_size == time_points == signal_window_size
        # assert (
        #     # self.subwindow_size == self.signal_window_size and
        #     self.time_slice_interval == 1 and
        #     # self.subwindows == 1 and
        #     self.time_points == self.signal_window_size and
        #     self.spatial_pool_size == 1
        # )

        # maxpools must be power of 2
        assert (
            np.log2(self.cnn_layer1_maxpool_spatial_size).is_integer() and
            np.log2(self.cnn_layer2_maxpool_spatial_size).is_integer() and
            np.log2(self.cnn_layer1_maxpool_time_size).is_integer() and
            np.log2(self.cnn_layer2_maxpool_time_size).is_integer()
        )

        # ensure valid maxpool in time dimension
        assert self.cnn_layer1_maxpool_time_size * self.cnn_layer2_maxpool_time_size <= self.time_points

        # kernel sizes must be odd
        assert (
            self.cnn_layer1_kernel_time_size % 2 == 1 and
            self.cnn_layer2_kernel_time_size % 2 == 1
            # self.cnn_layer1_kernel_spatial_size % 2 == 1 and
            # self.cnn_layer2_kernel_spatial_size % 2 == 1
        )

        input_shape = tuple([1]+list(self._input_size_after_timeslice_pooling))
        self.logger.info(f"CNN input after pre-pooling, pre-slicing: {input_shape}")

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
            # stride=(1, 1, 1),
            padding=((self.cnn_layer1_kernel_time_size-1)//2, 0, 0),  # pad time dimension
        )
        output_shape = [
            self.cnn_layer1_num_kernels,
            input_shape[1],  # time dim unchanged due to padding
            input_shape[2]-(self.cnn_layer1_kernel_spatial_size-1),  # contract
            input_shape[3]-(self.cnn_layer1_kernel_spatial_size-1),
        ]
        self.logger.info(f"CNN after conv #1: {output_shape}")
        test_bad_shape(output_shape)
        assert output_shape[2] % self.cnn_layer1_maxpool_spatial_size == 0


        # maxpool #1
        self.layer1_maxpool = nn.MaxPool3d(
            kernel_size=(
                self.cnn_layer1_maxpool_time_size,
                self.cnn_layer1_maxpool_spatial_size,
                self.cnn_layer1_maxpool_spatial_size,
            ),
        )
        output_shape = [
            output_shape[0],
            output_shape[1] // self.cnn_layer1_maxpool_time_size,
            output_shape[2] // self.cnn_layer1_maxpool_spatial_size,
            output_shape[3] // self.cnn_layer1_maxpool_spatial_size,
        ]
        self.logger.info(f"CNN after maxpool #1: {output_shape}")
        test_bad_shape(output_shape)

        # conv #2
        self.layer2_conv = nn.Conv3d(
            in_channels=self.cnn_layer1_num_kernels,
            out_channels=self.cnn_layer2_num_kernels,
            kernel_size=(
                self.cnn_layer2_kernel_time_size,
                self.cnn_layer2_kernel_spatial_size,
                self.cnn_layer2_kernel_spatial_size,
            ),
            stride=(1, 1, 1),
            padding=((self.cnn_layer2_kernel_time_size-1)//2, 0, 0),
        )
        output_shape = [
            self.cnn_layer2_num_kernels,
            output_shape[1],
            output_shape[2] - (self.cnn_layer2_kernel_spatial_size-1),
            output_shape[3] - (self.cnn_layer2_kernel_spatial_size-1),
        ]
        self.logger.info(f"CNN after conv #2: {output_shape}")
        test_bad_shape(output_shape)
        assert output_shape[2] % self.cnn_layer2_maxpool_spatial_size == 0

        # maxpool #2
        self.layer2_maxpool = nn.MaxPool3d(
            kernel_size=(
                self.cnn_layer2_maxpool_time_size,
                self.cnn_layer2_maxpool_spatial_size,
                self.cnn_layer2_maxpool_spatial_size,
            ),
        )
        output_shape = [
            output_shape[0],
            output_shape[1] // self.cnn_layer2_maxpool_time_size,
            output_shape[2] // self.cnn_layer2_maxpool_spatial_size,
            output_shape[3] // self.cnn_layer2_maxpool_spatial_size,
        ]
        self.logger.info(f"CNN after maxpool #2 (output): {output_shape}")
        test_bad_shape(output_shape)

        self.num_kernels = np.prod(output_shape, dtype=int)
        self.logger.info(f"CNN output shape: {output_shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._time_interval_and_pooling(x)
        x = self.activation(self.dropout(self.layer1_conv(x)))
        x = self.layer1_maxpool(x)
        x = self.activation(self.dropout(self.layer2_conv(x)))
        x = self.layer2_maxpool(x)
        return torch.flatten(x, 1)


@dataclasses.dataclass(eq=False)
class _FFT_Features_Dataclass(_Base_Features_Dataclass):
    fft_num_kernels: int = 0
    fft_nbins: int = 4


@dataclasses.dataclass(eq=False)
class FFT_Features(_FFT_Features_Dataclass, _Base_Features):

    def __post_init__(self):
        super().__post_init__()

        self.num_kernels = self.fft_num_kernels

        assert np.log2(self.fft_nbins) % 1 == 0  # ensure power of 2

        self.nfft = self.subwindow_size // self.fft_nbins
        self.nfreqs = self.nfft // 2 + 1

        filter_size = (
            self.nfreqs, 
            8 // self.spatial_pool_size,
            8 // self.spatial_pool_size,
        )

        # list of conv. filter banks (each with self.num_kernels) with size self.subwindow_bins
        self.conv = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels=1,
                    out_channels=self.fft_num_kernels,
                    kernel_size=filter_size,
                ) for _ in range(self.subwindows)
            ]
        )

    def forward(self, x):
        # x = x.to(self.device)  # needed for PowerPC architecture
        x = self._time_interval_and_pooling(x)
        fft_features_size = [
            x.shape[0],
            self.subwindows,
            self.fft_num_kernels,
            1,
            1,
            1,
        ]
        fft_features = torch.empty(size=fft_features_size, dtype=x.dtype, device=x.device)
        for i_sw in torch.arange(self.subwindows):
            fft_bins_size = [
                x.shape[0],
                self.fft_nbins,
                self.nfreqs,
                x.shape[3],
                x.shape[4],
            ]
            fft_bins = torch.empty(size=fft_bins_size, dtype=x.dtype, device=x.device)
            x_sw = x[:, :, i_sw*self.subwindow_size:(i_sw+1)*self.subwindow_size, :, :]
            for i_bin in torch.arange(self.fft_nbins):
                fft_bins[:, i_bin: i_bin + 1, :, :, :] = torch.abs(
                    torch.fft.rfft(
                        x_sw[:, :, i_bin * self.nfft:(i_bin+1) * self.nfft, :, :], 
                        dim=2,
                    )
                )
            fft_sw = torch.mean(fft_bins, dim=1, keepdim=True)
            fft_sw_features = self.conv[i_sw](fft_sw)
            fft_features[:, i_sw:i_sw+1, :, :, :, :] = \
                torch.unsqueeze(fft_sw_features, 1)
        output_features = self._flatten_activation_dropout(fft_features)
        return output_features


@dataclasses.dataclass(eq=False)
class _DCT_Features_Dataclass(_Base_Features_Dataclass):
    dct_num_kernels: int = 0
    dct_nbins: int = 2


@dataclasses.dataclass(eq=False)
class DCT_Features(_DCT_Features_Dataclass, _Base_Features):

    def __post_init__(self):
        super().__post_init__()

        self.num_kernels = self.dct_num_kernels

        assert np.log2(self.dct_nbins) % 1 == 0  # ensure power of 2

        self.ndct = self.subwindow_size // self.dct_nbins
        # self.nfreqs = self.ndct // 2 + 1
        self.nfreqs = self.ndct

        filter_size = (
            self.nfreqs, 
            8 // self.spatial_pool_size,
            8 // self.spatial_pool_size,
        )

        # list of conv. filter banks (each with self.num_kernels) with size self.subwindow_bins
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

        self.num_kernels = self.dwt_num_kernels

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

        # list of conv. filter banks (each with self.num_kernels) with size self.subwindow_bins
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
    # mlp_layer1_size: int = 32  # multi-layer perceptron (mlp)
    # mlp_layer2_size: int = 16
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
        ) is False

        if self.logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

        self.dense_features = \
            self.fft_features = \
            self.dct_features = \
            self.dwt_features = \
            self.cnn_features = None

        self_parameters = inspect.signature(self.__class__).parameters

        def get_feature_class_parameters(feature_class) -> dict:
            feature_parameters = inspect.signature(feature_class).parameters
            feature_kwargs = {}
            for param_name in feature_parameters:
                if param_name in self_parameters:
                    feature_kwargs[param_name] = getattr(self, param_name)
            return feature_kwargs

        if self.dense_num_kernels > 0:
            feature_kwargs = get_feature_class_parameters(Dense_Features)
            self.dense_features = Dense_Features(**feature_kwargs)

        if self.fft_num_kernels > 0:
            feature_kwargs = get_feature_class_parameters(FFT_Features)
            self.fft_features = FFT_Features(**feature_kwargs)

        if self.dct_num_kernels > 0:
            feature_kwargs = get_feature_class_parameters(DCT_Features)
            self.dct_features = DCT_Features(**feature_kwargs)

        if self.dwt_num_kernels > 0:
            feature_kwargs = get_feature_class_parameters(DWT_Features)
            self.dwt_features = DWT_Features(**feature_kwargs)

        if self.cnn_layer1_num_kernels > 0 and self.cnn_layer2_num_kernels > 0:
            feature_kwargs = get_feature_class_parameters(CNN_Features)
            self.cnn_features = CNN_Features(**feature_kwargs)

        self.total_features = sum(
            [
                features.num_kernels #* features.subwindow_nbins
                for features in [
                    self.dense_features,
                    self.fft_features,
                    self.dwt_features,
                    self.dct_features,
                    self.cnn_features,
                ]
                if features is not None
            ]
        )
        self.logger.info(f"Total features: {self.total_features}")

        hidden_layers = []
        in_features = self.total_features  # features are input layer to MLP
        for i_layer, layer_size in enumerate(self.mlp_hidden_layers):
            hidden_layers.append(
                nn.Linear(in_features=in_features, out_features=layer_size)
            )
            in_features = layer_size
            self.logger.info(f"MLP layer {i_layer+1} size: {layer_size}")
        self.hidden_layers = nn.ModuleList(hidden_layers)

        self.output_layer = nn.Linear(in_features=in_features, out_features=self.mlp_output_size)
        self.logger.info(f"MLP output size: {self.mlp_output_size}")

        self.activation_function = getattr(nn, self.activation_name)
        if self.activation_name == 'LeakyReLu':
            self.activation = self.activation_function(negative_slope=self.leakyrelu_negative_slope)
        else:
            self.activation = self.activation_function()
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        dense_features = self.dense_features(x) if self.dense_features else None
        fft_features = self.fft_features(x) if self.fft_features else None
        dwt_features = self.dwt_features(x) if self.dwt_features else None
        dct_features = self.dct_features(x) if self.dct_features else None
        cnn_features = self.cnn_features(x) if self.cnn_features else None

        all_features = [
            features
            for features in [dense_features, fft_features, dwt_features, dct_features, cnn_features]
            if features is not None
        ]

        x = torch.cat(all_features, dim=1)
        for hidden_layer in self.hidden_layers:
            x = self.activation(self.dropout(hidden_layer(x)))
        x = self.output_layer(x)

        return x

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
        input_shape = (1, 1, self.signal_window_size, 8, 8)

        # catpure torchinfo.summary() output
        tmp_io = io.StringIO()
        sys.stdout = tmp_io
        print()
        torchinfo.summary(self, input_size=input_shape, device=torch.device('cpu'))
        sys.stdout = sys.__stdout__
        self.logger.info(tmp_io.getvalue())
        # print model summary
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"Model contains {n_params} trainable parameters")
        test_input = torch.rand(*input_shape).to(torch.device('cpu'))
        self.logger.info(f'Single input size: {test_input.shape}')
        test_output = self(test_input)
        self.logger.info(f"Single output size: {test_output.shape}")


if __name__ == '__main__':
    m = Multi_Features_Model(
        dense_num_kernels=8,
        fft_num_kernels=8,
        dwt_num_kernels=8,
        dct_num_kernels=8,
        cnn_layer1_num_kernels=8,
        cnn_layer2_num_kernels=8,
    )
