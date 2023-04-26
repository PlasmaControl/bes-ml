from __future__ import annotations
import dataclasses
from typing import Iterable

import numpy as np
import torch
import torch.nn
import torch.utils.data


@dataclasses.dataclass(eq=False)
class Torch_Base(torch.nn.Module):
    signal_window_size: int = 128  # power of 2; ~16-512
    leaky_relu_slope: float = 1e-2

    def __post_init__(self):
        super().__init__()

        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

        assert np.log2(self.signal_window_size).is_integer(), 'Signal window must be power of 2'


@dataclasses.dataclass(eq=False)
class Torch_MLP_Mixin(Torch_Base):
    mlp_layers: tuple = (64, 32)
    mlp_dropout: float = 0.5

    def make_mlp(self, mlp_in_features: int, mlp_out_features: int = 1) -> torch.nn.Module:
        # MLP layers
        print("Constructing MLP layers")
        mlp_layers = torch.nn.Sequential(torch.nn.Flatten())
        n_layers = len(self.mlp_layers)
        for i, layer_size in enumerate(self.mlp_layers):
            in_features = mlp_in_features if i==0 else self.mlp_layers[i-1]
            print(f"  MLP layer {i} with in/out features: {in_features}/{layer_size} (LeakyReLU activ.)")
            mlp_layers.extend([
                torch.nn.Dropout(p=self.mlp_dropout) if i!=n_layers-1 else torch.nn.Identity(),
                torch.nn.Linear(
                    in_features=in_features,
                    out_features=layer_size,
                ),
                torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope) if i!=n_layers-1 else torch.nn.Identity(),
            ])

        # output layer
        print(f"  MLP output layer with in/out features {self.mlp_layers[-1]}/{mlp_out_features} (no activ.)")
        mlp_layers.append(
            torch.nn.Linear(
                in_features=self.mlp_layers[-1], 
                out_features=mlp_out_features,
            )
        )
        return mlp_layers


@dataclasses.dataclass(eq=False)
class Torch_CNN_Mixin(Torch_Base):
    cnn_nlayers: int = 3
    cnn_num_kernels: Iterable|int = 16
    cnn_kernel_time_size: Iterable|int = 4
    cnn_kernel_spatial_size: Iterable|int = 3
    cnn_padding: Iterable|int|str = 0
    cnn_input_channels: int = 1
    cnn_dropout: float = 0.1

    def make_cnn(self) -> tuple:
        for attr_name in [
            'cnn_num_kernels',
            'cnn_kernel_time_size',
            'cnn_kernel_spatial_size',
            'cnn_padding',
        ]:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, Iterable) and not isinstance(attr_value, str):
                assert len(attr_value) == self.cnn_nlayers, f"{attr_name} {attr_value}"
            else:
                new_attr_value = tuple([attr_value]*self.cnn_nlayers)
                setattr(self, attr_name, new_attr_value)

        for time_dim in self.cnn_kernel_time_size:
            assert np.log2(time_dim).is_integer(), 'Kernel time dims must be power of 2'

        print("Constructing CNN layers")

        data_shape = (self.cnn_input_channels, self.signal_window_size, 8, 8)
        self.input_data_shape = tuple(data_shape)
        print(f"  Input data shape {data_shape}  (size {np.prod(data_shape)})")

        # CNN layers
        cnn = torch.nn.Sequential()
        for i in range(self.cnn_nlayers):
            kernel = (
                self.cnn_kernel_time_size[i],
                self.cnn_kernel_spatial_size[i],
                self.cnn_kernel_spatial_size[i],
            )
            stride = (self.cnn_kernel_time_size[i], 1, 1)
            print(f"  CNN Layer {i}")
            print(f"    Kernel {kernel}")
            print(f"    Stride {stride}")
            print(f"    Padding {self.cnn_padding[i]}")
            conv3d = torch.nn.Conv3d(
                in_channels=self.cnn_num_kernels[i-1] if i!=0 else self.cnn_input_channels,
                out_channels=self.cnn_num_kernels[i],
                kernel_size=kernel,
                stride=stride,
                padding=self.cnn_padding[i],
                padding_mode='reflect',
            )
            data_shape = tuple(conv3d(torch.zeros(size=data_shape)).size())
            print(f"    Output data shape: {data_shape}  (size {np.prod(data_shape)})")
            assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape} after CNN layer {i}"
            cnn.extend([
                torch.nn.Dropout(p=self.cnn_dropout),
                conv3d,
                torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            ])

        num_features = np.prod(data_shape)
        print(f"  CNN output features: {num_features}")

        return cnn, num_features, data_shape

    def make_cnn_decoder(self, input_data_shape: Iterable) -> torch.nn.Module:
        decoder = torch.nn.Sequential()
        data_shape = input_data_shape
        for i in range(self.cnn_nlayers-1, -1, -1):
            kernel = (
                self.cnn_kernel_time_size[i],
                self.cnn_kernel_spatial_size[i],
                self.cnn_kernel_spatial_size[i],
            )
            stride = (self.cnn_kernel_time_size[i], 1, 1)
            print(f"  Decoder Layer {i}")
            print(f"    Kernel {kernel}")
            print(f"    Stride {stride}")
            print(f"    Padding {self.cnn_padding[i]}")
            conv3d = torch.nn.ConvTranspose3d(
                in_channels=self.cnn_num_kernels[i],
                out_channels=self.cnn_num_kernels[i-1] if i!=0 else self.cnn_input_channels,
                kernel_size=kernel,
                stride=stride,
                padding=self.cnn_padding[i],
            )
            data_shape = tuple(conv3d(torch.zeros(size=data_shape)).size())
            print(f"    Output data shape: {data_shape}  (size {np.prod(data_shape)})")
            assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape} after Decoder layer {i}"
            decoder.extend([
                torch.nn.Dropout(p=self.cnn_dropout) if i!=0 else torch.nn.Identity(),
                conv3d,
                torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope) if i!=0 else torch.nn.Identity(),
            ])
        assert np.array_equal(
            self.input_data_shape,
            data_shape,
        )
        return decoder


@dataclasses.dataclass(eq=False)
class Torch_CNN_Model(
    Torch_CNN_Mixin,
    Torch_MLP_Mixin,
):
    
    def __post_init__(self):
        super().__post_init__()

        self.cnn, cnn_features, _ = self.make_cnn()
        self.mlp = self.make_mlp(mlp_in_features=cnn_features)
            
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters {total_parameters:,}")
        cnn_parameters = sum(p.numel() for p in self.cnn.parameters() if p.requires_grad)
        mlp_parameters = sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
        print(f"  CNN parameters {cnn_parameters:,}")
        print(f"  MLP parameters {mlp_parameters:,}")

    def forward(self, signals: torch.Tensor):
        features = self.cnn(signals)
        prediction = self.mlp(features)
        return prediction


@dataclasses.dataclass(eq=False)
class Torch_AE_Model(
    Torch_CNN_Mixin,
):
    
    def __post_init__(self):
        super().__post_init__()

        self.encoder, _, encoder_output_shape = self.make_cnn()
        self.decoder = self.make_cnn_decoder(
            input_data_shape=encoder_output_shape,
        )
            
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters {total_parameters:,}")
        encoder_parameters = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f"  Encoder parameters {encoder_parameters:,}")
        decoder_parameters = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        print(f"  Decoder parameters {decoder_parameters:,}")

    def forward(self, signals: torch.Tensor):
        features = self.encoder(signals)
        reconstruction = self.decoder(features)
        return reconstruction
