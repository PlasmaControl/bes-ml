from __future__ import annotations
import dataclasses
from typing import Iterable

import numpy as np
import torch
import torch.nn
import torch.utils.data


@dataclasses.dataclass(eq=False)
class Torch_Base(torch.nn.Module):
    signal_window_size: int = 128  # power of 2; ~64-512
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

    def initialize_layers(self):
        # initialize trainable parameters
        print("Initializing model layers")
        for name, param in self.named_parameters():
            if name.endswith(".bias"):
                print(f"  {name}: initialized to zeros (numel {param.data.numel()})")
                param.data.fill_(0)
            elif name.endswith(".weight"):
                n_in = np.prod(param.shape[1:])
                sqrt_k = np.sqrt(3. / n_in)
                print(f"  {name}: initialized to uniform +- {sqrt_k:.1e} (numel {param.data.numel()})")
                param.data.uniform_(-sqrt_k, sqrt_k)
                print(f"    n_in*var: {n_in*torch.var(param.data):.3f}")
       

@dataclasses.dataclass(eq=False)
class Torch_MLP_Mixin(Torch_Base):
    mlp_layers: tuple = (64, 32)
    mlp_dropout: float = 0.5

    def make_mlp(
            self, 
            mlp_in_features: int, 
            mlp_out_features: int = 1,
            with_sigmoid: bool = False,
    ) -> torch.nn.Module:
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

        # Logit or probability output?
        if with_sigmoid:
            print(f"  Applying sigmoid at MLP output for probability with range [0,1]")
            mlp_layers.append(torch.nn.Sigmoid())
        else:
            print(f"  Logit output (log odds, log(p/(1-p))) with range [-inf,inf]")

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

    def make_cnn(self) -> tuple[torch.nn.Module,int,tuple]:
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
                conv3d,
                torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope) if i!=0 else torch.nn.Identity(),
            ])
    
        assert np.array_equal(self.input_data_shape, data_shape)
    
        return decoder


@dataclasses.dataclass(eq=False)
class Torch_CNN_Model(
    Torch_CNN_Mixin,
    Torch_MLP_Mixin,
):
    autoencoder_reconstruction: bool = True
    time_to_elm_regression: bool = True
    active_elm_classification: bool = True
    # frontends: torch.nn.ModuleDict = dataclasses.field(default=None, init=False)
    # frontends_active: dict = dataclasses.field(default=None, init=False)
    
    def __post_init__(self):
        super().__post_init__()

        # CNN encoder `backend` to featurize input data
        self.cnn_encoder, cnn_features, cnn_output_shape = self.make_cnn()

        # `frontends` for regression, classification, and self-supervised learning
        # self.frontends: dict[str, torch.nn.Module] = {}
        self.frontends = torch.nn.ModuleDict()
        self.frontends_active: dict[str, bool] = {}
        if self.time_to_elm_regression:
            key = 'time_to_elm_regression'
            self.frontends.update({key: self.make_mlp(mlp_in_features=cnn_features)})
            self.frontends_active[key] = True
        if self.autoencoder_reconstruction:
            key = 'autoencoder_reconstruction'
            self.frontends.update({key: self.make_cnn_decoder(input_data_shape=cnn_output_shape)})
            self.frontends_active[key] = True
        if self.active_elm_classification:
            key = 'active_elm_classification'
            self.frontends.update({key: self.make_mlp(mlp_in_features=cnn_features)})
            self.frontends_active[key] = True
            
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters {total_parameters:,}")
        cnn_parameters = sum(p.numel() for p in self.cnn_encoder.parameters() if p.requires_grad)
        print(f"  CNN encoder parameters {cnn_parameters:,}")

        for key, frontend in self.frontends.items():
            n_parameters = sum(p.numel() for p in frontend.parameters() if p.requires_grad)
            print(f"  Frontend `{key}` parameters: {n_parameters:,}")

        self.initialize_layers()

    def forward(self, signals: torch.Tensor) -> dict[torch.Tensor]:
        results = {}
        features = self.cnn_encoder(signals)
        for key, frontend in self.frontends.items():
            if self.frontends_active[key]:
                results[key] = frontend(features)
        return results
