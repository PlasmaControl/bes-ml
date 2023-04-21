import dataclasses
from typing import Iterable

import numpy as np
import torch
import torch.nn
import torch.utils.data


@dataclasses.dataclass(eq=False)
class Torch_Model_Base_Dataclass:
    signal_window_size: int = 128  # power of 2; ~16-512
    leaky_relu_slope: float = 1e-2
    dropout: float = 0.1
    mlp_layers: tuple = (64, 32)

    def make_mlp(self, mlp_in_features: int, mlp_out_features: int = 1) -> torch.nn.Module:
        # MLP layers
        print("Constructing MLP layers")
        mlp_layers = torch.nn.Sequential(torch.nn.Flatten())
        for i, layer_size in enumerate(self.mlp_layers):
            in_features = mlp_in_features if i==0 else self.mlp_layers[i-1]
            print(f"  MLP layer {i} with in/out features: {in_features}/{layer_size} (LeakyReLU activ.)")
            mlp_layers.extend([
                torch.nn.Linear(
                    in_features=in_features,
                    out_features=layer_size,
                ),
                torch.nn.Dropout(p=self.dropout),
                torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
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
class Torch_Model_CNN01(
    torch.nn.Module,
    Torch_Model_Base_Dataclass,
):
    cnn_layer1_num_kernels: int = 8
    cnn_layer1_kernel_time_size: int = 5
    cnn_layer1_kernel_spatial_size: int = 3
    cnn_layer1_maxpool_time: int = 4
    cnn_layer2_num_kernels: int = 8
    cnn_layer2_kernel_time_size: int = 5
    cnn_layer2_kernel_spatial_size: int = 3
    cnn_layer2_maxpool_time: int = 4
    mlp_layer1_size: int = 32
    mlp_layer2_size: int = 16
    
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
        assert self.cnn_layer1_kernel_time_size % 2 == 1, 'Kernel time size must be odd'
        assert self.cnn_layer2_kernel_time_size % 2 == 1, 'Kernel time size must be odd'
        assert self.cnn_layer1_maxpool_time%2 == 0
        assert self.cnn_layer2_maxpool_time%2 == 0
        in_channels = 1
        data_shape = [in_channels, self.signal_window_size, 8, 8]
        print(f"  Input data shape {data_shape}")

        # CNN and maxpool 1
        cnn_layer1_kernel = (
            self.cnn_layer1_kernel_time_size,
            self.cnn_layer1_kernel_spatial_size,
            self.cnn_layer1_kernel_spatial_size,
        )
        print(f"  CNN 1 kernel shape {cnn_layer1_kernel}")
        print(f"  CNN 1 kernel number {self.cnn_layer1_num_kernels}")
        cnn_layer1_padding = ((self.cnn_layer1_kernel_time_size-1) // 2, 0, 0)
        data_shape[0] = self.cnn_layer1_num_kernels
        data_shape[-2] = data_shape[-2]-(self.cnn_layer1_kernel_spatial_size-1)
        data_shape[-1] = data_shape[-1]-(self.cnn_layer1_kernel_spatial_size-1)
        print(f"    Data shape after CNN 1 {data_shape}")
        assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape}"
        maxpool_layer1_kernel = (self.cnn_layer1_maxpool_time, 1, 1)
        print(f"  Maxpool 1 shape {maxpool_layer1_kernel}")
        data_shape[1] = data_shape[1] // self.cnn_layer1_maxpool_time
        print(f"    Data shape after maxpool 1 {data_shape}")
        assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape}"

        # CNN and maxpool 2
        cnn_layer2_kernel = (
            self.cnn_layer2_kernel_time_size,
            self.cnn_layer2_kernel_spatial_size,
            self.cnn_layer2_kernel_spatial_size,
        )
        print(f"  CNN 2 kernel shape {cnn_layer2_kernel}")
        print(f"  CNN 2 kernel number {self.cnn_layer2_num_kernels}")
        cnn_layer2_padding = ((self.cnn_layer2_kernel_time_size-1) // 2, 0, 0)
        data_shape[0] = self.cnn_layer2_num_kernels
        data_shape[-2] = data_shape[-2]-(self.cnn_layer2_kernel_spatial_size-1)
        data_shape[-1] = data_shape[-1]-(self.cnn_layer2_kernel_spatial_size-1)
        print(f"    Data shape after CNN 2 {data_shape}")
        assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape}"
        maxpool_layer2_kernel = (self.cnn_layer2_maxpool_time, 1, 1)
        print(f"  Maxpool 2 shape {maxpool_layer2_kernel}")
        data_shape[1] = data_shape[1] // self.cnn_layer2_maxpool_time
        assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape}"
        print(f'    Data shape after maxpool 2 {data_shape}')

        # CNN output features
        cnn_features = np.prod(data_shape)
        print(f"  CNN output features {cnn_features}")

        # CNN model
        self.featurize = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=self.cnn_layer1_num_kernels,
                kernel_size=cnn_layer1_kernel,
                padding=cnn_layer1_padding,
            ),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            torch.nn.MaxPool3d(kernel_size=maxpool_layer1_kernel),
            torch.nn.Conv3d(
                in_channels=self.cnn_layer1_num_kernels,
                out_channels=self.cnn_layer2_num_kernels,
                kernel_size=cnn_layer2_kernel,
                padding=cnn_layer2_padding,
            ),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            torch.nn.MaxPool3d(kernel_size=maxpool_layer2_kernel),
        )

        # MLP
        print("Constructing MLP")
        print(f"  MLP layer 1 size {self.mlp_layer1_size}")
        print(f"  MLP layer 2 size {self.mlp_layer2_size}")
        self.mlp = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                in_features=cnn_features,
                out_features=self.mlp_layer1_size,
            ),
            torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(
                in_features=self.mlp_layer1_size, 
                out_features=self.mlp_layer2_size,
            ),
            torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(
                in_features=self.mlp_layer2_size, 
                out_features=1,
            ),
        )

        # parameter count
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters {total_parameters:,}")
        cnn_parameters = sum(p.numel() for p in self.featurize.parameters() if p.requires_grad)
        mlp_parameters = sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
        print(f"  CNN parameters {cnn_parameters:,}")
        print(f"  MLP parameters {mlp_parameters:,}")

    def forward(self, signals: torch.Tensor):
        features = self.featurize(signals)
        prediction = self.mlp(features)
        return prediction


@dataclasses.dataclass(eq=False)
class Torch_Model_CNN02(
    torch.nn.Module,
    Torch_Model_Base_Dataclass,
):
    cnn_nlayers: int = 3
    cnn_num_kernels: Iterable|int = 16
    cnn_kernel_time_size: Iterable|int = 4
    cnn_kernel_spatial_size: Iterable|int = 3
    cnn_padding: str|int|Iterable = 0
    cnn_padding_mode: str = 'zeros'
    
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

        for attr_name in ['cnn_num_kernels','cnn_kernel_time_size','cnn_kernel_spatial_size','cnn_padding']:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, Iterable) and not isinstance(attr_value, str):
                assert len(attr_value) == self.cnn_nlayers, f"{attr_name} {attr_value}"
            else:
                new_attr_value = tuple([attr_value]*self.cnn_nlayers)
                setattr(self, attr_name, new_attr_value)

        for time_dim in self.cnn_kernel_time_size:
            assert np.log2(time_dim).is_integer(), 'Kernel time dims must be power of 2'

        print("Constructing CNN layers")

        in_channels = 1
        data_shape = [in_channels, self.signal_window_size, 8, 8]
        print(f"  Input data shape {data_shape}")

        # CNN layers
        self.cnn = torch.nn.Sequential()
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
            print(f"    Padding {self.cnn_padding[i]} with mode `{self.cnn_padding_mode}`")
            conv3d = torch.nn.Conv3d(
                in_channels=in_channels if i==0 else self.cnn_num_kernels[i-1],
                out_channels=self.cnn_num_kernels[i],
                kernel_size=kernel,
                stride=stride,
                padding=self.cnn_padding[i],
                padding_mode=self.cnn_padding_mode,
            )
            data_shape = conv3d(torch.zeros(size=data_shape)).size()
            print(f"    Data shape after CNN layer {i}: {tuple(data_shape)}  (size {np.prod(data_shape)})")
            assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape} after CNN layer {i}"
            self.cnn.extend([
                conv3d,
                torch.nn.Dropout(p=self.dropout),
                torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            ])

        cnn_features = np.prod(data_shape)
        print(f"  CNN output features: {cnn_features}")

        self.mlp = self.make_mlp(mlp_in_features=cnn_features)
            
        # parameter count
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
class Torch_Model_AE01(
    torch.nn.Module,
    Torch_Model_Base_Dataclass,
):
    cnn_layer1_num_kernels: int = 8
    cnn_layer1_kernel_time_size: int = 5
    cnn_layer1_kernel_spatial_size: int = 3
    cnn_layer1_maxpool_time: int = 4
    cnn_layer2_num_kernels: int = 8
    cnn_layer2_kernel_time_size: int = 5
    cnn_layer2_kernel_spatial_size: int = 3
    cnn_layer2_maxpool_time: int = 4
    mlp_layer1_size: int = 32
    mlp_layer2_size: int = 16
    
    def __post_init__(self):
        super().__init__()
        print('Constructing CNN')
        assert np.log2(self.signal_window_size).is_integer(), 'Signal window must be power of 2'
        assert self.cnn_layer1_kernel_time_size % 2 == 1, 'Kernel time size must be odd'
        assert self.cnn_layer2_kernel_time_size % 2 == 1, 'Kernel time size must be odd'
        assert self.cnn_layer1_maxpool_time%2 == 0
        assert self.cnn_layer2_maxpool_time%2 == 0
        in_channels = 1
        data_shape = [in_channels, self.signal_window_size, 8, 8]
        print(f"  Input data shape {data_shape}")

        # CNN and maxpool 1
        cnn_layer1_kernel = (
            self.cnn_layer1_kernel_time_size,
            self.cnn_layer1_kernel_spatial_size,
            self.cnn_layer1_kernel_spatial_size,
        )
        print(f"  CNN 1 kernel shape {cnn_layer1_kernel}")
        print(f"  CNN 1 kernel number {self.cnn_layer1_num_kernels}")
        cnn_layer1_padding = ((self.cnn_layer1_kernel_time_size-1) // 2, 0, 0)
        data_shape[0] = self.cnn_layer1_num_kernels
        data_shape[-2] = data_shape[-2]-(self.cnn_layer1_kernel_spatial_size-1)
        data_shape[-1] = data_shape[-1]-(self.cnn_layer1_kernel_spatial_size-1)
        print(f"    Data shape after CNN 1 {data_shape}")
        assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape}"
        maxpool_layer1_kernel = (self.cnn_layer1_maxpool_time, 1, 1)
        print(f"  Maxpool 1 shape {maxpool_layer1_kernel}")
        data_shape[1] = data_shape[1] // self.cnn_layer1_maxpool_time
        print(f"    Data shape after maxpool 1 {data_shape}")
        assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape}"
        self.data_shape_after_maxpool1 = data_shape.copy()

        # CNN and maxpool 2
        cnn_layer2_kernel = (
            self.cnn_layer2_kernel_time_size,
            self.cnn_layer2_kernel_spatial_size,
            self.cnn_layer2_kernel_spatial_size,
        )
        print(f"  CNN 2 kernel shape {cnn_layer2_kernel}")
        print(f"  CNN 2 kernel number {self.cnn_layer2_num_kernels}")
        cnn_layer2_padding = ((self.cnn_layer2_kernel_time_size-1) // 2, 0, 0)
        data_shape[0] = self.cnn_layer2_num_kernels
        data_shape[-2] = data_shape[-2]-(self.cnn_layer2_kernel_spatial_size-1)
        data_shape[-1] = data_shape[-1]-(self.cnn_layer2_kernel_spatial_size-1)
        print(f"    Data shape after CNN 2 {data_shape}")
        assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape}"
        maxpool_layer2_kernel = (self.cnn_layer2_maxpool_time, 1, 1)
        print(f"  Maxpool 2 shape {maxpool_layer2_kernel}")
        data_shape[1] = data_shape[1] // self.cnn_layer2_maxpool_time
        assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape}"
        print(f'    Data shape after maxpool 2 {data_shape}')
        self.data_shape_after_maxpool2 = data_shape.copy()
        # CNN output features
        cnn_features = np.prod(data_shape)
        print(f"  CNN output features {cnn_features}")

        # CNN model
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=self.cnn_layer1_num_kernels,
                kernel_size=cnn_layer1_kernel,
                padding=cnn_layer1_padding,
            ),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            torch.nn.MaxPool3d(kernel_size=maxpool_layer1_kernel, return_indices=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=self.cnn_layer1_num_kernels,
                out_channels=self.cnn_layer2_num_kernels,
                kernel_size=cnn_layer2_kernel,
                padding=cnn_layer2_padding,
            ),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
            torch.nn.MaxPool3d(kernel_size=maxpool_layer2_kernel, return_indices=True),
        )
        self.max_unpool2 = torch.nn.MaxUnpool3d(kernel_size=maxpool_layer2_kernel)
        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                in_channels=self.cnn_layer2_num_kernels,
                out_channels=self.cnn_layer1_num_kernels,
                kernel_size=cnn_layer2_kernel,
                padding=cnn_layer2_padding,
            ),
        )
        self.max_unpool1 = torch.nn.MaxUnpool3d(kernel_size=maxpool_layer1_kernel)
        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(
                in_channels=self.cnn_layer1_num_kernels,
                out_channels=in_channels,
                kernel_size=cnn_layer1_kernel,
                padding=cnn_layer1_padding,
            ),
        )

        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters {total_parameters:,}")

    def forward(self, signals: torch.Tensor):
        out, indices1 = self.conv1(signals)
        out, indices2 = self.conv2(out)
        out = self.max_unpool2(out, indices2)
        out = self.deconv2(out)
        out = self.max_unpool1(out, indices1)
        reconstructed_signals = self.deconv1(out)
        return reconstructed_signals
