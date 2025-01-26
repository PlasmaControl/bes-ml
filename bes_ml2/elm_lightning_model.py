from __future__ import annotations
import dataclasses
import os
import time
from typing import Iterable
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from scipy.signal import spectrogram
import matplotlib.colors as mcolors 
from sklearn.manifold import TSNE
import itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from collections import defaultdict


import torch
import torch.nn
import torch.distributed as dist
from lightning.pytorch import LightningModule, loggers
import torchmetrics

import pickle
import h5py

import psutil

from torch.autograd import Function

class CustomRFFT(Function):
    @staticmethod
    def forward(ctx, input):
        # Perform the FFT operation and remove the DC component
        output = torch.fft.rfft(input, dim=2)[:, :, 1:, :, :]
        return output

    @staticmethod
    def symbolic(g, input):
        # ONNX placeholder using Identity, ensure the correct usage in runtime handling
        return g.op("Identity", input)
        
class BCEWithLogit(torchmetrics.Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = True
    bce: torch.Tensor
    counts: torch.Tensor

    def __init__(self):
        super().__init__()
        self.add_state("bce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("counts", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, input: torch.Tensor, target: torch.Tensor):
        self.bce += torch.nn.functional.binary_cross_entropy_with_logits(
            input=input, 
            target=target.type_as(input),
            reduction='sum',
        )
        self.counts += target.numel()

    def compute(self):
        return self.bce / self.counts

class CrossEntropy(torchmetrics.Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = True
    ce: torch.Tensor
    counts: torch.Tensor

    def __init__(self):
        super().__init__()
        self.add_state("ce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("counts", default=torch.tensor(0), dist_reduce_fx="sum")
        self.class_weights = torch.tensor([0.0853, 0.2436, 0.0622, 0.6089])

    def update(self, input: torch.Tensor, target: torch.Tensor):
        class_weights = self.class_weights.to(input.device)
        self.ce += torch.nn.functional.cross_entropy(
            input=input, 
            target=target,
            # weight=class_weights,
            reduction='sum',
        )
        self.counts += target.numel()

    def compute(self):
        return self.ce / self.counts

class TemperatureScaledSoftmax(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(TemperatureScaledSoftmax, self).__init__()
        self.temperature = temperature

    def forward(self, logits):
        scaled_logits = logits / self.temperature
        return torch.nn.functional.softmax(scaled_logits, dim=1)

class TemperatureScaledSigmoid(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(TemperatureScaledSigmoid, self).__init__()
        self.temperature = temperature

    def forward(self, logits):
        scaled_logits = logits / self.temperature
        return torch.nn.Sigmoid()(scaled_logits)

@dataclasses.dataclass(eq=False)
class Torch_Base(LightningModule):
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

        # assert np.log2(self.signal_window_size).is_integer(), 'Signal window must be power of 2'

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
class Torch_RCN_Mixin(Torch_Base):
    rcn_reservoir_size: int = 500  # Number of neurons in the reservoir
    rcn_spectral_radius: float = 0.9  # Reservoir stability parameter
    rcn_sparsity: float = 0.1  # Fraction of reservoir connections that are non-zero
    rcn_input_scaling: float = 1.0  # Scaling for input weights
    rcn_leaky_rate: float = 0.5  # Leak rate for reservoir state update

    def make_rcn_encoder(self, input_dim: int, output_dim: int) -> torch.nn.Module:
        """
        Constructs a Reservoir Computing Network (RCN) for encoding.
        
        Parameters:
        - input_dim: Dimensionality of the input features.
        - output_dim: Dimensionality of the output features.

        Returns:
        - An RCN model.
        """
        # class RCN(torch.nn.Module):
        #     def __init__(self, reservoir_size, spectral_radius, sparsity, input_scaling, leaky_rate):
        #         super().__init__()
        #         self.reservoir_size = reservoir_size
        #         self.spectral_radius = spectral_radius
        #         self.sparsity = sparsity
        #         self.input_scaling = input_scaling
        #         self.leaky_rate = leaky_rate

        #         # Input-to-Reservoir Weights
        #         self.input_weights = torch.randn(reservoir_size, input_dim) * input_scaling

        #         # Reservoir-to-Reservoir Weights
        #         self.reservoir_weights = torch.rand(reservoir_size, reservoir_size) - 0.5
        #         self.reservoir_weights[torch.rand_like(self.reservoir_weights) > sparsity] = 0
        #         eigenvalues, _ = torch.linalg.eig(self.reservoir_weights)
        #         max_eigenvalue = torch.abs(eigenvalues).max()
        #         self.reservoir_weights *= spectral_radius / max_eigenvalue

        #         # Reservoir state initialization
        #         self.reservoir_state = torch.zeros(reservoir_size)

        #     def forward(self, input_sequence):
        #         """
        #         Forward pass through the RCN.

        #         Parameters:
        #         - input_sequence: Tensor of shape (batch_size, time_steps, input_dim).

        #         Returns:
        #         - Outputs aggregated from the reservoir.
        #         """
        #         batch_size, time_steps, _ = input_sequence.size()
        #         outputs = []
        #         for t in range(time_steps):
        #             input_t = input_sequence[:, t, :]  # Shape: (batch_size, input_dim)
        #             input_t = input_t @ self.input_weights.T  # Map to reservoir
        #             reservoir_t = torch.tanh(input_t + self.reservoir_state @ self.reservoir_weights.T)
        #             self.reservoir_state = (1 - self.leaky_rate) * self.reservoir_state + self.leaky_rate * reservoir_t
        #             outputs.append(self.reservoir_state)

        #         outputs = torch.stack(outputs, dim=1)  # Shape: (batch_size, time_steps, reservoir_size)
        #         return outputs.mean(dim=1)  # Aggregate over time (e.g., mean or final state)


        class RCN(torch.nn.Module):
            def __init__(self, input_dim, reservoir_size, spectral_radius, sparsity, input_scaling, leaky_rate):
                super().__init__()
                self.input_dim = input_dim
                self.reservoir_size = reservoir_size
                self.spectral_radius = spectral_radius
                self.sparsity = sparsity
                self.input_scaling = input_scaling
                self.leaky_rate = leaky_rate

                # Input-to-Reservoir Weights
                self.input_weights = torch.nn.Parameter(
                    torch.randn(reservoir_size, input_dim) * input_scaling, requires_grad=False
                )

                # Reservoir-to-Reservoir Weights
                self.reservoir_weights = torch.nn.Parameter(
                    torch.rand(reservoir_size, reservoir_size) - 0.5, requires_grad=False
                )
                self.reservoir_weights[torch.rand_like(self.reservoir_weights) > sparsity] = 0
                eigenvalues, _ = torch.linalg.eig(self.reservoir_weights)
                max_eigenvalue = torch.abs(eigenvalues).max()
                self.reservoir_weights.data *= spectral_radius / max_eigenvalue

                # Reservoir state initialization (register as buffer for correct device placement)
                self.register_buffer("reservoir_state", torch.zeros(reservoir_size))

            def forward(self, input_sequence):
                # Ensure all tensors are on the same device
                self.reservoir_state = self.reservoir_state.to(input_sequence.device)
                self.input_weights = self.input_weights.to(input_sequence.device)
                self.reservoir_weights = self.reservoir_weights.to(input_sequence.device)

                batch_size, time_steps, _ = input_sequence.size()
                outputs = []

                # Process input sequence
                for t in range(time_steps):
                    input_t = input_sequence[:, t, :]  # Shape: (batch_size, input_dim)
                    input_t = input_t @ self.input_weights.T  # Map to reservoir
                    reservoir_t = torch.tanh(input_t + self.reservoir_state @ self.reservoir_weights.T)
                    self.reservoir_state = (1 - self.leaky_rate) * self.reservoir_state + self.leaky_rate * reservoir_t
                    outputs.append(self.reservoir_state)

                outputs = torch.stack(outputs, dim=1)  # Shape: (batch_size, time_steps, reservoir_size)
                return outputs.mean(dim=1)  # Aggregate over time (e.g., mean or final state)

        return RCN(
            input_dim=input_dim,
            reservoir_size=self.rcn_reservoir_size,
            spectral_radius=self.rcn_spectral_radius,
            sparsity=self.rcn_sparsity,
            input_scaling=self.rcn_input_scaling,
            leaky_rate=self.rcn_leaky_rate
        )

@dataclasses.dataclass(eq=False)
class Torch_MLP_Mixin(Torch_Base):
    mlp_layers: tuple = (64, 32)
    mlp_dropout: float = 0.1
    temperature: float = 1.0  # Default is 1, which means no scaling. Higher Temperature (T > 1) makes the probabilities more uniform. Lower Temperature (0 < T < 1) makes the output probabilities more extreme (either closer to 0 or 1), i.e. increases the model's confidence in its predictions.
    
    def make_mlp(
            self, 
            mlp_in_features: int, 
            mlp_out_features: int = 4, # number of classes
            output_activation: str = "logits", # Options: "sigmoid", "softmax", "logits"
    ) -> torch.nn.Module:
        # MLP layers
        print("Constructing MLP layers")
        mlp_layers = torch.nn.Sequential(torch.nn.Flatten())
        n_layers = len(self.mlp_layers)
        for i, layer_size in enumerate(self.mlp_layers):
            in_features = mlp_in_features if i==0 else self.mlp_layers[i-1]
            print(f"  MLP layer {i} with in/out features: {in_features}/{layer_size} (LeakyReLU activ.)")
            mlp_layers.extend([
                # torch.nn.Dropout(p=self.mlp_dropout) if i!=n_layers-1 else torch.nn.Identity(),
                torch.nn.Linear(
                    in_features=in_features,
                    out_features=layer_size,
                ),
                torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope) if i!=n_layers-1 else torch.nn.Identity(),
                # torch.nn.ReLU() if i!=n_layers-1 else torch.nn.Identity(),
            ])

        # output layer
        print(f"  MLP output layer with in/out features {self.mlp_layers[-1]}/{mlp_out_features} (no activ.)")
        mlp_layers.append(
            torch.nn.Linear(
                in_features=self.mlp_layers[-1], 
                out_features=mlp_out_features,
            )
        )

        # Determine output activation with temperature scaling
        if output_activation == "sigmoid":
            print(f"  Applying temperature-scaled sigmoid at MLP output")
            mlp_layers.append(TemperatureScaledSigmoid(self.temperature))
        elif output_activation == "softmax":
            print(f"  Applying temperature-scaled softmax at MLP output")
            mlp_layers.append(TemperatureScaledSoftmax(self.temperature))
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
    cnn_maxpool_time_size: Iterable|int = 4
    cnn_maxpool_spatial_size: Iterable|int = 3
    cnn_input_channels: int = 1
    cnn_dropout: float = 0.1
    # FFT CNN parameters
    fft_nlayers: int = 1
    fft_num_kernels: Iterable|int = 8
    fft_kernel_freq_size: Iterable|int = 5
    fft_kernel_spatial_size: Iterable = (3, 3)
    fft_stride_freq_size: Iterable|int = 1
    fft_padding: Iterable|int|str = 0
    fft_dropout: float = 0.1
    fft_maxpool_freq_size: Iterable|int = 1
    fft_maxpool_spatial_size: Iterable = (1, 1)
    # FFT paramters
    use_phase: bool = False 
    fft_subwindows: int = 1 # split signal into N subwindows and average later
    # fft_nbins: The number of bins into which each subwindow of the time dimension is divided for FFT analysis.
    # This allows for capturing frequency-domain features at different intervals within each subwindow.
    # Each bin will undergo its own FFT, and the results will be averaged to represent the subwindow.
    fft_nbins: int = 2 

    def __post_init__(self):
        super().__post_init__()
        self.subwindow_size = self.signal_window_size // self.fft_subwindows
        # nfft: The number of time points used for each individual FFT computation.
        # It is determined by dividing the subwindow size by the number of FFT bins (fft_nbins).
        # The choice of nfft affects the frequency resolution of the FFT.
        self.nfft = self.subwindow_size // self.fft_nbins
        # nfreqs: The number of unique frequency components expected from the FFT of a real-valued signal.
        # It includes the DC component (0 Hz) and half of the remaining frequency components,
        # as the spectrum is symmetric around the Nyquist frequency.
        self.nfreqs = self.nfft // 2 + 1

    def forward_with_fft(self, x):
        # FFT-based feature extraction
        batch_size, num_channels, time_dim, spatial_dim1, spatial_dim2 = x.shape

        # Conditionally allocate memory for fft_bins and fft_subwindows based on use_phase
        if self.use_phase:
            fft_bins = torch.empty((batch_size, self.fft_nbins, 2 * (self.nfreqs - 1), spatial_dim1, spatial_dim2), dtype=x.dtype, device=x.device)
            fft_subwindows = torch.empty((batch_size, self.fft_subwindows, 2 * (self.nfreqs - 1), spatial_dim1, spatial_dim2), dtype=x.dtype, device=x.device)
        else:
            fft_bins = torch.empty((batch_size, self.fft_nbins, self.nfreqs - 1, spatial_dim1, spatial_dim2), dtype=x.dtype, device=x.device)
            fft_subwindows = torch.empty((batch_size, self.fft_subwindows, self.nfreqs - 1, spatial_dim1, spatial_dim2), dtype=x.dtype, device=x.device)
        
        for i_subwindow, subwindow in enumerate(x.split(self.subwindow_size, dim=2)):
            for i_bin in range(self.fft_nbins):
                bin_data = subwindow[:, :, i_bin * self.nfft:(i_bin + 1) * self.nfft, :, :]
                fft_output = torch.fft.rfft(bin_data, dim=2)[:, :, 1:, :, :]  # Remove DC component

                magnitude = torch.abs(fft_output)
                fft_bins[:, i_bin:i_bin+1, :self.nfreqs-1, :, :] = magnitude ** 2

                if self.use_phase:
                    phase = torch.angle(fft_output)
                    normalized_phase = (phase + np.pi) / (2 * np.pi)
                    fft_bins[:, i_bin:i_bin+1, self.nfreqs-1:, :, :] = normalized_phase

            fft_subwindows[:, i_subwindow:i_subwindow+1, :, :, :] = torch.mean(fft_bins, dim=1, keepdim=True)

        # Apply logarithm to magnitude part, keep phase part as is
        # fft_subwindows[:, :, :self.nfreqs-1, :, :][fft_subwindows[:, :, :self.nfreqs-1, :, :] < 1e-5] = 1e-5
        # fft_subwindows[:, :, :self.nfreqs-1, :, :] = torch.log10(fft_subwindows[:, :, :self.nfreqs-1, :, :])
        
        fft_subwindows[fft_subwindows < 1e-5] = 1e-5        
        fft_subwindows = torch.log10(fft_subwindows)
        
        # Pass FFT features through the FFT-specific CNN encoder
        fft_features = self.cnn_encoder(fft_subwindows)

        return fft_features
    
    def forward_with_fpga_fft(self, x):
        # FFT-based feature extraction
        batch_size, num_channels, time_dim, spatial_dim1, spatial_dim2 = x.shape

        # Adjust for the combined spatial dimension and conditional memory allocation
        if self.use_phase:
            fft_bins = torch.empty((batch_size, self.fft_nbins, 2 * (self.nfreqs - 1), spatial_dim1 * spatial_dim2), dtype=x.dtype, device=x.device)
            fft_subwindows = torch.empty((batch_size, self.fft_subwindows, 2 * (self.nfreqs - 1), spatial_dim1 * spatial_dim2), dtype=x.dtype, device=x.device)
        else:
            fft_bins = torch.empty((batch_size, self.fft_nbins, self.nfreqs - 1, spatial_dim1 * spatial_dim2), dtype=x.dtype, device=x.device)
            fft_subwindows = torch.empty((batch_size, self.fft_subwindows, self.nfreqs - 1, spatial_dim1 * spatial_dim2), dtype=x.dtype, device=x.device)

        for i_subwindow, subwindow in enumerate(x.split(self.subwindow_size, dim=2)):
            for i_bin in range(self.fft_nbins):
                bin_data = subwindow[:, :, i_bin * self.nfft:(i_bin + 1) * self.nfft, :, :]
                fft_output = torch.fft.rfft(bin_data, dim=2)[:, :, 1:, :, :]  # Remove DC component
                # fft_output = CustomRFFT.apply(bin_data)  # Placeholder for onnx

                # Flatten spatial dimensions
                magnitude = torch.abs(fft_output).reshape(batch_size, -1, self.nfreqs - 1, spatial_dim1 * spatial_dim2)
                fft_bins[:, i_bin:i_bin+1, :self.nfreqs-1, :] = magnitude ** 2

                if self.use_phase:
                    phase = torch.angle(fft_output).reshape(batch_size, -1, self.nfreqs - 1, spatial_dim1 * spatial_dim2)
                    normalized_phase = (phase + np.pi) / (2 * np.pi)
                    fft_bins[:, i_bin:i_bin+1, self.nfreqs-1:, :] = normalized_phase

            fft_subwindows[:, i_subwindow:i_subwindow+1, :, :] = torch.mean(fft_bins, dim=1, keepdim=True)

        fft_subwindows[fft_subwindows < 1e-5] = 1e-5        
        fft_subwindows = torch.log10(fft_subwindows)
        
        # Reshape for 2D convolution input
        fft_subwindows = fft_subwindows.reshape(batch_size, self.fft_subwindows*(self.nfreqs - 1), spatial_dim1, spatial_dim2)
        fft_subwindows_reshaped = fft_subwindows.permute(0, 1, 3, 2).reshape(batch_size, 1, self.fft_subwindows*(self.nfreqs - 1), spatial_dim1 * spatial_dim2)

        # fft_subwindows_reshaped = fft_subwindows.reshape(batch_size, 1, self.fft_subwindows*(self.nfreqs - 1), spatial_dim1 * spatial_dim2) # use this
        # fft_subwindows_reshaped = fft_subwindows.reshape(batch_size, self.fft_subwindows, (self.nfreqs - 1), spatial_dim1 * spatial_dim2)

        # Pass FFT features through the FFT-specific CNN encoder
        fft_features = self.cnn_encoder(fft_subwindows_reshaped)

        return fft_features
    
    def reshape_signals(self, signals):
        batch_size, num_channels, time_dim, spatial_dim1, spatial_dim2 = signals.shape
        
        # Reshape for 2D convolution input
        signals = signals.reshape(batch_size, time_dim*spatial_dim1*spatial_dim2)

        return signals

    def reshape_signals_2(self, signals):
        batch_size, num_channels, time_dim, spatial_dim1 = signals.shape
        
        signals = signals.reshape(batch_size, time_dim*spatial_dim1)

        return signals
    
    def reshape_signals_rcn(self, signals):
        batch_size, num_channels, time_dim, spatial_dim1 = signals.shape

        # Flatten the feature dimensions (e.g., spatial_dim1)
        signals = signals.view(batch_size, time_dim, spatial_dim1)  # Shape: (batch_size, time_dim, input_dim)

        return signals

    def make_fft_cnn_encoder(self) -> tuple[torch.nn.Module, int, tuple]:
        # Generalize attributes to be iterables
        for attr_name in [
            'fft_num_kernels',
            'fft_kernel_freq_size',
            'fft_kernel_spatial_size',
            'fft_stride_freq_size',
            'fft_padding',
            'fft_maxpool_freq_size',
            'fft_maxpool_spatial_size',
        ]:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, Iterable) and not isinstance(attr_value, str):
                assert len(attr_value) == self.fft_nlayers, f"{attr_name} {attr_value}"
            else:
                new_attr_value = tuple([attr_value] * self.fft_nlayers)
                setattr(self, attr_name, new_attr_value)
            
        # Initialize the CNN layers
        cnn = torch.nn.Sequential()
        if self.use_phase:
            data_shape = (self.fft_subwindows, 2*(self.nfreqs-1), self.n_rows, self.n_cols)  # Adjust this based on your actual input shape
        else:
            data_shape = (self.fft_subwindows, self.nfreqs-1, self.n_rows, self.n_cols)  # Adjust this based on your actual input shape

        for i in range(self.fft_nlayers):
            kernel = (
                self.fft_kernel_freq_size[i],
                self.fft_kernel_spatial_size[i][0],
                self.fft_kernel_spatial_size[i][1],
            )
            padding = self.fft_padding[i]
            stride = (self.fft_stride_freq_size[i], 1, 1)
            # stride = (self.fft_kernel_freq_size[i], 1, 1)
            print(f"  FFT CNN Layer {i}")
            print(f"    Kernel {kernel}")
            print(f"    Stride {stride}")
            print(f"    Padding {padding}")
            conv3d = torch.nn.Conv3d(
                in_channels=self.fft_num_kernels[i-1] if i != 0 else self.fft_subwindows,
                out_channels=self.fft_num_kernels[i],
                kernel_size=kernel,
                padding=padding,
                stride=stride,
                groups=self.fft_subwindows,
            )

            # Add Batch Normalization after Convolution
            batch_norm = torch.nn.BatchNorm3d(num_features=self.fft_num_kernels[i])

            # Update data_shape based on the Conv3D layer
            data_shape = tuple(conv3d(torch.zeros(size=(1,) + data_shape)).size()[1:])
            
            # Add MaxPool3d layer
            maxpool = torch.nn.MaxPool3d(
                kernel_size=(
                self.fft_maxpool_freq_size[i],
                self.fft_maxpool_spatial_size[i][0],
                self.fft_maxpool_spatial_size[i][1],
                ), 
            )
            print(f"    Maxpool ({self.fft_maxpool_freq_size[i]}, {self.fft_maxpool_spatial_size[i][0]}, {self.fft_maxpool_spatial_size[i][1]})")

            # Update data_shape based on the maxpool layer
            data_shape = tuple(maxpool(torch.zeros(size=(1,) + data_shape)).size()[1:])

            cnn.extend([
                torch.nn.Dropout(p=self.fft_dropout),
                conv3d,
                batch_norm,
                torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
                maxpool,
            ])

        num_features = np.prod(data_shape)
        return cnn, num_features, data_shape
    
    def make_fpga_fft_cnn_encoder(self) -> tuple[torch.nn.Module, int, tuple]:
        # Adjust attributes for 2D convolution
        for attr_name in [
            'fft_num_kernels',
            'fft_kernel_freq_size',  # This will now be the height of the kernel
            'fft_kernel_spatial_size',  # This will be the width of the kernel
        ]:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, Iterable) and not isinstance(attr_value, str):
                assert len(attr_value) == self.fft_nlayers, f"{attr_name} {attr_value}"
            else:
                new_attr_value = tuple([attr_value] * self.fft_nlayers)
                setattr(self, attr_name, new_attr_value)

        # Initialize the CNN layers for 2D convolution
        cnn = torch.nn.Sequential()

        # Adjusted for 2D input shape. Either flatten subwindows or keep them as channels.
        data_shape = (1, self.fft_subwindows*(self.nfreqs - 1), self.n_rows * self.n_cols)  # The 1 here represents the single-channel input
        # data_shape = (self.fft_subwindows, (self.nfreqs - 1), self.n_rows * self.n_cols)  # number of subwindows are number of channels

        for i in range(self.fft_nlayers):
            kernel = (
                self.fft_kernel_freq_size[i],
                self.fft_kernel_spatial_size[i]
            )
            padding = 0  # Use 0 for 'valid' padding in PyTorch
            stride = (1, 1)
            print(f"  FFT CNN Layer {i}")
            print(f"    Kernel {kernel}")
            print(f"    Stride {stride}")
            print(f"    Padding {padding}")
            conv2d = torch.nn.Conv2d(
                in_channels=self.fft_num_kernels[i-1] if i != 0 else 1,
                # in_channels=self.fft_num_kernels[i-1] if i != 0 else self.fft_subwindows,
                out_channels=self.fft_num_kernels[i],
                kernel_size=kernel,
                padding=padding,
                stride=stride,
            )

            # Add Batch Normalization after Convolution
            batch_norm = torch.nn.BatchNorm2d(num_features=self.fft_num_kernels[i])

            # Dynamically compute the new data shape after each convolution
            with torch.no_grad():
                dummy_input = torch.zeros((1,) + data_shape)
                data_shape = conv2d(dummy_input).shape[1:]

            # Add MaxPool2d layer
            maxpool = torch.nn.MaxPool2d(kernel_size=(2,2))
            print(f"    Maxpool (2, 2)")

            # Dynamically compute the new data shape after pooling
            with torch.no_grad():
                dummy_input = torch.zeros((1,) + data_shape)
                data_shape = maxpool(dummy_input).shape[1:]

            cnn.extend([
                conv2d,
                # batch_norm,
                torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
                maxpool,
                torch.nn.Dropout(p=self.fft_dropout),
            ])

        # Adjusted to correctly reflect the flattened output dimensions after the final layer
        num_features = np.prod(data_shape[0:])  # Exclude the batch dimension

        return cnn, num_features, data_shape
    
    def make_cnn_encoder(self) -> tuple[torch.nn.Module,int,tuple]:
        for attr_name in [
            'cnn_num_kernels',
            'cnn_kernel_time_size',
            'cnn_kernel_spatial_size',
            'cnn_padding',
            'cnn_maxpool_time_size',
            'cnn_maxpool_spatial_size',
        ]:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, Iterable) and not isinstance(attr_value, str):
                assert len(attr_value) == self.cnn_nlayers, f"{attr_name} {attr_value}"
            else:
                # Ensuring that kernel sizes and other iterable attributes are tuples of appropriate length
                new_attr_value = tuple([attr_value]*self.cnn_nlayers) if attr_name != 'cnn_kernel_spatial_size' else (attr_value, attr_value)
                setattr(self, attr_name, new_attr_value)
                # new_attr_value = tuple([attr_value]*self.cnn_nlayers)
                # setattr(self, attr_name, new_attr_value)

        for time_dim in self.cnn_kernel_time_size:
            assert np.log2(time_dim).is_integer(), 'Kernel time dims must be power of 2'

        print("Constructing CNN layers")

        data_shape = (self.cnn_input_channels, self.signal_window_size, self.n_rows, self.n_cols)
        self.input_data_shape = tuple(data_shape)
        print(f"  Input data shape {data_shape}  (size {np.prod(data_shape)})")

        # CNN layers
        cnn = torch.nn.Sequential()
        for i in range(self.cnn_nlayers):
            kernel = (
                self.cnn_kernel_time_size[i],
                self.cnn_kernel_spatial_size[0] if len(self.cnn_kernel_spatial_size)>0 else self.cnn_kernel_spatial_size[i],
                self.cnn_kernel_spatial_size[1] if len(self.cnn_kernel_spatial_size)>0 else self.cnn_kernel_spatial_size[i],
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
            # print(f"    Output data shape: {data_shape}  (size {np.prod(data_shape)})")
            assert np.all(np.array(data_shape) >= 1), f"Bad data shape {data_shape} after CNN layer {i}"
             # Add MaxPool3d layer
            maxpool = torch.nn.MaxPool3d(
                kernel_size=(
                    self.cnn_maxpool_time_size[i],
                    self.cnn_maxpool_spatial_size[i][0],
                    self.cnn_maxpool_spatial_size[i][1],
                )
            )
            print(f"    Maxpool ({self.cnn_maxpool_time_size[i]}, {self.cnn_maxpool_spatial_size[i][0]}, {self.cnn_maxpool_spatial_size[i][1]})")
            # Update data_shape after MaxPooling
            data_shape = tuple(maxpool(torch.zeros(size=(1,) + data_shape)).size()[1:])
            print(f"    Output data shape: {data_shape}  (size {np.prod(data_shape)})")
            cnn.extend([
                torch.nn.Dropout(p=self.cnn_dropout),
                conv3d,
                torch.nn.BatchNorm3d(self.cnn_num_kernels[i]),
                torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
                maxpool,
            ])

        num_features = np.prod(data_shape)
        print(f"  CNN output features: {num_features}")

        return cnn, num_features, data_shape

    def make_2d_cnn_encoder(self) -> tuple[torch.nn.Module,int,tuple]:
        # Ensure all attributes are tuples of appropriate length.
        # Now that we only have one spatial dimension, many of these 
        # can be simplified compared to the 3D version.
        for attr_name in [
            'cnn_num_kernels',
            'cnn_kernel_time_size',
            'cnn_kernel_spatial_size',
            'cnn_padding',
            'cnn_maxpool_time_size',
            'cnn_maxpool_spatial_size',
        ]:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, Iterable) and not isinstance(attr_value, str):
                assert len(attr_value) == self.cnn_nlayers, f"{attr_name} {attr_value}"
            else:
                # For kernel and pooling sizes, we need tuples for each layer
                # In the old code, we replicated single values to match cnn_nlayers.
                # If you only have one spatial dimension, ensure spatial sizes are 
                # single integers (or convert them into a tuple of length 1 if needed).
                new_attr_value = tuple([attr_value]*self.cnn_nlayers)
                setattr(self, attr_name, new_attr_value)

        # Check kernel time sizes (if needed)
        for time_dim in self.cnn_kernel_time_size:
            assert np.log2(time_dim).is_integer(), 'Kernel time dims must be power of 2'

        print("Constructing 2D CNN layers")

        # Our data shape: (channels, time_dim, spatial_dim)
        data_shape = (self.cnn_input_channels, self.signal_window_size, 8)
        self.input_data_shape = data_shape
        print(f"  Input data shape {data_shape} (size {np.prod(data_shape)})")

        cnn = torch.nn.Sequential()
        input_channels = self.cnn_input_channels

        for i in range(self.cnn_nlayers):
            # Kernel and other parameters for this layer
            kernel = (self.cnn_kernel_time_size[i], self.cnn_kernel_spatial_size[i])
            stride = (self.cnn_kernel_time_size[i], 1)  # adjust as needed
            padding = (self.cnn_padding[i], self.cnn_padding[i]) if isinstance(self.cnn_padding[i], int) else self.cnn_padding[i]

            print(f"  CNN Layer {i}")
            print(f"    Kernel {kernel}")
            print(f"    Stride {stride}")
            print(f"    Padding {padding}")

            conv2d = torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=self.cnn_num_kernels[i],
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                padding_mode='reflect'
            )
            
            # Update data_shape after the conv layer.
            # Fake input to determine output shape
            with torch.no_grad():
                test_out = conv2d(torch.zeros((1,) + data_shape))
                data_shape = test_out.shape[1:]  # excluding batch dimension
                # data_shape is now (channels_out, time_out, spatial_out)

            # Now add the MaxPool2d layer
            maxpool_kernel = (self.cnn_maxpool_time_size[i], self.cnn_maxpool_spatial_size[i])
            print(f"    Maxpool {maxpool_kernel}")

            maxpool2d = torch.nn.MaxPool2d(kernel_size=maxpool_kernel)
            with torch.no_grad():
                test_out = maxpool2d(torch.zeros((1,) + data_shape))
                data_shape = test_out.shape[1:]

            print(f"    Output data shape: {data_shape} (size {np.prod(data_shape)})")

            # Add layers to the sequence
            cnn.extend([
                torch.nn.Dropout(p=self.cnn_dropout),
                conv2d,
                torch.nn.BatchNorm2d(self.cnn_num_kernels[i]),
                torch.nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
                maxpool2d,
            ])

            # Update input_channels for next layer
            input_channels = self.cnn_num_kernels[i]

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
class Lightning_Model(
    Torch_CNN_Mixin,
    Torch_MLP_Mixin,
    Torch_RCN_Mixin,
):
    encoder_lr: float = 1e-3
    decoder_lr: float = 1e-5
    lr_scheduler_patience: int = 20
    lr_scheduler_threshold: float = 1e-3
    weight_decay: float = 1e-6
    monitor_metric: str = 'sum_loss/val'
    log_dir: str = dataclasses.field(default='.', init=False)
    encoder_type: str = 'raw' # 'raw', 'fft', 'fpga_fft', 'both', 'none', 'rcn'
    # the following must be listed in `_frontend_names`
    reconstruction_decoder: bool = False
    classifier_mlp: bool = False
    multiclass_classifier_mlp: bool = False
    time_to_elm_mlp: bool = False
    velocimetry_mlp: bool = False
    separatrix_mlp: bool = False
    _frontend_names = ['reconstruction_decoder', 'classifier_mlp', 'multiclass_classifier_mlp', 'time_to_elm_mlp', 'velocimetry_mlp', 'separatrix_mlp']
    num_classes: int = 4 # number of classes for multiclass_classifier_mlp
    # capture outputs from penultimate layer to perform tSNE
    penultimate_outputs: list = dataclasses.field(default_factory=list)
    visualize_embeddings: bool = False
    save_test_data: bool = False
    n_rows: int = 6
    n_cols: int = 8

    def __post_init__(self):
        super().__post_init__()
        self.save_hyperparameters()

        print(f'Initiating {self.__class__.__name__}')
        class_fields_dict = {field.name: field for field in dataclasses.fields(self.__class__)}
        for field_name in dataclasses.asdict(self):
            value = getattr(self, field_name)
            field_str = f"  {field_name}: {value}"
            default_value = class_fields_dict[field_name].default
            if value != default_value:
                field_str += f" (default {default_value})"
            print(field_str)

        if self.encoder_type == 'raw':
            # self.cnn_encoder, cnn_features, cnn_output_shape = self.make_cnn_encoder()
            self.encoder, features, output_shape = self.make_2d_cnn_encoder()
        elif self.encoder_type == 'fft':
            self.encoder, features, output_shape = self.make_fft_cnn_encoder()
        elif self.encoder_type == 'fpga_fft':
            self.encoder, features, output_shape = self.make_fpga_fft_cnn_encoder()
        elif self.encoder_type == 'both':
            self.raw_cnn_encoder, raw_cnn_features, raw_cnn_output_shape = self.make_cnn_encoder()
            self.fft_cnn_encoder, fft_cnn_features, fft_cnn_output_shape = self.make_fft_cnn_encoder()
            features = raw_cnn_features + fft_cnn_features
        elif self.encoder_type == 'none':
            # self.cnn_encoder, cnn_features, cnn_output_shape = None, self.signal_window_size*self.n_rows*self.n_cols, None
            self.encoder, features, output_shape = None, self.signal_window_size*self.n_rows, None
        elif self.encoder_type == 'rcn':  # Add the RCN option
            self.encoder = self.make_rcn_encoder(
                input_dim=self.n_cols,  # Spatial dimension
                output_dim=self.rcn_reservoir_size,  # Reservoir size
            )
            # Output features of the RCN encoder
            features = self.rcn_reservoir_size
        else:
            raise ValueError("Invalid encoder_type")

        # `frontends` for regression, classification, and self-supervised learning
        self.frontends = torch.nn.ModuleDict()
        self.frontends_active = {}
        for frontend_key in self._frontend_names:
            if getattr(self, frontend_key) is True:
                self.frontends_active[frontend_key] = True
                if 'mlp' in frontend_key:
                    if 'time_to_elm' in frontend_key:
                        setattr(self, f"{frontend_key}_mse_loss", torchmetrics.MeanSquaredError())
                        setattr(self, f"{frontend_key}_r2_score", torchmetrics.R2Score())
                    elif 'classifier' in frontend_key:
                        if 'multiclass' in frontend_key:
                            new_module = self.make_mlp(mlp_in_features=features, mlp_out_features=self.num_classes, output_activation="softmax")
                            self.frontends.update({frontend_key: new_module})
                            setattr(self, f"{frontend_key}_ce_loss", CrossEntropy())
                            setattr(self, f"{frontend_key}_f1_score", torchmetrics.F1Score(task='multiclass',  num_classes=self.num_classes, average='macro'))
                            # Initialize per-class F1 scores
                            self.per_class_f1_scores = torch.nn.ModuleList([
                                torchmetrics.F1Score(num_classes=1, average='none', task='binary') for _ in range(self.num_classes)  # Assuming 4 classes
                            ])
                            # setattr(self, f"{frontend_key}_accuracy", torchmetrics.Accuracy(task='multiclass',  num_classes=4))
                        else:
                            setattr(self, f"{frontend_key}_bce_loss", BCEWithLogit())
                            setattr(self, f"{frontend_key}_f1_score", torchmetrics.F1Score(task='binary'))
                    elif 'velocimetry' in frontend_key:
                        # Create a simple MLP for velocimetry with multiple outputs
                        # vZ_shear_profile is np.gradient(vZ.mean(axis=1), dR, axis=1) where vZ has dim (time, poloidal, radial)
                        # new_module = self.make_mlp(mlp_in_features=cnn_features, mlp_out_features=1 * self.n_cols, output_activation="linear")
                        new_module = self.make_mlp(mlp_in_features=features, mlp_out_features=1, output_activation="linear")
                        self.frontends.update({frontend_key: new_module})
                        for label in ['vZ']:
                            setattr(self, f"{frontend_key}_{label}_mse_loss", torchmetrics.MeanSquaredError())
                            setattr(self, f"{frontend_key}_{label}_r2_score", torchmetrics.R2Score(num_outputs=1))
                    elif 'separatrix' in frontend_key:
                        # Create a simple MLP to identify 6 closest (R,Z) points on separatrix
                        new_module = self.make_mlp(mlp_in_features=features, mlp_out_features=2 * 6, output_activation="linear")
                        self.frontends.update({frontend_key: new_module})
                        setattr(self, f"{frontend_key}_mse_loss", torchmetrics.MeanSquaredError())
                        setattr(self, f"{frontend_key}_r2_score", torchmetrics.R2Score())
                    else:
                        raise KeyError
                elif 'decoder' in frontend_key:
                    new_module = self.make_cnn_decoder(input_data_shape=output_shape)
                    self.frontends.update({frontend_key: new_module})
                    if 'reconstruction' in frontend_key:
                        setattr(self, f"{frontend_key}_mse_loss", torchmetrics.MeanSquaredError())
                    else:
                        raise KeyError
                else:
                    raise KeyError
            
        # cnn_parameters = sum(p.numel() for p in self.cnn_encoder.parameters() if p.requires_grad)
        # print(f"  CNN encoder parameters {cnn_parameters:,}")
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters {total_parameters:,}")

        encoder_parameters = 0  # Default value if no encoder is present
        if getattr(self, 'encoder', None) is not None:
            encoder_parameters = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            print(f"  Encoder parameters: {encoder_parameters:,}")
            
        # if hasattr(self, 'raw_cnn_encoder'):
        #     raw_cnn_parameters = sum(p.numel() for p in self.raw_cnn_encoder.parameters() if p.requires_grad)
        #     print(f"  Raw CNN encoder parameters {raw_cnn_parameters:,}")

        # if hasattr(self, 'fft_cnn_encoder'):
        #     fft_cnn_parameters = sum(p.numel() for p in self.fft_cnn_encoder.parameters() if p.requires_grad)
        #     print(f"  FFT CNN encoder parameters {fft_cnn_parameters:,}")

        # if hasattr(self, 'rcn_encoder'):
        #     rcn_parameters = sum(p.numel() for p in self.rcn_encoder.parameters() if p.requires_grad)
        #     print(f"  RCN encoder parameters {rcn_parameters:,}")

        for key, frontend_key in self.frontends.items():
            n_parameters = sum(p.numel() for p in frontend_key.parameters() if p.requires_grad)
            print(f"  Frontend `{key}` parameters: {n_parameters:,}")

        # self.example_input_array = torch.zeros(
        #     (1, 1, self.signal_window_size, self.n_rows, self.n_cols), 
        #     dtype=torch.float32,
        # )
        self.example_input_array = torch.zeros(
            (1, 1, self.signal_window_size, self.n_rows), 
            dtype=torch.float32,
        )

        self.initialize_layers()

    def visualize_fft_features(self, x, num_top_freq_indices=3, subwindow_idx=0):
        with torch.no_grad():
            # Perform FFT-based feature extraction
            batch_size, num_channels, time_dim, spatial_dim1, spatial_dim2 = x.shape
            fft_bins = torch.empty((batch_size, self.fft_nbins, self.nfreqs - 1, spatial_dim1, spatial_dim2), dtype=x.dtype, device=x.device)
            fft_subwindows = torch.empty((batch_size, self.fft_subwindows, self.nfreqs - 1, spatial_dim1, spatial_dim2), dtype=x.dtype, device=x.device)
        
            for i_subwindow, subwindow in enumerate(x.split(self.subwindow_size, dim=2)):
                for i_bin in range(self.fft_nbins):
                    bin_data = subwindow[:, :, i_bin * self.nfft:(i_bin + 1) * self.nfft, :, :]
                    fft_output = torch.fft.rfft(bin_data, dim=2)[:, :, 1:, :, :]  # Remove DC component

                    magnitude = torch.abs(fft_output)
                    fft_bins[:, i_bin:i_bin+1, :self.nfreqs-1, :, :] = magnitude ** 2

                fft_subwindows[:, i_subwindow:i_subwindow+1, :, :, :] = torch.mean(fft_bins, dim=1, keepdim=True)

            fft_subwindows[fft_subwindows < 1e-5] = 1e-5        
            fft_subwindows = torch.log10(fft_subwindows)

            # Now working with fft_subwindows instead of fft_features
            subwindow_features = fft_subwindows[:, subwindow_idx, :, :, :]
            print("Subwindow features shape:", subwindow_features.shape)  # Debugging step 2

            # Getting the top frequency indices based on magnitude
            magnitude = subwindow_features.abs()
            top_freq_indices = torch.topk(magnitude.reshape(magnitude.shape[0], -1), num_top_freq_indices, dim=1)[1]

            # Ensuring the frequency indices are within the valid range
            top_freq_indices = torch.clamp(top_freq_indices, 0, self.nfreqs - 2)

            # Visualization for each index in top_freq_indices
            for i in range(num_top_freq_indices):
                freq_index = top_freq_indices[0, i]  # Assuming visualization of the first item in the batch
                self.plot_contour(subwindow_features[0, freq_index], title=f"Subwindow {subwindow_idx}, Freq Index {freq_index}_{i}")

    def plot_contour(self, data, title):
        """
        Plot a contour map for the given data.
        
        Args:
            data: 2D Tensor representing the spatial distribution of magnitudes.
            title: Title for the plot.
        """
        # Ensure the data is at least 2D and matches the spatial grid dimensions
        if data.ndim == 1:
            # Reshape or expand the 1D data to 2D
            data = data.view(self.n_rows, self.n_cols)  # Adjust dimensions as needed

        if data.shape[0] < 2 or data.shape[1] < 2:
            # Handle cases where reshaped data is still not (2, 2) or larger
            raise ValueError("Data for contour plot must be at least (2, 2) in shape.")

        data_np = data.cpu().numpy()
        plt.figure(figsize=(10, 8))
        X, Y = np.meshgrid(range(data_np.shape[1]), range(data_np.shape[0]))  # Spatial grid
        plt.contourf(X, Y, np.flip(data_np, axis=0), levels=20, cmap="cubehelix")
        plt.title(title)
        plt.xlabel('Radial')
        plt.ylabel('Poloidal')
        plt.colorbar(label='Log Magnitude')
        filepath = os.path.join(self.log_dir, title)
        print(f"Saving figures {filepath}")
        plt.savefig(filepath)

    def configure_optimizers(self):
        # Define the different learning rates
        encoder_lr = self.encoder_lr  
        mlp_lr = self.decoder_lr 

        # Parameters with their respective learning rates
        # param_groups = [
        #     # {'params': self.cnn_encoder.parameters(), 'lr': cnn_encoder_lr},
        #     {'params': self.frontends["multiclass_classifier_mlp"].parameters(), 'lr': mlp_classifier_lr},
        # ]
        # if hasattr(self, 'rcn_encoder'):
        #     parameters = self.rcn_encoder.parameters()
        # elif hasattr(self, 'raw_cnn_encoder'):
        #     parameters = self.cnn_encoder.parameters()

        # param_groups = [
        #     {'params': self.encoder.parameters(), 'lr': encoder_lr},
        #     {'params': self.frontends["velocimetry_mlp"].parameters(), 'lr': mlp_lr},
        #     # {'params': self.frontends["separatrix_mlp"].parameters(), 'lr': mlp_classifier_lr},

        # ]
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=mlp_lr,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=0.5,
            patience=self.lr_scheduler_patience,
            threshold=self.lr_scheduler_threshold,
            min_lr=1e-7,
            mode='min' if 'loss' in self.monitor_metric else 'max',
        )
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'monitor': self.monitor_metric,
        }

    def forward(self, signals: torch.Tensor) -> dict[str, torch.Tensor]:
        results = {}
        
        if self.encoder_type == 'raw':
            features = self.encoder(signals)
        elif self.encoder_type == 'fft':
            features = self.forward_with_fft(signals)
        elif self.encoder_type == 'fpga_fft':
            features = self.forward_with_fpga_fft(signals)
        elif self.encoder_type == 'both':
            raw_features = self.raw_cnn_encoder(signals)
            fft_features = self.forward_with_fft(signals)
            print("Raw features shape:", raw_features.shape)
            print("FFT features shape:", fft_features.shape)
            features = torch.cat([raw_features, fft_features], dim=1)  # Concatenate along the feature dimension
        elif self.encoder_type == 'none':
            # features = self.reshape_signals(signals)
            features = self.reshape_signals_2(signals)
        elif self.encoder_type == 'rcn':
            reshaped_signals = self.reshape_signals_rcn(signals)
            # Process signals through the RCN encoder
            features = self.encoder(reshaped_signals)
        else:
            raise ValueError("Invalid encoder_type")
        
        for frontend_key, frontend in self.frontends.items():
            if self.frontends_active[frontend_key]:
                results[frontend_key] = frontend(features)
                
        return results
    
    def update_step(self, batch, batch_idx) -> torch.Tensor:
        signals, labels, _ = batch
        results = self(signals)
        sum_loss = None

        # Define metric suffixes for each task
        metric_suffices_dict = {
            'time_to_elm': ['mse_loss', 'r2_score'],
            'reconstruction': ['mse_loss'],
            'classifier_multiclass': ['ce_loss', 'f1_score'],
            'classifier_binary': ['bce_loss', 'f1_score'],
            'velocimetry': ['mse_loss', 'r2_score'],
            'separatrix': ['mse_loss', 'r2_score'],
        }

        for frontend_key, frontend_is_active in self.frontends_active.items():
            if not frontend_is_active:
                continue

            frontend_result = results[frontend_key]

            # Determine the type of task and corresponding metrics
            if 'time_to_elm' in frontend_key:
                metric_suffices = metric_suffices_dict['time_to_elm']
                target = labels
            elif 'reconstruction' in frontend_key:
                metric_suffices = metric_suffices_dict['reconstruction']
                target = signals
            elif 'classifier' in frontend_key:
                if 'multiclass' in frontend_key:
                    metric_suffices = metric_suffices_dict['classifier_multiclass']
                    target = labels
                else:
                    metric_suffices = metric_suffices_dict['classifier_binary']
                    target = labels  # class_labels
            elif 'velocimetry' in frontend_key:
                metric_suffices = metric_suffices_dict['velocimetry']
            elif 'separatrix' in frontend_key:
                metric_suffices = metric_suffices_dict['separatrix']
            else:
                raise ValueError(f"Unknown frontend_key: {frontend_key}")

            for metric_suffix in metric_suffices:
                if 'velocimetry' in frontend_key:
                    # For velocimetry tasks, assume single output per sample: frontend_result shape: (batch, 1)
                    # labels shape: (batch,)
                    # Squeeze the output to match labels' shape: (batch,)
                    frontend_result_squeezed = frontend_result.float().squeeze(-1)
                    target = labels.float()
                    # Handle velocimetry task
                    # frontend_result = frontend_result.reshape(-1, 2, 8, 8)
                    for idx, label_key in enumerate(['vZ']):
                        metric_name = f"{frontend_key}_{label_key}_{metric_suffix}"
                        metric = getattr(self, metric_name)

                        # Ensure shapes match, both (batch,)
                        assert frontend_result_squeezed.shape == target.shape, \
                            f"Shape mismatch: preds {frontend_result_squeezed.shape}, target {target.shape}"

                        metric_value = metric(frontend_result_squeezed, target)
                        
                        if 'loss' in metric_name:
                            sum_loss = metric_value if sum_loss is None else sum_loss + metric_value
                elif 'separatrix' in frontend_key:
                    # Handle separatrix task
                    frontend_result = frontend_result.reshape(-1, 6, 2)
                    metric_name = f"{frontend_key}_{metric_suffix}"
                    metric = getattr(self, metric_name)
                    target_flat = labels.float().flatten()
                    frontend_result_flat = frontend_result.float().flatten()

                    metric_value = metric(frontend_result_flat, target_flat)

                    if 'loss' in metric_name:
                        sum_loss = metric_value if sum_loss is None else sum_loss + metric_value
                else:
                    # Handle other tasks
                    metric_name = f"{frontend_key}_{metric_suffix}"
                    metric = getattr(self, metric_name)

                    if 'multiclass' in metric_name:
                        target = target.squeeze()
                        if 'f1_score' in metric_name:
                            target = target.argmax(dim=1) if target.dim() == 2 else target
                        metric_value = metric(frontend_result, target)
                    else:
                        metric_value = metric(frontend_result, target)

                    if 'loss' in metric_name:
                        sum_loss = metric_value if sum_loss is None else sum_loss + metric_value

        return sum_loss

    def compute_log_reset(self, stage: str):
        sum_loss = None
        for frontend_key, frontend_is_active in self.frontends_active.items():
            if frontend_is_active is False:
                continue
            if 'time_to_elm' in frontend_key or 'velocimetry' in frontend_key or 'separatrix' in frontend_key:
                metric_suffices = ['mse_loss', 'r2_score']
            elif 'reconstruction' in frontend_key:
                metric_suffices = ['mse_loss']
            elif 'classifier' in frontend_key:
                if 'multiclass' in frontend_key:
                    metric_suffices = ['ce_loss', 'f1_score']
                else:
                    metric_suffices = ['bce_loss', 'f1_score']
            else:
                raise ValueError
            for metric_suffix in metric_suffices:
                if 'velocimetry' in frontend_key:
                    for label in ['vZ']:
                        metric_name = f"{frontend_key}_{label}_{metric_suffix}"
                        metric: torchmetrics.Metric = getattr(self, metric_name)
                        metric_value = metric.compute()
                        self.log(f"{metric_name}/{stage}", metric_value, sync_dist=True)
                else:
                    metric_name = f"{frontend_key}_{metric_suffix}"
                    metric: torchmetrics.Metric = getattr(self, metric_name)
                    metric_value = metric.compute()
                    self.log(f"{metric_name}/{stage}", metric_value, sync_dist=True)
                if 'loss' in metric_name:
                    sum_loss = metric_value if sum_loss is None else sum_loss + metric_value
                metric.reset()
        self.log(f"sum_loss/{stage}", sum_loss, sync_dist=True)

    def on_fit_start(self) -> None:
        self.t_fit_start = time.time()

    def on_train_epoch_start(self):
        self.t_train_epoch_start = time.time()
        print('The CPU usage is: ', psutil.cpu_percent(4))
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

        print(f"Epoch {self.current_epoch} start")

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self.update_step(batch, batch_idx)

    def on_train_batch_end(self, *args, **kwargs):
        self.compute_log_reset(stage='train')

    def on_train_epoch_end(self) -> None:
        for name, param in self.named_parameters():
            if 'weight' not in name:
                continue
            values = param.data.detach()
            mean = torch.mean(values).item()
            std = torch.std(values).item()
            z_scores = (values-mean)/std
            skew = torch.mean(z_scores**3).item()
            kurt = torch.mean(z_scores**4).item()
            self.log(f"param_mean/{name}", mean, sync_dist=True)
            self.log(f"param_std/{name}", std, sync_dist=True)
            self.log(f"param_skew/{name}", skew, sync_dist=True)
            self.log(f"param_kurt/{name}", kurt, sync_dist=True)
        print(f"Epoch {self.current_epoch} elapsed train time: {(time.time()-self.t_train_epoch_start)/60:0.1f} min")

    def on_validation_epoch_start(self) -> None:
        self.t_val_epoch_start = time.time()

    def validation_step(self, batch, batch_idx) -> None:
        self.update_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        self.compute_log_reset(stage='val')
        print(f"Epoch {self.current_epoch} elapsed valid. time: {(time.time()-self.t_val_epoch_start)/60:0.1f} min")

    def on_fit_end(self) -> None:
        print(f"Fit elapsed time {(time.time()-self.t_fit_start)/60:0.1f} min")

    def capture_penultimate_output(self, module, input, output):
        self.penultimate_outputs.append(output.cpu().detach().numpy())
    
    def on_test_start(self) -> None:
        if 'velocimetry_mlp' in self.frontends_active and self.save_test_data:
            self.collected_data = []

        if self.visualize_embeddings:
            self.penultimate_outputs.clear()  # Clear any previous outputs
            self.labels = []  # Initialize the list for storing labels
            # Register the hook on the penultimate layer
            self.handle = self.frontends['multiclass_classifier_mlp'][7].register_forward_hook(self.capture_penultimate_output)
        self.t_test_start = time.time()

    def test_step(self, batch, batch_idx) -> None:
        signals, labels, time_points = batch
        results = self(signals)

        if 'multiclass_classifier_mlp' in results:
            frontend_result = results['multiclass_classifier_mlp']
        
            # Convert probabilities to class indices
            class_preds = frontend_result.argmax(dim=1)  # Assuming frontend_result is [batch_size, num_classes]

            # Prepare target labels in a similar manner as you typically do
            target = labels.squeeze()  # Adjust shape if necessary
            class_labels = target.argmax(dim=1) if target.dim() == 2 else target  # Convert one-hot to indices if needed

            for i, f1_metric in enumerate(self.per_class_f1_scores):
                # For per-class F1 calculation, compare binary representations
                per_class_preds = (class_preds == i).int()  # Binary predictions for each class
                per_class_labels = (class_labels == i).int()  # Binary labels for each class
                f1_metric.update(per_class_preds, per_class_labels)
        
        # old approach for saving inference data
        if 'velocimetry_mlp' in results and self.save_test_data:
            # Reshape to [batch_size, 4, 8, 8] assuming the order is [vR, vR_std, vZ, vZ_std]
            # frontend_result = results['velocimetry_mlp'].detach().view(-1, 2, 8, 8)
            frontend_result = results['velocimetry_mlp'].detach().view(-1, self.n_cols)
            # components = ['vZ', 'vZ_std']
            components = ['vZ_shear_profile']
            # Create a dictionary to store predictions and actual labels
            processed_data = {}
            # for i, component in enumerate(components):
            #     processed_data[f"{component}_pred"] = frontend_result[:, i, :, :].cpu().numpy()
            #     processed_data[f"{component}_true"] = labels[component].cpu().numpy()

            processed_data[f"vZ_shear_profile_pred"] = frontend_result[:, :].cpu().numpy()
            processed_data[f"vZ_shear_profile_true"] = labels.cpu().numpy()

            processed_data['signals'] = signals.cpu()
            processed_data['time_points'] = time_points.cpu()
            
            self.collected_data.append((batch_idx, processed_data))

        self.update_step(batch, batch_idx)
        if self.visualize_embeddings:
            self.labels.extend(labels.cpu().numpy())

    def on_test_epoch_end(self)-> None:
        if 'multiclass_classifier_mlp' in self.frontends_active:
            per_class_f1 = [f1_metric.compute() for f1_metric in self.per_class_f1_scores]
            for i, f1_score in enumerate(per_class_f1):
                self.log(f'per_class_f1_class_{i}', f1_score, on_step=False, on_epoch=True, sync_dist=True)
            if self.visualize_embeddings:
                # Remove the hook
                self.handle.remove()

                # Convert to numpy arrays
                all_outputs = np.concatenate(self.penultimate_outputs, axis=0)
                all_labels = np.array(self.labels)

                all_labels = np.argmax(all_labels, axis=1)

                # Perform t-SNE visualization
                tsne_2d = TSNE(n_components=2, random_state=0)
                tsne_results_2d = tsne_2d.fit_transform(all_outputs)

                tsne_3d = TSNE(n_components=3, random_state=0)
                tsne_results_3d = tsne_3d.fit_transform(all_outputs)

                # 2D Visualization
                fig_2d = plt.figure(figsize=(16, 16))
                scatter_2d = plt.scatter(tsne_results_2d[:, 0], tsne_results_2d[:, 1], c=all_labels, alpha=0.5, cmap='Set1')
                plt.colorbar(scatter_2d, ticks=range(4))
                plt.title('2D t-SNE Visualization')
                plt.grid(True)

                fig_3d = plt.figure(figsize=(16, 16))
                ax_3d = fig_3d.add_subplot(111, projection='3d')
                scatter_3d = ax_3d.scatter(tsne_results_3d[:, 0], tsne_results_3d[:, 1], tsne_results_3d[:, 2], c=all_labels, alpha=0.5, cmap='Set1')
                plt.colorbar(scatter_3d, ticks=range(4))
                plt.title('3D t-SNE Visualization')
                
                # Save and log the figure
                filename = '21033900_tsne_penultimate_layer'
                filepath = os.path.join(self.log_dir, filename)
                plt.savefig(filepath+'.pdf', format='pdf', transparent=True)
                plt.savefig(filepath+'.png', format='png', transparent=True)
                plt.savefig(filepath+'_3d.pdf', format='pdf', transparent=True)
                plt.savefig(filepath+'_3d.png', format='png', transparent=True)

                # Save tsne_results and all_labels to a .pkl file
                with open(filepath+'.pkl', 'wb') as f:
                    pickle.dump({'tsne_results_2d': tsne_results_2d, 'tsne_results_3d': tsne_results_3d, 'labels': all_labels}, f)

                # Logging part
                for logger in self.loggers:
                    if isinstance(logger, loggers.TensorBoardLogger):
                        logger.experiment.add_figure(f"tsne/{filename}", fig_2d, close=False)
                        logger.experiment.add_figure(f"tsne/{filename}", fig_3d, close=False)
                    elif isinstance(logger, loggers.WandbLogger):
                        logger.log_image(key='tsne', images=[filepath+'.png'])

        # old approach
        if 'velocimetry_mlp' in self.frontends_active and self.save_test_data:
            print("Number of data batches collected:", len(self.collected_data))
            save_filename = 'velocimetry_inference_data.hdf5'
            data_filepath = os.path.join(self.log_dir, save_filename)
            # Open or create an HDF5 file
            with h5py.File(data_filepath, 'a') as f:
                for batch_idx, data_tuple in enumerate(self.collected_data):
                    _, data = data_tuple  # Unpack the tuple to get the data dictionary
                    group_name = f'batch_{batch_idx}'
                    if group_name in f:
                        print(f"Group {group_name} already exists. Overwriting...")
                        del f[group_name]  # Remove the existing group to avoid conflicts
                    grp = f.create_group(group_name)
                    for key, value in data.items():
                        # Ensure value is a suitable format for HDF5
                        if isinstance(value, (np.ndarray, list)):  # Ensure lists are converted to arrays
                            value = np.array(value)
                        grp.create_dataset(key, data=value, compression="gzip")
            print(f"Velocimetry test data successfully saved to HDF5 at {data_filepath}.")

        self.compute_log_reset(stage='test')
        print(f"Test elapsed time {(time.time()-self.t_test_start)/60:0.1f} min")

    def on_predict_start(self):
        """
        Called at the start of the predict loop.
        Initializes containers to store predictions, labels, and time points.
        """
        self.predictions = []
        self.true_labels = []
        self.time_points = []
        self.shot_ids = []
        self.radial_positions = []

    def predict_step(self, batch, batch_idx):
        """
        Called for each batch during prediction.
        Perform inference and store predictions, true labels, time points, and shot IDs.
        """
        signals, labels, time_points, shot_ids, radial_positions = batch        
        results = self(signals)
        predictions = results["velocimetry_mlp"].detach().cpu().numpy()  # Assuming 'velocimetry_mlp' is used
        predictions = np.squeeze(predictions)  # Ensure shape is (batch_size,) if needed

        self.predictions.append(predictions)
        self.true_labels.append(labels.cpu().numpy().squeeze())
        self.time_points.append(time_points.cpu().numpy().squeeze())
        self.shot_ids.append(shot_ids)  # Collect shot IDs for later grouping
        self.radial_positions.append(radial_positions.cpu().numpy().squeeze())

    def on_predict_end(self):
        """
        Called at the end of the predict loop. Aggregates and saves predictions vs truth,
        as well as optional HDF5 saving for further analysis.
        """
        if self.trainer.is_global_zero:  # Only execute on the main process
            print("Aggregating predictions...")

            # Combine predictions, labels, and times
            predictions = np.concatenate(self.predictions, axis=0)  # Shape: (total_windows, n_cols)
            true_labels = np.concatenate(self.true_labels, axis=0)  # Shape: (total_windows, n_cols)
            times = np.concatenate(self.time_points, axis=0)        # Shape: (total_windows,)
            shots = np.concatenate(self.shot_ids, axis=0)           # Shape: (total_windows,)
            r_positions = np.concatenate(self.radial_positions, axis=0)  # shape: (N,)

            print(f"Predictions shape: {predictions.shape}")
            print(f"True labels shape: {true_labels.shape}")
            print(f"Times shape: {times.shape}")
            print(f"Shots shape: {shots.shape}")
            print(f"Radial positions shape: {r_positions.shape}")  # New

            # Ensure the log directory exists
            if not os.path.exists(self.log_dir):
                print(f"Creating log directory at {self.log_dir}")
                os.makedirs(self.log_dir)

            # Create an HDF5 file for storing predictions
            hdf5_filepath = os.path.join(self.log_dir, "predictions.hdf5")
            print(f"Saving predictions to HDF5 file at {hdf5_filepath}")
            with h5py.File(hdf5_filepath, "w") as h5_file:
                unique_shots = np.unique(shots)
                for shot_id in unique_shots:
                    shot_mask = shots == shot_id
                    shot_group = h5_file.create_group(str(shot_id))
                    shot_group.create_dataset(
                        "predictions",
                        data=predictions[shot_mask],
                        compression="gzip",
                        chunks=True
                    )
                    shot_group.create_dataset(
                        "true_labels",
                        data=true_labels[shot_mask],
                        compression="gzip",
                        chunks=True
                    )
                    shot_group.create_dataset(
                        "times",
                        data=times[shot_mask],
                        compression="gzip",
                        chunks=True
                    )
                    # Optionally store radial positions as well
                    shot_group.create_dataset(
                        "radial_positions",
                        data=r_positions[shot_mask],
                        compression="gzip",
                        chunks=True
                    )
                    print(f"Saved data for shot {shot_id} to HDF5 file, predicting done.")

            # Create individual plots for each shot
            # unique_shots = np.unique(shots)
            # for shot_id in unique_shots:
            #     shot_mask = shots == shot_id
            #     print(f"Plotting for shot {shot_id}")
            #     self.plot_predictions_vs_truth(
            #         predictions=predictions[shot_mask],
            #         true_labels=true_labels[shot_mask],
            #         times=times[shot_mask],
            #         shot_id=shot_id,
            #     )

    def plot_predictions_vs_truth(self, predictions, true_labels, times, shot_id):
        """
        Generate a plot comparing model predictions to true labels for a specific shot and save the figures.
        """

        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in range(predictions.shape[1]):  # Iterate over radial points
            ax.plot(
                times,
                2*np.pi*true_labels[:, col],
                label=f"True Radial {col+1}",
                linestyle="--",
                linewidth=1.5,
                marker='o',
                markersize=3,
            )
            ax.plot(
                times,
                2*np.pi*predictions[:, col],
                label=f"Pred Radial {col+1}",
                linestyle="-",
                linewidth=2.0,
                marker='o',
                markersize=3,
            )

        # Set dynamic y-limits
        ymin = min(np.min(predictions), np.min(true_labels))
        ymax = max(np.max(predictions), np.max(true_labels))
        margin = 0.1 * (ymax - ymin)
        # ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_ylim(-150, 150)

        # Customize the plot
        ax.set_title(f"Shot {shot_id}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(r"$v_{BES}^Z$ [km/s]")
        ax.legend()
        ax.grid()

        # Save the figure
        filename = f"predictions_vs_truth_shot_{shot_id}"
        filepath = os.path.join(self.log_dir, filename)
        plt.savefig(filepath + ".png", format="png", bbox_inches="tight", transparent=True, dpi=150)
        plt.savefig(filepath + ".pdf", format="pdf", bbox_inches="tight", transparent=True, dpi=150)
        plt.savefig(filepath + ".jpg", format="jpg", bbox_inches="tight", transparent=True, dpi=150)

        # Log the plot to TensorBoard and WandB
        for logger in self.loggers:
            if isinstance(logger, loggers.TensorBoardLogger):
                logger.experiment.add_figure(f"predictions/{filename}", fig, close=False)
            elif isinstance(logger, loggers.WandbLogger):
                logger.log_image(key=f"predictions/{filename}", images=[filepath + ".png"])

        plt.close(fig)
        

    def save_inference_data(self, data_list, filename):
        data_filepath = os.path.join(self.log_dir, filename)
        with h5py.File(data_filepath, 'w') as f:
            for idx, data in enumerate(data_list):
                group_name = f'batch_{idx}'
                grp = f.create_group(group_name)
                for key, value in data.items():
                    # Ensure value is converted to NumPy array
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    elif isinstance(value, list):
                        value = np.array(value)
                    grp.create_dataset(key, data=value, compression="gzip")

    def manual_predict_separatrix(self, test_dataloader, test_dataset, debug=False, save_filename=None, plot_inference=False):
        print(f"Starting manual_predict ... debug mode: {debug} ... save data? {save_filename}")
        world_size = self.trainer.world_size
        rank = self.trainer.global_rank
        local_labels = {}
        local_predictions = {}
        
        self.eval()  # set the model to evaluation mode

        with torch.no_grad():  # disable gradient computation
            for batch_idx, batch in enumerate(test_dataloader):
                batch_signals, batch_labels, shot_numbers, time_points = batch

                # Move data to the appropriate device
                batch_signals = batch_signals.to(self.device)
                batch_labels = batch_labels.to(self.device)
                shot_numbers = shot_numbers.to(self.device)
                time_points = time_points.to(self.device)

                # Forward pass
                batch_predictions = self(batch_signals)

                # Process 'separatrix_mlp' data
                if 'separatrix_mlp' in batch_predictions:
                    for element_idx in range(batch_signals.size(0)):
                        shot_number = shot_numbers[element_idx].item()
                        time_point = time_points[element_idx].item()
                        prediction = batch_predictions['separatrix_mlp'][element_idx].cpu().numpy()
                        label = batch_labels[element_idx].cpu().numpy()

                        # Use (shot_number, time_point) as the key
                        key = (shot_number, time_point)
                        if key not in local_labels:
                            local_labels[key] = []
                            local_predictions[key] = []

                        local_labels[key].append(label)
                        local_predictions[key].append(prediction)
                else:
                    print("Warning: 'separatrix_mlp' not found in batch_predictions.")
                    continue  # Skip if 'separatrix_mlp' is not in predictions

        # Collecting data from all processes
        all_labels = [None for _ in range(world_size)]
        all_predictions = [None for _ in range(world_size)]

        dist.all_gather_object(all_labels, local_labels)
        dist.all_gather_object(all_predictions, local_predictions)

        if rank == 0:
            # Aggregate data from all processes
            aggregated_labels = {}
            aggregated_predictions = {}

            for proc_labels in all_labels:
                for key, labels in proc_labels.items():
                    if key not in aggregated_labels:
                        aggregated_labels[key] = []
                    aggregated_labels[key].extend(labels)

            for proc_predictions in all_predictions:
                for key, preds in proc_predictions.items():
                    if key not in aggregated_predictions:
                        aggregated_predictions[key] = []
                    aggregated_predictions[key].extend(preds)

            # Now, organize data by shot_number and time_point
            data_by_shot = {}
            for (shot_number, time_point), labels in aggregated_labels.items():
                preds = aggregated_predictions[(shot_number, time_point)]
                shot_str = str(shot_number)
                time_point_str = f"{time_point:.3f}"

                if shot_str not in data_by_shot:
                    data_by_shot[shot_str] = {}

                if time_point_str not in data_by_shot[shot_str]:
                    data_by_shot[shot_str][time_point_str] = {
                        'coords_preds': [],
                        'coords_true': []
                    }

                data_by_shot[shot_str][time_point_str]['coords_preds'].extend(preds)
                data_by_shot[shot_str][time_point_str]['coords_true'].extend(labels)

            # Save data to HDF5
            if save_filename:
                data_filepath = os.path.join(self.log_dir, save_filename)
                print(f"Saving labels and predictions to {data_filepath}")
                with h5py.File(data_filepath, 'w') as f:
                    for shot_str, time_points_data in data_by_shot.items():
                        shot_group = f.create_group(shot_str)
                        for time_point_str, data_dict in time_points_data.items():
                            # Replace periods in time_point_str to comply with HDF5 group naming rules
                            time_group_name = time_point_str.replace('.', '_')
                            time_group = shot_group.create_group(time_group_name)
                            coords_preds = np.array(data_dict['coords_preds'])
                            coords_true = np.array(data_dict['coords_true'])
                            time_group.create_dataset('coords_preds', data=coords_preds, compression="gzip")
                            time_group.create_dataset('coords_true', data=coords_true, compression="gzip")
                print(f"Separatrix test data successfully saved to HDF5 at {data_filepath}.")

        else:
            pass  # Non-zero ranks don't need to do anything else

    def manual_predict(self, test_dataloader, test_dataset, debug=False, save_filename=None, plot_inference=False):
        print(f"Starting manual_predict ... debug mode: {debug} ... save data? {save_filename}")
        world_size = self.trainer.world_size
        rank = self.trainer.global_rank
        local_labels = {}
        local_predictions = {}
        local_confinement_ids = []
        
        self.eval()  # set the model to evaluation mode
        if debug:
            batch_limit = 1000
        else:
            batch_limit = len(test_dataloader)

        print_interval = 100

        with torch.no_grad():  # disable gradient computation
            for batch_idx, (batch_signals, batch_labels, confinement_mode_ids) in enumerate(test_dataloader):
                if batch_idx >= batch_limit:
                    break

                # if (batch_idx + 1) % print_interval == 0:
                #     print(f'batch_idx: {batch_idx}')
                #     print('The CPU usage is: ', psutil.cpu_percent(4))
                #     # Getting % usage of virtual_memory ( 3rd field)
                #     print('RAM memory % used:', psutil.virtual_memory()[2])
                #     # Getting usage of virtual_memory in GB ( 4th field)
                #     print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

                batch_signals = batch_signals.to(self.device)  # Move to GPU
                batch_predictions = self(batch_signals)  # Forward pass

                for element_idx, c_id in enumerate(confinement_mode_ids.cpu().numpy()):
                    c_id_hashable = tuple(c_id)  # Convert to tuple
                    if c_id_hashable not in local_labels:
                        local_labels[c_id_hashable] = []
                        local_predictions[c_id_hashable] = []

                    indexed_label = (batch_idx, element_idx, batch_labels[element_idx].cpu().numpy())
                    indexed_prediction = (batch_idx, element_idx, batch_predictions['multiclass_classifier_mlp'][element_idx].cpu().numpy())
                    # indexed_prediction = (batch_idx, element_idx, batch_predictions['separatrix_mlp'][element_idx].cpu().numpy())

                    local_labels[c_id_hashable].append(indexed_label)
                    local_predictions[c_id_hashable].append(indexed_prediction)
                    
                local_confinement_ids.extend(confinement_mode_ids.cpu().numpy())
            local_confinement_ids = np.array(local_confinement_ids)

        # Collecting from all processes to rank 0
        all_labels = [{} for _ in range(world_size)]
        all_predictions = [{} for _ in range(world_size)]
        all_confinement_ids = [[] for _ in range(world_size)]

        # Check the type and size of local_labels and all_labels for debugging
        if rank == 0:
            print("Before all_gather - local_labels:", type(local_labels), len(local_labels))
            print("After all_gather - all_labels:", type(all_labels), len(all_labels))

        # Use all_gather or gather based on your needs to collect data
        dist.all_gather_object(all_labels, local_labels)
        dist.all_gather_object(all_predictions, local_predictions)
        dist.all_gather_object(all_confinement_ids, local_confinement_ids)

        if rank == 0:
            # aggregate gathered all_labels, all_predictions, all_confinement_ids
            aggregated_labels = {}
            aggregated_predictions = {}
            aggregated_confinement_ids = []
            
            for i in range(world_size):
                aggregated_confinement_ids.extend(all_confinement_ids[i])
                
                for k, v in all_labels[i].items():
                    if k not in aggregated_labels:
                        aggregated_labels[k] = []
                    aggregated_labels[k].extend(v)
                # print(f"After aggregating data from world {i}, aggregated_labels has {len(aggregated_labels)} keys.")
                    
                for k, v in all_predictions[i].items():
                    if k not in aggregated_predictions:
                        aggregated_predictions[k] = []
                    aggregated_predictions[k].extend(v)

            for c_id in aggregated_labels.keys():
                # print(f"Converting aggregated_labels and aggregated_predictions for c_id = {c_id}")
                
                # Sort based on batch_idx and element_idx
                sorted_labels = sorted(aggregated_labels[c_id], key=lambda x: (x[0], x[1]))
                sorted_predictions = sorted(aggregated_predictions[c_id], key=lambda x: (x[0], x[1]))
                
                # Extract the actual label and prediction data for concatenation
                actual_labels = [x[2] for x in sorted_labels]
                actual_predictions = [x[2] for x in sorted_predictions]
                                
                # Now, concatenate or stack them
                aggregated_labels[c_id] = np.concatenate(actual_labels)
                aggregated_predictions[c_id] = np.stack(actual_predictions)

            aggregated_labels_by_shot = defaultdict(list)
            aggregated_predictions_by_shot = defaultdict(list)

            for c_id, labels in aggregated_labels.items():
                shot_num = self.extract_event_metadata(c_id)[0]
                aggregated_labels_by_shot[shot_num].extend(labels)
                aggregated_predictions_by_shot[shot_num].extend(aggregated_predictions[c_id])

            # Save to a .pkl file
            if save_filename: 
                data_filepath = os.path.join(self.log_dir, save_filename)
                print(f"Saving labels and predictions to {data_filepath}")
                with open(data_filepath, 'wb') as f:
                    pickle.dump({
                        'aggregated_confinement_ids': aggregated_confinement_ids,
                        'aggregated_labels': aggregated_labels,
                        'aggregated_preds': aggregated_predictions,
                        'aggregated_labels_by_shot': aggregated_labels_by_shot,
                        'aggregated_preds_by_shot': aggregated_predictions_by_shot
                    }, f)
            
            matplotlib.rcParams['font.family'] = "sans-serif"
            class_labels=['L', 'L->H', 'H', 'H->L', 'QH', 'QH->WP', 'WPQH']
            # class_labels=['L-mode', 'LH-transition within 100 ms', 'H-mode', 'HL-transition within 100 ms']
            # class_labels=['QH-mode', 'Transition to WP within 100 ms', 'WPQH-mode']
            self.plot_and_log_metrics(
                all_labels=aggregated_labels,
                all_predictions=aggregated_predictions,
                all_labels_by_shot=aggregated_labels_by_shot,
                all_predictions_by_shot=aggregated_predictions_by_shot,
                class_labels=class_labels,
            )
            if plot_inference:
                settings = {
                    'window': 'hann',
                    'nperseg': 'auto',
                    'detrend': 'linear',
                    'color_limits': [1e-04, 1e-02], 
                    'colormap': 'viridis', #'coolwarm'
                    'colorscale': 'log',
                    'do_bandpass': False,
                    'do_comb': True,
                    'overlap': 0.5,
                    'freq_limits': [2.5, 200],
                    'numtaps': 501,
                    'use_transfer_function': False
                }
                self.plot_multiclass_classifier_inference(
                    all_confinement_ids=aggregated_confinement_ids, 
                    all_labels=aggregated_labels, 
                    all_predictions=aggregated_predictions, 
                    dataset=test_dataset,
                    settings=settings,
                    class_labels=class_labels,
                )

    def shot_accuracy(self, aggregated_labels, aggregated_predictions, threshold=0.9):
        """
        Calculates the proportion of shots with accuracy above a given threshold.
        """
        
        unique_shots = list(aggregated_labels.keys())
        above_threshold_count = 0  # count of shots with accuracy above the threshold
        
        for shot in unique_shots:
            shot_true_labels = np.array(aggregated_labels[shot])
            shot_true_labels = np.argmax(shot_true_labels, axis=-1)
            shot_pred_labels = np.array(aggregated_predictions[shot])
            shot_pred_labels = np.argmax(shot_pred_labels, axis=-1)

            shot_accuracy = np.mean((shot_true_labels == shot_pred_labels).astype(int))
            
            if shot_accuracy >= threshold:  # adjust this threshold as needed
                above_threshold_count += 1
                
        proportion_above_threshold = above_threshold_count / len(unique_shots)
        
        return proportion_above_threshold
    
    def extract_event_metadata(self, confinement_id):
        """
        Extracts the shot number, start time, and unique id from a confinement_id.
        e.g. (19603714200,) becomes '196037', 1420, '196037'/'1420'
        """
        # Extract string from tuple
        confinement_id_str = str(confinement_id[0])
        shot = confinement_id_str[:6]
        start_time = int(confinement_id_str[6:]) # in ms
        formatted_unique_id = f"{shot}/{start_time}"

        # start_time = int(confinement_id_str[6:-1]) # in ms
        # label = confinement_id_str[-1] # 0,1,2,3
        # formatted_unique_id = f"{shot}_{start_time}_{label}"

        # Split string and get shot number
        return shot, start_time, formatted_unique_id
        # return shot, start_time, label, formatted_unique_id

    def plot_multiclass_classifier_inference(self, all_confinement_ids, all_labels, all_predictions, dataset, settings, class_labels):
        i_page = 1
        fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(2, 2, width_ratios=[9, 1])

        unique_confinement_ids = np.unique(all_confinement_ids, axis=0)
        print(f"Unique Confinement IDs: {unique_confinement_ids.shape}")
        # class_labels = ['QH-mode', 'Transition to WP within 100 ms', 'WPQH-mode']
        # colors = ['#ed717e', '#b56ae8', '#68afb1', '#f3ff2c']
        colors = ['#ed717e', '#b56ae8', '#f3ff2c']

        for i_confinement_mode, unique_id in enumerate(unique_confinement_ids):
            print(f"Confinement Mode: {i_confinement_mode}, {unique_id}")
            unique_id_tuple = tuple(unique_id)

            # Extract the parts from the string and fetch the full signal
            shot, start_time, formatted_unique_id = self.extract_event_metadata(unique_id) 
            signal = dataset.get_full_signal_by_id(formatted_unique_id)

            # only plot events >= 100 ms
            if signal is None or signal.shape[0] < 1e3:
                continue

            # Calculate the spectrogram
            fs = 1000.
            freqs, times, Sxx = self.calculate_spectrogram(signal, fs, settings)
            if times.size == 0 or freqs.size == 0 or Sxx.size == 0:
                print("One of the arrays is empty.")
                continue

            if i_confinement_mode % 2 == 0 and i_confinement_mode != 0:
                i_page += 1
                plt.close(fig)
                fig = plt.figure(figsize=(16, 16))
                gs = gridspec.GridSpec(2, 2, width_ratios=[9, 1])

            row = i_confinement_mode % 2
            gs_inner = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[row, 0], height_ratios=[0.25, 2, 4, 0.25], hspace=0.0)

            ax1 = fig.add_subplot(gs_inner[0])  # Prediction subplot
            ax_prob = fig.add_subplot(gs_inner[1])  # Raw probabilities subplot
            ax2 = fig.add_subplot(gs_inner[2])  # Main spectrogram plot
            ax3 = fig.add_subplot(gs_inner[3])  # Ground truth subplot

            end_time = start_time + signal.shape[0] / 1000
            time_array = np.linspace(start_time, end_time, signal.shape[0])

            # Adjust times to align with the signal
            times = time_array[0] + times

            # Plot the spectrogram on ax2
            ax2.pcolormesh(times, freqs, Sxx, shading='gouraud', cmap=settings['colormap'],
                        norm=mcolors.LogNorm(vmin=settings['color_limits'][0], vmax=settings['color_limits'][1]))
            ax2.set_ylabel('Frequency (kHz)', fontsize=16, fontweight='bold')
            
            # Fetch labels and predictions corresponding to this confinement mode
            labels = all_labels.get(unique_id_tuple, np.array([]))
            predictions = all_predictions.get(unique_id_tuple, np.array([]))

            argmax_predictions = np.argmax(predictions, axis=1)
            argmax_labels = np.argmax(labels, axis=1)

            # Align the time array with the predictions array for prediction is for backward-looking signal window
            prediction_start_index = time_array.shape[0] - predictions.shape[0]
            aligned_time_array = time_array[prediction_start_index:]

            assert argmax_labels.shape[0] == argmax_predictions.shape[0]

            print(f"times: {times[0]}, {times[-1]}, {times.shape}")
            print(f"time_array: {time_array[0]}, {time_array[-1]}, {time_array.shape}")
            print(f"Length of signal: {signal.shape[0]}")
            print(f"Length of time_array: {time_array.shape[0]}")
            print(f"Number of predictions: {argmax_predictions.shape[0]}")
            print(f"Number of labels: {argmax_labels.shape[0]}")

            if argmax_predictions.shape[0] > time_array.shape[0]:
                print(f"skipping {unique_id} as predictions are longer than time array")
                continue

            # Plot background color for predicted classes on ax1
            for i in range(argmax_predictions.shape[0] - 1):
                t_start = aligned_time_array[i]
                t_end = aligned_time_array[i + 1]
                ax1.axvspan(t_start, t_end, facecolor=colors[argmax_predictions[i]], alpha=0.55)

            # Plot raw probabilities on ax_prob
            for i, label in enumerate(class_labels):
                ax_prob.plot(aligned_time_array, predictions[:, i], label=label, color=colors[i], alpha=0.75)
            ax_prob.set_ylabel('P(regime | BES)', fontsize=16, fontweight='bold')

            # Shade the ground truth subplot (ax3)
            for i in range(argmax_labels.shape[0] - 1):
                t_start = aligned_time_array[i]
                t_end = aligned_time_array[i + 1]
                ax3.axvspan(t_start, t_end, facecolor=colors[argmax_labels[i]], alpha=0.55)

            ax1.set_xlim(time_array[0], time_array[-1])
            ax_prob.set_xlim(time_array[0], time_array[-1])
            ax2.set_xlim(time_array[0], time_array[-1])
            ax3.set_xlim(time_array[0], time_array[-1])

            # Set up the axes, legends, and titles
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title(f"Classification Result for Discharge {shot} at time {start_time} ms", fontsize=12)
            ax1.text(0.5, 0.5, 'Classifications', fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontweight='bold')

            ax_prob.set_xticks([])

            ax2.set_xticks([])
            ax2.set_ylim(settings['freq_limits'])

            ax3.set_yticks([])
            ax3.set_xlabel('Time (ms)', fontsize=16, fontweight='bold')
            legend_handles = [ax1.scatter([], [], color=colors[idx], label=f'{label}') for idx, label in enumerate(class_labels)]
            ax3.legend(handles=legend_handles, loc='lower left', markerscale=2, fontsize=12)
            ax3.text(0.5, 0.5, 'Ground Truth', fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, fontweight='bold')

            if i_confinement_mode % 2 == 1 or i_confinement_mode == unique_confinement_ids.shape[0] - 1:
                plt.tight_layout()
                filename = f'inference_{i_page:02d}'
                filepath = os.path.join(self.log_dir, filename)
                print(f"Saving figures {filepath}")
                plt.savefig(filepath + '.pdf', format='pdf', transparent=True, dpi=150)
                plt.savefig(filepath + '.png', format='png', transparent=True, dpi=150)
                for logger in self.loggers:
                    if isinstance(logger, loggers.TensorBoardLogger):
                        logger.experiment.add_figure(f"inference/{filename}", fig, close=False)
                    elif isinstance(logger, loggers.WandbLogger):
                        logger.log_image(key='inference', images=[filepath + '.png'])
                plt.close(fig)

    def calculate_spectrogram(self, signals, fs, settings):
        # Ensure signals is a numpy array
        if isinstance(signals, torch.Tensor):
            signals = signals.cpu().numpy()  # Convert tensor to numpy array

        if isinstance(fs, torch.Tensor):
            fs = fs.cpu().numpy()

        nperseg = settings['nperseg'] if settings['nperseg'] != 'auto' else int(np.sqrt(2 * len(signals)))
        noverlap = int(settings['overlap'] * nperseg)

        if len(signals) < nperseg:
            print("Warning: Signal length is shorter than nperseg.")
            nperseg = len(signals)

        freqs, times, Sxx = spectrogram(
            signals, fs=fs, window=settings['window'],
            nperseg=nperseg, noverlap=noverlap, detrend=settings['detrend'],
            scaling='spectrum', mode='magnitude'
        )

        if 'freq_limits' in settings:
            freq_mask = (freqs >= settings['freq_limits'][0]) & (freqs <= settings['freq_limits'][1])
            Sxx = Sxx[freq_mask, :]
            freqs = freqs[freq_mask]

        return freqs, times, Sxx
    
    def fast_rolling_window_majority(self, arr, window_size):
        # Ensure the array is a numpy array
        arr = np.asarray(arr)
        
        # Pad array with minimal value if not divisible by window_size
        if arr.size % window_size:
            pad_size = window_size - arr.size % window_size
            arr = np.append(arr, np.full(pad_size, np.min(arr)))
        
        # Reshape the array to have a window on each row
        reshaped = arr.reshape(-1, window_size)
        
        # Use NumPy's bincount to find the most frequent value in each window
        majority_values = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=reshaped)
        
        return majority_values

    def plot_and_log_metrics(self,
                            all_labels: dict,
                            all_predictions: dict,
                            all_labels_by_shot,
                            all_predictions_by_shot,
                            class_labels: list = ['QH-mode', 'Transition to WP within 100 ms', 'WPQH-mode'],
                            ):

        # Convert lists to NumPy arrays for easier manipulation
        # Flattening all labels and all predictions into a single NumPy array
        all_labels = np.concatenate([v for v in all_labels.values()])
        all_predictions_prob = np.concatenate([v for v in all_predictions.values()])
        all_argmax_predictions = np.argmax(all_predictions_prob, axis=1)

        # one-hot labels
        all_labels = np.argmax(all_labels, axis=1)
        # all_labels_by_shot = np.argmax(all_labels_by_shot, axis=1)

        # Identify NaNs in Labels and update corresponding predictions to NaN
        nan_indices = np.isnan(all_labels)
        all_argmax_predictions = np.where(nan_indices, np.nan, all_argmax_predictions)

        # Check for NaNs in Labels and Probabilities
        if nan_indices.any():
            print("Warning: NaN values found in 'all_labels'")
        if np.isnan(all_predictions_prob).any():
            print("Warning: NaN values found in 'all_predictions_prob'")
        
        # Check Array Shapes
        print("Shape of all_labels:", all_labels.shape)
        print("Shape of all_predictions_prob:", all_predictions_prob.shape)

        # Ensure that the two arrays have compatible shapes
        if all_labels.shape[0] != all_predictions_prob.shape[0]:
            print("Warning: Shape mismatch between 'all_labels' and 'all_predictions_prob'")

        # Ensure there's at least one positive and one negative sample for each class
        for i in range(all_predictions_prob.shape[1]):
            if np.sum(all_labels == i) == 0 or np.sum(all_labels != i) == 0:
                print(f"Warning: Not enough samples for class {i} to compute ROC curve")        
                
        # Calculate overall accuracy, precision, and recall, excluding NaNs
        valid_indices = ~np.isnan(all_labels) & ~np.isnan(all_argmax_predictions)
        accuracy = accuracy_score(all_labels[valid_indices], all_argmax_predictions[valid_indices])
        precision = precision_score(all_labels[valid_indices], all_argmax_predictions[valid_indices], average='weighted', zero_division=0)
        recall = recall_score(all_labels[valid_indices], all_argmax_predictions[valid_indices], average='weighted', zero_division=0)

        print(f"Overall accuracy: {accuracy:.3f}")
        print(f"Overall precision: {precision:.3f}")
        print(f"Overall recall: {recall:.3f}")

        # Log these metrics
        for logger in self.loggers:
            if isinstance(logger, loggers.TensorBoardLogger):
                logger.experiment.add_scalar('summary/accuracy', accuracy)
                logger.experiment.add_scalar('summary/precision', precision)
            elif isinstance(logger, loggers.WandbLogger):
                logger.log_metrics({
                    'summary/accuracy': accuracy,
                    'summary/precision': precision,
                })

        # Calculate per-class precision, recall, and F1 score, excluding NaNs
        precision_per_class = precision_score(all_labels[valid_indices], all_argmax_predictions[valid_indices], average=None, zero_division=0)
        recall_per_class = recall_score(all_labels[valid_indices], all_argmax_predictions[valid_indices], average=None, zero_division=0)
        f1_per_class = f1_score(all_labels[valid_indices], all_argmax_predictions[valid_indices], average=None, zero_division=0)

        # Print or log per-class metrics
        for i, (prec, rec, f1) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            print(f"Class {i} - Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

            # Log these metrics
            for logger in self.loggers:
                if isinstance(logger, loggers.TensorBoardLogger):
                    logger.experiment.add_scalar(f'class_{i}/precision', prec)
                    logger.experiment.add_scalar(f'class_{i}/recall', rec)
                    logger.experiment.add_scalar(f'class_{i}/f1_score', f1)
                elif isinstance(logger, loggers.WandbLogger):
                    logger.log_metrics({
                        f'class_{i}/precision': prec,
                        f'class_{i}/recall': rec,
                        f'class_{i}/f1_score': f1,
                    })

        # Create a summary figure with Confusion Matrix, ROC curve, and PR curve
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))  # Adjusted for 3 subplots

        # List of marker styles
        marker_styles = ['o', '^', 's', 'D', 'P', '*', 'v', '<', '>', 'p', 'H', '+', 'x', '|']
        linestyles = ['solid', 'dotted', 'dashed', 'dashdot']

        # Plot confusion matrix on ax1
        cm = confusion_matrix(all_labels, all_argmax_predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cax = ax1.matshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(cax, ax=ax1)
        # ax1.set_title('Confusion Matrix', fontsize=16)
        ax1.set_xlabel('Predicted Class', fontsize=12)
        ax1.set_ylabel('True Class', fontsize=12)
        ax1.set_xticks(range(len(class_labels)))
        ax1.set_yticks(range(len(class_labels)))
        ax1.set_xticklabels(class_labels, fontsize=12)
        ax1.set_yticklabels(class_labels, fontsize=12)

        # Code for adding text annotations
        thresh = np.mean(cm_normalized)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax1.text(j, i, f"{cm[i, j]:d}\n({cm_normalized[i, j]*100:.1f}%)",
                    horizontalalignment="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=12)

        # Plot ROC curve on ax2
        optimal_tpr_fpr_per_class = []
        worst_distance = 0
        worst_point_annotation = ''  # Annotation text for the worst point
        worst_point_coordinates = (1, 0)  # Coordinates for the worst point
        for i, label in enumerate(class_labels):
            fpr, tpr, _ = roc_curve(all_labels == i, np.array(all_predictions_prob)[:, i])
            roc_auc = auc(fpr, tpr)

            # Find the optimal point for this class
            optimal_idx = np.argmin(np.sqrt(np.square(1-tpr) + np.square(fpr)))
            optimal_fpr, optimal_tpr = fpr[optimal_idx], tpr[optimal_idx]
            optimal_tpr_fpr_per_class.append((optimal_tpr, optimal_fpr))

            ax2.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})', linestyle=linestyles[i % len(linestyles)])
            # ax2.plot(optimal_fpr, optimal_tpr, marker='*', markersize=10)

            # Check if this is the worst point so far
            distance = np.sqrt(np.square(1-optimal_tpr) + np.square(optimal_fpr))
            if distance > worst_distance:
                worst_distance = distance
                worst_point_annotation = f'(Class: {label})\nTPR: {optimal_tpr:.2f}, FPR: {optimal_fpr:.2f}'
                worst_point_coordinates = (optimal_fpr, optimal_tpr)
                ax2.plot(optimal_fpr, optimal_tpr, marker='*', markersize=10)

            # Log the optimal points
            for logger in self.loggers:
                if isinstance(logger, loggers.TensorBoardLogger):
                    logger.experiment.add_scalar(f'class_{i}/optimal_tpr', optimal_tpr)
                    logger.experiment.add_scalar(f'class_{i}/optimal_fpr', optimal_fpr)
                elif isinstance(logger, loggers.WandbLogger):
                    logger.log_metrics({
                        f'class_{i}/optimal_tpr': optimal_tpr,
                        f'class_{i}/optimal_fpr': optimal_fpr,
                    })

        # Annotate the worst optimal point
        ax2.plot(0, 1, marker='*', markersize=12, color='black') # most optimal point
        # ax2.annotate(worst_point_annotation, worst_point_coordinates, textcoords="offset points", xytext=(10,-10), ha='center', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.65)
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.tick_params(axis='both', which='major', direction="in", labelsize=12)
        ax2.set_xlim(0, 0.2)
        ax2.set_ylim(0.9, 1)
        ax2.legend(fontsize=12)
        # ax2.legend(prop={'size': 16})
        # ax2.set_title('Receiver Operating Characteristic')

        # Plot PR curve on ax3
        # Create an inset_axes in ax2 for the PR curve
        # ax3 = inset_axes(ax2, width="40%", height="40%", loc='lower right', borderpad=2)
        worst_distance = 0
        worst_point_annotation = ''  # Annotation text for the worst point
        worst_point_coordinates = (0, 0)  # Coordinates for the worst point
        for i, label in enumerate(class_labels):
            precision, recall, _ = precision_recall_curve(all_labels == i, np.array(all_predictions_prob)[:, i])

            # Find the optimal point for this class
            optimal_idx = np.argmin(np.sqrt(np.square(1-precision) + np.square(1-recall)))
            optimal_recall, optimal_precision = recall[optimal_idx], precision[optimal_idx]

            ax3.plot(recall, precision, label=f'{label}', linestyle=linestyles[i % len(linestyles)])
            # ax3.plot(optimal_recall, optimal_precision, marker='*', markersize=10)

            # Check if this is the worst point so far
            distance = np.sqrt(np.square(1-optimal_precision) + np.square(1-optimal_recall))
            if distance > worst_distance:
                worst_distance = distance
                worst_point_annotation = f'(Class: {label})\nPrecision: {optimal_precision:.2f}, Recall: {optimal_recall:.2f}'
                worst_point_coordinates = (optimal_recall, optimal_precision)
                ax3.plot(optimal_recall, optimal_precision, marker='*', markersize=10)

        # Annotate the worst optimal point
        ax3.plot(1, 1, marker='*', markersize=12, color='black') # most optimal point
        # ax3.annotate(worst_point_annotation, worst_point_coordinates, textcoords="offset points", xytext=(10,-10), ha='center', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.65)
        ax3.set_xlabel('Recall', fontsize=12)
        ax3.set_ylabel('Precision', fontsize=12)
        ax3.tick_params(axis='both', which='major', direction="in", labelsize=12)
        ax3.legend(prop={'size': 12})
        # ax3.set_title('Precision-Recall Curve')

        # # Compute proportions for each threshold
        # thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        # proportions = [self.shot_accuracy(all_labels_by_shot, all_predictions_by_shot, threshold=t) for t in thresholds]

        # # Plot Threshold-Proportion curve on ax4
        # ax4.plot(thresholds, proportions, marker='^', color='darkred', linestyle='-')
        # ax4.grid(True, linestyle='--', alpha=0.65)
        # ax4.set_xlabel('Threshold for Classification Accuracy', fontsize=12)
        # ax4.set_ylabel('Proportion of Test Discharges Correctly Classified', fontsize=12)
        # ax4.axvline(x=0.75, linestyle='--', color='teal')

        plt.tight_layout()

        # Save and log the figure
        filename = 'summary_metrics'
        filepath = os.path.join(self.log_dir, filename)
        plt.savefig(filepath+'.png', format='png', bbox_inches='tight', transparent=True, dpi=150)
        plt.savefig(filepath+'.pdf', format='pdf', bbox_inches='tight', transparent=True, dpi=150)
        plt.savefig(filepath+'.jpg', format='jpg')

        for logger in self.loggers:
            if isinstance(logger, loggers.TensorBoardLogger):
                logger.experiment.add_figure(f"summary/{filename}", fig, close=False)
            elif isinstance(logger, loggers.WandbLogger):
                logger.log_image(key='summary', images=[filepath+'.png'])

        plt.close(fig)
    