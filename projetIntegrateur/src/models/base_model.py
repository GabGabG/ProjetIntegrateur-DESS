import torch
import torch.nn as nn
from typing import Union
import numpy as np


class SpeckleCNN(nn.Module):
    def __init__(self, input_sizes: tuple = (128, 128), out_channels: tuple = (16,),
                 kernel_sizes: Union[tuple, int] = 3, strides: Union[tuple, int] = 1, paddings: Union[tuple, int] = 1,
                 mlp_out: int = 64, padding_mode: str = "reflect"):
        super().__init__()

        dims = (1, *out_channels)
        n_convs = len(out_channels)
        if isinstance(kernel_sizes, int):
            kernel_sizes = (kernel_sizes,) * n_convs
        if isinstance(strides, int):
            strides = (strides,) * n_convs
        if isinstance(paddings, int):
            paddings = (paddings,) * n_convs
        self.spatial_feature_extractor = nn.Sequential()
        for i in range(n_convs):
            c = nn.Conv2d(dims[i], dims[i + 1], kernel_sizes[i], strides[i], paddings[i], padding_mode=padding_mode)
            self.spatial_feature_extractor.append(c)
            r = nn.ReLU()
            self.spatial_feature_extractor.append(r)

        # Réduire le nombre de channels
        # conv 1x1xC -> 1 channel
        self.channel_projection = nn.Conv2d(dims[-1], 1, 1)

        mlp_in = np.prod(input_sizes)
        self.attention_mlp = nn.Linear(mlp_in, mlp_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_feature_extractor(x)
        x = self.channel_projection(x)
        x = x.view(x.size(0), -1)
        x = self.attention_mlp(x)
        return x


class SpeckleRNN(nn.Module):

    def __init__(self, input_size: int, hidden_size: int = 128, *gru_args, **gru_kwargs):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, *gru_args, **gru_kwargs)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x


class SpeckleNN(nn.Module):

    def __init__(self, input_sizes: tuple = (128, 128), cnn_out_channels: tuple = (16,),
                 cnn_kernel_sizes: Union[tuple, int] = 3, cnn_strides: Union[tuple, int] = 1,
                 cnn_paddings: Union[tuple, int] = 1, cnn_mlp_out: int = 64, cnn_padding_mode: str = "reflect",
                 gru_hidden_size: int = 128, *gru_args, **gru_kwargs):
        super().__init__()
        self.cnn = SpeckleCNN(input_sizes, cnn_out_channels, cnn_kernel_sizes, cnn_strides, cnn_paddings, cnn_mlp_out,
                              cnn_padding_mode)
        rnn_in = cnn_mlp_out + 1  # +1 pour temps d'intégration comme feature
        self.rnn = SpeckleRNN(rnn_in, gru_hidden_size, *gru_args, **gru_kwargs)

    def forward(self, x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        B, N_chunk, C_in, H, W = x.shape  # B: batch size, N_chunk: len chunk, C_in: channels in, ...
        x = x.reshape(B * N_chunk, C_in, H, W)  # On regroupe les batches et les chunks en 1 dimensions pour le CNN
        x = self.cnn(x)
        x = x.reshape(B, N_chunk, -1)  # On remet dans les bonnes dimensions pour le RNN
        T_feature = torch.log(T).unsqueeze(1).repeat(1, N_chunk, 1)  # Temps d'intégration en feature
        x = torch.cat([x, T_feature], dim=-1)  # On concat les features
        x = self.rnn(x)
        return x
