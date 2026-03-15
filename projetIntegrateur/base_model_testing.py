from src.models.base_model import SpeckleNN
from src.models.dataset import MetadataSplitter, SpeckleDataset
from src.models.training import SpeckleTrainingLoop, SpeckleCallback
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
from torch import nn


def train_and_valid_Adam(device: torch.device, metadata_name: str, data_root: str, model_saves, lr, batch_size,
                         chunk_size, n_epochs, train_split_frac, callbacks_base_name: str, predictions_save: str = None,
                         *nn_args, **nn_kwargs):
    metadata_path = os.path.join(data_root, metadata_name)
    splitter = MetadataSplitter.from_csv(metadata_path)

    train, val = splitter(train_split_frac)

    train_dataset = SpeckleDataset(data_root, train.metadata, chunk_size=chunk_size)
    valid_dataset = SpeckleDataset(data_root, val.metadata, chunk_size=chunk_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)

    model = SpeckleNN(*nn_args, **nn_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.L1Loss()

    callback = SpeckleCallback(callbacks_base_name, model_saves, max_checkpoints=5, keep_when_multiple=10)
    trainer = SpeckleTrainingLoop(model, optimizer, loss, train_loader, valid_loader, predictions_save)
    losses = trainer.train(n_epochs, callback=callback)
    return losses


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_saves = "checkpoints"
    data_root_500 = "speckle_data/data_500"
    data_root_5000 = "speckle_data/data_5000"
    batch_size = 8

    # v1
    n_epochs = 100
    lr = 1e-4
    callback_base_name = "checkpoints_base_model_v1"
    train_and_valid_Adam(device, "metadata.csv", data_root_500, model_saves, lr, batch_size, np.inf, n_epochs, 0.8,
                         callback_base_name, "t1.npz", cnn_out_channels=(16, 32, 64))

    # v2
    n_epochs = 100
    lr = 1e-4
    callback_base_name = "checkpoints_base_model_v2"
    train_and_valid_Adam(device, "metadata.csv", data_root_500, model_saves, lr, batch_size, np.inf, n_epochs, 0.8,
                         callback_base_name, cnn_out_channels=(8, 8), cnn_mlp_out=32, gru_hidden_size=64)

    # v3
    n_epochs = 100
    lr = 1e-4
    callback_base_name = "checkpoints_base_model_v3"
    train_and_valid_Adam(device, "metadata.csv", data_root_5000, model_saves, lr, batch_size, np.inf, n_epochs, 0.8,
                         callback_base_name, cnn_out_channels=(8, 16))

    # v4
    n_epochs = 100
    lr = 1e-3
    callback_base_name = "checkpoints_base_model_v4"
    train_and_valid_Adam(device, "metadata.csv", data_root_5000, model_saves, lr, batch_size, np.inf, n_epochs, 0.8,
                         callback_base_name, cnn_out_channels=(8, 16))

    # v5
    n_epochs = 100
    lr = 5e-4
    callback_base_name = "checkpoints_base_model_v5"
    train_and_valid_Adam(device, "metadata.csv", data_root_5000, model_saves, lr, batch_size, np.inf, n_epochs, 0.8,
                         callback_base_name, cnn_out_channels=(8, 16))
