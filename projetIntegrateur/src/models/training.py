# TODO: peut-être déplacer dans un endroit plus "propice"

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Union
import os
import tempfile
from tqdm import tqdm
from base_model import SpeckleNN
from dataset import MetadataSplitter, SpeckleDataset


class SpeckleCallback:

    def __init__(self, name: str, checkpoints_root: str = "", checkpoint_basename: str = None,
                 max_checkpoints: int = -1, keep_when_multiple: int = 50):
        # TODO: pour simplifier, mettre max_checkpoints à np.inf?
        self.__name = name
        self.__checkpoints_root = checkpoints_root
        self.__checkpoint_basename = checkpoint_basename if checkpoint_basename is not None else self.__name
        self.__max_checkpoints = max_checkpoints
        self.__keep_when_multiple = keep_when_multiple

    def __checkpoint_model(self, model: SpeckleNN, optimizer: torch.optim.Optimizer,
                           scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, mean_train_losses: list,
                           mean_val_losses: list) -> None:
        """
        Function called to save the current important information.
        :param model: SpeckleNN. Model to save.
        :param optimizer: Optimizer from PyTorch. Optimizer of the model to save.
        :param scheduler: LRScheduler from PyTorch. Scheduler of the model to save. `None` when no scheduler is used.
        :param epoch: int. Current epoch.
        :param mean_train_losses: list. Mean train losses of the model, from beginning to current epoch.
        :param mean_val_losses: list. Mean validation losses of the model, from beginning to current epoch.
        :return: Nothing
        """
        current_name = f"{self.__checkpoint_basename}_epoch_{epoch}.pt"
        current_name = os.path.join(self.__checkpoints_root, current_name)
        scheduler_state_dict = None
        if scheduler is not None:
            scheduler_state_dict = scheduler.state_dict()
        to_save = {"epoch": epoch,
                   "mean_train_loss": mean_train_losses,
                   "mean_val_loss": mean_val_losses,
                   "model_state_dict": model.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(),
                   "scheduler_state_dict": scheduler_state_dict}
        torch.save(to_save, current_name)

    def __clear_oldest_file(self) -> None:
        """
        Removes the oldest checkpoint file.
        :return: Nothing.
        """
        listdir = os.listdir(self.__checkpoints_root)
        all_files = [os.path.join(self.__checkpoints_root, f) for f in listdir if
                     f.endswith(".pt") and f.startswith(self.__checkpoint_basename)]
        if all_files and len(all_files) >= self.__max_checkpoints:
            oldest = min(all_files, key=os.path.getmtime)
            os.remove(oldest)

    def __call__(self, model: SpeckleNN, optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, mean_train_losses: list,
                 mean_val_losses: list) -> None:
        """
        Callback main function. Check is saving conditions are respected, then saves the current important informations
        in a .pt file. Saves the model state dict, the optimizer state dict, the scheduler state dict (if applicable),
        the current epoch, the mean train losses and the mean validation losses.
        :param model: SpeckleNN. Model to save.
        :param optimizer: Optimizer from PyTorch. Optimizer of the model to save.
        :param scheduler: LRScheduler from PyTorch. Scheduler of the model to save. `None` when no scheduler is used.
        :param epoch: int. Current epoch.
        :param mean_train_losses: list. Mean train losses of the model, from beginning to current epoch.
        :param mean_val_losses: list. Mean validation losses of the model, from beginning to current epoch.
        :return: Nothing
        """
        # On sauvegarde si et seulement si l'époque courante est un multiple d'une certaine valeur ET on veut garder
        # un nombre de checkpoints non nul.
        if epoch % self.__keep_when_multiple == 0 and self.__max_checkpoints != 0:
            # Si le nombre de checkpoints maximal à garder est -1, on garde tout.
            if self.__max_checkpoints == -1:
                self.__checkpoint_model(model, optimizer, scheduler, epoch, mean_train_losses, mean_val_losses)
            else:
                self.__clear_oldest_file()
                self.__checkpoint_model(model, optimizer, scheduler, epoch, mean_train_losses, mean_val_losses)
            msg = f"Epoch: {epoch}\n{mean_train_losses[-1]}\n{mean_val_losses[-1]}"
            print(msg)


class SpeckleTrainingLoop:

    def __init__(self, model: SpeckleNN, optimizer: torch.optim.Optimizer, loss_function: callable,
                 dataloader_train: DataLoader, dataloader_valid: DataLoader = None):
        """
        Initializes the training loop.
        :param model: SpeckleRNN. The model to train.
        :param optimizer: Optimizer from PyTorch. Optimizer used to update the model parameters.
        :param loss_function: Loss function from PyTorch. Loss function used to compute the loss between the predicted
        values and the target values.
        :param dataloader_train: DataLoader from PyTorch. Dataloader used for training data.
        :param dataloader_valid: DataLoader from PyTorch. Dataloader used for validation data. If None, no validation
        is done (default).
        """
        self.__device = next(model.parameters()).device
        self.__train_losses_all = []
        self.__valid_losses_all = []
        self.__model = model
        self.__optimizer = optimizer
        self.__loss_function = loss_function
        self.__dataloader_train = dataloader_train
        self.__dataloader_valid = dataloader_valid

    def __core(self, x: torch.Tensor, T: torch.Tensor, target: torch.Tensor,
               return_predictions: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Inner code for the training and validation step. Puts data on the device, evaluates the model and computes the
        loss.
        :param x: torch.Tensor. Tensor containing time series. Must be of the shape
        (batch_size, time steps, height, width).
        :param T: torch.Tensor. Tensor containing the integration times associated with each sequence of time steps.
        :param target: torch.Tensor. Tensor containing the target correlation time associated with each sequence of
        time steps.
        :param return_predictions: bool. Whether to return the predictions of the model or not. Default is False,
        no predictions are returned.
        :return: torch.Tensor or tuple of two torch.Tensor. If `return_predictions` is False, returns only the mean
        loss. If `return_predictions` is True, returns the mean loss and the (log) predictions.
        """
        x = x.to(self.__device, non_blocking=True)
        T = T.to(self.__device, non_blocking=True)
        target = target.to(self.__device, non_blocking=True)
        if target.ndim == 1:
            target.unsqueeze_(1)
        log_target = torch.log(target)
        log_pred = self.__model(x, T)
        loss = self.__loss_function(log_pred, log_target)
        if return_predictions:
            return loss, log_pred
        return loss

    def __train_epoch(self) -> float:
        """
        Trains the model for a single epoch.
        :return: the average training loss.
        """
        self.__model.train()
        losses = []

        for (x, T), target in tqdm(self.__dataloader_train, desc="Training", leave=True):
            self.__optimizer.zero_grad()
            loss = self.__core(x, T, target)
            loss.backward()

            self.__optimizer.step()
            losses.append(loss.detach())

        mean_loss = torch.mean(torch.stack(losses)).item()
        return mean_loss

    def __valid_epoch(self) -> float:
        """
        Calculates validation for a single epoch.
        :return: the average validation loss.
        """
        self.__model.eval()
        losses = []
        with torch.no_grad():
            for (x, T), target in tqdm(self.__dataloader_valid, desc="Validation", leave=True):
                loss = self.__core(x, T, target)
                losses.append(loss.detach())

            mean_loss = torch.mean(torch.stack(losses)).item()
        return mean_loss

    def train(self, num_epochs: int, scheduler: torch.optim.lr_scheduler.LRScheduler = None,
              callback: SpeckleCallback = None) -> tuple[list, list]:
        """
        Trains the model with optional validation.
        :param num_epochs: int. Number of epochs to train.
        :param scheduler: LRScheduler from PyTorch. Scheduler used to update the learning rate.
        Default is None, no scheduler.
        :param callback: callable. Function called every epoch, optional. If provided, must take six positional
        arguments: model, optimizer, scheduler, epoch number, mean training loss and mean validation loss for the
        current epoch. Must consider the case where no scheduler is provided. No callback by default.
        :return: a tuple of lists. The first list contains the training losses and the second list contains the
        validation losses.
        """
        train_mean_losses = []
        val_mean_losses = []
        val_mean_loss = None

        for epoch in range(1, num_epochs + 1):
            train_mean_loss = self.__train_epoch()

            if self.__dataloader_valid is not None:
                val_mean_loss = self.__valid_epoch()
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_mean_loss)
                    else:
                        scheduler.step()

            train_mean_losses.append(train_mean_loss)
            val_mean_losses.append(val_mean_loss)

            if callback is not None:
                callback(self.__model, self.__optimizer, scheduler, epoch, train_mean_losses, val_mean_losses)

        return train_mean_losses, val_mean_losses

    def test_model(self, test_dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tests the current model.
        :param test_dataloader: DataLoader from PyTorch. Dataloader used to test the model with test data.
        :return: a tuple of torch.Tensor. The first element contains the test losses, the second element contains
        the (log) predictions of all the test data, and the last element contains the targets. To compare the (log)
        predictions and the targets, one must use torch.exp on the (log) predictions. Tensors are returned to CPU.
        """
        self.__model.eval()
        losses = []
        log_predictions_all = []
        targets = []
        with torch.no_grad():
            for (x, T), target in tqdm(test_dataloader, desc="Test phase", leave=True):
                loss, log_predictions = self.__core(x, T, target, True)
                losses.append(loss.detach().item())
                log_predictions_all.append(log_predictions.detach())
                targets.append(target.detach())
        losses = torch.tensor(losses, device=torch.device("cpu"))
        log_predictions_all = torch.cat(log_predictions_all).cpu()
        targets = torch.cat(targets).cpu()
        return losses, log_predictions_all, targets


class Overfit:

    def __init__(self, model: SpeckleNN, optimizer: torch.optim.Optimizer, loss_function: callable,
                 data_simulator: callable, *data_sim_args, **data_sim_kwargs):
        """
        Class used to test overfitting of SpeckleNN on few.
        :param model: SpeckleNN. The model to test.
        :param optimizer: Optimizer from PyTorch. Optimizer used to update the model parameters.
        :param loss_function: Loss function from PyTorch. Loss function used to compute the loss between the predicted
        values and the target values.
        :param data_simulator: callable. Function used to simulate data. Must accept at least one keyword argument, the
        root where to save data.
        :param data_sim_args: Arguments to pass to the data simulator function. Optional.
        :param data_sim_kwargs: Keyword arguments to pass to the data simulator. Optional.
        """
        self.__model = model
        self.__optimizer = optimizer
        self.__loss_function = loss_function
        self.__data_simulator = data_simulator
        self.__data_sim_args = data_sim_args
        self.__data_sim_kwargs = data_sim_kwargs

    def __call__(self, num_epochs: int, scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                 callback: SpeckleCallback = None, dataset_args: tuple = (), dataset_kwargs: dict = None,
                 dataloader_args: tuple = (), dataloader_kwargs: dict = None):
        """
        Method to run the overfitting training procedure.
        :param num_epochs: int. Number of epochs to train.
        :param scheduler: LRScheduler from PyTorch. Scheduler used to update the learning rate.
        Default is None, no scheduler.
        :param callback: callable. Function called every epoch, optional. If provided, must take six positional
        arguments: model, optimizer, scheduler, epoch number, mean training loss and mean validation loss for the
        current epoch. Must consider the case where no scheduler is provided. No callback by default.
        :param dataset_args: tuple. Arguments to pass to the SpeckleDataset class. Optional.
        :param dataset_kwargs: dict. Keyword arguments to pass to the SpeckleDataset class. Optional.
        :param dataloader_args: tuple. Arguments to pass to the DataLoader class. Optional.
        :param dataloader_kwargs: dict. Keyword arguments to pass to the DataLoader class. Optional.
        :return: a list of train losses for all epochs.
        """
        if dataloader_kwargs is None:
            dataloader_kwargs = {}
        if dataset_kwargs is None:
            dataset_kwargs = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = self.__data_simulator(root=temp_dir, *self.__data_sim_args, **self.__data_sim_kwargs)
            dataset = SpeckleDataset(temp_dir, metadata, *dataset_args, **dataset_kwargs)
            dataloader = DataLoader(dataset, *dataloader_args, **dataloader_kwargs)
            train = SpeckleTrainingLoop(self.__model, self.__optimizer, self.__loss_function, dataloader)
            losses, _ = train.train(num_epochs, scheduler=scheduler, callback=callback)
        return losses


if __name__ == '__main__':
    from ..simulations.time_integrated_sims import MultipleTimeIntegratedTimeSeriesGenerator
    from ..simulations.correlation_functions import expon

    batch_size = 8
    lr = 1e-3
    n_epochs = 300
    data_root = r"C:\Users\goubi\OtherGit\code_article_gabriel\source\speckles\data"
    model_saves = r"C:\Users\goubi\OtherGit\code_article_gabriel\source\speckles\callbacks"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_data = 16
    n_repeats = 2

    # Régime tau_c < T
    T = np.array([1])
    tau_cs = np.linspace(1e-2, 0.5, n_data // n_repeats)

    model = SpeckleNN(cnn_out_channels=(16, 32, 64)).to(device)
    gen = MultipleTimeIntegratedTimeSeriesGenerator(tau_cs, T, [expon], n_repeats)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.L1Loss()
    o = Overfit(model, optimizer, loss, gen.generate, sim_width=128, speckle_size=3, time_series_length=50,
                correlation_function_sampling=100)
    lr_str = str(lr).replace(".", "p")
    callback = SpeckleCallback(f"data_len_{n_data}_L1_lr_{lr_str}", model_saves, max_checkpoints=5,
                               keep_when_multiple=10)
    losses = o(n_epochs, None, callback,
               dataloader_kwargs={"batch_size": batch_size, "shuffle": False, "num_workers": 4, "pin_memory": True})

    # load = torch.load(r"C:\Users\goubi\OtherGit\code_article_gabriel\source\speckles\callbacks\range_10_L1_epoch_4.pt")
    # print(load.keys())
    # print(load["epoch"])
    # print(load["mean_train_loss"])
    # print(load["mean_val_loss"])
    # print(load["scheduler_state_dict"])
    #
    # exit()
    metadata_name = "metadata.csv"
    metadata_path = os.path.join(data_root, metadata_name)

    metadata_splitter = MetadataSplitter.from_csv(metadata_path)
    train, val = metadata_splitter(0.8)

    train_dataset = SpeckleDataset(data_root, train.metadata)
    val_dataset = SpeckleDataset(data_root, val.metadata)
    range_ = 64
    mini_train = Subset(train_dataset, range(range_))
    mini_train_loader = DataLoader(mini_train, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    lr_str = str(lr).replace(".", "p")
    callback = SpeckleCallback(f"range_{range_}_L1_lr_{lr_str}", model_saves, max_checkpoints=5,
                               keep_when_multiple=10)

    trainer = SpeckleTrainingLoop(model, optimizer, loss, mini_train_loader)
    trainer.train(num_epochs=n_epochs, callback=callback)
