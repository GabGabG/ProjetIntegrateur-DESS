# TODO: peut-être déplacer dans un endroit plus "propice"

import torch
from torch.utils.data import DataLoader
from typing import Union
import os
from tqdm import tqdm
from base_model import SpeckleNN


class SpeckleCallback:

    def __init__(self, name: str, checkpoints_root: str = "", checkpoint_basename: str = None,
                 max_checkpoints: int = -1, keep_when_multiple: int = 50):
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
        if len(listdir) >= self.__max_checkpoints:
            all_files = [os.path.join(self.__checkpoints_root, f) for f in listdir if
                         f.endswith(".pt")]
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


class SpeckleTrainingLoop:

    def __init__(self, model: SpeckleNN, optimizer: torch.optim.Optimizer, loss_function: torch._Loss,
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
        current epoch. Must consider the case where no scheduler is provided.
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
