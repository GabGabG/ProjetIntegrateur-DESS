from src.simulations.time_integrated_sims import MultipleTimeIntegratedTimeSeriesGenerator
from src.simulations.correlation_functions import expon, gaussian
from src.models.base_model import SpeckleNN
from src.models.training import SpeckleCallback, Overfit
import numpy as np
import torch
from torch import nn
from typing import Callable


class Regime:
    def __init__(self, min_val: float, max_val: float, n_samples: int = 8, T: float = 1):
        self.__T = T
        self.__tau_cs = np.linspace(min_val, max_val, n_samples)

    @property
    def T(self) -> float:
        return self.__T

    @property
    def tau_cs(self) -> np.ndarray:
        return self.__tau_cs


class PlusPetit(Regime):
    def __init__(self, min_val: float = 1e-2, max_val: float = 0.5, n_samples: int = 8, T: float = 1):
        super().__init__(min_val, max_val, n_samples, T)


class PlusGrand(Regime):
    def __init__(self, min_val: float = 1.5, max_val: float = 20, n_samples: int = 8, T: float = 1):
        super().__init__(min_val, max_val, n_samples, T)


class Approx(Regime):
    def __init__(self, min_val: float = 0.8, max_val: float = 1.2, n_samples: int = 8, T: float = 1):
        super().__init__(min_val, max_val, n_samples, T)


class RegimeTrials:

    def __init__(self, n_trials: int, regime: Regime, n_repeats: int = 2, *speckles_generator_args,
                 **speckles_generator_kwargs):
        self.__n_trials = n_trials
        self.__tau_cs = regime.tau_cs
        self.__n_repeats = n_repeats
        self.__T = np.array([regime.T])
        self.__speckles_generator_args = speckles_generator_args
        self.__speckles_generator_kwargs = speckles_generator_kwargs

    def __core(self, g1: callable, lr: float, trial: int, model_class: Callable[..., SpeckleNN],
               optimizer_class: Callable[..., torch.optim.Optimizer], loss_func: nn.Module, batch_size: int,
               n_epochs: int, model_creation_args: tuple, model_creation_kwargs: dict, optimizer_creation_args: tuple,
               optimizer_creation_kwargs: dict, callback_creation_args: tuple, callback_creation_kwargs: dict) -> None:
        generator = MultipleTimeIntegratedTimeSeriesGenerator(self.__tau_cs, self.__T, [g1],
                                                              self.__n_repeats)

        model = model_class(*model_creation_args, **model_creation_kwargs)
        optimizer = optimizer_class(model.parameters(), lr, *optimizer_creation_args,
                                    **optimizer_creation_kwargs)
        overfit = Overfit(model, optimizer, loss_func, generator.generate, *self.__speckles_generator_args,
                          **self.__speckles_generator_kwargs)
        lr_str = str(lr).replace(".", "p")
        callback_name = f"plus_petit_lr_{lr_str}_g1_{g1.__name__}_data_trial_{trial + 1}"
        callback = SpeckleCallback(callback_name, *callback_creation_args, **callback_creation_kwargs)
        dataloader_kwargs = {"batch_size": batch_size, "shuffle": False, "num_workers": 4,
                             "pin_memory": True}
        overfit(n_epochs, None, callback, dataloader_kwargs=dataloader_kwargs)

    def __single_lr(self, lr: float, trial: int, model_class: Callable[..., SpeckleNN],
                    optimizer_class: Callable[..., torch.optim.Optimizer], loss_func: nn.Module,
                    batch_size: int, g1_s: list[callable], n_epochs: int, model_creation_args: tuple,
                    model_creation_kwargs: dict, optimizer_creation_args: tuple, optimizer_creation_kwargs: dict,
                    callback_creation_args: tuple, callback_creation_kwargs: dict) -> None:
        for g1 in g1_s:
            self.__core(g1, lr, trial, model_class, optimizer_class, loss_func, batch_size, n_epochs,
                        model_creation_args, model_creation_kwargs, optimizer_creation_args, optimizer_creation_kwargs,
                        callback_creation_args, callback_creation_kwargs)

    def __single_trial(self, trial: int, model_class: Callable[..., SpeckleNN],
                       optimizer_class: Callable[..., torch.optim.Optimizer],
                       loss_func: nn.Module, batch_size: int, lr_s: list[float], g1_s: list[callable], n_epochs: int,
                       model_creation_args: tuple, model_creation_kwargs: dict, optimizer_creation_args: tuple,
                       optimizer_creation_kwargs: dict, callback_creation_args: tuple,
                       callback_creation_kwargs: dict) -> None:
        for lr in lr_s:
            self.__single_lr(lr, trial, model_class, optimizer_class, loss_func, batch_size, g1_s, n_epochs,
                             model_creation_args, model_creation_kwargs, optimizer_creation_args,
                             optimizer_creation_kwargs, callback_creation_args, callback_creation_kwargs)

    def trials(self, model_class: Callable[..., SpeckleNN], optimizer_class: Callable[..., torch.optim.Optimizer],
               loss_func: nn.Module, batch_size: int, lr_s: list[float], g1_s: list[callable], n_epochs: int,
               model_creation_args: tuple = (), model_creation_kwargs: dict = None, optimizer_creation_args: tuple = (),
               optimizer_creation_kwargs: dict = None, callback_creation_args: tuple = (),
               callback_creation_kwargs: dict = None) -> None:
        if optimizer_creation_kwargs is None:
            optimizer_creation_kwargs = {}
        if callback_creation_kwargs is None:
            callback_creation_kwargs = {}
        if model_creation_kwargs is None:
            model_creation_kwargs = {}

        for trial in range(self.__n_trials):
            self.__single_trial(trial, model_class, optimizer_class, loss_func, batch_size, lr_s, g1_s, n_epochs,
                                model_creation_args, model_creation_kwargs, optimizer_creation_args,
                                optimizer_creation_kwargs, callback_creation_args, callback_creation_kwargs)


if __name__ == '__main__':
    batch_size = 8
    n_epochs = 300
    model_saves = r"C:\Users\goubi\OtherGit\code_article_gabriel\source\speckles\overfits"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_repeats = 2
    corrfuncs = [expon, gaussian]
    lr_s = [1e-3, 1e-4]
    n_trials = 10
    model_class = lambda *args, **kwargs: SpeckleNN(*args, **kwargs).to(device)
    model_kwargs = {"cnn_out_channels": (16, 32, 64)}
    callbacks_kwargs = {"checkpoints_root": model_saves, "max_checkpoints": 5, "keep_when_multiple": 10}

    plus_petit = RegimeTrials(n_trials, PlusPetit(), sim_width=128, speckle_size=3, time_series_length=50,
                              correlation_function_sampling=100)
    plus_petit.trials(model_class, torch.optim.Adam, nn.L1Loss(), batch_size, lr_s, corrfuncs, n_epochs,
                      model_creation_kwargs=model_kwargs, callback_creation_kwargs=callbacks_kwargs)

    approx = RegimeTrials(n_trials, Approx(), sim_width=128, speckle_size=3, time_series_length=50,
                          correlation_function_sampling=100)
    approx.trials(model_class, torch.optim.Adam, nn.L1Loss(), batch_size, lr_s, corrfuncs, n_epochs,
                  model_creation_kwargs=model_kwargs, callback_creation_kwargs=callbacks_kwargs)

    plus_grand = RegimeTrials(n_trials, PlusGrand(), sim_width=128, speckle_size=3, time_series_length=50,
                              correlation_function_sampling=100)
    plus_grand.trials(model_class, torch.optim.Adam, nn.L1Loss(), batch_size, lr_s, corrfuncs, n_epochs,
                      model_creation_kwargs=model_kwargs, callback_creation_kwargs=callbacks_kwargs)

    plus_petit_v2 = RegimeTrials(n_trials, PlusPetit(), sim_width=128, speckle_size=3, time_series_length=50,
                                 correlation_function_sampling=100)
    plus_petit_v2.trials(model_class, torch.optim.Adam, nn.L1Loss(), batch_size, [1e-2], [expon], 200,
                         model_creation_kwargs=model_kwargs, callback_creation_kwargs=callbacks_kwargs)
