import torch
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})


class OverfitAnalysis:

    def __init__(self, root: str, regime: str, lr_s: list[float], g1_s: list[str], n_trials: int,
                 file_template: callable):
        self.__root = root
        self.__regime = regime
        self.__lr_s = lr_s
        self.__g1_s = g1_s
        self.__n_trials = n_trials
        self.__file_template = file_template
        self.__all_losses = self.__extract_all_losses()
        self.__means = np.mean(self.__all_losses, axis=1)
        self.__n_epochs = self.__means.shape[-1]

    @property
    def all_losses(self):
        return self.__all_losses

    @property
    def means(self):
        return self.__means

    @property
    def n_epochs(self):
        return self.__n_epochs

    def __template(self, lr: float, g1: str, trial: int) -> str:
        lr_str = str(lr).replace(".", "p")
        file = self.__file_template(self.__regime, lr_str, g1, trial)
        return os.path.join(self.__root, file)

    def __extract_all_losses(self) -> np.ndarray:
        all_losses = []
        for g1 in self.__g1_s:
            for lr in self.__lr_s:
                current_losses = []
                for trial in range(self.__n_trials):
                    current_file = self.__template(lr, g1, trial + 1)
                    data = torch.load(current_file)
                    current_losses.append(data["mean_train_loss"])
                all_losses.append(current_losses)
        return np.array(all_losses)

    def plot_all_losses(self, savename: str = None, subplots_kw: dict = None, plot_kw: dict = None,
                        savefig_kw: dict = None, show: bool = True):
        n_rows = len(self.__g1_s)
        n_cols = len(self.__lr_s)
        titles = [f"{g1}, lr = {lr}" for g1 in self.__g1_s for lr in self.__lr_s]
        subplots_kw = subplots_kw if subplots_kw is not None else dict()
        plot_kw = plot_kw if plot_kw is not None else dict()
        savefig_kw = savefig_kw if savefig_kw is not None else dict()

        fig, axes = plt.subplots(n_rows, n_cols, **subplots_kw)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        epochs = range(1, self.__n_epochs + 1)

        for i, ax in enumerate(axes):
            ax.plot(epochs, self.__all_losses[i].T, **plot_kw)
            ax.set_title(titles[i])
            if (i // n_rows == 1) or (len(axes) == 1):
                ax.set_xlabel("Époque")
            if i % n_cols == 0:
                ax.set_ylabel("Perte LEA")
            ax.set_yscale("log")

        fig.tight_layout()

        if savename is not None:
            fig.savefig(savename, **savefig_kw)
        if show:
            plt.show()
        return fig, axes

    def plot_mean_losses(self, savename: str = None, coloring_function: callable = None,
                         linestyle_function: callable = None, subplots_kw: dict = None, savefig_kw: dict = None):
        labels = [f"{g1}, lr = {lr}" for g1 in self.__g1_s for lr in self.__lr_s]
        subplots_kw = subplots_kw if subplots_kw is not None else dict()
        savefig_kw = savefig_kw if savefig_kw is not None else dict()

        fig, ax = plt.subplots(**subplots_kw)
        epochs = range(1, self.__n_epochs + 1)
        coloring_function = coloring_function if coloring_function is not None else lambda _: None
        linestyle_function = linestyle_function if linestyle_function is not None else lambda _: None

        for i, mean in enumerate(self.__means):
            ax.plot(epochs, mean, label=labels[i], color=coloring_function(i), ls=linestyle_function(i))
        ax.set_xlabel("Époque")
        ax.set_ylabel("Perte LEA")
        ax.set_yscale("log")
        ax.legend()

        fig.tight_layout()
        if savename is not None:
            fig.savefig(savename, **savefig_kw)
        plt.show()


if __name__ == '__main__':
    root_plus_petit = root_approx = root_plus_grand = root_plus_petit_v2 = "overfits"
    images_root = r"tex\images"
    lr1 = 1e-4
    lr2 = 1e-3
    g1_1 = "expon"
    g1_2 = "gaussian"
    lr_s = [lr1, lr2]
    g1_s = [g1_1, g1_2]


    def template_1(regime: str, lr_str: str, g1: str, trial: str):
        return f"{regime}_lr_{lr_str}_g1_{g1}_data_len_16_trial_{trial}_epoch_300.pt"


    plus_petit = OverfitAnalysis(root_plus_petit, "plus_petit", lr_s, g1_s, 10, template_1)

    savename = os.path.join(images_root, "all_losses_overfit_plus_petit.pdf")
    plus_petit.plot_all_losses(savename, subplots_kw={"sharey": True, "sharex": True, "figsize": (15, 8)},
                               savefig_kw={"format": "pdf"})

    savename = os.path.join(images_root, "moyennes_overfit_plus_petit.pdf")
    plus_petit.plot_mean_losses(savename, lambda i: ["red", "green"][i // 2], lambda i: [":", None][i % 2],
                                subplots_kw={"figsize": (15, 8)}, savefig_kw={"format": "pdf"})

    approx = OverfitAnalysis(root_approx, "approx", lr_s, g1_s, 10, template_1)

    savename = os.path.join(images_root, "all_losses_overfit_approx.pdf")
    approx.plot_all_losses(savename, subplots_kw={"sharey": True, "sharex": True, "figsize": (15, 8)},
                           savefig_kw={"format": "pdf"})

    savename = os.path.join(images_root, "moyennes_overfit_approx.pdf")
    approx.plot_mean_losses(savename, lambda i: ["red", "green"][i // 2], lambda i: [":", None][i % 2],
                            subplots_kw={"figsize": (15, 8)}, savefig_kw={"format": "pdf"})

    plus_grand = OverfitAnalysis(root_plus_grand, "plus_grand", lr_s, g1_s, 10, template_1)

    savename = os.path.join(images_root, "all_losses_overfit_plus_grand.pdf")
    plus_grand.plot_all_losses(savename, subplots_kw={"sharey": True, "sharex": True, "figsize": (15, 8)},
                               savefig_kw={"format": "pdf"})

    savename = os.path.join(images_root, "moyennes_overfit_plus_grand.pdf")
    plus_grand.plot_mean_losses(savename, lambda i: ["red", "green"][i // 2], lambda i: [":", None][i % 2],
                                subplots_kw={"figsize": (15, 8)}, savefig_kw={"format": "pdf"})

    plus_petit_v2 = OverfitAnalysis(root_plus_petit_v2, "plus_petit", [1e-2], ["expon"], 10, template_1)
    fig, axes = plus_petit_v2.plot_all_losses(subplots_kw={"figsize": (15, 8)}, show=False)
    means = plus_petit_v2.means
    n_epochs = plus_petit_v2.n_epochs
    epochs = range(1, n_epochs + 1)
    axes[0].plot(epochs, means.squeeze(), color="black", ls="--", label="Moyenne")
    axes[0].legend()
    fig.savefig(os.path.join(images_root, "plus_petit_0p01_overfit.pdf"), format="pdf")
    plt.show()
    print(means)
