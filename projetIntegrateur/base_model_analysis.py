import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

plt.rcParams.update({'font.size': 18})


def template_checkpoints(root: str, version: str, epoch: int) -> str:
    base = f"checkpoints_base_model_{version}_epoch_{epoch}.pt"
    return os.path.join(root, base)


def template_preds(root: str, version: str) -> str:
    base = f"predictions_{version}.npz"
    return os.path.join(root, base)


def losses_from_checkpoint(root: str, version: str, epoch: int) -> tuple[np.ndarray, np.ndarray]:
    data = torch.load(template_checkpoints(root, version, epoch))
    train_losses = data["mean_train_loss"]
    val_losses = data["mean_val_loss"]
    return train_losses, val_losses


class BaseModelAnalysis:

    def __init__(self, predictions_root: str, version: str, file_template: callable):
        self.___predictions_root = predictions_root
        self.__version = version
        self.__file_template = file_template
        self.__train_preds, self.__val_preds = self.__read_file()

    def r2_train(self, epoch: int, log: bool = False) -> float:
        targets = self.__train_preds[epoch, 1]
        preds = self.__train_preds[epoch, 0]
        if log:
            targets = np.log(targets)
            preds = np.log(preds)
        return r2_score(targets, preds)

    def r2_val(self, epoch: int, log: bool = False) -> float:
        targets = self.__val_preds[epoch, 1]
        preds = self.__val_preds[epoch, 0]
        if log:
            targets = np.log(targets)
            preds = np.log(preds)
        return r2_score(targets, preds)

    def r2(self, log: bool = False) -> tuple[list, list]:
        n_epochs = self.__train_preds.shape[0]
        scores_train = []
        scores_val = []
        for e in range(n_epochs):
            scores_train.append(self.r2_train(e, log))
            scores_val.append(self.r2_val(e, log))
        return scores_train, scores_val

    def __read_file(self) -> tuple[np.ndarray, np.ndarray]:
        file = self.__file_template(self.___predictions_root, self.__version)
        preds = np.load(file)
        train_preds = preds["train_preds"]
        val_preds = preds["val_preds"]
        return train_preds, val_preds

    def __unique_indices(self, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        _, unique_idx_train = np.unique(self.__train_preds[epoch, 1], True)
        _, unique_idx_val = np.unique(self.__val_preds[epoch, 1], True)
        return unique_idx_train, unique_idx_val

    def __uniques(self, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        unique_idx_train, unique_idx_val = self.__unique_indices(epoch)
        current_uniques_train = self.__train_preds[epoch, :, unique_idx_train]
        current_uniques_val = self.__val_preds[epoch, :, unique_idx_val]
        return current_uniques_train, current_uniques_val

    def __single_plot_preds_vs_targs(self, ax_train: plt.Axes, ax_val: plt.Axes, epoch: int, val_marker: str,
                                     log_r2: bool = False, **plot_kw) -> None:
        current_uniques_train, current_uniques_val = self.__uniques(epoch)
        label_base = f"Époque {epoch + 1}"
        r2 = "$R^2$ (log) = " if log_r2 else "$R^2$ = "
        label_train = f"{label_base}, {r2}{self.r2_train(epoch, log_r2):.2f}"
        label_val = f"{label_base}, {r2}{self.r2_val(epoch, log_r2):.2f}"
        ax_train.plot(current_uniques_train[:, 1], current_uniques_train[:, 0], label=label_train, **plot_kw)
        ax_val.plot(current_uniques_val[:, 1], current_uniques_val[:, 0], label=label_val, marker=val_marker,
                    **plot_kw)

    def __single_plot_LAE_vs_targs(self, ax_train: plt.Axes, ax_val: plt.Axes, epoch: int, val_marker: str,
                                   **plot_kw) -> None:
        current_uniques_train, current_uniques_val = self.__uniques(epoch)
        label = f"Époque {epoch + 1}"
        log_preds_train = np.log(current_uniques_train)
        log_preds_val = np.log(current_uniques_val)
        LAE_train = np.abs(log_preds_train[:, 0] - log_preds_train[:, 1])
        LAE_val = np.abs(log_preds_val[:, 0] - log_preds_val[:, 1])
        ax_train.plot(current_uniques_train[:, 1], LAE_train, label=label, **plot_kw)
        ax_val.plot(current_uniques_val[:, 1], LAE_val, label=label, marker=val_marker, **plot_kw)

    def plot_predictions_fct_targets(self, epochs: np.ndarray[int] = None, savename: str = None,
                                     val_markers: bool = True, subplots_kw: dict = None, plot_kw: dict = None,
                                     savefig_kw: dict = None, show: bool = True, log_scale: bool = False) -> None:

        if epochs is None:
            epochs = range(self.__train_preds.shape[0])
        marker = "x" if val_markers is not None else None
        subplots_kw = subplots_kw if subplots_kw is not None else dict()
        plot_kw = plot_kw if plot_kw is not None else dict()
        savefig_kw = savefig_kw if savefig_kw is not None else dict()

        fig, (ax_train, ax_val) = plt.subplots(1, 2, **subplots_kw)
        for e in epochs:
            self.__single_plot_preds_vs_targs(ax_train, ax_val, e, marker, log_scale, **plot_kw)
        ax_train.axline((0, 0), slope=1, label="Pente 1", color="black", ls="--")
        ax_val.axline((0, 0), slope=1, label="Pente 1", color="black", ls="--")
        ax_train.set_xlabel("Cibles")
        ax_train.set_ylabel("Prédictions")
        if log_scale:
            ax_train.set_xscale("log")
            ax_train.set_yscale("log")
            ax_val.set_xscale("log")
            ax_val.set_yscale("log")
        ax_train.legend(fontsize=15)
        ax_train.set_title("Entraînement")

        ax_val.set_xlabel("Cibles")
        ax_val.legend(fontsize=15)
        ax_val.set_title("Validation")
        fig.tight_layout()
        if savename is not None:
            fig.savefig(savename, **savefig_kw)
        if show:
            plt.show()

    def plot_LAE_fct_targets(self, epochs: np.ndarray[int] = None, savename: str = None,
                             val_markers: bool = True, subplots_kw: dict = None, plot_kw: dict = None,
                             savefig_kw: dict = None, show: bool = True, log_scale: bool = False) -> None:
        if epochs is None:
            epochs = range(self.__train_preds.shape[0])
        marker = "x" if val_markers is not None else None
        subplots_kw = subplots_kw if subplots_kw is not None else dict()
        plot_kw = plot_kw if plot_kw is not None else dict()
        savefig_kw = savefig_kw if savefig_kw is not None else dict()
        fig, (ax_train, ax_val) = plt.subplots(1, 2, **subplots_kw)
        for e in epochs:
            self.__single_plot_LAE_vs_targs(ax_train, ax_val, e, marker, **plot_kw)
        ax_train.set_xlabel("Cibles")
        ax_train.set_ylabel("Perte LEA")
        if log_scale:
            ax_train.set_yscale("log")
            ax_val.set_yscale("log")

        ax_train.legend(fontsize=15)
        ax_train.set_title("Entraînement")

        ax_val.set_xlabel("Cibles")
        ax_val.legend(fontsize=15)
        ax_val.set_title("Validation")
        fig.tight_layout()
        if savename is not None:
            fig.savefig(savename, **savefig_kw)
        if show:
            plt.show()

    def learning_curves(self, train_losses: np.ndarray, val_losses: np.ndarray, savename: str = None,
                        subplots_kw: dict = None, plot_kw: dict = None, savefig_kw: dict = None,
                        show: bool = True) -> None:
        subplots_kw = subplots_kw if subplots_kw is not None else dict()
        plot_kw = plot_kw if plot_kw is not None else dict()
        savefig_kw = savefig_kw if savefig_kw is not None else dict()
        fig, ax = plt.subplots(1, 1, **subplots_kw)
        n_epochs = len(train_losses)
        epochs = range(1, n_epochs + 1)
        ax.plot(epochs, train_losses, label="Entraînement", **plot_kw)
        ax.plot(epochs, val_losses, label="Validation", **plot_kw)
        ax.set_xlabel("Époque")
        ax.set_ylabel("Pert LEA")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        if savename is not None:
            fig.savefig(savename, **savefig_kw)
        if show:
            plt.show()


if __name__ == '__main__':
    epoch = 100
    root = r"D:\Gab\DESS\checkpoints"
    root_preds = r"D:\Gab\DESS"
    image_root = r"C:\Users\goubi\OtherGit\ProjetIntegrateur-DESS\projetIntegrateur\tex\images"
    v1_name_preds_vs_targs = os.path.join(image_root, "predsVsTargs_v1.pdf")
    v2_name_preds_vs_targs = os.path.join(image_root, "predsVsTargs_v2.pdf")
    v3_name_preds_vs_targs = os.path.join(image_root, "predsVsTargs_v3.pdf")
    v4_name_preds_vs_targs = os.path.join(image_root, "predsVsTargs_v4.pdf")
    v5_name_preds_vs_targs = os.path.join(image_root, "predsVsTargs_v5.pdf")

    v1_name_preds_vs_targs_log = os.path.join(image_root, "predsVsTargs_v1_log.pdf")
    v2_name_preds_vs_targs_log = os.path.join(image_root, "predsVsTargs_v2_log.pdf")
    v3_name_preds_vs_targs_log = os.path.join(image_root, "predsVsTargs_v3_log.pdf")
    v4_name_preds_vs_targs_log = os.path.join(image_root, "predsVsTargs_v4_log.pdf")
    v5_name_preds_vs_targs_log = os.path.join(image_root, "predsVsTargs_v5_log.pdf")

    v1_LAE_vs_targs = os.path.join(image_root, "LAEVsTargs_v1.pdf")
    v2_LAE_vs_targs = os.path.join(image_root, "LAEVsTargs_v2.pdf")
    v3_LAE_vs_targs = os.path.join(image_root, "LAEVsTargs_v3.pdf")
    v4_LAE_vs_targs = os.path.join(image_root, "LAEVsTargs_v4.pdf")
    v5_LAE_vs_targs = os.path.join(image_root, "LAEVsTargs_v5.pdf")

    v1_learning_curves = os.path.join(image_root, "learning_curves_v1.pdf")
    v2_learning_curves = os.path.join(image_root, "learning_curves_v2.pdf")
    v3_learning_curves = os.path.join(image_root, "learning_curves_v3.pdf")
    v4_learning_curves = os.path.join(image_root, "learning_curves_v4.pdf")
    v5_learning_curves = os.path.join(image_root, "learning_curves_v5.pdf")

    p = BaseModelAnalysis(root_preds, "v1", template_preds)
    epochs = np.array([1, 25, 50, 75, 100], dtype=int) - 1
    p.plot_predictions_fct_targets(epochs, v1_name_preds_vs_targs, True, {"figsize": (16, 8), "sharey": True},
                                   savefig_kw={"format": "pdf"})
    p.plot_predictions_fct_targets(epochs, v1_name_preds_vs_targs_log, True, {"figsize": (16, 8), "sharey": True},
                                   savefig_kw={"format": "pdf"}, log_scale=True)
    p.plot_LAE_fct_targets(epochs, v1_LAE_vs_targs, True, {"figsize": (16, 8), "sharey": True}, log_scale=True,
                           savefig_kw={"format": "pdf"})
    p.learning_curves(*losses_from_checkpoint(root, "v1", 100), savename=v1_learning_curves,
                      subplots_kw={"figsize": (15, 8)}, savefig_kw={"format": "pdf"})

    p = BaseModelAnalysis(root_preds, "v2", template_preds)
    p.plot_predictions_fct_targets(epochs, v2_name_preds_vs_targs, True, {"figsize": (16, 8), "sharey": True},
                                   savefig_kw={"format": "pdf"})
    p.plot_predictions_fct_targets(epochs, v2_name_preds_vs_targs_log, True, {"figsize": (16, 8), "sharey": True},
                                   savefig_kw={"format": "pdf"}, log_scale=True)
    p.plot_LAE_fct_targets(epochs, v2_LAE_vs_targs, True, {"figsize": (16, 8), "sharey": True}, log_scale=True,
                           savefig_kw={"format": "pdf"})
    p.learning_curves(*losses_from_checkpoint(root, "v2", 100), savename=v2_learning_curves,
                      subplots_kw={"figsize": (15, 8)}, savefig_kw={"format": "pdf"})

    p = BaseModelAnalysis(root_preds, "v3", template_preds)
    p.plot_predictions_fct_targets(epochs, v3_name_preds_vs_targs, True, {"figsize": (16, 8), "sharey": True},
                                   savefig_kw={"format": "pdf"})
    p.plot_predictions_fct_targets(epochs, v3_name_preds_vs_targs_log, True, {"figsize": (16, 8), "sharey": True},
                                   savefig_kw={"format": "pdf"}, log_scale=True)
    p.plot_LAE_fct_targets(epochs, v3_LAE_vs_targs, True, {"figsize": (16, 8), "sharey": True}, log_scale=True,
                           savefig_kw={"format": "pdf"})
    p.learning_curves(*losses_from_checkpoint(root, "v3", 100), savename=v3_learning_curves,
                      subplots_kw={"figsize": (15, 8)}, savefig_kw={"format": "pdf"})

    p = BaseModelAnalysis(root_preds, "v4", template_preds)
    p.plot_predictions_fct_targets(epochs, v4_name_preds_vs_targs, True, {"figsize": (16, 8), "sharey": True},
                                   savefig_kw={"format": "pdf"})
    p.plot_predictions_fct_targets(epochs, v4_name_preds_vs_targs_log, True, {"figsize": (16, 8), "sharey": True},
                                   savefig_kw={"format": "pdf"}, log_scale=True)
    p.plot_LAE_fct_targets(epochs, v4_LAE_vs_targs, True, {"figsize": (16, 8), "sharey": True}, log_scale=True,
                           savefig_kw={"format": "pdf"})
    p.learning_curves(*losses_from_checkpoint(root, "v4", 100), savename=v4_learning_curves,
                      subplots_kw={"figsize": (15, 8)}, savefig_kw={"format": "pdf"})

    p = BaseModelAnalysis(root_preds, "v5", template_preds)
    p.plot_predictions_fct_targets(epochs, v5_name_preds_vs_targs, True, {"figsize": (16, 8), "sharey": True},
                                   savefig_kw={"format": "pdf"})
    p.plot_predictions_fct_targets(epochs, v5_name_preds_vs_targs_log, True, {"figsize": (16, 8), "sharey": True},
                                   savefig_kw={"format": "pdf"}, log_scale=True)
    p.plot_LAE_fct_targets(epochs, v5_LAE_vs_targs, True, {"figsize": (16, 8), "sharey": True}, log_scale=True,
                           savefig_kw={"format": "pdf"})
    p.learning_curves(*losses_from_checkpoint(root, "v5", 100), savename=v5_learning_curves,
                      subplots_kw={"figsize": (15, 8)}, savefig_kw={"format": "pdf"})
