from src.simulations.time_integrated_sims import MultipleTimeIntegratedTimeSeriesGenerator
from src.simulations.correlation_functions import expon, gaussian
from src.models.dataset import SpeckleDataset, MetadataSplitter
import numpy as np


def generate_all_data(T, tau_cs, g1, root, n_repeats=2, **speckle_generation_kwargs):
    T = np.array([T])
    g1 = [g1]
    gen = MultipleTimeIntegratedTimeSeriesGenerator(tau_cs, T, g1, n_repeats)
    df = gen.generate(root=root, clear_root=True, **speckle_generation_kwargs)
    return df


if __name__ == '__main__':
    data_root_500 = "speckle_data/data_500"
    data_root_5000 = "speckle_data/data_5000"

    T = 1
    tau_cs = np.logspace(-1, 1, 100)
    g1 = expon
    N = 128
    speckle_size = 3
    M = 50
    ell = 100  # sampling of eigenvalues

    df = generate_all_data(T, tau_cs, g1, data_root_500, n_repeats=5, sim_width=N, speckle_size=3, time_series_length=M,
                           correlation_function_sampling=ell)

    tau_cs = np.logspace(-1, 1, 1_000)
    df = generate_all_data(T, tau_cs, g1, data_root_5000, n_repeats=5, sim_width=N, speckle_size=3,
                           time_series_length=M, correlation_function_sampling=ell)
