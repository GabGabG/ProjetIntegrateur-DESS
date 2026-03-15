import numpy as np
import cupy as cp
from cupyx.scipy.linalg import toeplitz
from typing import Union
from tqdm import tqdm
import os
from itertools import product
from random import sample, shuffle
import pandas as pd

class MultipleTimeIntegratedTimeSeriesGenerator:

    def __init__(self, possible_tau_cs: np.ndarray, possible_Ts: np.ndarray, possible_corrfuncs: list,
                 n_repeats: int = 1):
        self.__possible_tau_cs = possible_tau_cs
        self.__possible_Ts = possible_Ts
        self.__possible_corrfuncs = possible_corrfuncs
        self.__n_repeats = n_repeats

    def __generate_single_time_series(self, verbose: bool = True, *time_series_args,
                                      **time_series_kwargs):
        # TODO: arranger verbose
        generator = TimeIntegratedCorrelatedLaserSpeckles(*time_series_args, **time_series_kwargs)
        verbose_lvl = 1 if verbose else 0
        series = generator.simulate(return_numpy=True, verbose_lvl=verbose_lvl)
        return series

    def generate(self, verbose_lvl: int = 3, root: str = "", clear_root: bool = False, return_all: bool = False,
                 metadata_name: str = "metadata.csv", base_name: str = "speckles", shuffle_combinaisons: bool = False,
                 to_keep: int = -1, *time_series_args,
                 **time_series_kwargs) -> Union[tuple[np.ndarray, pd.DataFrame], pd.DataFrame]:
        if clear_root:
            for file in os.listdir(root):
                os.remove(os.path.join(root, file))
        combinaisons = list(product(self.__possible_tau_cs, self.__possible_Ts, self.__possible_corrfuncs))
        if shuffle_combinaisons:
            if to_keep > 0:
                combinaisons = sample(combinaisons, to_keep)
            else:
                shuffle(combinaisons)
        leave_outer = verbose_lvl == 3
        disable = verbose_lvl == 0
        inner_verbose = verbose_lvl == 2
        all_info = []
        all_data = []
        metadata_columns = ["File path", "Correlation function", "Integration time", "Correlation time"]
        metadata_path = os.path.join(root, metadata_name)
        for i, combinaison in enumerate(tqdm(combinaisons, disable=disable, leave=leave_outer)):
            tau_c, T, corrfunc = combinaison
            for j in tqdm(range(self.__n_repeats), disable=self.__n_repeats == 1, leave=True):
                savename = f"{base_name}_{i}_v{j + 1}.npy"
                current_info = [savename, corrfunc.__name__, T, tau_c]
                all_info.append(current_info)
                series = self.__generate_single_time_series(inner_verbose, correlation_time=tau_c,
                                                            correlation_function=corrfunc, integration_time=T,
                                                            *time_series_args, **time_series_kwargs)
                full_path = os.path.join(root, savename)
                np.save(full_path, series)
                if return_all:
                    all_data.append(series)
        metadata = pd.DataFrame(all_info, columns=metadata_columns)
        metadata.to_csv(metadata_path)
        if return_all:
            return np.ndarray(all_data), metadata
        return metadata

class TimeIntegratedCorrelatedLaserSpeckles:
    def __init__(self, sim_width: int, speckle_size: float, time_series_length: int, integration_time: float,
                 correlation_function: callable, correlation_time: float, correlation_function_sampling: int,
                 *correlation_function_params, **correlation_function_kwargs):
        """
        Initializes the time-integrated correlated laser speckle time series generation with important parameters.
        :param sim_width: a non-negative integer. The width of the simulation. Is also the height for now.
        :param speckle_size: a non-negative float. The average linear size of the speckles.
        :param time_series_length: a non-negative integer. The length of the time series.
        :param integration_time: a non-negative float. The integration time of the system.
        :param correlation_function: a callable function. A correlation function through time. Must accept (at least)
        two arguments: the lag and the correlation time, also called characteristic time.
        :param correlation_time: a non-negative float. The correlation time, also called the characteristic time.
        :param correlation_function_sampling: a non-negative integer. Number used to sample the correlation function
        in  order to simulate time integration. The bigger, the more precise, but more computationally expensive.
        :param correlation_function_params: arguments to feed to the correlation function.
        :param correlation_function_kwargs: keyword arguments to feed to the correlation function.
        """
        self.__N = sim_width
        self.__speckle_size = speckle_size
        self.__radius = sim_width / (2 * speckle_size)
        self.__M = time_series_length
        self.__T = integration_time
        self.__tau_c = correlation_time
        self.__corrfunc = correlation_function
        self.__n_corrfunc_sampling = correlation_function_sampling
        self.__corrfunc_args = correlation_function_params
        self.__corrfunc_kwargs = correlation_function_kwargs

    @property
    def sim_width(self):
        return self.__N

    @property
    def speckle_size(self):
        return self.__speckle_size

    @property
    def radius(self):
        return self.__radius

    @property
    def tau_c(self):
        return self.__tau_c

    @property
    def corrfunc(self):
        return self.__corrfunc

    @property
    def T(self):
        return self.__T

    def __len__(self):
        return self.__M

    def __create_mask(self) -> cp.ndarray:
        """
        Creates a mask simulating a circular physical aperture.
        :return: a real CuPy array of the circular mask.
        """
        X = cp.arange(self.__N, dtype=cp.uint16) - self.__N / 2
        X2 = X * X
        r2 = self.__radius * self.__radius
        mask = (cp.add.outer(X2, X2) <= r2).astype(cp.uint8)
        return mask

    def __g1_sampling(self) -> cp.ndarray:
        """
        Samples the correlation function through time on a single integration time period.
        :return: a real CuPy array containing the 2D discrete correlation matrix.
        """
        t1 = cp.linspace(0, self.__T, self.__n_corrfunc_sampling, dtype=cp.float32)
        t = cp.abs(cp.subtract.outer(t1, t1))
        g1_mat = self.__corrfunc(t, self.__tau_c, *self.__corrfunc_args, **self.__corrfunc_kwargs)
        return g1_mat

    def __g1_eigenvals(self) -> cp.ndarray:
        """
        Computes the eigenvalues of the discrete correlation function through time on a single time integration period.
        :return: a real CuPy array containing the (square root) eigenvalues.
        """
        g1_mat = self.__g1_sampling()
        eigenvals = cp.linalg.eigvalsh(g1_mat) / self.__n_corrfunc_sampling
        eigenvals = eigenvals.astype(cp.float32)
        cp.clip(eigenvals, 0, None, eigenvals)
        cp.sqrt(eigenvals, out=eigenvals)
        return eigenvals

    def __cholesky(self) -> cp.ndarray:
        """
        Cholesky decomposition of the discrete correlation function through time.
        :return: a real CuPy array of the Cholesky decomposition matrix.
        """
        first_line = self.__corrfunc(self.__T * cp.arange(self.__M, dtype=cp.float64), self.__tau_c,
                                     *self.__corrfunc_args, **self.__corrfunc_kwargs)
        cov = toeplitz(first_line)
        cov += 1e-10 * cp.eye(cov.shape[0])
        L = cp.linalg.cholesky(cov).T.astype(cp.float32)
        return L

    def __complex_amplitudes(self, cholesky: cp.ndarray) -> cp.ndarray:
        """
        Creates a complex random normal array with correlation given through a Cholesky decomposition matrix.
        :param cholesky: a real CuPy array. Cholesky decomposition matrix used to correlate the complex values generated
        with independent standard normal realizations.
        :return: a complex CuPy array with random complex values correlated along the first axis.
        """
        reals = cp.random.randn(self.__N, self.__N, self.__M, dtype=cp.float32)
        reals @= cholesky
        imags = cp.random.randn(self.__N, self.__N, self.__M, dtype=cp.float32)
        imags @= cholesky
        return (reals + 1j * imags).transpose(2, 0, 1)

    def __propagate(self, cholesky: cp.ndarray, mask: cp.ndarray, sqrt_eigenval: cp.float32) -> cp.ndarray:
        """
        Propagates a random complex amplitude matrix through space and a circular physical aperture.
        :param cholesky: a real CuPy array. Cholesky decomposition matrix used to correlate the complex amplitudes
        through time.
        :param mask: a real CuPy array. Circular mask used to simulate a circular aperture.
        :param sqrt_eigenval: a float 32. The square root of an eigenvalue of the discrete temporal correlation matrix
        required to simulate time integration.
        :return: a complex CuPy array of the amplitude matrix propagated through the circular aperture.
        """
        amplitudes = self.__complex_amplitudes(cholesky)
        amplitudes = cp.fft.fft2(amplitudes, axes=(-2, -1)) * sqrt_eigenval
        amplitudes = cp.fft.fftshift(amplitudes, axes=(-2, -1))
        amplitudes *= mask
        amplitudes = cp.fft.ifftshift(amplitudes, axes=(-2, -1))
        amplitudes = cp.fft.ifft2(amplitudes, axes=(-2, -1))
        return amplitudes

    def simulate(self, return_numpy: bool = False, verbose_lvl: int = 2) -> Union[cp.ndarray, np.ndarray]:
        """
        Simulated a time series of time integrated laser speckles following a certain correlation function in the
        time domain.
        :param return_numpy: bool. Whether to return the intensity matrix directly on CPU (NumPy)
        :param verbose_lvl: int. Level of verbose wanted. 0 is silent, 1 is a progressbar that disappear after
        completion and 2 (default) is a progressbar that stays after completion.
        :return: Either a NumPy or CuPy array, depending on if return_numpy is True or False.
        """
        s_eigenvals = self.__g1_eigenvals()  # Get eigenvalues used to simulate time integration
        mask = self.__create_mask()  # Circular mask used to simulate a propagation through a circular aperture
        L = self.__cholesky()  # Cholesky matrix
        intensity_matrix = cp.zeros((self.__M, self.__N, self.__N), dtype=cp.float32)  # Final intensity matrix
        disable = verbose_lvl == 0
        leave = verbose_lvl == 2
        for s_eigenval in tqdm(s_eigenvals, desc="Simulating", leave=True, disable=disable):
            amplitudes = self.__propagate(L, mask, s_eigenval)  # Propagates a random complex array of amplitudes
            amplitudes = cp.abs(amplitudes, dtype=cp.float32)  # Computes the real amplitudes after propagation
            intensity_matrix += amplitudes * amplitudes  # Accumulates the intensities after propagation for each eigval
        if return_numpy:
            return intensity_matrix.get()
        return intensity_matrix