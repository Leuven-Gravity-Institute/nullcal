from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..detector.networks import InterferometerList
from .time_frequency_map import construct_time_frequency_map

logger = logging.getLogger("nullcal")


def _get_neighbours(i: int, j: int, mask: np.ndarray) -> list:
    """Get the neighbors of a pixel at (i,j ).

    Args:
        i (int): Time index.
        j (int): Frequency index.
        mask (np.ndarray): A time-frequency mask.

    Returns:
        list: A list of time-frequency coordinates of the neighbours.
    """
    neighbours = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            if x == 0 and y == 0:
                continue
            if 0 <= i + x < mask.shape[0] and 0 <= j + y < mask.shape[1]:
                neighbours.append((i + x, j + y))
    return neighbours


# Depth-first search
def _dfs(i: int, j: int, mask: np.ndarray, visited: np.ndarray) -> list:
    """Depth-first search.

    Args:
        i (int): Time index.
        j (int): Frequency index.
        mask (np.ndarray): The time-frequency mask.
        visited (np.ndarray): An array to indicate the pixels that have been visited.

    Returns:
        list: A list of time-frequency coordinates in the cluster.
    """
    stack = [(i, j)]
    cluster = []
    while stack:
        i, j = stack.pop()
        if visited[i, j]:
            continue
        visited[i, j] = 1
        cluster.append((i, j))
        for neighbour in _get_neighbours(i, j, mask):
            if mask[neighbour[0], neighbour[1]]:
                stack.append(neighbour)
    return cluster


def clustering(filter: np.ndarray, dt: float, df: float, padding_time: float = 0.1, padding_freq: float = 10, **kwargs):
    """
    Find the largest cluster in the filter.

    Args:
        filter (np.ndarray): A binary mask in shape (n_time, n_freq).
        dt (float): The time resolution in seconds.
    df (float): The frequency resolution in Hz.
    padding_time (float, optional): The padding in time direction in seconds. Default is 0.1.
    padding_freq (float, optional): The padding in frequency direction in Hz. Default is 10.

    Returns:
    np.ndarray: A mask with the largest cluster in shape (n_time, n_freq).
    """

    # find clusters
    visited = np.zeros(filter.shape, dtype=np.uint8)
    clusters = []
    for i in range(filter.shape[0]):
        for j in range(filter.shape[1]):
            if filter[i, j] and not visited[i, j]:
                clusters.append(_dfs(i, j, filter, visited))

    # find the largest cluster
    largest_cluster = max(clusters, key=len)
    mask = np.zeros(filter.shape, dtype=np.uint8)
    for i, j in largest_cluster:
        mask[i, j] = 1

    # add padding
    padding_time = int(np.ceil(padding_time / dt))
    padding_freq = int(np.ceil(padding_freq / df))
    for i, j in largest_cluster:
        for x in range(-padding_time, padding_time + 1):
            for y in range(-padding_freq, padding_freq + 1):
                if 0 <= i + x < mask.shape[0] and 0 <= j + y < mask.shape[1]:
                    mask[i + x, j + y] = 1

    return mask


def single_clustering_by_quantile(
    interferometers: InterferometerList,
    frequency_resolution: float,
    nx: float,
    quantile: float,
    padding_time: float = 0.05,
    padding_freq: float = 0.0,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
) -> np.ndarray:
    """Perform clustering with the threshold set by the quantile of the power.

    This function only selects the largest cluster.

    Args:
        interferometers (InterferometerList): A list of interferometers.
        frequency_resolution (float): The frequency resolution in Hz.
        nx (float): The sharpness of wavelet.
        quantile (float): The quantile to define the threshold.
        padding_time (float, optional): The time window to pad at both ends. Defaults to 0.05.
        padding_freq (float, optional): The frequency window to pad at both ends. Defaults to 0.0.
        minimum_frequency (Optional[float], optional): Minimum frequency. Defaults to None.
        maximum_frequency (Optional[float], optional): Maximum frequency. Defaults to None.

    Returns:
        np.ndarray: A boolean time-frequency mask.
    """
    time_frequency_map = construct_time_frequency_map(
        interferometers=interferometers, frequency_resolution=frequency_resolution, nx=nx
    )
    # Zero the components beyond the frequency range
    if minimum_frequency is not None:
        freq_low_idx = int(np.ceil(minimum_frequency / frequency_resolution))
        time_frequency_map[:, :freq_low_idx] = 0.0
    if maximum_frequency is not None:
        freq_high_idx = int(np.floor(maximum_frequency / frequency_resolution))
        if freq_high_idx == time_frequency_map.shape[1] - 1:
            logger.warning(f"The freq_high_idx = {freq_high_idx} contains the Nyquist frequency.")
            freq_high_idx -= 1
            logger.warning(f"freq_high_idx is set to {freq_high_idx}.")
        time_frequency_map[:, freq_high_idx + 1 :] = 0.0
    threshold = np.quantile(time_frequency_map[time_frequency_map > 0.0], quantile)
    filter = time_frequency_map > threshold
    Nf = int(interferometers[0].sampling_frequency / 2 / frequency_resolution)
    Nt = int(len(interferometers[0].time_array) / Nf)
    dt = interferometers[0].duration / Nt
    output = clustering(filter, dt, frequency_resolution, padding_time=padding_time, padding_freq=padding_freq)
    return output.astype(bool)


def single_clustering_by_threshold(
    interferometers: InterferometerList,
    frequency_resolution: float,
    nx: float,
    threshold: float,
    padding_time: float = 0.05,
    padding_freq: float = 0.0,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
) -> np.ndarray:
    time_frequency_map = construct_time_frequency_map(
        interferometers=interferometers, frequency_resolution=frequency_resolution, nx=nx
    )
    # Zero the components beyond the frequency range
    if minimum_frequency is not None:
        freq_low_idx = int(np.ceil(minimum_frequency / frequency_resolution))
        time_frequency_map[:, :freq_low_idx] = 0.0
    if maximum_frequency is not None:
        freq_high_idx = int(np.floor(maximum_frequency / frequency_resolution))
        if freq_high_idx == time_frequency_map.shape[1] - 1:
            logger.warning(f"The freq_high_idx = {freq_high_idx} contains the Nyquist frequency.")
            freq_high_idx -= 1
            logger.warning(f"freq_high_idx is set to {freq_high_idx}.")
        time_frequency_map[:, freq_high_idx:] = 0.0
    filter = time_frequency_map > threshold
    Nf = int(interferometers[0].sampling_frequency / 2 / frequency_resolution)
    Nt = int(len(interferometers[0].time_array) / Nf)
    dt = interferometers[0].duration / Nt
    output = clustering(filter, dt, frequency_resolution, padding_time=padding_time, padding_freq=padding_freq)
    return output.astype(bool)
