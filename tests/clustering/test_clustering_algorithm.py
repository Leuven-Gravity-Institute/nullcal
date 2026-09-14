"""Tests for the connected-component search that turns a thresholded power map into a mask.

These exercise the algorithm on hand-built masks: no detectors, no transform, no randomness. The
expected answers are worked out by inspection of the input, which is what makes them anchors rather
than recordings of current behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullcal.clustering.single import _dfs, _get_neighbors, clustering


class TestGetNeighbors:
    """Neighbourhood used by the search: eight-connected, clipped at the array border."""

    @pytest.mark.unit
    def test_interior_pixel_has_eight_neighbours(self):
        """Eight-connectivity, and the pixel itself is not its own neighbour."""
        neighbours = _get_neighbors(2, 2, np.zeros((5, 5)))

        assert sorted(neighbours) == [
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 3),
        ]
        assert (2, 2) not in neighbours

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("i", "j", "expected_count"),
        [(0, 0, 3), (0, 2, 5), (4, 4, 3), (2, 0, 5), (2, 2, 8)],
    )
    def test_border_pixels_are_clipped_not_wrapped(self, i, j, expected_count):
        """Corners have three neighbours, edges five, interior eight.

        Clipping rather than wrapping matters physically: the time axis of the map is not periodic
        for clustering purposes, so a cluster at the last time row must not be joined to one at the
        first.
        """
        neighbours = _get_neighbors(i, j, np.zeros((5, 5)))

        assert len(neighbours) == expected_count
        assert all(0 <= x < 5 and 0 <= y < 5 for x, y in neighbours)


class TestDepthFirstSearch:
    """``_dfs`` collects one connected component and marks it visited."""

    @pytest.mark.unit
    def test_collects_a_contiguous_block(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[1:3, 1:3] = True
        visited = np.zeros(mask.shape, dtype=np.uint8)

        cluster = _dfs(1, 1, mask, visited)

        assert sorted(cluster) == [(1, 1), (1, 2), (2, 1), (2, 2)]
        assert visited.sum() == 4

    @pytest.mark.unit
    def test_diagonally_touching_pixels_are_one_cluster(self):
        """Consistent with the eight-connected neighbourhood above.

        A chirp crosses the time-frequency plane diagonally, so four-connectivity would split one
        signal into a string of separate clusters and the largest-cluster rule would keep only a
        fragment of it.
        """
        mask = np.array([[1, 0], [0, 1]], dtype=bool)

        cluster = _dfs(0, 0, mask, np.zeros((2, 2), dtype=np.uint8))

        assert sorted(cluster) == [(0, 0), (1, 1)]

    @pytest.mark.unit
    def test_does_not_cross_a_gap(self):
        """Two blocks separated by an empty row or column are two clusters."""
        mask = np.zeros((6, 6), dtype=bool)
        mask[0, 0] = True
        mask[4, 4] = True

        cluster = _dfs(0, 0, mask, np.zeros(mask.shape, dtype=np.uint8))

        assert cluster == [(0, 0)]

    @pytest.mark.unit
    def test_already_visited_pixels_are_not_revisited(self):
        """The visited array is the recursion guard; without it the search would not terminate."""
        mask = np.ones((3, 3), dtype=bool)
        visited = np.ones(mask.shape, dtype=np.uint8)

        assert _dfs(1, 1, mask, visited) == []


class TestClustering:
    """``clustering`` keeps the largest connected component and pads it."""

    @pytest.mark.unit
    def test_keeps_only_the_largest_cluster(self):
        """The documented behaviour: one signal per analysis, so the runner-up is discarded."""
        tf_filter = np.zeros((10, 10), dtype=bool)
        tf_filter[2:4, 2:4] = True  # four pixels
        tf_filter[7, 7] = True  # one pixel

        mask = clustering(tf_filter, dt=1.0, df=1.0, padding_time=0.0, padding_freq=0.0)

        assert mask.sum() == 4
        assert sorted(map(tuple, np.argwhere(mask))) == [(2, 2), (2, 3), (3, 2), (3, 3)]

    @pytest.mark.unit
    def test_padding_is_converted_from_physical_units_to_pixels(self):
        """``padding_time`` is seconds and ``padding_freq`` is hertz; both are ceiled to pixels.

        With ``dt = 0.05`` s a padding of 0.1 s is exactly two time pixels, and with ``df = 5`` Hz a
        padding of 10 Hz is exactly two frequency pixels. A single pixel is grown to a 5x5 block.
        The conversion is the part that can be wrong in a way no shape check would catch -- a
        padding applied in pixels rather than seconds would silently scale with the resolution.
        """
        tf_filter = np.zeros((21, 21), dtype=bool)
        tf_filter[10, 10] = True

        mask = clustering(tf_filter, dt=0.05, df=5.0, padding_time=0.1, padding_freq=10.0)

        assert mask.sum() == 25
        corners = np.argwhere(mask)
        assert corners.min(axis=0).tolist() == [8, 8]
        assert corners.max(axis=0).tolist() == [12, 12]

    @pytest.mark.unit
    def test_padding_rounds_up_to_a_whole_pixel(self):
        """A padding smaller than one pixel still pads by one -- ``ceil``, not ``round``.

        Rounding down would make a sub-pixel padding request a silent no-op.
        """
        tf_filter = np.zeros((9, 9), dtype=bool)
        tf_filter[4, 4] = True

        mask = clustering(tf_filter, dt=1.0, df=1.0, padding_time=0.01, padding_freq=0.01)

        assert mask.sum() == 9

    @pytest.mark.unit
    def test_padding_is_clipped_at_the_array_border(self):
        """Padding a cluster in the corner must not wrap or write out of bounds."""
        tf_filter = np.zeros((5, 5), dtype=bool)
        tf_filter[0, 0] = True

        mask = clustering(tf_filter, dt=1.0, df=1.0, padding_time=2.0, padding_freq=2.0)

        assert mask.shape == (5, 5)
        assert sorted(map(tuple, np.argwhere(mask))) == [(i, j) for i in range(3) for j in range(3)]

    @pytest.mark.unit
    def test_zero_padding_leaves_the_cluster_untouched(self):
        tf_filter = np.zeros((7, 7), dtype=bool)
        tf_filter[3, 3] = True

        mask = clustering(tf_filter, dt=1.0, df=1.0, padding_time=0.0, padding_freq=0.0)

        assert mask.sum() == 1
        assert mask[3, 3] == 1

    @pytest.mark.unit
    def test_returns_an_integer_mask_of_the_input_shape(self):
        """Callers convert the result with ``.astype(bool)``; a float return would still convert
        but would make ``sum()`` mean something different to every reader."""
        tf_filter = np.zeros((8, 6), dtype=bool)
        tf_filter[1:3, 1:3] = True

        mask = clustering(tf_filter, dt=1.0, df=1.0)

        assert mask.shape == (8, 6)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 1}

    @pytest.mark.unit
    def test_an_empty_filter_raises_rather_than_returning_an_empty_mask(self):
        """A map in which nothing passed the threshold is a real configuration error.

        Current behaviour is a ``ValueError`` from ``max()`` over no clusters. It is recorded here
        because it is the boundary case of the function and the alternative -- returning an all-zero
        mask -- would send an empty filter downstream, where it would produce a null stream with no
        pixels and a likelihood of zero rather than an error. The message is numpy's rather than a
        described one; a clearer message would be an improvement, not a behaviour change.
        """
        with pytest.raises(ValueError, match="empty"):
            clustering(np.zeros((4, 4), dtype=bool), dt=1.0, df=1.0)
