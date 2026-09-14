"""Tests for the WDM window function, anchored on its published definition.

The transform implemented in ``nullcal.time_frequency_transform`` is the Wilson-Daubechies-Meyer
(WDM) wavelet transform. Its defining object is the frequency-domain window :math:`\\tilde\\phi(\\omega)`
-- a Meyer-type window whose taper is a cosine of the regularised incomplete beta function
:math:`I_x(d, d)`, with the steepness :math:`d` given here by ``nx``.

External references for every property asserted below (none of these numbers come from running
this package):

* N. J. Cornish, "Time-frequency analysis of gravitational wave data",
  Phys. Rev. D **102**, 124038 (2020), arXiv:2009.00043 -- the formulation this implementation
  follows, including the incomplete-beta taper and the half-density time-frequency tiling.
* V. Necula, S. Klimenko and G. Mitselmakher, "Transient analysis with fast Wilson-Daubechies
  time-frequency transform", J. Phys. Conf. Ser. **363**, 012032 (2012) -- the original WDM
  transform for gravitational-wave burst analysis.

The properties tested here are consequences of that definition that can be written down
independently of any implementation:

1. the window is flat at :math:`1/\\sqrt{\\Delta\\Omega}` over the passband;
2. at the layer half-spacing the taper argument is exactly :math:`1/2`, and
   :math:`I_{1/2}(d, d) = 1/2` for every :math:`d` by the symmetry of the Beta distribution,
   so :math:`\\tilde\\phi(\\Delta\\Omega/2) = \\cos(\\pi/4)/\\sqrt{\\Delta\\Omega}` exactly and
   independently of the steepness;
3. the window has compact support :math:`|\\omega| < 3\\Delta\\Omega/4`, which is what limits
   spectral leakage to immediately adjacent frequency layers;
4. the translated squares sum to a constant -- the partition of unity that makes the WDM
   wavepackets an orthonormal basis, and therefore the reason Parseval's theorem holds for the
   transform.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullcal.time_frequency_transform.transform_freq_funcs import phitilde_vec, phitilde_vec_norm

# Float64 machine epsilon is 2.2e-16. Every tolerance below is stated as a multiple of it rather
# than read off a measurement, so that none of them can be quietly widened to make a test pass.
EPS = np.finfo(np.float64).eps
STEEPNESS_VALUES = (2.0, 4.0, 8.0)


def layer_spacing(n_f: int) -> float:
    """Angular frequency spacing between WDM layers, :math:`\\Delta\\Omega = \\pi / n_f`.

    This is the Nyquist angular frequency :math:`\\pi` divided into ``n_f`` equal layers, the
    tiling of the published WDM definition.
    """
    return np.pi / n_f


@pytest.mark.unit
@pytest.mark.parametrize("n_f", [16, 32, 128])
@pytest.mark.parametrize("nx", STEEPNESS_VALUES)
def test_window_passband_is_flat_at_inverse_sqrt_layer_spacing(n_f, nx):
    """Inside the passband the window equals :math:`1/\\sqrt{\\Delta\\Omega}` (Cornish 2020).

    The flat value is fixed by the definition, not by this implementation: it is what makes the
    window an isometry on the band it passes.
    """
    d_omega = layer_spacing(n_f)
    # The flat region of the published window is |omega| < a, with a = d_omega / 4.
    passband = np.linspace(-d_omega / 4, d_omega / 4, 257, endpoint=False)

    values = phitilde_vec(passband, n_f, nx)

    expected = 1.0 / np.sqrt(d_omega)
    assert np.max(np.abs(values - expected)) <= 8 * EPS * expected


@pytest.mark.unit
@pytest.mark.parametrize("n_f", [16, 32, 128])
@pytest.mark.parametrize("nx", STEEPNESS_VALUES)
def test_window_at_half_layer_spacing_is_analytic_and_steepness_independent(n_f, nx):
    """At :math:`\\omega = \\Delta\\Omega/2` the window is :math:`1/\\sqrt{2\\Delta\\Omega}` exactly.

    This is the sharpest available closed-form anchor on the taper. At that frequency the taper
    argument is :math:`x = 1/2`; the regularised incomplete beta function satisfies
    :math:`I_{1/2}(d, d) = 1/2` for every :math:`d > 0` because the Beta(d, d) distribution is
    symmetric about 1/2. The window there is therefore
    :math:`\\cos(\\pi/4)/\\sqrt{\\Delta\\Omega} = 1/\\sqrt{2\\Delta\\Omega}` for any steepness --
    a value no choice of ``nx`` may move.
    """
    d_omega = layer_spacing(n_f)

    value = phitilde_vec(np.array([d_omega / 2]), n_f, nx)[0]

    expected = 1.0 / np.sqrt(2 * d_omega)
    assert abs(value - expected) <= 8 * EPS * expected


@pytest.mark.unit
@pytest.mark.parametrize("nx", STEEPNESS_VALUES)
def test_window_support_confines_overlap_to_adjacent_layers(nx):
    """The window vanishes for :math:`|\\omega| \\ge 3\\Delta\\Omega/4`.

    Support narrower than :math:`\\Delta\\Omega` is the structural property that makes the WDM
    transform useful: a frequency layer overlaps only its immediate neighbours, so a narrowband
    signal cannot leak into next-nearest layers. A window that lost its compact support would
    still round-trip, so this is not implied by the round-trip test.
    """
    n_f = 32
    d_omega = layer_spacing(n_f)
    cutoff = 3 * d_omega / 4

    outside = np.concatenate([np.linspace(-4 * d_omega, -cutoff, 2001), np.linspace(cutoff, 4 * d_omega, 2001)])

    assert np.all(phitilde_vec(outside, n_f, nx) == 0.0)
    # ... and it is genuinely non-zero just inside, so the assertion above is not vacuous.
    assert phitilde_vec(np.array([cutoff * 0.99]), n_f, nx)[0] > 0.0


@pytest.mark.unit
@pytest.mark.parametrize("nx", STEEPNESS_VALUES)
def test_window_is_even_in_frequency(nx):
    """:math:`\\tilde\\phi(-\\omega) = \\tilde\\phi(\\omega)`, as the definition depends only on
    :math:`|\\omega|`. An even window is what makes the corresponding time-domain wavelet real."""
    n_f = 32
    omega = np.linspace(0.0, 2 * layer_spacing(n_f), 4001)

    assert np.array_equal(phitilde_vec(omega, n_f, nx), phitilde_vec(-omega, n_f, nx))


@pytest.mark.unit
@pytest.mark.parametrize("n_f", [16, 32])
@pytest.mark.parametrize("nx", STEEPNESS_VALUES)
def test_translated_window_squares_form_a_partition_of_unity(n_f, nx):
    """:math:`\\sum_m \\tilde\\phi(\\omega - m\\Delta\\Omega)^2 = 1/\\Delta\\Omega` for all
    :math:`\\omega`.

    This is the orthonormality (Parseval) condition of the WDM basis: it says the layers tile the
    spectrum without gaps or double counting, and it is the reason the transform conserves energy.
    Testing it here separates "the window is right" from "the transform built on it is right" --
    the energy test in ``test_wavelet_transforms.py`` would fail for either cause, this one only
    for the first.

    The sum is taken over enough neighbours to cover the window's compact support
    (:math:`3\\Delta\\Omega/4`, so two layers either side is already generous).
    """
    d_omega = layer_spacing(n_f)
    omega = np.linspace(0.0, d_omega, 501)

    total = sum(phitilde_vec(omega - m * d_omega, n_f, nx) ** 2 for m in range(-3, 4))

    expected = 1.0 / d_omega
    # Peak-relative, never per-sample relative: the deviation is judged against the constant the
    # sum should equal.
    assert np.max(np.abs(total - expected)) / expected <= 64 * EPS


@pytest.mark.unit
@pytest.mark.parametrize("nx", STEEPNESS_VALUES)
def test_window_taper_decreases_monotonically(nx):
    """The taper falls monotonically from the passband edge to zero.

    :math:`\\cos(\\pi y / 2)` is decreasing on :math:`y \\in [0, 1]` and :math:`I_x(d, d)` is
    increasing in :math:`x`, so the composition must decrease. A non-monotone window would mean
    the incomplete-beta taper had been mis-assembled while still hitting the endpoint values the
    tests above pin.
    """
    n_f = 32
    d_omega = layer_spacing(n_f)
    taper = np.linspace(d_omega / 4, 3 * d_omega / 4, 2001)

    values = phitilde_vec(taper, n_f, nx)

    assert np.all(np.diff(values) <= 0.0)
    assert values[0] > values[-1]  # the taper actually falls; not a constant


@pytest.mark.unit
@pytest.mark.parametrize("n_t", [32, 64])
@pytest.mark.parametrize("nx", STEEPNESS_VALUES)
def test_phitilde_vec_norm_returns_a_unit_normalised_window(n_t, nx):
    """``phitilde_vec_norm`` divides by exactly the norm that makes its own norm functional 1.

    The source comments the intent as "nrm should be 1". Recomputing that functional on the
    returned array is the check: it is a property of the output, not a value copied from a run.
    """
    n_f = 32
    n_d = n_f * n_t

    phif = phitilde_vec_norm(n_f, n_t, nx)

    assert phif.shape == (n_t // 2 + 1,)
    norm = np.sqrt((2 * np.sum(phif[1:] ** 2) + phif[0] ** 2) * 2 * np.pi / n_d)
    norm /= np.pi ** (3 / 2) / np.pi
    assert abs(norm - 1.0) <= 64 * EPS


@pytest.mark.unit
def test_normalised_window_depends_on_steepness_only_through_its_taper():
    """``nx`` leaves the passband untouched and reshapes the taper.

    The split matters, and it is why the forward/inverse pair must be evaluated at the *same*
    ``nx``. The flat passband is :math:`1/\\sqrt{\\Delta\\Omega}` for every steepness, so a
    mismatched ``nx`` is invisible in the low-frequency layers; the taper moves by over 10% of the
    window's peak, so it is not invisible anywhere else. ``test_wavelet_transforms.py`` records the
    defect that this makes reachable through ``WaveletTransform``.
    """
    n_f, n_t = 32, 64
    d_omega = layer_spacing(n_f)
    # The sampled angular frequencies of phitilde_vec_norm, so passband and taper can be separated.
    oms = 2 * np.pi / (n_f * n_t) * np.arange(0, n_t // 2 + 1)
    in_passband = np.abs(oms) < d_omega / 4

    reference = phitilde_vec_norm(n_f, n_t, 4.0)

    assert in_passband.any()  # the passband is actually sampled
    assert (~in_passband).any()  # the taper is actually sampled
    for nx in (2.0, 8.0):
        other = phitilde_vec_norm(n_f, n_t, nx)
        peak = np.max(np.abs(reference))
        assert np.max(np.abs(other[in_passband] - reference[in_passband])) / peak <= 64 * EPS
        # The taper genuinely responds to nx -- so a mismatched nx is a real numerical error and
        # not a cosmetic one.
        assert np.max(np.abs(other[~in_passband] - reference[~in_passband])) / peak > 0.1
