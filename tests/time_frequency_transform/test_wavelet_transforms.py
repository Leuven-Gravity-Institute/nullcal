"""Tests for the WDM wavelet transform, anchored outside the implementation.

Every claim below is checked against something the package did not compute: a theorem (Parseval),
an identity (forward followed by inverse is the identity map), an analytically known answer (a
monochromatic tone occupies the frequency layer containing its frequency), or the published
definition of the transform. No test here compares the transform against a stored value that a
previous run of this same code produced -- such a comparison would pin the current behaviour
without saying whether it is right.

References for the definition and its properties:

* N. J. Cornish, "Time-frequency analysis of gravitational wave data",
  Phys. Rev. D **102**, 124038 (2020), arXiv:2009.00043.
* V. Necula, S. Klimenko and G. Mitselmakher, "Transient analysis with fast Wilson-Daubechies
  time-frequency transform", J. Phys. Conf. Ser. **363**, 012032 (2012).

Tolerances. The transform is a sequence of FFTs in float64. The standard round-off bound for an
FFT of length :math:`N` grows like :math:`\\varepsilon \\log_2 N` with
:math:`\\varepsilon = 2.2\\times10^{-16}`; for the :math:`N = 2048` used here that is
:math:`\\approx 2.4\\times10^{-15}`, and a forward-plus-inverse round trip doubles it. The
tolerance used for exact-identity claims is therefore ``ROUND_TRIP_TOL = 1e-12``, some two
hundred times that bound. It is derived from the bound, not from the measured residual, so it
cannot be silently widened to rescue a regression: a change that degrades the transform by three
orders of magnitude and still passes would have to be a change that broke the bound itself.

Comparisons are **peak-relative** (``max|diff| / max|reference|``), never per-sample relative. A
per-sample relative test on a transform output is meaningless because the output contains pixels
that are legitimately zero to round-off. Where absolute tolerances appear they are ``0.0``: at
strain amplitudes (~1e-24) ``numpy``'s default ``atol=1e-8`` would make any two such arrays
compare equal and the assertion could never fail. ``test_scale_covariance_at_strain_amplitude``
demonstrates that failure mode rather than just asserting around it.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullcal.time_frequency_transform import (
    get_shape_of_wavelet_transform,
    transform_wavelet_freq,
    transform_wavelet_freq_quadrature,
    transform_wavelet_freq_time,
    transform_wavelet_freq_time_quadrature,
)
from nullcal.time_frequency_transform.wavelet_transforms import (
    WaveletTransform,
    inverse_wavelet_freq,
    inverse_wavelet_freq_time,
)

SAMPLING_FREQUENCY = 512.0
DURATION = 4.0
FREQUENCY_RESOLUTION = 8.0
N_SAMPLES = int(SAMPLING_FREQUENCY * DURATION)

# See the module docstring: 1e-12 is ~200x the float64 FFT round-off bound eps*log2(N) for N=2048.
ROUND_TRIP_TOL = 1e-12


def peak_relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    """``max|actual - reference| / max|reference|``.

    The accuracy measure used throughout: it compares the largest discrepancy against the largest
    value present, so it is unaffected by pixels that are legitimately zero and is invariant to an
    overall rescaling of the data.
    """
    return float(np.max(np.abs(actual - reference)) / np.max(np.abs(reference)))


@pytest.fixture
def shape():
    """``(n_t, n_f)`` for the fixture configuration: 64 time rows by 32 frequency layers."""
    return get_shape_of_wavelet_transform(
        t_length=N_SAMPLES,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_resolution=FREQUENCY_RESOLUTION,
    )


@pytest.fixture
def time_array():
    """Sample times of the fixture time series."""
    return np.arange(N_SAMPLES) / SAMPLING_FREQUENCY


@pytest.fixture
def noise():
    """A fixed-seed Gaussian time series. Seeded so a failure is reproducible, not to pin values."""
    return np.random.default_rng(20260914).normal(size=N_SAMPLES)


@pytest.mark.unit
def test_tiling_satisfies_the_wdm_half_density_condition(shape):
    """:math:`\\Delta t \\, \\Delta f = 1/2`, the defining tiling of the WDM basis.

    A WDM decomposition of :math:`N` samples produces :math:`n_t \\times n_f = N` real
    coefficients, so the pixel area must be exactly one half in natural units -- the transform is
    critically sampled, neither redundant nor lossy. Both cited references state this condition.
    Getting it wrong would mean every frequency axis the package draws is mislabelled.
    """
    n_t, n_f = shape

    assert n_t * n_f == N_SAMPLES

    delta_t = DURATION / n_t
    delta_f = SAMPLING_FREQUENCY / (2 * n_f)
    assert delta_f == pytest.approx(FREQUENCY_RESOLUTION, rel=1e-15)
    assert delta_t * delta_f == pytest.approx(0.5, rel=1e-15)


@pytest.mark.unit
def test_transform_conserves_energy_by_a_constant_factor(shape, noise, time_array):
    """Parseval: the wavelet coefficients carry the signal's energy, up to one fixed constant.

    The WDM wavepackets form an orthonormal basis (the partition of unity checked in
    ``test_wdm_window.py``), so :math:`\\sum_{n,m} w_{nm}^2` must be proportional to
    :math:`\\sum_j x_j^2` with a proportionality constant that depends only on the normalisation
    convention -- never on the data. That data-independence is the whole content of the theorem
    and is what this test asserts: the ratio is measured for four inputs with completely different
    spectral content and required to be the same number.

    The constant this implementation uses is :math:`N`, i.e. :math:`w/\\sqrt{N}` are
    orthonormal-basis coefficients. That value is asserted separately so a change of convention is
    caught rather than absorbed.
    """
    n_t, n_f = shape
    impulse = np.zeros(N_SAMPLES)
    impulse[N_SAMPLES // 3] = 1.0
    inputs = {
        "gaussian noise": noise,
        "tone": np.sin(2 * np.pi * 10 * FREQUENCY_RESOLUTION * time_array),
        "impulse": impulse,
        "chirp": np.sin(2 * np.pi * (20.0 + 12.0 * time_array) * time_array),
    }

    ratios = {}
    for name, data in inputs.items():
        wave = transform_wavelet_freq_time(data, n_f=n_f, n_t=n_t)
        ratios[name] = np.sum(wave**2) / np.sum(data**2)

    values = np.array(list(ratios.values()))
    assert peak_relative_error(values, np.full_like(values, N_SAMPLES)) <= ROUND_TRIP_TOL, (
        f"energy ratio is not data-independent: {ratios}"
    )


@pytest.mark.unit
def test_forward_then_inverse_recovers_the_frequency_domain_data(shape, noise):
    """Round trip in the frequency domain is the identity, to the FFT round-off bound.

    Invertibility is the property that says no information was destroyed. It is an identity, so
    the target is exact equality and the tolerance is pure arithmetic round-off -- there is no
    physics in it to justify loosening.
    """
    n_t, n_f = shape
    frequency_domain_data = np.fft.rfft(noise)

    wave = transform_wavelet_freq(frequency_domain_data, n_f=n_f, n_t=n_t)
    recovered = inverse_wavelet_freq(wave, n_f=n_f, n_t=n_t)

    assert recovered.shape == frequency_domain_data.shape
    assert peak_relative_error(recovered, frequency_domain_data) <= ROUND_TRIP_TOL


@pytest.mark.unit
def test_forward_then_inverse_recovers_the_time_domain_data(shape, noise):
    """The same identity taken all the way back to the time series."""
    n_t, n_f = shape

    wave = transform_wavelet_freq_time(noise, n_f=n_f, n_t=n_t)
    recovered = inverse_wavelet_freq_time(wave, n_f=n_f, n_t=n_t)

    assert recovered.shape == noise.shape
    assert peak_relative_error(recovered, noise) <= ROUND_TRIP_TOL


@pytest.mark.unit
@pytest.mark.parametrize("nx", [2.0, 4.0, 8.0])
def test_round_trip_holds_at_every_filter_steepness(shape, noise, nx):
    """Invertibility does not depend on ``nx`` when forward and inverse agree on it.

    The free functions take ``nx`` explicitly, so this test exercises the transform as designed and
    establishes that the *algorithm* is steepness-agnostic. It is the control for the defect
    recorded in ``test_wavelet_transform_class_ignores_its_steepness_argument``: there, the round
    trip fails not because the mathematics depends on ``nx`` but because the two halves are
    evaluated at different values of it.
    """
    n_t, n_f = shape
    frequency_domain_data = np.fft.rfft(noise)

    wave = transform_wavelet_freq(frequency_domain_data, n_f=n_f, n_t=n_t, nx=nx)
    recovered = inverse_wavelet_freq(wave, n_f=n_f, n_t=n_t, nx=nx)

    assert peak_relative_error(recovered, frequency_domain_data) <= ROUND_TRIP_TOL


@pytest.mark.unit
@pytest.mark.parametrize("layer", [5, 9, 12, 20])
def test_monochromatic_tone_occupies_its_own_frequency_layer(shape, time_array, layer):
    """A tone at :math:`f = m \\Delta f` puts essentially all its energy in layer :math:`m`.

    This is the analytically known answer that fixes the frequency *calibration* of the transform:
    Parseval and the round trip are both satisfied by a transform whose layers are permuted or
    offset by one, and this test is not. The expected layer index follows from the tiling --
    :math:`\\Delta f = f_s / (2 n_f)` -- not from running the code.
    """
    n_t, n_f = shape
    frequency = layer * FREQUENCY_RESOLUTION

    wave = transform_wavelet_freq_time(np.sin(2 * np.pi * frequency * time_array), n_f=n_f, n_t=n_t)

    power_per_layer = np.sum(wave**2, axis=0)
    assert int(np.argmax(power_per_layer)) == layer
    # Leakage is bounded by the window's compact support, so the containment is near-total.
    assert power_per_layer[layer] / np.sum(power_per_layer) > 0.999


@pytest.mark.unit
def test_tone_layer_index_tracks_the_requested_frequency_resolution(time_array):
    """Halving ``frequency_resolution`` doubles the layer index of the same tone.

    A transform that happened to land tones in the right layer for one configuration but did not
    scale with the requested resolution would pass the test above at a single setting. The
    relationship asserted here is the tiling's, so it holds for any pair of resolutions.
    """
    frequency = 80.0

    layers = {}
    for frequency_resolution in (8.0, 4.0):
        n_t, n_f = get_shape_of_wavelet_transform(
            t_length=N_SAMPLES,
            sampling_frequency=SAMPLING_FREQUENCY,
            frequency_resolution=frequency_resolution,
        )
        wave = transform_wavelet_freq_time(np.sin(2 * np.pi * frequency * time_array), n_f=n_f, n_t=n_t)
        layers[frequency_resolution] = int(np.argmax(np.sum(wave**2, axis=0)))

    assert layers[8.0] == int(frequency / 8.0)
    assert layers[4.0] == int(frequency / 4.0)
    assert layers[4.0] == 2 * layers[8.0]


@pytest.mark.unit
def test_short_burst_lands_in_the_expected_time_row(shape, time_array):
    """A burst centred at :math:`t_c` peaks in the time row containing :math:`t_c`.

    The counterpart of the tone test for the other axis: it fixes the transform's time
    calibration. The tolerance is one time pixel, which is the resolution of the representation
    itself -- :math:`\\Delta t` -- and so is the tightest claim the transform can support.
    """
    n_t, n_f = shape
    delta_t = DURATION / n_t
    expected_row = 20
    centre_time = (expected_row + 0.5) * delta_t
    burst = np.exp(-((time_array - centre_time) ** 2) / (2 * 0.03**2)) * np.sin(
        2 * np.pi * 10 * FREQUENCY_RESOLUTION * time_array
    )

    wave = transform_wavelet_freq_time(burst, n_f=n_f, n_t=n_t)

    row, layer = np.unravel_index(int(np.argmax(np.abs(wave))), wave.shape)
    assert layer == 10
    assert abs(row - centre_time / delta_t) <= 1.0


@pytest.mark.unit
def test_transform_is_linear(shape, noise, time_array):
    """:math:`W(a x + b y) = a W(x) + b W(y)`.

    Linearity is assumed everywhere downstream -- the null stream is a linear combination of
    detector strains that the package transforms term by term, so a non-linear transform would
    make the whole null-stream construction invalid.
    """
    n_t, n_f = shape
    other = np.sin(2 * np.pi * 96.0 * time_array)
    a, b = 2.5, -0.75

    combined = transform_wavelet_freq_time(a * noise + b * other, n_f=n_f, n_t=n_t)
    separate = a * transform_wavelet_freq_time(noise, n_f=n_f, n_t=n_t) + b * transform_wavelet_freq_time(
        other, n_f=n_f, n_t=n_t
    )

    assert peak_relative_error(combined, separate) <= ROUND_TRIP_TOL


@pytest.mark.unit
def test_scale_covariance_at_strain_amplitude(shape, noise):
    """Scaling the data scales the coefficients, checked at gravitational-wave strain amplitude.

    Two things are being established.

    First the physics: the transform must be exactly covariant under rescaling, because the data it
    is applied to are strains of order 1e-24 and any amplitude-dependent behaviour would be a
    catastrophic, silent bias.

    Second, the test's own validity. ``numpy``'s ``allclose``/``isclose`` default to
    ``atol=1e-8``, which is twelve orders of magnitude *larger* than the quantities compared here;
    with that default, any two strain-scale arrays compare equal and the assertion cannot fail. The
    body below asserts that the comparison actually discriminates -- the default-tolerance
    comparison is shown to accept an obviously wrong answer (an array of zeros), while the
    peak-relative comparison used for the real assertion rejects it.
    """
    n_t, n_f = shape
    strain_scale = 1e-24

    reference = transform_wavelet_freq_time(noise, n_f=n_f, n_t=n_t)
    scaled = transform_wavelet_freq_time(noise * strain_scale, n_f=n_f, n_t=n_t)

    expected = strain_scale * reference
    assert peak_relative_error(scaled, expected) <= ROUND_TRIP_TOL

    # The assertion above must be capable of failing. An array of zeros is not the right answer,
    # and numpy's default absolute tolerance says it is.
    assert np.allclose(scaled, np.zeros_like(scaled)), (
        "precondition of this test: at strain scale numpy's default atol accepts anything"
    )
    assert not np.allclose(scaled, np.zeros_like(scaled), rtol=0.0, atol=0.0)
    assert peak_relative_error(np.zeros_like(scaled), expected) > ROUND_TRIP_TOL


@pytest.mark.unit
def test_quadrature_transform_conserves_energy_by_the_same_constant(shape, noise):
    """The quadrature transform obeys the same Parseval constant as the in-phase transform.

    The two are the real and imaginary parts of one analytic decomposition, so they must share a
    normalisation. A mismatch would put a constant factor into the time-frequency power map that
    the clustering threshold is compared against.
    """
    n_t, n_f = shape

    wave = transform_wavelet_freq_time_quadrature(noise, n_f=n_f, n_t=n_t)

    assert wave.shape == (n_t, n_f)
    assert np.sum(wave**2) / np.sum(noise**2) == pytest.approx(N_SAMPLES, rel=ROUND_TRIP_TOL)


@pytest.mark.unit
def test_quadrature_pair_gives_a_phase_independent_envelope(shape, time_array):
    """:math:`w^2 + w_{\\rm quad}^2` is the same for a cosine and a sine of the same frequency.

    This is what "quadrature" has to mean: the two transforms are a Hilbert pair, so their squared
    sum is the signal's envelope, which for a steady tone is independent of the tone's phase. It is
    an external property -- it follows from the definition of an analytic signal, not from this
    code -- and it is the property ``construct_time_frequency_map`` relies on when it adds the two
    powers to get a phase-insensitive energy map.
    """
    n_t, n_f = shape
    frequency = 10 * FREQUENCY_RESOLUTION

    envelopes = []
    for data in (np.cos(2 * np.pi * frequency * time_array), np.sin(2 * np.pi * frequency * time_array)):
        in_phase = transform_wavelet_freq_time(data, n_f=n_f, n_t=n_t)
        quadrature = transform_wavelet_freq_time_quadrature(data, n_f=n_f, n_t=n_t)
        envelopes.append(in_phase**2 + quadrature**2)

    # Looser than ROUND_TRIP_TOL: this is a cancellation between two O(1) terms rather than an
    # identity, so it carries the conditioning of the subtraction. Still eleven orders of magnitude
    # below the envelope itself.
    assert peak_relative_error(envelopes[0], envelopes[1]) <= 1e-11


@pytest.mark.unit
def test_quadrature_transform_is_not_the_in_phase_transform(shape, noise):
    """The two transforms genuinely differ.

    Without this, every quadrature assertion above would also be satisfied by an implementation
    that returned the in-phase transform twice.
    """
    n_t, n_f = shape

    in_phase = transform_wavelet_freq(np.fft.rfft(noise), n_f=n_f, n_t=n_t)
    quadrature = transform_wavelet_freq_quadrature(np.fft.rfft(noise), n_f=n_f, n_t=n_t)

    assert peak_relative_error(quadrature, in_phase) > 0.1


class TestWaveletTransformClass:
    """The object-oriented surface used by ``NullStream`` and the clustering code."""

    @pytest.fixture
    def transform(self):
        return WaveletTransform(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            frequency_resolution=FREQUENCY_RESOLUTION,
        )

    @pytest.mark.unit
    def test_shape_matches_the_free_function(self, transform, shape):
        """The class and the free helper must agree on the grid, since callers mix the two."""
        assert transform.shape == shape

    @pytest.mark.unit
    def test_round_trip_through_the_class(self, transform, noise):
        """``frequency_to_wavelet`` then ``wavelet_to_frequency`` is the identity at default ``nx``."""
        frequency_domain_data = np.fft.rfft(noise)

        recovered = transform.wavelet_to_frequency(transform.frequency_to_wavelet(frequency_domain_data))

        assert peak_relative_error(recovered, frequency_domain_data) <= ROUND_TRIP_TOL

    @pytest.mark.unit
    def test_rejects_frequency_domain_data_of_the_wrong_length(self, transform):
        """A length mismatch is a silent misalignment of the frequency axis if it is not caught."""
        with pytest.raises(ValueError, match="does not match the expected length"):
            transform.frequency_to_wavelet(np.zeros(transform._f_length - 1, dtype=complex))

    @pytest.mark.unit
    def test_quadrature_rejects_frequency_domain_data_of_the_wrong_length(self, transform):
        with pytest.raises(ValueError, match="does not match the expected length"):
            transform.frequency_to_wavelet_quadrature(np.zeros(transform._f_length + 1, dtype=complex))

    @pytest.mark.unit
    def test_rejects_wavelet_data_of_the_wrong_shape(self, transform):
        with pytest.raises(ValueError, match="does not match the expected shape"):
            transform.wavelet_to_frequency(np.zeros((transform.shape[0], transform.shape[1] + 1)))

    @pytest.mark.unit
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Defect: WaveletTransform.frequency_to_wavelet does not forward self.nx to "
            "transform_wavelet_freq, so the forward transform always runs at the default steepness "
            "4.0 while wavelet_to_frequency uses the requested value. Not fixed here: the transform "
            "kernels are numerically load-bearing and a change to them is a separate, reviewed "
            "decision."
        ),
    )
    def test_wavelet_transform_class_ignores_its_steepness_argument(self, noise):
        """The forward transform should depend on the ``nx`` the object was constructed with.

        ``test_round_trip_holds_at_every_filter_steepness`` shows the underlying algorithm handles
        any steepness correctly, so this is purely a plumbing fault -- and a consequential one: at
        ``nx=2.0`` the class's round trip is wrong by about 10% of the data's peak, silently.
        """
        frequency_domain_data = np.fft.rfft(noise)

        at_default = WaveletTransform(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            frequency_resolution=FREQUENCY_RESOLUTION,
            nx=4.0,
        ).frequency_to_wavelet(frequency_domain_data)
        at_sharper = WaveletTransform(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            frequency_resolution=FREQUENCY_RESOLUTION,
            nx=2.0,
        ).frequency_to_wavelet(frequency_domain_data)

        assert peak_relative_error(at_sharper, at_default) > ROUND_TRIP_TOL

    @pytest.mark.unit
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Defect: consequence of the unforwarded nx above -- the class's forward and inverse "
            "transforms use different windows whenever nx != 4.0, so the round trip is not the "
            "identity."
        ),
    )
    def test_class_round_trip_holds_at_non_default_steepness(self, noise):
        """Invertibility should not be a property only of the default ``nx``."""
        transform = WaveletTransform(
            duration=DURATION,
            sampling_frequency=SAMPLING_FREQUENCY,
            frequency_resolution=FREQUENCY_RESOLUTION,
            nx=2.0,
        )
        frequency_domain_data = np.fft.rfft(noise)

        recovered = transform.wavelet_to_frequency(transform.frequency_to_wavelet(frequency_domain_data))

        assert peak_relative_error(recovered, frequency_domain_data) <= ROUND_TRIP_TOL
