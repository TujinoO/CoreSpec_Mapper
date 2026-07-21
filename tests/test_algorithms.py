import unittest

import numpy as np

from corespec_mapper.algorithms import (
    absorption_feature_metrics,
    continuum_remove,
    continuum_remove_linear,
    directional_stripe_mask,
    remove_elongated_components,
    robust_column_bias,
    savgol_smooth,
    spectral_angles,
    spectral_feature_fit,
)


class AlgorithmTests(unittest.TestCase):
    def test_sam_identity_and_orthogonal(self):
        pixels = np.array([[1.0, 0.0], [0.0, 1.0]])
        references = np.array([[1.0, 0.0]])
        angles = spectral_angles(pixels, references)
        self.assertTrue(np.isclose(angles[0, 0], 0.0))
        self.assertTrue(np.isclose(angles[1, 0], np.pi / 2))

    def test_continuum_removal_uses_upper_hull(self):
        wavelengths = np.array([0.0, 1.0, 2.0])
        spectrum = np.array([1.0, 0.5, 1.0])
        removed = continuum_remove(spectrum, wavelengths)
        np.testing.assert_allclose(removed, [1.0, 0.5, 1.0])

    def test_linear_continuum_is_vectorized_and_preserves_absorption_center(self):
        wavelengths = np.linspace(2100.0, 2300.0, 101)
        continuum = 0.35 + 0.0004 * (wavelengths - wavelengths[0])
        absorption = 1.0 - 0.18 * np.exp(-0.5 * ((wavelengths - 2206.0) / 8.0) ** 2)
        spectra = np.vstack([continuum * absorption, 1.7 * continuum * absorption])

        removed = continuum_remove_linear(spectra, wavelengths)

        self.assertEqual(removed.shape, spectra.shape)
        centers = wavelengths[np.argmin(removed, axis=1)]
        np.testing.assert_allclose(centers, [2206.0, 2206.0], atol=2.0)
        np.testing.assert_allclose(removed[0], removed[1], atol=1e-12)

    def test_linear_continuum_splits_large_wavelength_gaps(self):
        wavelengths = np.array([1000.0, 1010.0, 1020.0, 2000.0, 2010.0, 2020.0])
        spectrum = np.array([1.0, 0.8, 1.0, 3.0, 2.4, 3.0])

        removed = continuum_remove_linear(spectrum, wavelengths)

        np.testing.assert_allclose(removed, [1.0, 0.8, 1.0, 1.0, 0.8, 1.0])

    def test_sff_self_fit(self):
        reference = np.array([[0.0, 0.2, 0.0]])
        result = spectral_feature_fit(reference.copy(), reference)
        np.testing.assert_allclose(result.scale, [[1.0]], atol=1e-12)
        np.testing.assert_allclose(result.rms, [[0.0]], atol=1e-12)
        self.assertGreater(result.quality[0, 0], 1e6)

    def test_sff_quality_is_scale_over_rms(self):
        pixel = np.array([[0.0, 0.2, 0.1]])
        reference = np.array([[0.0, 0.2, 0.0]])
        result = spectral_feature_fit(pixel, reference)
        np.testing.assert_allclose(result.quality, result.scale / result.rms)

    def test_absorption_feature_metrics_tracks_center_and_ratio(self):
        wavelengths = np.array([2100.0, 2150.0, 2200.0, 2250.0, 2300.0])
        continuum_removed = np.array(
            [
                [1.0, 0.98, 0.80, 0.95, 1.0],
                [1.0, 0.90, 0.96, 0.70, 1.0],
            ]
        )
        metrics = absorption_feature_metrics(continuum_removed, wavelengths, [2175.0, 2225.0])
        np.testing.assert_allclose(metrics.center_nm, [2200.0, 2200.0])
        np.testing.assert_allclose(metrics.center_depth, [0.20, 0.04])
        np.testing.assert_allclose(metrics.full_depth, [0.20, 0.30])
        np.testing.assert_allclose(metrics.depth_ratio, [1.0, 0.04 / 0.30])

    def test_savgol_shape_and_finiteness(self):
        x = np.linspace(0, 1, 11)
        spectra = np.vstack([x, x**2])
        smoothed = savgol_smooth(spectra, 5, 2)
        self.assertEqual(smoothed.shape, spectra.shape)
        self.assertTrue(np.all(np.isfinite(smoothed)))

    def test_robust_column_bias_recovers_fixed_detector_offset(self):
        cube = np.ones((40, 9, 3), dtype=np.float64)
        cube[:, 4, :] += np.array([0.1, 0.2, 0.3])
        bias = robust_column_bias(cube, np.ones((40, 9), dtype=bool), radius=2)
        np.testing.assert_allclose(bias[4], [0.1, 0.2, 0.3], atol=1e-12)

    def test_directional_filter_separates_vertical_line_from_patch(self):
        valid = np.ones((81, 31), dtype=bool)
        candidate = np.zeros_like(valid)
        candidate[5:76, 5] = True
        candidate[35:42, 20:27] = True
        stripes = directional_stripe_mask(
            candidate,
            valid,
            vertical_window=31,
            min_vertical_density=0.3,
            column_ratio=2.0,
            column_excess=0.1,
            max_lateral_support=2,
        )
        self.assertGreater(np.count_nonzero(stripes[:, 5]), 50)
        self.assertEqual(np.count_nonzero(stripes[:, 20:27]), 0)
        cleaned = remove_elongated_components(candidate, min_height=20, max_width=2, min_aspect=5.0)
        self.assertEqual(np.count_nonzero(cleaned[:, 5]), 0)
        self.assertEqual(np.count_nonzero(cleaned[:, 20:27]), 49)


if __name__ == "__main__":
    unittest.main()
