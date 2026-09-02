from pathlib import Path
import tempfile
import unittest

import numpy as np

from corespec_mapper.empirical_vnir import (
    candidate_neighborhood_support,
    empirical_absorption_features,
    local_radiometric_stability_mask,
    policy_candidate,
    read_envi_ascii_spectrum,
    seeded_candidate_region_grow,
    spectral_angles,
    time_normalized_mask,
)


class EmpiricalVnirTests(unittest.TestCase):
    def test_reads_envi_ascii_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.txt"
            path.write_text(
                "ENVI ASCII Plot File\nColumn 1: Wavelength\nColumn 2: source pixel\n"
                "  700.0  0.2\n  710.0  0.18\n  720.0  0.21\n",
                encoding="utf-8",
            )
            result = read_envi_ascii_spectrum(path)
        self.assertEqual(result.source_label, "source pixel")
        np.testing.assert_allclose(result.wavelengths_nm, (700.0, 710.0, 720.0))

    def test_feature_angle_is_albedo_invariant(self):
        x = np.linspace(0.0, 1.0, 81)
        spectrum = 0.2 + 0.15 * x - 0.035 * np.exp(-((x - 0.55) / 0.05) ** 2)
        features = empirical_absorption_features(np.vstack((spectrum, spectrum * 2.5)))
        angle = spectral_angles(features[1:], features[:1])[0, 0]
        self.assertLess(angle, 1e-6)

    def test_time_mapping_keeps_endpoints_and_width(self):
        mask = np.zeros((5, 3), dtype=bool)
        mask[0, 0] = True
        mask[-1, -1] = True
        mapped, indices = time_normalized_mask(mask, 9)
        self.assertEqual(mapped.shape, (9, 3))
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 4)
        self.assertTrue(mapped[0, 0])
        self.assertTrue(mapped[-1, -1])

    def test_policy_candidates_are_nested(self):
        angle = np.asarray([[0.1, 0.25, 0.34, 0.5]])
        strength = np.full_like(angle, 0.03)
        valid = np.ones_like(angle, dtype=bool)
        conservative = policy_candidate(
            angle, strength, valid, maximum_angle_rad=0.2,
            minimum_strength=0.01, maximum_strength=0.2,
        )
        balanced = policy_candidate(
            angle, strength, valid, maximum_angle_rad=0.3,
            minimum_strength=0.008, maximum_strength=0.2,
        )
        sensitive = policy_candidate(
            angle, strength, valid, maximum_angle_rad=0.4,
            minimum_strength=0.005, maximum_strength=0.2,
        )
        self.assertTrue(np.all(conservative <= balanced))
        self.assertTrue(np.all(balanced <= sensitive))

    def test_radiometric_stability_keeps_uniform_interior_and_rejects_edge(self):
        reflectance = np.full((21, 21), 0.30, dtype=np.float32)
        reflectance[:, 11:] = 0.08
        foreground = np.ones_like(reflectance, dtype=bool)
        stable, support, coefficient_of_variation, log_gradient = local_radiometric_stability_mask(
            reflectance,
            foreground,
            window_size=5,
            minimum_valid_fraction=0.8,
            maximum_coefficient_of_variation=0.3,
            maximum_log_gradient=0.25,
        )
        self.assertTrue(stable[10, 4])
        self.assertTrue(stable[10, 17])
        self.assertFalse(stable[10, 10])
        self.assertGreater(coefficient_of_variation[10, 10], 0.3)
        self.assertGreater(log_gradient[10, 10], 0.25)
        self.assertEqual(support[10, 10], 1.0)

    def test_radiometric_stability_requires_neighborhood_support(self):
        reflectance = np.full((9, 9), 0.2, dtype=np.float32)
        foreground = np.zeros_like(reflectance, dtype=bool)
        foreground[2:7, 2:7] = True
        stable, support, _, _ = local_radiometric_stability_mask(
            reflectance, foreground, minimum_valid_fraction=0.8
        )
        self.assertTrue(stable[4, 4])
        self.assertFalse(stable[2, 2])
        self.assertLess(support[2, 2], 0.8)

    def test_candidate_support_removes_thin_line_and_keeps_patch(self):
        candidate = np.zeros((11, 11), dtype=bool)
        candidate[2, 1:8] = True
        candidate[6:9, 6:9] = True
        supported, counts = candidate_neighborhood_support(candidate, minimum_pixels=4)
        self.assertFalse(np.any(supported[2, 1:8]))
        self.assertTrue(supported[7, 7])
        self.assertEqual(counts[7, 7], 9)

    def test_seeded_growth_expands_connected_domain_without_creating_islands(self):
        seeds = np.zeros((9, 12), dtype=bool)
        seeds[4, 3] = True
        domain = np.zeros_like(seeds)
        domain[3:6, 2:7] = True
        domain[1:3, 9:11] = True
        grown = seeded_candidate_region_grow(seeds, domain)
        self.assertTrue(np.all(grown[3:6, 2:7]))
        self.assertFalse(np.any(grown[1:3, 9:11]))

    def test_seeded_growth_requires_matching_two_dimensional_masks(self):
        with self.assertRaises(ValueError):
            seeded_candidate_region_grow(np.ones((3, 3), bool), np.ones((3, 3, 1), bool))


if __name__ == "__main__":
    unittest.main()
