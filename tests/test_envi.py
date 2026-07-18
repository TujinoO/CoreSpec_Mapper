from pathlib import Path
import tempfile
import unittest

import numpy as np

from corespec_mapper.envi import EnviDataset, SpectralLibrary, classification_palette, parse_envi_header, write_envi


class EnviTests(unittest.TestCase):
    def test_reads_bil_raster(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            cube = np.arange(2 * 3 * 4, dtype="<f4").reshape(2, 3, 4)
            data_path = tmp_path / "cube.dat"
            cube.tofile(data_path)
            (tmp_path / "cube.hdr").write_text(
                "ENVI\n"
                "samples = 4\n"
                "lines = 2\n"
                "bands = 3\n"
                "data type = 4\n"
                "interleave = bil\n"
                "byte order = 0\n"
                "wavelength = {1000, 1100, 1200}\n",
                encoding="utf-8",
            )
            dataset = EnviDataset(data_path)
            rows = dataset.read_rows(0, 2)
            self.assertEqual(rows.shape, (2, 4, 3))
            np.testing.assert_array_equal(rows[0, :, 0], cube[0, 0, :])
            dataset.close()

    def test_spectral_library_unit_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            spectra = np.array([[0.8, 0.7, 0.9], [0.6, 0.5, 0.7]], dtype="<f4")
            data_path = tmp_path / "library.sli"
            spectra.tofile(data_path)
            (tmp_path / "library.hdr").write_text(
                "ENVI\n"
                "samples = 3\n"
                "lines = 2\n"
                "bands = 1\n"
                "data type = 4\n"
                "interleave = bsq\n"
                "byte order = 0\n"
                "file type = ENVI Spectral Library\n"
                "wavelength units = Micrometers\n"
                "wavelength = {1000, 1100, 1200}\n"
                "spectra names = {A, B}\n",
                encoding="utf-8",
            )
            library = SpectralLibrary.open(data_path)
            self.assertEqual(library.spectra.shape, (2, 3))
            self.assertIsNotNone(library.warning)
            np.testing.assert_allclose(library.wavelengths_nm, [1000, 1100, 1200])

    def test_micrometer_library_extending_past_twenty_micrometers(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data_path = tmp_path / "library.sli"
            np.array([[0.8, 0.7, 0.9]], dtype=">f4").tofile(data_path)
            (tmp_path / "library.hdr").write_text(
                "ENVI\n"
                "samples = 3\n"
                "lines = 1\n"
                "bands = 1\n"
                "data type = 4\n"
                "interleave = bsq\n"
                "byte order = 1\n"
                "file type = ENVI Spectral Library\n"
                "wavelength units = Micrometers\n"
                "wavelength = {2.08, 10.0, 25.0}\n"
                "spectra names = {A}\n",
                encoding="utf-8",
            )
            library = SpectralLibrary.open(data_path)
            np.testing.assert_allclose(library.wavelengths_nm, [2080.0, 10000.0, 25000.0])

    def test_classification_header_contains_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "classes.dat"
            write_envi(
                np.array([[0, 1], [2, 3]], dtype=np.uint8),
                data_path,
                class_names=["Unclassified", "Calcite", "Dolomite", "Masked Pixels"],
            )
            header = parse_envi_header(data_path.with_suffix(".hdr"))
            self.assertEqual(header["classes"], 4)
            self.assertEqual(len(header["class lookup"]), 12)
            self.assertEqual(classification_palette(header["class names"])[3:6], [255, 0, 0])


if __name__ == "__main__":
    unittest.main()
