from pathlib import Path
import json
import unittest

from corespec_mapper.envi import EnviDataset, SpectralLibrary, wavelength_indices


ROOT = Path(r"F:\NC-1-31_40")


@unittest.skipUnless(ROOT.exists(), "NC-1 dataset is not available")
class RealDataTests(unittest.TestCase):
    def test_nc1_metadata_smoke(self):
        image = EnviDataset(ROOT / r"2026_07_02_16_48_52-NC-1-31.0_40.0-1,717.0_2,023.1\SWIR-20260702_164852-00000.dat")
        self.assertEqual((image.info.samples, image.info.lines, image.info.bands), (320, 5446, 212))
        self.assertEqual(image.info.interleave, "bil")
        library = SpectralLibrary.open(ROOT / r"FILL\Calcite_Dolomite.sli")
        self.assertEqual(library.spectra.shape, (4, 212))
        self.assertEqual(len(library.names), 4)

    def test_envi_equivalent_sam_band_windows(self):
        config_path = Path(__file__).parents[1] / "configs" / "nc1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        image = EnviDataset(config["analysis_image"])
        wavelengths = image.info.wavelengths_nm
        self.assertIsNotNone(wavelengths)
        expected = {
            "carbonates": (174, 195, 22),
            "sulfates": (51, 188, 131),
            "clays": (154, 175, 22),
        }
        for group in config["groups"]:
            indices = wavelength_indices(wavelengths, group["sam_windows"])
            first, last, count = expected[group["name"]]
            self.assertEqual((int(indices[0]), int(indices[-1]), int(indices.size)), (first, last, count))
        image.close()


if __name__ == "__main__":
    unittest.main()
