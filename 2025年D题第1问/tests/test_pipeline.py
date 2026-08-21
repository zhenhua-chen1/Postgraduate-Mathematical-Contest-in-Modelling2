import unittest
from pathlib import Path

import numpy as np

import main


class EndToEndPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
        cls.profiler = main.load_wind_profiler(cls.raw_dir)
        cls.radiometer = main.load_radiometers(cls.raw_dir)
        cls.model_a = main.compute_model_a(cls.profiler, cls.radiometer)
        cls.model_b_table = main.add_temporal_features(main.add_model_b_shear(cls.model_a))

    def test_official_question_one_file_counts(self):
        self.assertEqual(self.profiler["profiler_source"].nunique(), 12)
        self.assertEqual(self.profiler["profile_id"].nunique(), 12)
        # The official raw files contain 453 valid rows. The 2025 manual table
        # dropped nine upper-level station-A observations to force a common grid.
        self.assertEqual(len(self.profiler), 453)

    def test_model_a_is_finite_and_complete(self):
        self.assertEqual(len(self.model_a), 453)
        self.assertTrue(np.isfinite(self.model_a["ri_model_a"]).all())
        self.assertTrue(self.model_a["ri_target"].between(0, 20).all())

    def test_station_b_missing_initial_radiometer_time_is_tracked(self):
        station_b = self.model_a[self.model_a["station_name"] == "b"]
        first_time = station_b["time"].min()
        first_profile = station_b[station_b["time"] == first_time]
        self.assertEqual(float(first_profile["nearest_radiometer_gap_min"].iloc[0]), 2.0)

    def test_temporal_features_are_finite(self):
        columns = [
            "horizontal_wind_tendency_mps2",
            "vertical_wind_tendency_mps2",
            "shear_tendency_per_s2",
        ]
        self.assertTrue(np.isfinite(self.model_b_table[columns].to_numpy()).all())
        self.assertEqual(int(self.model_b_table["temporal_support_count"].max()), 6)


if __name__ == "__main__":
    unittest.main()
