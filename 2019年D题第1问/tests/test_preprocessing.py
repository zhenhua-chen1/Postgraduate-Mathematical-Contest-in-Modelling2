import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from preprocessing import preprocess_vehicle_data, validate_processed_data


def sample(times, speeds):
    return pd.DataFrame(
        {
            "时间": [pd.Timestamp(value).strftime("%Y/%m/%d %H:%M:%S.000.") for value in times],
            "GPS车速": speeds,
            "经度": np.arange(len(times), dtype=float),
        }
    )


class PreprocessingTests(unittest.TestCase):
    def test_two_second_gap_uses_linear_midpoint(self):
        raw = sample(["2017-01-01 00:00:00", "2017-01-01 00:00:02"], [9.1, 24.3])
        result, detail, metrics = preprocess_vehicle_data(raw)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result.loc[1, "GPS车速"], 16.7)
        self.assertAlmostEqual(result.loc[1, "经度"], 0.5)
        self.assertEqual(metrics["插值新增数"], 1)
        self.assertEqual(detail.iloc[0]["处理类型"], "新增插值记录")

    def test_acceleration_limits_keep_kmh_units(self):
        raw = sample(pd.date_range("2017-01-01", periods=3, freq="s"), [36.0, 72.0, 0.0])
        result, _, metrics = preprocess_vehicle_data(raw)
        self.assertAlmostEqual(result.loc[1, "GPS车速"], 50.256)
        self.assertAlmostEqual(result.loc[2, "GPS车速"], 21.456)
        self.assertEqual(metrics["车速修正数"], 2)
        validate_processed_data(result)

    def test_long_low_speed_run_is_capped_at_180_records(self):
        raw = sample(pd.date_range("2017-01-01", periods=185, freq="s"), [5.0] * 185)
        result, detail, metrics = preprocess_vehicle_data(raw)
        self.assertEqual(len(result), 180)
        self.assertEqual(metrics["长低速删除数"], 5)
        self.assertEqual((detail["处理类型"] == "删除长低速记录").sum(), 5)
        validate_processed_data(result)

    def test_long_gap_starts_new_segment_and_is_not_interpolated(self):
        raw = sample(
            ["2017-01-01 00:00:00", "2017-01-01 00:10:00", "2017-01-01 00:10:01"],
            [0.0, 100.0, 0.0],
        )
        result, _, metrics = preprocess_vehicle_data(raw)
        self.assertEqual(len(result), 3)
        self.assertEqual(metrics["保留的长间隔数"], 1)
        self.assertAlmostEqual(result.loc[1, "GPS车速"], 100.0)
        self.assertAlmostEqual(result.loc[2, "GPS车速"], 71.2)

    def test_invalid_time_order_duplicate_and_negative_speed_are_rejected(self):
        cases = [
            sample(["2017-01-01 00:00:01", "2017-01-01 00:00:00"], [0, 0]),
            sample(["2017-01-01 00:00:00", "2017-01-01 00:00:00"], [0, 0]),
            sample(["2017-01-01 00:00:00", "2017-01-01 00:00:01"], [0, -1]),
            sample(["2017-01-01 00:00:00", "2017-01-01 00:00:01"], [0, np.inf]),
        ]
        for frame in cases:
            with self.subTest(frame=frame):
                with self.assertRaises(ValueError):
                    preprocess_vehicle_data(frame)

    def test_input_is_not_modified(self):
        raw = sample(pd.date_range("2017-01-01", periods=3, freq="s"), [0, 1, 2])
        before = raw.copy(deep=True)
        preprocess_vehicle_data(raw)
        pd.testing.assert_frame_equal(raw, before)


if __name__ == "__main__":
    unittest.main()
