import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import main


class UtilityTests(unittest.TestCase):
    def test_timestamp_formats(self):
        self.assertEqual(
            main.timestamp_from_name(Path("wrfvar_output_1km.202507310200.nanjing.nc")),
            pd.Timestamp("2025-07-31 02:00"),
        )
        self.assertEqual(
            main.timestamp_from_name(Path("Z_RADR_I_Z9250_20250731015800_O.csv")),
            pd.Timestamp("2025-07-31 01:58"),
        )
        self.assertEqual(
            main.timestamp_from_name(Path("SURF_CHN_MUL_MIN_20250731_0230.txt")),
            pd.Timestamp("2025-07-31 02:30"),
        )

    def test_robust_unit_clips(self):
        values = main.robust_unit(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]), 0.0, 1.0)
        np.testing.assert_allclose(values, [0.0, 0.0, 0.5, 1.0, 1.0])

    def test_column_interpolation(self):
        z = np.broadcast_to(np.array([0.0, 100.0, 200.0])[:, None, None], (3, 2, 2))
        values = 2.0 * z + 5.0
        result = main.interpolate_columns(z, values, np.array([50.0, 150.0]))
        np.testing.assert_allclose(result[0], 105.0)
        np.testing.assert_allclose(result[1], 305.0)


class PortableDataTests(unittest.TestCase):
    def test_project_uses_bundled_data_only(self):
        self.assertEqual(main.DATA_DIR, main.SCRIPT_DIR / "data")
        files, grid, input_mode = main.discover_nwp_inputs(main.DATA_DIR)
        self.assertEqual(input_mode, "compact")
        self.assertEqual(len(files), 12)
        self.assertTrue(all(main.DATA_DIR in path.parents for _, path in files))
        self.assertEqual(grid.shape, (166, 94))

    def test_compact_inputs_are_compatible(self):
        files, grid, _ = main.discover_nwp_inputs(main.DATA_DIR)
        regular = main.load_compact_nwp_regular(files[0][1])
        self.assertEqual(regular["theta"].shape, (41, 166, 94))
        observed = main.load_compact_observed_c(main.DATA_DIR, grid)
        self.assertIsNotNone(observed)
        field, coverage, uncertainty, metadata = observed
        self.assertEqual(field.shape, (7, 41, 166, 94))
        self.assertEqual(coverage.shape, field.shape)
        self.assertEqual(uncertainty.shape, field.shape)
        self.assertEqual(metadata["input"], "data/model_c_validation_1km_50m.npz")


class RouteTests(unittest.TestCase):
    def test_astar_avoids_high_risk_barrier(self):
        turbulence = np.zeros((3, 7, 7), dtype=float)
        turbulence[:, 3, 3] = 1.0
        mask = np.ones((7, 7), dtype=bool)
        start, goal = (1, 3, 0), (1, 3, 6)
        route = main.astar_route(turbulence, mask, start, goal, 1000.0, 1000.0, 50.0, risk_weight=12.0)
        self.assertTrue(route.reached)
        self.assertNotIn((1, 3, 3), route.nodes)
        straight = main.straight_route_nodes(start, goal)
        route_metric = main.route_metrics(route.nodes, turbulence, 1000.0, 1000.0, 50.0)
        straight_metric = main.route_metrics(straight, turbulence, 1000.0, 1000.0, 50.0)
        self.assertLess(route_metric["integrated_risk_km"], straight_metric["integrated_risk_km"])


if __name__ == "__main__":
    unittest.main()
