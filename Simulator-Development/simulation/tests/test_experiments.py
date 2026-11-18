from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "ml")]

from data import assert_seed_splits_disjoint, build_common_support, seed_group_split
from experiments.validity import compute_metrics, load_config, paired_statistics, read_telemetry


class SplitTests(unittest.TestCase):
    def test_common_support_uses_identical_target_rows(self):
        rows = []
        features = ["gyro_x","gyro_y","gyro_z","rangefinder_alt","motor_rpm_0","motor_rpm_1","motor_rpm_2","motor_rpm_3","motor_current_0","motor_current_1","motor_current_2","motor_current_3","thrust_cmd_norm"]
        for seed in (1, 2):
            for i in range(8):
                row = {"seed":seed,"t":i/60,"ge_factor_truth_mean":seed+i/100,"ge_ff_analytical":1.0}
                row.update({c:float(i) for c in features}); rows.append(row)
        support = build_common_support(pd.DataFrame(rows), window=3, lead_rows=2, extra_cols=["ge_ff_analytical"])
        self.assertEqual(len(support["y"]), 8)
        self.assertTrue(np.all(support["target_t"] > support["source_t"]))
        self.assertEqual(set(support["seed"]), {1, 2})

    def test_seed_split_is_deterministic_and_disjoint(self):
        frame = pd.DataFrame({"seed": np.repeat(np.arange(1, 301), 2), "x": 1})
        first = seed_group_split(frame, random_seed=20260825)[3]
        second = seed_group_split(frame, random_seed=20260825)[3]
        self.assertEqual(first, second)
        self.assertTrue(assert_seed_splits_disjoint(first))
        self.assertEqual([len(first[k]) for k in ("train", "validation", "test")], [240, 30, 30])

    def test_overlap_fails(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_seed_splits_disjoint({"train": [1], "validation": [1], "test": [2]})


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "experiments/configs/ijira_validity.json")

    def test_metrics_and_paired_aggregation(self):
        t = np.arange(0, 12, .1)
        df = pd.DataFrame({"seed": 1, "t": t, "gyro_x": 3., "gyro_y": 4., "gyro_z": 0.,
                           "true_agl": .23, "min_rotor_clearance": .14,
                           "ge_ff_applied": 1., "ge_oracle_truth": 1.})
        metrics = compute_metrics(df, self.cfg)
        self.assertAlmostEqual(metrics["agl_rms_primary"], .01)
        self.assertAlmostEqual(metrics["gyro_rms"], 5.)
        self.assertGreater(metrics["low_clearance_duration_s"], 0.)
        rows = []
        for seed in range(3):
            for mode, value in (("a", .1 + seed), ("b", .2 + seed)):
                rows.append({"seed": seed, "mode": mode, **{k: value for k in
                    ("agl_rms_primary", "low_clearance_duration_s", "low_clearance_events",
                     "gyro_rms", "minimum_clearance_m", "p05_clearance_m")}})
        out = paired_statistics(pd.DataFrame(rows), self.cfg)
        self.assertEqual(set(out.n), {3})
        self.assertTrue(np.allclose(out.mean_difference_a_minus_b, -.1))
        self.assertTrue({"paired_t_p_raw", "paired_t_p_holm"}.issubset(out.columns))
        self.assertTrue((out.paired_t_p_holm >= out.paired_t_p_raw).all())

    def test_robustness_pairing_does_not_mix_design_cells(self):
        rows = []
        for slope in (.5, 1.):
            for seed in range(3):
                for mode, value in (("a", slope + seed), ("b", slope + seed + .1)):
                    rows.append({"seed": seed, "mode": mode, "slope_exponent": slope,
                        "curvature_gain": .6, **{k: value for k in
                        ("agl_rms_primary", "low_clearance_duration_s", "low_clearance_events",
                         "gyro_rms", "minimum_clearance_m", "p05_clearance_m")}})
        out = paired_statistics(pd.DataFrame(rows), self.cfg)
        self.assertEqual(set(out.n), {3})
        self.assertEqual(set(out.slope_exponent), {.5, 1.})

    def test_missing_and_malformed_telemetry_fail(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.csv"
            with self.assertRaises(FileNotFoundError): read_telemetry(path)
            path.write_text("seed,t\n1,not-a-number\n")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                read_telemetry(path)

    def test_controller_seeds_are_unseen(self):
        offline = set(range(1, 301))
        validation = set(self.cfg["controller_validation_seeds"])
        test = set(self.cfg["controller_test_seeds"])
        self.assertFalse(offline & validation)
        self.assertFalse(offline & test)
        self.assertFalse(validation & test)

    def test_oracle_uses_simulator_truth(self):
        simulator = ROOT / "build/bin/simulator"
        if not simulator.exists(): self.skipTest("build simulator first")
        with tempfile.TemporaryDirectory() as td:
            csv = Path(td) / "oracle.csv"
            env = dict(__import__("os").environ, GE_FF_MODE="oracle")
            subprocess.run([simulator, "--fc", ROOT / "build/bin/libfc.so", "--seed", "331",
                "--duration", ".2", "--fast", "--no-telemetry", "--csv", csv],
                cwd=ROOT, env=env, check=True, capture_output=True)
            frame = read_telemetry(csv)
            self.assertTrue(np.allclose(frame.ge_ff_applied, frame.ge_oracle_truth, atol=1e-6))
            self.assertEqual(set(frame.ge_ff_source), {3})

    def test_oracle_pulse_is_time_limited_and_logs_shadow_ml(self):
        simulator = ROOT / "build/bin/simulator"
        if not simulator.exists(): self.skipTest("build simulator first")
        with tempfile.TemporaryDirectory() as td:
            csv = Path(td) / "pulse.csv"
            env = dict(__import__("os").environ, GE_FF_MODE="oracle")
            subprocess.run([simulator, "--fc", ROOT / "build/bin/libfc.so", "--seed", "331",
                "--duration", ".5", "--fast", "--no-telemetry", "--csv", csv,
                "--oracle-bias", ".02", "--oracle-pulse-start", ".2",
                "--oracle-pulse-duration", ".1"], cwd=ROOT, env=env, check=True, capture_output=True)
            frame = read_telemetry(csv)
            active = frame.ge_oracle_bias != 0
            self.assertTrue(active.any())
            self.assertTrue(((frame.loc[active, "t"] >= .2) & (frame.loc[active, "t"] < .32)).all())
            self.assertTrue(frame.ge_ff_ai.notna().all())


if __name__ == "__main__":
    unittest.main()
