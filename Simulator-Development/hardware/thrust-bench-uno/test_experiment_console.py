import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from experiment_console import (
    GRAVITY, capture_calibration, compare_pair, enter_rpm, fit_calibration, run_trial, to_newtons,
    trial_specs, video_rpm,
)


class ExperimentConsoleTests(unittest.TestCase):
    def test_signed_calibration(self):
        calibration = fit_calibration([
            {"mass_g": 0, "mean_counts": 2000},
            {"mass_g": 200, "mean_counts": 1800},
            {"mass_g": 400, "mean_counts": 1600},
        ])
        self.assertAlmostEqual(to_newtons(2000, calibration), 0)
        self.assertAlmostEqual(to_newtons(1700, calibration), 0.3 * GRAVITY)

    def test_reverse_calibration_load_keeps_thrust_positive(self):
        uno = Mock()
        uno.read.side_effect = [f"{t},0,1940,0" for t in range(0, 2001, 100)]
        points = [{"mass_g": 0, "mean_counts": 2000}]
        with tempfile.TemporaryDirectory() as path:
            capture_calibration(uno, Path(path), points, -30)
        calibration = fit_calibration(points)
        self.assertAlmostEqual(to_newtons(1940, calibration), -0.03 * GRAVITY)
        self.assertAlmostEqual(to_newtons(2060, calibration), 0.03 * GRAVITY)

    def test_run_requires_ready_state(self):
        uno = Mock()
        uno.status.return_value = "# esc=off pwm_us=0 hx711_fresh=true"
        with self.assertRaisesRegex(RuntimeError, "ready tone"):
            run_trial(uno, None, [], {}, {})
        uno.send.assert_not_called()

    def test_steady_samples_and_independent_rpm(self):
        uno = Mock()
        uno.status.return_value = "# esc=ready pwm_us=1000 hx711_fresh=true"
        uno.read.side_effect = [
            f"{t},1100,200,0" for t in range(0, 1000, 100)
        ] + ["1000,1000,100,0"]
        specs = trial_specs(["B1", "far", "300", "3000", "1100"], 1000, 300)
        trials = []
        with tempfile.TemporaryDirectory() as path:
            directory = Path(path)
            run_trial(uno, directory, trials, specs, {"zero_counts": 100, "counts_per_n": 50})
            with (directory / "run_001.csv").open(newline="") as stream:
                raw = list(csv.DictReader(stream))
            self.assertEqual(raw[-1]["pwm_us"], "1000")
            self.assertEqual(trials[0]["steady_samples"], 7)
            self.assertEqual(trials[0]["thrust_N"], 2.0)
            self.assertIsNone(trials[0]["rpm"])
            self.assertEqual(enter_rpm(directory, trials, "run_001", "2990"), 1)
            with (directory / "measurements.csv").open(newline="") as stream:
                measurement = next(csv.DictReader(stream))
            self.assertEqual(measurement["rpm"], "2990.0")
        uno.send.assert_called_once_with("RUN 1100 1000")
        uno.stop.assert_called_once()

    def test_aborted_run_verifies_stop(self):
        uno = Mock()
        uno.status.return_value = "# esc=ready pwm_us=1000 hx711_fresh=true"
        uno.read.return_value = "# stopped"
        specs = trial_specs(["B1", "near", "50", "3000", "1100"], 1000, 300)
        trials = []
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                run_trial(uno, Path(path), trials, specs, {"zero_counts": 100, "counts_per_n": 50})
        self.assertEqual(trials[0]["status"], "incomplete")
        uno.stop.assert_called_once()

    def test_video_rpm_uses_capture_rate_and_elapsed_frames(self):
        self.assertAlmostEqual(video_rpm(1000, 120, 360, 40), 10000)
        with self.assertRaisesRegex(ValueError, "four captured"):
            video_rpm(1000, 120, 240, 40)
        with self.assertRaisesRegex(ValueError, "increasing"):
            video_rpm(1000, 120, 119, 1)

    def test_pair_checks_measured_speeds_and_propeller(self):
        far = {"run_id": "run_001", "status": "complete", "condition": "far",
               "height_m": 0.5, "rpm": 10000, "thrust_N": 2, "propeller": "1045"}
        near = {**far, "run_id": "run_002", "condition": "near",
                "height_m": 0.05, "rpm": 10040, "thrust_N": 2.1}
        matched = compare_pair([far, near], "run_001", "run_002", 0.005)
        self.assertAlmostEqual(matched["near_to_far_thrust_ratio"], 1.05)
        near["rpm"] = 10100
        unmatched = compare_pair([far, near], "run_001", "run_002", 0.005)
        self.assertFalse(unmatched["matched"])
        self.assertNotIn("near_to_far_thrust_ratio", unmatched)
        near["propeller"] = "8040"
        with self.assertRaisesRegex(ValueError, "propeller"):
            compare_pair([far, near], "run_001", "run_002", 0.005)
