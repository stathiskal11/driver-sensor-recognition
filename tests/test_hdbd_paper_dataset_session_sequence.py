from __future__ import annotations

import unittest

from src.data import hdbd_paper_dataset


class SessionSequenceTests(unittest.TestCase):
    def test_build_session_sequence_precomputes_modalities_and_normalized_signals(
        self,
    ) -> None:
        rows = [
            {
                "ImageFile": "img_a.png",
                "TimeStamp": "100",
                "navigation": "left",
                "transparency": "1",
                "weather": "0",
                "ECGtoHR": "70",
                "GSR": "0.5",
                "Throttle": "10",
                "RPM": "1000",
                "Steering": "5",
                "Speed": "30",
            },
            {
                "ImageFile": "img_b.png",
                "TimeStamp": "200",
                "navigation": "straight",
                "transparency": "2",
                "weather": "1",
                "ECGtoHR": "-1",
                "GSR": "",
                "Throttle": "20",
                "RPM": "1500",
                "Steering": "10",
                "Speed": "60",
            },
        ]
        signal_stats = {
            "physiology": {
                "0001": {
                    "ECGtoHR": {"mean": 60.0, "std": 5.0},
                    "GSR": {"mean": 0.4, "std": 0.1},
                }
            },
            "can_bus": {
                "Throttle": {"min": 0.0, "max": 20.0},
                "RPM": {"min": 500.0, "max": 1500.0},
                "Steering": {"min": 0.0, "max": 10.0},
                "Speed": {"min": 0.0, "max": 60.0},
            },
        }

        session = hdbd_paper_dataset._build_session_sequence(
            rows,
            member_name="Synced_csv_files-participant_level/0001/session.csv",
            signal_columns=list(hdbd_paper_dataset.DEFAULT_SIGNAL_COLUMNS),
            signal_stats=signal_stats,
        )

        self.assertEqual(session.image_files, ["img_a.png", "img_b.png"])
        self.assertEqual(session.heatmap_files, ["100.png", "200.png"])
        self.assertEqual(session.hmi_vectors.shape, (2, 9))
        self.assertEqual(session.normalized_signals.shape, (2, 6))

        self.assertAlmostEqual(float(session.normalized_signals[0, 0]), 2.0, places=6)
        self.assertAlmostEqual(float(session.normalized_signals[0, 1]), 1.0, places=6)
        self.assertAlmostEqual(float(session.normalized_signals[0, 2]), 0.5, places=6)
        self.assertAlmostEqual(float(session.normalized_signals[0, 3]), 0.5, places=6)
        self.assertAlmostEqual(float(session.normalized_signals[0, 4]), 0.5, places=6)
        self.assertAlmostEqual(float(session.normalized_signals[0, 5]), 0.5, places=6)

        self.assertAlmostEqual(float(session.normalized_signals[1, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(session.normalized_signals[1, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(session.normalized_signals[1, 2]), 1.0, places=6)
        self.assertAlmostEqual(float(session.normalized_signals[1, 3]), 1.0, places=6)
        self.assertAlmostEqual(float(session.normalized_signals[1, 4]), 1.0, places=6)
        self.assertAlmostEqual(float(session.normalized_signals[1, 5]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
