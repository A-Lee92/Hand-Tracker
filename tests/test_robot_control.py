import json
import unittest

from robot_control import (
    ControlConfig,
    Point3,
    RobotTarget,
    TargetSmoother,
    build_arm_metrics,
    build_calibration,
    distance,
    map_metrics_to_target,
    target_changed_enough,
    target_to_json_line,
)


class RobotControlTests(unittest.TestCase):
    def test_distance_uses_all_three_axes(self):
        self.assertAlmostEqual(
            distance(Point3(0, 0, 0), Point3(3, 4, 12)),
            13.0,
        )

    def test_metrics_include_angles_and_grip(self):
        config = ControlConfig(grip_closed_distance=0.02, grip_open_distance=0.12)
        metrics = build_arm_metrics(
            shoulder=Point3(0, 0, 0),
            elbow=Point3(1, 0, 0),
            wrist=Point3(2, 0, 0),
            index=Point3(0.12, 0, 0),
            thumb=Point3(0, 0, 0),
            config=config,
        )

        self.assertAlmostEqual(metrics.upper_arm_length, 1.0)
        self.assertAlmostEqual(metrics.forearm_length, 1.0)
        self.assertAlmostEqual(metrics.reach_distance, 2.0)
        self.assertAlmostEqual(metrics.elbow_angle_deg, 180.0)
        self.assertAlmostEqual(metrics.wrist_pitch_deg, 0.0)
        self.assertAlmostEqual(metrics.grip_open, 1.0)

    def test_metrics_map_to_clamped_robot_target(self):
        config = ControlConfig(
            human_full_reach=2.0,
            center_delta=Point3(2, 0, 0),
            robot_min_x=100,
            robot_max_x=200,
            robot_neutral_y=0,
            robot_min_y=-50,
            robot_max_y=50,
            robot_neutral_z=75,
            robot_min_z=25,
            robot_max_z=125,
            grip_closed_distance=0.02,
            grip_open_distance=0.12,
        )
        metrics = build_arm_metrics(
            shoulder=Point3(0, 0, 0),
            elbow=Point3(1, 0, 0),
            wrist=Point3(2, 0, 0),
            index=Point3(0.12, 0, 0),
            thumb=Point3(0, 0, 0),
            config=config,
        )

        target = map_metrics_to_target(metrics, config)

        self.assertAlmostEqual(target.x, 200.0)
        self.assertAlmostEqual(target.y, 0.0)
        self.assertAlmostEqual(target.z, 75.0)
        self.assertAlmostEqual(target.wrist, 0.0)
        self.assertAlmostEqual(target.grip, 1.0)

    def test_json_line_is_arduino_friendly(self):
        payload = json.loads(
            target_to_json_line(RobotTarget(x=120.123, y=-4.567, z=80, wrist=15, grip=0.5))
        )

        self.assertEqual(
            payload,
            {
                "type": "target",
                "x": 120.12,
                "y": -4.57,
                "z": 80,
                "wrist": 15,
                "grip": 0.5,
            },
        )

    def test_smoother_and_deadband_limit_small_updates(self):
        config = ControlConfig(deadband_mm=2.0, wrist_deadband_deg=2.0, grip_deadband=0.1)
        smoother = TargetSmoother(alpha=0.5)
        first = smoother.update(RobotTarget(100, 0, 50, 0, 0))
        second = smoother.update(RobotTarget(104, 0, 50, 4, 0.2))

        self.assertAlmostEqual(second.x, 102.0)
        self.assertTrue(target_changed_enough(second, first, config))
        self.assertFalse(
            target_changed_enough(
                RobotTarget(103.0, 0.5, 50.5, 3.5, 0.18),
                second,
                config,
            )
        )

    def test_calibration_averages_visible_samples(self):
        config = ControlConfig()
        sample_a = build_arm_metrics(
            shoulder=Point3(0, 0, 0),
            elbow=Point3(0.25, 0, 0),
            wrist=Point3(0.5, 0.1, 0),
            index=Point3(0.1, 0, 0),
            thumb=Point3(0, 0, 0),
            config=config,
        )
        sample_b = build_arm_metrics(
            shoulder=Point3(0, 0, 0),
            elbow=Point3(0.3, 0, 0),
            wrist=Point3(0.7, -0.1, 0),
            index=Point3(0.1, 0, 0),
            thumb=Point3(0, 0, 0),
            config=config,
        )

        calibrated = build_calibration([sample_a, sample_b], config)

        self.assertAlmostEqual(calibrated.center_delta.x, 0.6)
        self.assertAlmostEqual(calibrated.center_delta.y, 0.0)
        self.assertGreater(calibrated.human_full_reach, 0.5)


if __name__ == "__main__":
    unittest.main()
