from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


EPSILON = 1e-9


@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float

    @classmethod
    def from_iterable(cls, values):
        x, y, z = values
        return cls(float(x), float(y), float(z))

    def __sub__(self, other: "Point3") -> "Point3":
        return Point3(self.x - other.x, self.y - other.y, self.z - other.z)


@dataclass(frozen=True)
class ArmMetrics:
    shoulder: Point3
    elbow: Point3
    wrist: Point3
    index: Point3
    thumb: Point3
    upper_arm_length: float
    forearm_length: float
    reach_distance: float
    elbow_angle_deg: float
    wrist_pitch_deg: float
    grip_open: float

    @property
    def wrist_delta(self) -> Point3:
        return self.wrist - self.shoulder


@dataclass(frozen=True)
class RobotTarget:
    x: float
    y: float
    z: float
    wrist: float
    grip: float


@dataclass(frozen=True)
class ControlConfig:
    human_full_reach: float = 0.55
    center_delta: Point3 = field(default_factory=lambda: Point3(0.25, 0.0, 0.0))
    robot_min_x: float = 80.0
    robot_max_x: float = 220.0
    robot_neutral_y: float = 0.0
    robot_min_y: float = -120.0
    robot_max_y: float = 120.0
    robot_neutral_z: float = 90.0
    robot_min_z: float = 30.0
    robot_max_z: float = 180.0
    lateral_scale_mm: float = 450.0
    vertical_scale_mm: float = 450.0
    min_wrist_deg: float = -90.0
    max_wrist_deg: float = 90.0
    grip_closed_distance: float = 0.025
    grip_open_distance: float = 0.12
    smoothing_alpha: float = 0.35
    deadband_mm: float = 2.0
    wrist_deadband_deg: float = 2.0
    grip_deadband: float = 0.03

    @classmethod
    def from_dict(cls, data: dict) -> "ControlConfig":
        defaults = cls()
        values = asdict(defaults)
        values.update({key: value for key, value in data.items() if key in values})
        center_delta = values.get("center_delta", asdict(defaults.center_delta))
        if isinstance(center_delta, dict):
            values["center_delta"] = Point3(
                center_delta.get("x", defaults.center_delta.x),
                center_delta.get("y", defaults.center_delta.y),
                center_delta.get("z", defaults.center_delta.z),
            )
        else:
            values["center_delta"] = Point3.from_iterable(center_delta)
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


class TargetSmoother:
    def __init__(self, alpha: float):
        self.alpha = clamp(alpha, 0.0, 1.0)
        self._last_target: RobotTarget | None = None

    def reset(self) -> None:
        self._last_target = None

    def update(self, target: RobotTarget) -> RobotTarget:
        if self._last_target is None:
            self._last_target = target
            return target

        smoothed = RobotTarget(
            x=lerp(self._last_target.x, target.x, self.alpha),
            y=lerp(self._last_target.y, target.y, self.alpha),
            z=lerp(self._last_target.z, target.z, self.alpha),
            wrist=lerp(self._last_target.wrist, target.wrist, self.alpha),
            grip=lerp(self._last_target.grip, target.grip, self.alpha),
        )
        self._last_target = smoothed
        return smoothed


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def distance(point_a: Point3, point_b: Point3) -> float:
    return math.sqrt(
        (point_b.x - point_a.x) ** 2
        + (point_b.y - point_a.y) ** 2
        + (point_b.z - point_a.z) ** 2
    )


def angle_at(point_a: Point3, vertex: Point3, point_c: Point3) -> float:
    vector_a = point_a - vertex
    vector_c = point_c - vertex
    magnitude_a = distance(point_a, vertex)
    magnitude_c = distance(point_c, vertex)
    if magnitude_a < EPSILON or magnitude_c < EPSILON:
        return 0.0

    dot = (
        vector_a.x * vector_c.x
        + vector_a.y * vector_c.y
        + vector_a.z * vector_c.z
    )
    cosine = clamp(dot / (magnitude_a * magnitude_c), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def forearm_pitch_deg(elbow: Point3, wrist: Point3) -> float:
    delta = wrist - elbow
    horizontal = math.sqrt(delta.x**2 + delta.z**2)
    return math.degrees(math.atan2(-delta.y, max(horizontal, EPSILON)))


def grip_openness(thumb: Point3, index: Point3, config: ControlConfig) -> float:
    span = config.grip_open_distance - config.grip_closed_distance
    if abs(span) < EPSILON:
        return 0.0
    return clamp((distance(thumb, index) - config.grip_closed_distance) / span, 0.0, 1.0)


def build_arm_metrics(
    shoulder: Point3,
    elbow: Point3,
    wrist: Point3,
    index: Point3,
    thumb: Point3,
    config: ControlConfig,
) -> ArmMetrics:
    return ArmMetrics(
        shoulder=shoulder,
        elbow=elbow,
        wrist=wrist,
        index=index,
        thumb=thumb,
        upper_arm_length=distance(shoulder, elbow),
        forearm_length=distance(elbow, wrist),
        reach_distance=distance(shoulder, wrist),
        elbow_angle_deg=angle_at(shoulder, elbow, wrist),
        wrist_pitch_deg=forearm_pitch_deg(elbow, wrist),
        grip_open=grip_openness(thumb, index, config),
    )


def map_metrics_to_target(metrics: ArmMetrics, config: ControlConfig) -> RobotTarget:
    full_reach = max(config.human_full_reach, EPSILON)
    reach_ratio = clamp(metrics.reach_distance / full_reach, 0.0, 1.0)
    wrist_delta = metrics.wrist_delta

    target_x = lerp(config.robot_min_x, config.robot_max_x, reach_ratio)
    target_y = (
        config.robot_neutral_y
        + (wrist_delta.x - config.center_delta.x) * config.lateral_scale_mm
    )
    target_z = (
        config.robot_neutral_z
        + (config.center_delta.y - wrist_delta.y) * config.vertical_scale_mm
    )

    return RobotTarget(
        x=clamp(target_x, config.robot_min_x, config.robot_max_x),
        y=clamp(target_y, config.robot_min_y, config.robot_max_y),
        z=clamp(target_z, config.robot_min_z, config.robot_max_z),
        wrist=clamp(metrics.wrist_pitch_deg, config.min_wrist_deg, config.max_wrist_deg),
        grip=clamp(metrics.grip_open, 0.0, 1.0),
    )


def target_changed_enough(
    current: RobotTarget,
    previous: RobotTarget | None,
    config: ControlConfig,
) -> bool:
    if previous is None:
        return True

    linear_change = max(
        abs(current.x - previous.x),
        abs(current.y - previous.y),
        abs(current.z - previous.z),
    )
    return (
        linear_change >= config.deadband_mm
        or abs(current.wrist - previous.wrist) >= config.wrist_deadband_deg
        or abs(current.grip - previous.grip) >= config.grip_deadband
    )


def target_to_json_line(target: RobotTarget) -> str:
    payload = {
        "type": "target",
        "x": round(target.x, 2),
        "y": round(target.y, 2),
        "z": round(target.z, 2),
        "wrist": round(target.wrist, 2),
        "grip": round(target.grip, 3),
    }
    return json.dumps(payload, separators=(",", ":")) + "\n"


def build_calibration(samples: list[ArmMetrics], base_config: ControlConfig) -> ControlConfig:
    if not samples:
        raise ValueError("At least one visible arm sample is required for calibration.")

    sample_count = len(samples)
    reach = sum(sample.reach_distance for sample in samples) / sample_count
    delta_x = sum(sample.wrist_delta.x for sample in samples) / sample_count
    delta_y = sum(sample.wrist_delta.y for sample in samples) / sample_count
    delta_z = sum(sample.wrist_delta.z for sample in samples) / sample_count

    return replace(
        base_config,
        human_full_reach=max(reach, EPSILON),
        center_delta=Point3(delta_x, delta_y, delta_z),
    )


def load_config(path: Path) -> ControlConfig:
    if not path.exists():
        return ControlConfig()

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return ControlConfig.from_dict(data)


def save_config(path: Path, config: ControlConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
