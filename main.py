import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from robot_control import (
    Point3,
    TargetSmoother,
    build_arm_metrics,
    build_calibration,
    load_config,
    map_metrics_to_target,
    save_config,
    target_changed_enough,
    target_to_json_line,
)

try:
    import cv2
    import numpy as np
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
except ImportError as exc:
    missing_dependency = exc.name or "required dependency"
    print(f"Missing dependency: {missing_dependency}")
    print("Install dependencies with: python -m pip install -r requirements.txt")
    sys.exit(1)


VISIBILITY_THRESHOLD = 0.75
FRAME_INTERVAL_MS = 33
DEFAULT_BAUD_RATE = 115200
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "robot_config.json"

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DOWNLOADS = {
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/"
        "pose_landmarker_lite.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}

ARM_LANDMARKS = {
    "right": (
        vision.PoseLandmark.RIGHT_SHOULDER.value,
        vision.PoseLandmark.RIGHT_ELBOW.value,
        vision.PoseLandmark.RIGHT_WRIST.value,
        vision.PoseLandmark.RIGHT_INDEX.value,
        vision.PoseLandmark.RIGHT_THUMB.value,
    ),
    "left": (
        vision.PoseLandmark.LEFT_SHOULDER.value,
        vision.PoseLandmark.LEFT_ELBOW.value,
        vision.PoseLandmark.LEFT_WRIST.value,
        vision.PoseLandmark.LEFT_INDEX.value,
        vision.PoseLandmark.LEFT_THUMB.value,
    ),
}
POSE_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS
)
HAND_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS
)


class TargetTransport:
    def __init__(self, serial_port, baud_rate, dry_run):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.dry_run = dry_run or not serial_port
        self.connection = None

        if self.dry_run:
            print("Dry run enabled; target JSON will print to stdout.")
            return

        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is required for Arduino output. "
                "Install dependencies with: python -m pip install -r requirements.txt"
            ) from exc

        try:
            self.connection = serial.Serial(serial_port, baud_rate, timeout=0.01)
            time.sleep(2.0)
        except Exception as exc:
            raise RuntimeError(f"Could not open serial port {serial_port}: {exc}") from exc

        print(f"Serial output connected on {serial_port} at {baud_rate} baud.")

    @property
    def label(self):
        if self.dry_run:
            return "dry-run stdout"
        return f"{self.serial_port} @ {self.baud_rate}"

    def send(self, target):
        line = target_to_json_line(target)
        if self.dry_run:
            print(line, end="")
            return

        self.connection.write(line.encode("utf-8"))

    def read_reply(self):
        if self.dry_run or not self.connection or self.connection.in_waiting <= 0:
            return None

        reply = self.connection.readline().decode("utf-8", errors="replace").strip()
        return reply or None

    def close(self):
        if self.connection:
            self.connection.close()


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Track an arm with MediaPipe and stream robot targets to Arduino."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--arm", choices=("right", "left"), default="right")
    parser.add_argument("--serial-port", help="Arduino serial port, for example COM3.")
    parser.add_argument("--baud-rate", type=int, default=DEFAULT_BAUD_RATE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target JSON instead of writing to serial.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Calibration/config JSON path.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Collect visible arm samples and write calibration config.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=positive_int,
        default=60,
        help="Number of visible frames to average during calibration.",
    )
    parser.add_argument(
        "--send-every-frames",
        type=positive_int,
        default=3,
        help="Send at most one target every N camera frames.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without opening the OpenCV preview window.",
    )
    return parser.parse_args(argv)


def ensure_model(model_name):
    model_path = MODEL_DIR / model_name
    if model_path.exists():
        return model_path

    model_url = MODEL_DOWNLOADS[model_name]
    temporary_path = model_path.with_suffix(model_path.suffix + ".tmp")
    MODEL_DIR.mkdir(exist_ok=True)

    print(f"Downloading {model_name}...")
    try:
        urllib.request.urlretrieve(model_url, temporary_path)
        temporary_path.replace(model_path)
    except (OSError, urllib.error.URLError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {model_name}. Download it manually from "
            f"{model_url} and save it to {model_path}."
        ) from exc

    return model_path


def create_pose_landmarker(model_path):
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def create_hand_landmarker(model_path):
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def landmark_to_point(landmark):
    return Point3(landmark.x, landmark.y, landmark.z)


def is_landmark_visible(landmark):
    if landmark.x is None or landmark.y is None or landmark.z is None:
        return False

    visibility = getattr(landmark, "visibility", None)
    if visibility is not None and visibility < VISIBILITY_THRESHOLD:
        return False

    presence = getattr(landmark, "presence", None)
    if presence is not None and presence < VISIBILITY_THRESHOLD:
        return False

    return True


def collect_visible_landmarks(landmarks, landmark_indices):
    points = []

    for landmark_index in landmark_indices:
        landmark = landmarks[landmark_index]
        if not is_landmark_visible(landmark):
            return None
        points.append(landmark_to_point(landmark))

    return points


def extract_arm_metrics(pose_result, arm, config):
    if not pose_result.pose_landmarks:
        return None

    points = collect_visible_landmarks(
        pose_result.pose_landmarks[0],
        ARM_LANDMARKS[arm],
    )
    if not points:
        return None

    shoulder, elbow, wrist, index, thumb = points
    return build_arm_metrics(
        shoulder=shoulder,
        elbow=elbow,
        wrist=wrist,
        index=index,
        thumb=thumb,
        config=config,
    )


def normalized_to_pixel(landmark, width, height):
    if landmark.x is None or landmark.y is None:
        return None

    x = min(max(landmark.x, 0.0), 1.0)
    y = min(max(landmark.y, 0.0), 1.0)
    return int(x * (width - 1)), int(y * (height - 1))


def draw_landmarks(frame, landmarks, connections, landmark_color, connection_color):
    height, width = frame.shape[:2]
    points = [normalized_to_pixel(landmark, width, height) for landmark in landmarks]

    for start, end in connections:
        if start >= len(points) or end >= len(points):
            continue
        start_point = points[start]
        end_point = points[end]
        if start_point and end_point:
            cv2.line(frame, start_point, end_point, connection_color, 2)

    for point in points:
        if point:
            cv2.circle(frame, point, 3, landmark_color, -1)


def draw_detection_results(frame, pose_result, hand_result):
    for pose_landmarks in pose_result.pose_landmarks:
        draw_landmarks(
            frame,
            pose_landmarks,
            POSE_CONNECTIONS,
            landmark_color=(0, 180, 255),
            connection_color=(255, 255, 255),
        )

    for hand_landmarks in hand_result.hand_landmarks:
        draw_landmarks(
            frame,
            hand_landmarks,
            HAND_CONNECTIONS,
            landmark_color=(80, 220, 120),
            connection_color=(180, 255, 210),
        )


def draw_status(frame, lines):
    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (240, 250, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24


def process_frame(frame, pose_landmarker, hand_landmarker, timestamp_ms):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
    pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    return pose_result, hand_result


def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return None
    return cap


def show_frame(window_name, frame, no_window):
    if no_window:
        return False

    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF == ord("q")


def format_target_status(target):
    return (
        f"Target x={target.x:.1f} y={target.y:.1f} z={target.z:.1f} "
        f"wrist={target.wrist:.1f} grip={target.grip:.2f}"
    )


def run_calibration(args, cap, pose_landmarker, hand_landmarker, config):
    print(
        "Calibration: hold your tracked arm fully extended along the center line "
        "of the robot workspace until sampling completes."
    )
    samples = []
    frame_count = 0

    while len(samples) < args.calibration_samples:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            return 1

        pose_result, hand_result = process_frame(
            frame,
            pose_landmarker,
            hand_landmarker,
            frame_count * FRAME_INTERVAL_MS,
        )
        metrics = extract_arm_metrics(pose_result, args.arm, config)

        draw_detection_results(frame, pose_result, hand_result)
        if metrics:
            samples.append(metrics)
            status = f"Calibration samples: {len(samples)}/{args.calibration_samples}"
        else:
            status = "Calibration: show shoulder, elbow, wrist, index, and thumb"

        draw_status(frame, [status, "Press q to cancel"])
        if show_frame("Hand Tracking Calibration", frame, args.no_window):
            print("Calibration cancelled.")
            return 1

        frame_count += 1

    calibrated = build_calibration(samples, config)
    save_config(args.config, calibrated)
    print(f"Saved calibration to {args.config}")
    print(
        "human_full_reach="
        f"{calibrated.human_full_reach:.4f}, "
        "center_delta="
        f"({calibrated.center_delta.x:.4f}, "
        f"{calibrated.center_delta.y:.4f}, "
        f"{calibrated.center_delta.z:.4f})"
    )
    return 0


def run_tracking(args, cap, pose_landmarker, hand_landmarker, config):
    try:
        transport = TargetTransport(
            serial_port=args.serial_port,
            baud_rate=args.baud_rate,
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    smoother = TargetSmoother(config.smoothing_alpha)
    frame_count = 0
    last_sent_target = None
    last_reply = None
    tracking_lost = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                return 1

            pose_result, hand_result = process_frame(
                frame,
                pose_landmarker,
                hand_landmarker,
                frame_count * FRAME_INTERVAL_MS,
            )
            metrics = extract_arm_metrics(pose_result, args.arm, config)
            target = None

            draw_detection_results(frame, pose_result, hand_result)

            if metrics:
                if tracking_lost:
                    print("Tracking reacquired.")
                    tracking_lost = False

                raw_target = map_metrics_to_target(metrics, config)
                target = smoother.update(raw_target)

                if (
                    frame_count % args.send_every_frames == 0
                    and target_changed_enough(target, last_sent_target, config)
                ):
                    transport.send(target)
                    last_sent_target = target

                reply = transport.read_reply()
                if reply:
                    last_reply = reply
            else:
                if not tracking_lost:
                    print("Tracking lost; holding last robot target.")
                    tracking_lost = True
                smoother.reset()

            status_lines = [f"Output: {transport.label}"]
            if target:
                status_lines.append(format_target_status(target))
                status_lines.append(
                    f"Elbow={metrics.elbow_angle_deg:.1f} reach={metrics.reach_distance:.3f}"
                )
            else:
                status_lines.append("No visible control arm")
            if last_reply:
                status_lines.append(f"Arduino: {last_reply[:70]}")
            status_lines.append("Press q to quit")
            draw_status(frame, status_lines)

            if show_frame("Hand Tracking", frame, args.no_window):
                break

            frame_count += 1
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        transport.close()

    return 0


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    config = load_config(args.config)

    try:
        pose_model_path = ensure_model("pose_landmarker_lite.task")
        hand_model_path = ensure_model("hand_landmarker.task")
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    cap = open_camera(args.camera_index)
    if cap is None:
        return 1

    try:
        with create_pose_landmarker(pose_model_path) as pose_landmarker:
            with create_hand_landmarker(hand_model_path) as hand_landmarker:
                if args.calibrate:
                    return run_calibration(
                        args,
                        cap,
                        pose_landmarker,
                        hand_landmarker,
                        config,
                    )

                return run_tracking(
                    args,
                    cap,
                    pose_landmarker,
                    hand_landmarker,
                    config,
                )
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        print(f"Error: {exc}")
        return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
