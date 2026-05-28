# Hand-Tracker

## Description

This project tracks your arm with MediaPipe, converts the landmarks into robot-space end-effector targets, and streams those targets to an Arduino-controlled robotic arm. The Python app handles vision, calibration, smoothing, and serial output. The Arduino sketch reads JSON target messages, runs inverse kinematics, and moves the servos.

## Installation

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

On first run, the app downloads the required MediaPipe model files into the `models` folder.

## Python Usage

Run without Arduino first. This opens the webcam preview and prints JSON targets to the terminal:

```bash
python main.py --dry-run
```

Calibrate your human arm mapping:

```bash
python main.py --calibrate --config robot_config.json
```

During calibration, hold your tracked arm fully extended along the center line of the robot workspace. The generated `robot_config.json` is ignored by git because it is specific to your body, camera, and robot setup.

Send targets to an Arduino:

```bash
python main.py --serial-port COM3 --baud-rate 115200
```

Useful options:

```bash
python main.py --arm left
python main.py --camera-index 1
python main.py --no-window --serial-port COM3
python main.py --send-every-frames 2 --serial-port COM3
```

## Serial Contract

Python sends one JSON object per line:

```json
{"type":"target","x":120.5,"y":40.0,"z":85.2,"wrist":15.0,"grip":0.7}
```

Fields:

- `x`, `y`, `z`: robot-space end-effector target in millimeters
- `wrist`: wrist pitch target in degrees
- `grip`: gripper openness from `0.0` closed to `1.0` open

The Arduino replies with either servo targets:

```json
{"ok":true,"base":90.0,"shoulder":72.0,"elbow":118.0,"wrist":95.0,"gripper":70.0}
```

or an error:

```json
{"ok":false,"error":"target_out_of_reach"}
```

## Arduino Setup

Open `arduino/robot_arm_controller/robot_arm_controller.ino` in the Arduino IDE.

Install these Arduino libraries:

- `Servo`
- `ArduinoJson`

Before attaching the arm under load, tune the sketch constants for your hardware:

- servo pins
- upper-arm and forearm link lengths
- shoulder height and forward offset
- servo min/max angles
- servo zero positions and inversion flags
- gripper open/closed angles

The sketch assumes a base yaw joint, shoulder joint, elbow joint, wrist joint, and gripper. It rejects unreachable targets, clamps servo commands to configured limits, and steps servos gradually instead of jumping directly.

## Safety

Test in this order:

1. Run `python main.py --dry-run` and confirm target values move smoothly.
2. Upload the Arduino sketch with servos detached or the arm unloaded.
3. Send one target at a time from Serial Monitor and tune servo directions/limits.
4. Run Python with `--serial-port` only after the arm moves correctly from manual test commands.

Press `q` in the video window to quit.

## Author

**Aaron Lee**
