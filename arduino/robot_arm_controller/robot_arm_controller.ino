#include <ArduinoJson.h>
#include <Servo.h>

// Tune these values for the physical arm before moving under load.
const float UPPER_ARM_MM = 100.0;
const float FOREARM_MM = 100.0;
const float SHOULDER_HEIGHT_MM = 60.0;
const float SHOULDER_FORWARD_OFFSET_MM = 20.0;

const int BASE_PIN = 3;
const int SHOULDER_PIN = 5;
const int ELBOW_PIN = 6;
const int WRIST_PIN = 9;
const int GRIPPER_PIN = 10;

const float BASE_CENTER_DEG = 90.0;
const float SHOULDER_HORIZONTAL_DEG = 90.0;
const float ELBOW_STRAIGHT_DEG = 90.0;
const float WRIST_LEVEL_DEG = 90.0;
const float GRIPPER_CLOSED_DEG = 35.0;
const float GRIPPER_OPEN_DEG = 105.0;

const bool BASE_INVERTED = false;
const bool SHOULDER_INVERTED = false;
const bool ELBOW_INVERTED = false;
const bool WRIST_INVERTED = false;

const float BASE_MIN_DEG = 0.0;
const float BASE_MAX_DEG = 180.0;
const float SHOULDER_MIN_DEG = 10.0;
const float SHOULDER_MAX_DEG = 170.0;
const float ELBOW_MIN_DEG = 10.0;
const float ELBOW_MAX_DEG = 170.0;
const float WRIST_MIN_DEG = 0.0;
const float WRIST_MAX_DEG = 180.0;
const float GRIPPER_MIN_DEG = 20.0;
const float GRIPPER_MAX_DEG = 130.0;

const unsigned long SERVO_STEP_INTERVAL_MS = 20;
const float SERVO_STEP_DEG = 2.0;

Servo baseServo;
Servo shoulderServo;
Servo elbowServo;
Servo wristServo;
Servo gripperServo;

struct ServoTargets {
  float base;
  float shoulder;
  float elbow;
  float wrist;
  float gripper;
};

ServoTargets currentTarget = {
  BASE_CENTER_DEG,
  SHOULDER_HORIZONTAL_DEG,
  ELBOW_STRAIGHT_DEG,
  WRIST_LEVEL_DEG,
  GRIPPER_CLOSED_DEG
};
ServoTargets desiredTarget = currentTarget;

char serialLine[160];
size_t serialLineLength = 0;
unsigned long lastServoStepMs = 0;

float clampFloat(float value, float minimum, float maximum) {
  if (value < minimum) {
    return minimum;
  }
  if (value > maximum) {
    return maximum;
  }
  return value;
}

float radiansToDegrees(float radians) {
  return radians * 180.0 / PI;
}

float jointToServo(float jointDeg, float zeroDeg, bool inverted, float minimum, float maximum) {
  float command = zeroDeg + (inverted ? -jointDeg : jointDeg);
  return clampFloat(command, minimum, maximum);
}

float lerp(float start, float end, float amount) {
  return start + (end - start) * amount;
}

void replyError(const char* error) {
  Serial.print("{\"ok\":false,\"error\":\"");
  Serial.print(error);
  Serial.println("\"}");
}

void replyOk(const ServoTargets& target) {
  Serial.print("{\"ok\":true,\"base\":");
  Serial.print(target.base, 1);
  Serial.print(",\"shoulder\":");
  Serial.print(target.shoulder, 1);
  Serial.print(",\"elbow\":");
  Serial.print(target.elbow, 1);
  Serial.print(",\"wrist\":");
  Serial.print(target.wrist, 1);
  Serial.print(",\"gripper\":");
  Serial.print(target.gripper, 1);
  Serial.println("}");
}

bool computeIk(float x, float y, float z, float wristDeg, float grip, ServoTargets& target) {
  float baseJointDeg = radiansToDegrees(atan2(y, x));
  float planar = sqrt(x * x + y * y) - SHOULDER_FORWARD_OFFSET_MM;
  float vertical = z - SHOULDER_HEIGHT_MM;

  if (planar < 0.0) {
    planar = 0.0;
  }

  float reach = sqrt(planar * planar + vertical * vertical);
  float maxReach = UPPER_ARM_MM + FOREARM_MM;
  float minReach = fabs(UPPER_ARM_MM - FOREARM_MM);
  if (reach > maxReach || reach < minReach || reach <= 0.0) {
    return false;
  }

  float cosShoulder = (
    UPPER_ARM_MM * UPPER_ARM_MM
    + reach * reach
    - FOREARM_MM * FOREARM_MM
  ) / (2.0 * UPPER_ARM_MM * reach);
  cosShoulder = clampFloat(cosShoulder, -1.0, 1.0);

  float shoulderBaseRad = atan2(vertical, planar);
  float shoulderOffsetRad = acos(cosShoulder);
  float shoulderJointDeg = radiansToDegrees(shoulderBaseRad + shoulderOffsetRad);

  float cosElbow = (
    UPPER_ARM_MM * UPPER_ARM_MM
    + FOREARM_MM * FOREARM_MM
    - reach * reach
  ) / (2.0 * UPPER_ARM_MM * FOREARM_MM);
  cosElbow = clampFloat(cosElbow, -1.0, 1.0);

  float elbowInternalDeg = radiansToDegrees(acos(cosElbow));
  float elbowBendDeg = 180.0 - elbowInternalDeg;
  float wristJointDeg = wristDeg - shoulderJointDeg + elbowBendDeg;

  target.base = jointToServo(
    baseJointDeg,
    BASE_CENTER_DEG,
    BASE_INVERTED,
    BASE_MIN_DEG,
    BASE_MAX_DEG
  );
  target.shoulder = jointToServo(
    shoulderJointDeg,
    SHOULDER_HORIZONTAL_DEG,
    SHOULDER_INVERTED,
    SHOULDER_MIN_DEG,
    SHOULDER_MAX_DEG
  );
  target.elbow = jointToServo(
    elbowBendDeg,
    ELBOW_STRAIGHT_DEG,
    ELBOW_INVERTED,
    ELBOW_MIN_DEG,
    ELBOW_MAX_DEG
  );
  target.wrist = jointToServo(
    wristJointDeg,
    WRIST_LEVEL_DEG,
    WRIST_INVERTED,
    WRIST_MIN_DEG,
    WRIST_MAX_DEG
  );
  target.gripper = clampFloat(
    lerp(GRIPPER_CLOSED_DEG, GRIPPER_OPEN_DEG, clampFloat(grip, 0.0, 1.0)),
    GRIPPER_MIN_DEG,
    GRIPPER_MAX_DEG
  );

  return true;
}

void processLine(char* line) {
  StaticJsonDocument<192> document;
  DeserializationError error = deserializeJson(document, line);
  if (error) {
    replyError("bad_json");
    return;
  }

  const char* type = document["type"] | "";
  if (strcmp(type, "target") != 0) {
    replyError("bad_type");
    return;
  }

  if (
    document["x"].isNull()
    || document["y"].isNull()
    || document["z"].isNull()
    || document["wrist"].isNull()
    || document["grip"].isNull()
  ) {
    replyError("missing_field");
    return;
  }

  ServoTargets nextTarget;
  bool reachable = computeIk(
    document["x"].as<float>(),
    document["y"].as<float>(),
    document["z"].as<float>(),
    document["wrist"].as<float>(),
    document["grip"].as<float>(),
    nextTarget
  );

  if (!reachable) {
    replyError("target_out_of_reach");
    return;
  }

  desiredTarget = nextTarget;
  replyOk(desiredTarget);
}

void readSerialLines() {
  while (Serial.available() > 0) {
    char incoming = Serial.read();

    if (incoming == '\n') {
      serialLine[serialLineLength] = '\0';
      if (serialLineLength > 0) {
        processLine(serialLine);
      }
      serialLineLength = 0;
      return;
    }

    if (incoming == '\r') {
      continue;
    }

    if (serialLineLength < sizeof(serialLine) - 1) {
      serialLine[serialLineLength] = incoming;
      serialLineLength++;
    } else {
      serialLineLength = 0;
      replyError("line_too_long");
    }
  }
}

float stepToward(float current, float desired) {
  float delta = desired - current;
  if (fabs(delta) <= SERVO_STEP_DEG) {
    return desired;
  }
  return current + (delta > 0.0 ? SERVO_STEP_DEG : -SERVO_STEP_DEG);
}

void stepServos() {
  unsigned long now = millis();
  if (now - lastServoStepMs < SERVO_STEP_INTERVAL_MS) {
    return;
  }
  lastServoStepMs = now;

  currentTarget.base = stepToward(currentTarget.base, desiredTarget.base);
  currentTarget.shoulder = stepToward(currentTarget.shoulder, desiredTarget.shoulder);
  currentTarget.elbow = stepToward(currentTarget.elbow, desiredTarget.elbow);
  currentTarget.wrist = stepToward(currentTarget.wrist, desiredTarget.wrist);
  currentTarget.gripper = stepToward(currentTarget.gripper, desiredTarget.gripper);

  baseServo.write(round(currentTarget.base));
  shoulderServo.write(round(currentTarget.shoulder));
  elbowServo.write(round(currentTarget.elbow));
  wristServo.write(round(currentTarget.wrist));
  gripperServo.write(round(currentTarget.gripper));
}

void setup() {
  Serial.begin(115200);

  baseServo.attach(BASE_PIN);
  shoulderServo.attach(SHOULDER_PIN);
  elbowServo.attach(ELBOW_PIN);
  wristServo.attach(WRIST_PIN);
  gripperServo.attach(GRIPPER_PIN);

  baseServo.write(round(currentTarget.base));
  shoulderServo.write(round(currentTarget.shoulder));
  elbowServo.write(round(currentTarget.elbow));
  wristServo.write(round(currentTarget.wrist));
  gripperServo.write(round(currentTarget.gripper));

  Serial.println("{\"ok\":true,\"status\":\"ready\"}");
}

void loop() {
  readSerialLines();
  stepServos();
}
