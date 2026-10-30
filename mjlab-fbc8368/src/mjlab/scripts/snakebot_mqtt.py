"""Mirror the keyboard simulation's servo targets using the ESP MQTT protocol."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable

import tyro

import mjlab
from mjlab.scripts.snakebot_keyboard import (
  KeyboardConfig,
  KeyboardGoalViewer,
  module_joint_angles_deg,
  run_keyboard,
)

FEEDBACK_FIELDS = ("request_id", "command_ts", "roll", "pitch", "gx", "gy", "gz")


@dataclass(frozen=True)
class MqttKeyboardConfig(KeyboardConfig):
  broker: str = "10.49.2.223"
  port: int = 1883
  topic: str = "servos/sync_command"
  module_ids: tuple[str, ...] = ("ESP_01", "ESP_02", "ESP_03", "ESP_04", "ESP_05")
  """IDs in simulation module order. Use one ID to send only module 1."""
  neutral_angles: tuple[int, ...] = (90, 90)
  """Bottom/top neutral degrees: one shared pair or a pair per module."""
  servo_directions: tuple[int, ...] = (1, 1)
  """Bottom/top signs (+1/-1): one shared pair or a pair per module."""
  servo_min: int = 65
  servo_max: int = 115
  publish_interval: float = 0.05
  """Minimum wall-clock spacing. The policy itself runs at 20 Hz."""
  initial_delay: float = 1.0
  """Schedule the first command this many seconds ahead, as in the sender."""
  command_delay: float = 0.05
  """Schedule later commands this many seconds ahead, using Unix milliseconds."""
  require_feedback: bool = True
  feedback_topic: str = "servos/ESP_01/feedback"
  feedback_timeout: float = 5.0
  feedback_csv: Path = Path("logs/hardware/mpu_feedback.csv")
  dry_run: bool = False
  """Print outgoing payloads without connecting to MQTT or moving hardware."""


def _expand_pairs(values: tuple[int, ...], count: int, name: str) -> tuple[int, ...]:
  if len(values) == 2:
    return values * count
  if len(values) != 2 * count:
    raise ValueError(f"{name} needs two values or two values per selected module")
  return values


def validate_mqtt_config(cfg: MqttKeyboardConfig) -> None:
  if not 1 <= len(cfg.module_ids) <= 5 or len(set(cfg.module_ids)) != len(
    cfg.module_ids
  ):
    raise ValueError("module_ids must contain one to five unique ESP IDs")
  if any(not name.strip() for name in cfg.module_ids):
    raise ValueError("Module IDs must not be empty")
  if not cfg.broker.strip() or not 1 <= cfg.port <= 65535:
    raise ValueError("A broker host and port between 1 and 65535 are required")
  if not cfg.topic or (cfg.require_feedback and not cfg.feedback_topic):
    raise ValueError("Command and feedback topics must not be empty")
  if not 0 <= cfg.servo_min < cfg.servo_max <= 180:
    raise ValueError("Servo limits must satisfy 0 <= min < max <= 180")
  neutrals = _expand_pairs(cfg.neutral_angles, len(cfg.module_ids), "neutral_angles")
  signs = _expand_pairs(cfg.servo_directions, len(cfg.module_ids), "servo_directions")
  if any(not cfg.servo_min <= angle <= cfg.servo_max for angle in neutrals):
    raise ValueError("Neutral angles must be within the servo limits")
  if any(sign not in (-1, 1) for sign in signs):
    raise ValueError("Servo directions must be +1 or -1")
  for name in ("publish_interval", "feedback_timeout"):
    value = getattr(cfg, name)
    if not math.isfinite(value) or value <= 0:
      raise ValueError(f"{name} must be finite and greater than zero")
  for name in ("initial_delay", "command_delay"):
    value = getattr(cfg, name)
    if not math.isfinite(value) or value < 0:
      raise ValueError(f"{name} must be finite and non-negative")


def servo_payload_data(
  targets_deg: list[list[float]], cfg: MqttKeyboardConfig
) -> dict[str, list[int]]:
  """Map simulated bottom/top servo targets to the sender's integer degrees."""
  if len(targets_deg) < len(cfg.module_ids):
    raise ValueError("Missing simulated module targets")
  neutrals = _expand_pairs(cfg.neutral_angles, len(cfg.module_ids), "neutral_angles")
  signs = _expand_pairs(cfg.servo_directions, len(cfg.module_ids), "servo_directions")
  data = {}
  for module, module_id in enumerate(cfg.module_ids):
    pair = targets_deg[module]
    if len(pair) != 2 or any(not math.isfinite(angle) for angle in pair):
      raise ValueError(f"Invalid bottom/top servo targets for {module_id}")
    # The old gait sender uses 90-int(angle), then sends [vertical, horizontal].
    # MJCF targets already equal -angle in bottom/top order: do not apply the
    # rolling-joint atan conversion or swap these two joints again.
    data[module_id] = [
      min(
        cfg.servo_max,
        max(
          cfg.servo_min,
          neutrals[2 * module + joint] + int(signs[2 * module + joint] * angle),
        ),
      )
      for joint, angle in enumerate(pair)
    ]
  return data


def _mqtt_client():
  # Optional hardware dependency; simulation and dry-run need no MQTT package.
  try:
    mqtt = import_module("paho.mqtt.client")
  except ModuleNotFoundError as exc:
    raise RuntimeError(
      "Install the MQTT dependency: uv pip install -r requirements-hardware.txt"
    ) from exc
  return mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    reconnect_on_failure=False,
  )


class MqttBridge:
  """One command in flight; feedback callbacks never block the viewer thread."""

  def __init__(
    self,
    cfg: MqttKeyboardConfig,
    client_factory: Callable[[], Any] = _mqtt_client,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
  ):
    validate_mqtt_config(cfg)
    self.cfg = cfg
    self._client_factory = client_factory
    self._clock, self._wall_clock = clock, wall_clock
    self._client = None
    self._ready = Event()
    self._lock = Lock()
    self._error: str | None = None
    self._pending: tuple[int, int, float] | None = None
    self._feedback: dict | None = None
    self._next_send = 0.0
    self._request_id = 0
    self._closing = False

  def _fail(self, message: str) -> None:
    with self._lock:
      self._error = message
    self._ready.set()

  def _on_connect(self, client, userdata, flags, reason_code, properties):
    if reason_code != 0:
      self._fail(f"MQTT connection refused: {reason_code}")
    elif self.cfg.require_feedback:
      result, _ = client.subscribe(self.cfg.feedback_topic)
      if result != 0:
        self._fail("MQTT feedback subscription failed")
    else:
      self._ready.set()

  def _on_subscribe(self, client, userdata, mid, reason_codes, properties):
    if any(getattr(code, "is_failure", False) for code in reason_codes):
      self._fail("Broker refused the feedback subscription")
    else:
      self._ready.set()

  def _on_disconnect(self, client, userdata, flags, reason_code, properties):
    if not self._closing:
      self._fail(f"MQTT disconnected: {reason_code}. Restart the hardware demo.")

  def _on_feedback(self, client, userdata, msg):
    try:
      feedback = json.loads(msg.payload.decode())
    except (ValueError, UnicodeDecodeError):
      return
    if not isinstance(feedback, dict) or type(feedback.get("request_id")) is not int:
      return
    with self._lock:
      if self._pending is not None and feedback["request_id"] == self._pending[0]:
        self._feedback = feedback

  def start(self) -> None:
    if self.cfg.dry_run:
      print("DRY RUN: MQTT payloads will be printed; no hardware connection.")
      return
    self._client = self._client_factory()
    client = self._client
    client.on_connect = self._on_connect
    client.on_subscribe = self._on_subscribe
    client.on_disconnect = self._on_disconnect
    client.message_callback_add(self.cfg.feedback_topic, self._on_feedback)
    client.max_queued_messages_set(1)
    client.connect(self.cfg.broker, self.cfg.port, 60)
    client.loop_start()
    if not self._ready.wait(timeout=5.0):
      raise TimeoutError("MQTT connection/feedback subscription timed out")
    self.poll()
    print(f"MQTT connected: {self.cfg.broker}:{self.cfg.port} -> {self.cfg.topic}")

  def _validate_feedback(self, feedback: dict, pending: tuple[int, int, float]) -> None:
    if feedback.get("sensor_valid") is not True or any(
      not isinstance(feedback.get(field), (int, float))
      or not math.isfinite(feedback[field])
      for field in ("roll", "pitch", "gx", "gy", "gz")
    ):
      message = f"Invalid MPU feedback for command #{pending[0]}"
      self._fail(message)
      raise RuntimeError(message)

  def _record_feedback(self, feedback: dict, pending: tuple[int, int, float]) -> None:
    path = self.cfg.feedback_csv
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
      writer = csv.DictWriter(stream, fieldnames=FEEDBACK_FIELDS)
      if write_header:
        writer.writeheader()
      row = {field: feedback.get(field) for field in FEEDBACK_FIELDS}
      row["command_ts"] = feedback.get("command_ts", pending[1])
      writer.writerow(row)
    print(
      f"Feedback #{pending[0]}: roll={feedback['roll']:.2f}, pitch={feedback['pitch']:.2f}"
    )

  def poll(self) -> bool:
    """Consume matching feedback and report whether another command can be sent."""
    with self._lock:
      error, pending, feedback = self._error, self._pending, self._feedback
      if feedback is not None:
        self._feedback = None
        self._pending = None
    if error:
      raise RuntimeError(error)
    if feedback is not None and pending is not None:
      self._validate_feedback(feedback, pending)
      self._record_feedback(feedback, pending)
      pending = None
    if pending is not None:
      if self._clock() - pending[2] > self.cfg.feedback_timeout:
        message = f"No matching feedback for command #{pending[0]} within {self.cfg.feedback_timeout:g}s"
        self._fail(message)
        raise TimeoutError(message)
      return False
    return self._clock() >= self._next_send

  def send(self, targets_deg: list[list[float]]) -> dict:
    if not self.poll():
      raise RuntimeError(
        "Previous command is pending or publish interval has not elapsed"
      )
    data = servo_payload_data(targets_deg, self.cfg)
    self._request_id += 1
    delay = self.cfg.initial_delay if self._request_id == 1 else self.cfg.command_delay
    timestamp = int((self._wall_clock() + delay) * 1000)
    payload = {"request_id": self._request_id, "ts": timestamp, "data": data}
    if not self.cfg.dry_run:
      if self._client is None or not self._client.is_connected():
        raise RuntimeError("MQTT is not connected")
      with self._lock:
        if self.cfg.require_feedback:
          self._pending = (self._request_id, timestamp, self._clock())
      result = self._client.publish(
        self.cfg.topic, json.dumps(payload), qos=0, retain=False
      )
      if result.rc != 0:
        self._fail(f"Failed to publish command #{self._request_id}")
        raise RuntimeError(f"Failed to publish command #{self._request_id}")
    # An early acknowledgement must not queue a second command ahead of the
    # first one's execution time (particularly the initial one-second delay).
    self._next_send = self._clock() + max(self.cfg.publish_interval, delay)
    print(f"{'DRY RUN' if self.cfg.dry_run else 'MQTT'} {json.dumps(payload)}")
    return payload

  def close(self) -> None:
    self._closing = True
    if self._client is not None:
      try:
        self._client.disconnect()
      finally:
        self._client.loop_stop()
        self._client = None


class MqttGoalViewer(KeyboardGoalViewer):
  def __init__(self, *args, bridge: MqttBridge, **kwargs):
    super().__init__(*args, **kwargs)
    self.bridge = bridge
    self._hardware_fault = False

  def _stop_on_fault(self, exc: Exception) -> None:
    if not self._hardware_fault:
      print(f"Hardware stream stopped: {exc}")
    self._hardware_fault = True
    self.pause()

  def resume(self) -> None:
    if self._hardware_fault:
      print("Restart the hardware demo after resolving the reported fault.")
      return
    super().resume()

  def tick(self) -> bool:
    if not self._hardware_fault:
      try:
        self.bridge.poll()  # Receive/log feedback even while paused.
      except (RuntimeError, TimeoutError, OSError) as exc:
        self._stop_on_fault(exc)
    return super().tick()

  def step_simulation(self) -> None:
    if self._is_paused or self._hardware_fault:
      return
    try:
      if not self.bridge.poll():
        return  # Keep rendering/keyboard responsive while awaiting feedback.
      super().step_simulation()
      if self.env.unwrapped.episode_length_buf[0].item() == 0:
        raise RuntimeError("Simulation reset after instability; no reset targets sent")
      targets = module_joint_angles_deg(self.env.unwrapped)[:, 2:].cpu().tolist()
      self.bridge.send(targets)
    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
      self._stop_on_fault(exc)


def run_mqtt(cfg: MqttKeyboardConfig) -> None:
  bridge = MqttBridge(cfg)

  def make_viewer(*args, **kwargs):
    bridge.start()
    return MqttGoalViewer(*args, bridge=bridge, **kwargs)

  try:
    run_keyboard(cfg, viewer_factory=make_viewer)
  finally:
    bridge.close()


def main() -> None:
  run_mqtt(tyro.cli(MqttKeyboardConfig, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
