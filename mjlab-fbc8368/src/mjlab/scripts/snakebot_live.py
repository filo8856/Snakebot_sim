"""Five-module launcher matching the supplied ESP32 main.cpp firmware."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

import mjlab
from mjlab.scripts.snakebot_keyboard import run_keyboard
from mjlab.scripts.snakebot_mpu_mqtt import (
  AllModuleMpuBridge,
  HybridPolicy,
  MpuFusion,
  MpuGoalViewer,
  MpuKeyboardConfig,
  MpuSample,
)
from mjlab.scripts.snakebot_mqtt import MqttGoalViewer, servo_payload_data


@dataclass(frozen=True)
class LiveSnakebotConfig(MpuKeyboardConfig):
  broker: str = "10.220.249.223"
  gyro_units: Literal["deg/s", "rad/s", "raw"] | None = "deg/s"
  """The supplied firmware divides raw gyro by 131 and publishes degrees/second."""
  mode: Literal["simulation", "imu-hybrid"] = "simulation"
  """simulation mirrors the model; imu-hybrid additionally feeds physical IMUs to RL."""
  angle_scale: float = 3.0
  """Hardware angle-offset multiplier; 3x gives roughly +/-15 degrees at the servos."""
  post_command_delay: float = 0.05
  """Hold 50 ms after command ACKs or a tolerated timeout before the next frame."""
  burst_size: int = 1
  """Simulation frames per burst; each frame contains ten angles. Use 1 for continuous."""
  burst_frame_interval: float = 0.05
  """Seconds between timestamped frames during ESP playback (20 Hz by default)."""
  burst_lead_time: float = 1.0
  """Upload lead time before the first queued frame executes."""
  max_missed_acks: int = 10
  """Pause simulation mode after this many consecutive missed commands per ESP."""
  ignore_ack_errors: bool = False
  """Ignore all command ACKs and their errors/waits in simulation mode."""
  feedback_csv: Path = Path("logs/hardware/live_head.csv")
  all_feedback_jsonl: Path = Path("logs/hardware/live_all_modules.jsonl")
  ack_csv: Path = Path("logs/hardware/live_command_ack.csv")
  """Command acknowledgements for simulation mode; no connected MPU is required."""
  ack_status_csv: Path = Path("logs/hardware/live_ack_status.csv")
  """Per-command ACK/miss status for every ESP, including missing replies."""
  max_steps: int | None = None
  """Optional number of policy steps before closing the viewer and disconnecting."""


class FirmwareMqttBridge(AllModuleMpuBridge):
  """Match both identifiers supplied by the firmware, across all five ESPs."""

  cfg: LiveSnakebotConfig

  def __init__(self, cfg: LiveSnakebotConfig, **kwargs):
    super().__init__(cfg, **kwargs)
    if cfg.ignore_ack_errors and cfg.mode != "simulation":
      raise ValueError("ignore_ack_errors requires simulation mode; hybrid needs IMUs")
    if set(cfg.module_ids) != {f"ESP_{i:02d}" for i in range(1, 6)}:
      raise ValueError("This firmware profile requires ESP_01 through ESP_05")
    if cfg.max_steps is not None and cfg.max_steps <= 0:
      raise ValueError("max_steps must be positive when set")
    if type(cfg.max_missed_acks) is not int or cfg.max_missed_acks < 1:
      raise ValueError("max_missed_acks must be a positive integer")
    if not 1 <= cfg.burst_size <= 180:
      raise ValueError(
        "burst_size must be between 1 and 180 (ESP queue capacity is 199)"
      )
    # Check a worst-case command against a conservative 256-byte MQTT packet
    # budget before opening any connection. The supplied MCU parses 512-byte JSON.
    probe = {
      "request_id": 2**32 - 1,
      "ts": 9999999999999,
      "data": servo_payload_data([[180.0, 180.0]] * 5, cfg),
    }
    packet_bytes = len(json.dumps(probe).encode()) + len(cfg.topic.encode()) + 7
    if packet_bytes > 256:
      raise ValueError(
        "Command topic/payload exceeds the 256-byte firmware packet budget"
      )
    self._reported_request = 0
    self._last_ack_outcome = 0
    self.missed_ack_streaks = {name: 0 for name in cfg.module_ids}

  def _on_feedback(self, client, userdata, msg):
    if self.cfg.ignore_ack_errors:
      return
    module = self._topics.get(msg.topic)
    if module is None:
      return
    try:
      data = json.loads(msg.payload.decode())
    except (ValueError, UnicodeDecodeError):
      return
    if not isinstance(data, dict):
      return
    with self._lock:
      pending = self._pending
    if pending is None or type(data.get("request_id")) is not int:
      return
    if data["request_id"] != pending[0]:
      return
    if data.get("esp_id") != module:
      self._fail(
        f"Feedback topic {msg.topic} has unexpected esp_id={data.get('esp_id')!r}"
      )
      return
    # request IDs restart at 1 each process. Matching command_ts also prevents a
    # delayed reply from an earlier run from acknowledging a new request #1.
    if type(data.get("command_ts")) is not int or data["command_ts"] != pending[1]:
      return
    if self.cfg.mode == "imu-hybrid":
      super()._on_feedback(client, userdata, msg)
      return
    # In simulation mode this packet acknowledges servo.write(), even when
    # no MPU is attached. Preserve sensor_valid=false and all raw telemetry;
    # never pretend missing measurements are real IMU data.
    sample = MpuSample(module, data["request_id"], self._clock(), data)
    with self._lock:
      if self._pending is None or self._pending[0] != sample.request_id:
        return
      self._samples[module] = sample
      if len(self._samples) == 5:
        self.latest_samples = tuple(self._samples[name] for name in self.cfg.module_ids)
        self._feedback = self.latest_samples[0].data

  def _validate_feedback(self, feedback: dict, pending: tuple[int, int, float]) -> None:
    if self.cfg.mode == "imu-hybrid":
      super()._validate_feedback(feedback, pending)
    # Simulation mode has already validated every ESP ID, request and timestamp.

  def _record_feedback(self, feedback: dict, pending: tuple[int, int, float]) -> None:
    if self.cfg.mode == "imu-hybrid":
      super()._record_feedback(feedback, pending)
      return
    samples = {s.module_id: s for s in self.latest_samples or ()}
    self._write_ack_csv(samples)
    self._record_ack_outcome(pending[0], pending[1], samples)

  def _write_ack_csv(self, samples: dict[str, MpuSample]) -> None:
    if not samples:
      return
    path = self.cfg.ack_csv
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    fields = ("esp_id", "request_id", "command_ts", "sensor_valid")
    with path.open("a", newline="", encoding="utf-8") as stream:
      writer = csv.DictWriter(stream, fieldnames=fields)
      if header:
        writer.writeheader()
      for sample in samples.values():
        writer.writerow({field: sample.data.get(field) for field in fields})

  def _record_ack_outcome(
    self, request_id: int, timestamp: int, samples: dict[str, MpuSample]
  ) -> None:
    """Resolve commands in request order; each frame affects each streak once."""
    if request_id <= self._last_ack_outcome:
      return
    self._last_ack_outcome = request_id
    missing = []
    for module in self.cfg.module_ids:
      previous = self.missed_ack_streaks[module]
      if module in samples:
        self.missed_ack_streaks[module] = 0
        if previous:
          print(f"{module} ACK recovered after {previous} missed commands")
      else:
        self.missed_ack_streaks[module] += 1
        missing.append(module)
    path = self.cfg.ack_status_csv
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
      writer = csv.writer(stream)
      if header:
        writer.writerow(
          ("esp_id", "request_id", "command_ts", "ack_received", "consecutive_misses")
        )
      for module in self.cfg.module_ids:
        writer.writerow(
          (
            module,
            request_id,
            timestamp,
            module in samples,
            self.missed_ack_streaks[module],
          )
        )
    if missing:
      detail = ", ".join(
        f"{module} ({self.missed_ack_streaks[module]}/{self.cfg.max_missed_acks})"
        for module in missing
      )
      print(f"Command #{request_id}: missing ACK after timeout: {detail}")
      failed = [
        module
        for module in missing
        if self.missed_ack_streaks[module] >= self.cfg.max_missed_acks
      ]
      if failed:
        message = (
          f"Command #{request_id}: {self.cfg.max_missed_acks} consecutive missing "
          f"acknowledgements from {', '.join(failed)}"
        )
        self._fail(message)
        raise TimeoutError(message)

  def _record_partial_acknowledgements(
    self, request_id: int, timestamp: int, samples: dict[str, MpuSample]
  ) -> None:
    # Log only actual received packets. Never synthesize an ACK or IMU sample.
    self._write_ack_csv(samples)
    if samples:
      path = self.cfg.all_feedback_jsonl
      path.parent.mkdir(parents=True, exist_ok=True)
      with path.open("a", encoding="utf-8") as stream:
        for sample in samples.values():
          stream.write(
            json.dumps(
              {
                "module_id": sample.module_id,
                "received_monotonic_s": sample.received_at,
                "feedback": sample.data,
              }
            )
            + "\n"
          )
    self._record_ack_outcome(request_id, timestamp, samples)

  def poll(self) -> bool:
    if self.cfg.ignore_ack_errors:
      # Time-based command spacing still applies; no reply is required or
      # recorded as successful. Transport errors remain fatal in parent poll().
      with self._lock:
        self._pending = None
        self._feedback = None
      return super().poll()
    expired = None
    if self.cfg.mode == "simulation":
      with self._lock:
        if (
          self._error is None
          and self._pending is not None
          and self._feedback is None
          and self._clock() - self._pending[2] > self.cfg.feedback_timeout
        ):
          expired = (self._pending, dict(self._samples))
          self._pending = None
      if expired is not None:
        pending, samples = expired
        self._record_partial_acknowledgements(pending[0], pending[1], samples)
        self._next_send = max(
          self._next_send, self._clock() + self.cfg.post_command_delay
        )
    ready = super().poll()
    if self.latest_samples is not None:
      request_id = self.latest_samples[0].request_id
      if request_id != self._reported_request:
        print(f"All five ESPs acknowledged command #{request_id}")
        self._reported_request = request_id
    return ready


def run_live(cfg: LiveSnakebotConfig) -> None:
  # Imported here to avoid a cycle: burst transport extends this firmware profile.
  from mjlab.scripts.snakebot_burst import BurstGoalViewer, BurstMqttBridge

  use_bursts = cfg.mode == "simulation" and cfg.burst_size > 1
  bridge = BurstMqttBridge(cfg) if use_bursts else FirmwareMqttBridge(cfg)

  def make_viewer(env, policy, *args):
    print(
      f"Live Snakebot: {cfg.mode} | broker {cfg.broker}:{cfg.port}\n"
      f"Head-to-tail ESP order: {', '.join(cfg.module_ids)}\n"
      "Servo 1 / pin 32: bottom; servo 2 / pin 33: top (existing sender order).\n"
      "WASD/arrows: move target | Space: pause/resume | J: angles | T: recenter\n"
      f"Ignore acknowledgement errors: {cfg.ignore_ack_errors}."
    )
    print(
      f"Hardware angle scale: {cfg.angle_scale:g}x; servo limits {cfg.servo_min}-{cfg.servo_max} degrees."
    )
    print(f"Continuous post-command hold: {cfg.post_command_delay * 1000:g} ms.")
    if cfg.mode == "imu-hybrid":
      print(
        "Hybrid mode uses physical IMUs; joint state and position remain model estimates."
      )
      if cfg.dry_run:
        print("DRY RUN: hybrid IMU packets are synthesized from the model.")
      fusion = MpuFusion(env.unwrapped, cfg)
      hybrid = HybridPolicy(env.unwrapped, policy, bridge, fusion)
      bridge.start()
      bridge.send([[0.0, 0.0]] * 5)
      return MpuGoalViewer(env, hybrid, *args, bridge=bridge, hybrid_policy=hybrid)
    bridge.start()
    print("SIMULATION ANGLES ONLY: MPU hardware and IMU readings are not required.")
    if cfg.ignore_ack_errors:
      print("ACK checks/waits disabled; playback assumes scheduled execution.")
    else:
      print(
        f"Missing ACKs are logged; pause after {cfg.max_missed_acks} consecutive misses per ESP."
      )
    # The initial neutral hold uses the same ACK timeout/miss policy as gait frames.
    bridge.send([[0.0, 0.0]] * 5)
    if use_bursts:
      assert isinstance(bridge, BurstMqttBridge)
      print(
        f"Collecting {cfg.burst_size} complete frames per burst; playback interval {cfg.burst_frame_interval:g}s."
      )
      print(
        "Space pauses future bursts; queued ESP commands finish their current playback."
      )
      return BurstGoalViewer(env, policy, *args, bridge=bridge)
    return MqttGoalViewer(env, policy, *args, bridge=bridge)

  try:
    run_keyboard(cfg, viewer_factory=make_viewer, num_steps=cfg.max_steps)
  finally:
    bridge.close()


def main() -> None:
  run_live(tyro.cli(LiveSnakebotConfig, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
