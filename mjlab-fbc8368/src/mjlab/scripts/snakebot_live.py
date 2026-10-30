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
  burst_size: int = 100
  """Simulation frames per burst; each frame contains ten angles. Use 1 for continuous."""
  burst_frame_interval: float = 0.05
  """Seconds between timestamped frames during ESP playback (20 Hz by default)."""
  burst_lead_time: float = 1.0
  """Upload lead time before the first queued frame executes."""
  feedback_csv: Path = Path("logs/hardware/live_head.csv")
  all_feedback_jsonl: Path = Path("logs/hardware/live_all_modules.jsonl")
  ack_csv: Path = Path("logs/hardware/live_command_ack.csv")
  """Command acknowledgements for simulation mode; no connected MPU is required."""
  max_steps: int | None = None
  """Optional number of policy steps before closing the viewer and disconnecting."""


class FirmwareMqttBridge(AllModuleMpuBridge):
  """Match both identifiers supplied by the firmware, across all five ESPs."""

  cfg: LiveSnakebotConfig

  def __init__(self, cfg: LiveSnakebotConfig, **kwargs):
    super().__init__(cfg, **kwargs)
    if set(cfg.module_ids) != {f"ESP_{i:02d}" for i in range(1, 6)}:
      raise ValueError("This firmware profile requires ESP_01 through ESP_05")
    if cfg.max_steps is not None and cfg.max_steps <= 0:
      raise ValueError("max_steps must be positive when set")
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

  def _on_feedback(self, client, userdata, msg):
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
    path = self.cfg.ack_csv
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    fields = ("esp_id", "request_id", "command_ts", "sensor_valid")
    with path.open("a", newline="", encoding="utf-8") as stream:
      writer = csv.DictWriter(stream, fieldnames=fields)
      if header:
        writer.writeheader()
      for sample in self.latest_samples or ():
        writer.writerow({field: sample.data.get(field) for field in fields})

  def poll(self) -> bool:
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
      "The live stream requires matching ESP IDs, request_id and command_ts from all five ESPs."
    )
    print(
      f"Hardware angle scale: {cfg.angle_scale:g}x; servo limits {cfg.servo_min}-{cfg.servo_max} degrees."
    )
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
    # Start with a known neutral hold. No gait command is sent until all five
    # modules have executed this timestamp and acknowledged the command.
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
