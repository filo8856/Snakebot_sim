"""Five-module launcher matching the supplied ESP32 main.cpp firmware."""

from __future__ import annotations

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
)
from mjlab.scripts.snakebot_mqtt import MqttGoalViewer, servo_payload_data


@dataclass(frozen=True)
class LiveSnakebotConfig(MpuKeyboardConfig):
  broker: str = "10.220.249.223"
  gyro_units: Literal["deg/s", "rad/s", "raw"] | None = "deg/s"
  """The supplied firmware divides raw gyro by 131 and publishes degrees/second."""
  mode: Literal["simulation", "imu-hybrid"] = "simulation"
  """simulation mirrors the model; imu-hybrid additionally feeds physical IMUs to RL."""
  feedback_csv: Path = Path("logs/hardware/live_head.csv")
  all_feedback_jsonl: Path = Path("logs/hardware/live_all_modules.jsonl")
  max_steps: int | None = None
  """Optional number of policy steps before closing the viewer and disconnecting."""


class FirmwareMqttBridge(AllModuleMpuBridge):
  """Match both identifiers supplied by the firmware, across all five ESPs."""

  def __init__(self, cfg: LiveSnakebotConfig, **kwargs):
    super().__init__(cfg, **kwargs)
    if set(cfg.module_ids) != {f"ESP_{i:02d}" for i in range(1, 6)}:
      raise ValueError("This firmware profile requires ESP_01 through ESP_05")
    if cfg.max_steps is not None and cfg.max_steps <= 0:
      raise ValueError("max_steps must be positive when set")
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
    super()._on_feedback(client, userdata, msg)

  def poll(self) -> bool:
    ready = super().poll()
    if self.latest_samples is not None:
      request_id = self.latest_samples[0].request_id
      if request_id != self._reported_request:
        print(f"All five ESPs acknowledged command #{request_id}")
        self._reported_request = request_id
    return ready


def run_live(cfg: LiveSnakebotConfig) -> None:
  bridge = FirmwareMqttBridge(cfg)

  def make_viewer(env, policy, *args):
    print(
      f"Live Snakebot: {cfg.mode} | broker {cfg.broker}:{cfg.port}\n"
      f"Head-to-tail ESP order: {', '.join(cfg.module_ids)}\n"
      "Servo 1 / pin 32: bottom; servo 2 / pin 33: top (existing sender order).\n"
      "WASD/arrows: move target | Space: pause/resume | J: angles | T: recenter\n"
      "The live stream requires matching request_id, command_ts and valid IMU data from all five ESPs."
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
    # Start with a known neutral hold. No gait command is sent until all five
    # modules have executed this timestamp and returned valid sensor snapshots.
    bridge.send([[0.0, 0.0]] * 5)
    return MqttGoalViewer(env, policy, *args, bridge=bridge)

  try:
    run_keyboard(cfg, viewer_factory=make_viewer, num_steps=cfg.max_steps)
  finally:
    bridge.close()


def main() -> None:
  run_live(tyro.cli(LiveSnakebotConfig, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
