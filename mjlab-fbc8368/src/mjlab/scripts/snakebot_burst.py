"""Capture model commands in batches and queue timestamped ESP playback."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Callable

from mjlab.scripts.snakebot_keyboard import KeyboardGoalViewer, module_joint_angles_deg
from mjlab.scripts.snakebot_live import FirmwareMqttBridge, LiveSnakebotConfig
from mjlab.scripts.snakebot_mpu_mqtt import MpuSample
from mjlab.scripts.snakebot_mqtt import MqttGoalViewer, servo_payload_data


@dataclass
class QueuedFrame:
  payload: dict
  due_at: float
  acknowledgements: dict[str, MpuSample] = field(default_factory=dict)


class BurstMqttBridge(FirmwareMqttBridge):
  """Upload one bounded batch, then await scheduled execution/acknowledgements."""

  UPLOAD_SPACING = 0.002

  def __init__(
    self,
    cfg: LiveSnakebotConfig,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
  ):
    super().__init__(cfg, **kwargs)
    if cfg.mode != "simulation":
      raise ValueError(
        "Burst playback is only for simulation mode; hybrid needs live IMUs"
      )
    if not math.isfinite(cfg.burst_frame_interval) or cfg.burst_frame_interval < 0.02:
      raise ValueError("burst_frame_interval must be finite and at least 0.02 seconds")
    if (
      not math.isfinite(cfg.burst_lead_time)
      or cfg.burst_lead_time < cfg.burst_size * self.UPLOAD_SPACING + 0.2
    ):
      raise ValueError("burst_lead_time must allow time to upload the complete batch")
    self._sleep = sleep
    self._burst_pending: dict[int, QueuedFrame] = {}
    self._burst_active = False
    self._burst_end = 0.0
    self._burst_total = 0

  @property
  def burst_active(self) -> bool:
    return self._burst_active

  def send_burst(self, frames: list[list[list[float]]]) -> list[dict]:
    if not self.poll():
      raise RuntimeError("Wait for the previous command/burst before uploading another")
    if not 1 <= len(frames) <= self.cfg.burst_size:
      raise ValueError("Burst must contain one to burst_size complete command frames")
    if not self.cfg.dry_run and (
      self._client is None or not self._client.is_connected()
    ):
      raise RuntimeError("MQTT is not connected")
    start = self._clock()
    first_ts = int((self._wall_clock() + self.cfg.burst_lead_time) * 1000)
    # Validate/scale every frame before publishing any portion of the batch.
    payloads = [
      {
        "request_id": self._request_id + i + 1,
        "ts": first_ts + round(i * self.cfg.burst_frame_interval * 1000),
        "data": servo_payload_data(frame, self.cfg),
      }
      for i, frame in enumerate(frames)
    ]
    with self._lock:
      self._burst_pending = (
        {}
        if self.cfg.dry_run
        else {
          p["request_id"]: QueuedFrame(
            p, start + self.cfg.burst_lead_time + i * self.cfg.burst_frame_interval
          )
          for i, p in enumerate(payloads)
        }
      )
      self._request_id = payloads[-1]["request_id"]
      self._burst_active = True
      self._burst_end = (
        start
        + self.cfg.burst_lead_time
        + (len(frames) - 1) * self.cfg.burst_frame_interval
      )
      self._burst_total = len(frames)
    for i, payload in enumerate(payloads):
      if not self.cfg.dry_run:
        assert self._client is not None
        result = self._client.publish(
          self.cfg.topic, json.dumps(payload), qos=0, retain=False
        )
        if result.rc != 0:
          self._fail(f"Failed to publish burst frame #{payload['request_id']}")
          raise RuntimeError(f"Failed to publish burst frame #{payload['request_id']}")
        if i + 1 < len(payloads):
          self._sleep(self.UPLOAD_SPACING)
    if self._clock() >= start + self.cfg.burst_lead_time:
      self._fail("Burst upload exceeded its lead time; queued frames may still execute")
      raise RuntimeError("Burst upload missed the first execution timestamp")
    duration = len(frames) * self.cfg.burst_frame_interval
    print(
      f"{'DRY RUN' if self.cfg.dry_run else 'MQTT'} BURST: {len(frames)} frames / "
      f"{len(frames) * 10} servo angles, {duration:.2f}s playback, "
      f"request IDs {payloads[0]['request_id']}-{payloads[-1]['request_id']}"
    )
    print(f"First hardware frame: {payloads[0]['data']}")
    return payloads

  def _on_feedback(self, client, userdata, msg):
    try:
      data = json.loads(msg.payload.decode())
    except (ValueError, UnicodeDecodeError):
      return
    if not isinstance(data, dict) or type(data.get("request_id")) is not int:
      return
    module = self._topics.get(msg.topic)
    with self._lock:
      frame = self._burst_pending.get(data["request_id"])
    if frame is None:
      super()._on_feedback(client, userdata, msg)  # Initial neutral hold.
      return
    if module is None:
      return
    if data.get("esp_id") != module:
      self._fail(f"Burst feedback on {msg.topic} has an unexpected ESP identity")
      return
    if (
      type(data.get("command_ts")) is not int
      or data["command_ts"] != frame.payload["ts"]
    ):
      return
    sample = MpuSample(module, data["request_id"], self._clock(), data)
    with self._lock:
      frame.acknowledgements[module] = sample

  def poll(self) -> bool:
    with self._lock:
      error = self._error
      completed = [
        request
        for request, frame in self._burst_pending.items()
        if len(frame.acknowledgements) == 5
      ]
      complete_frames = [self._burst_pending.pop(request) for request in completed]
      overdue = [
        (request, frame)
        for request, frame in self._burst_pending.items()
        if self._clock() > frame.due_at + self.cfg.feedback_timeout
      ]
    if error:
      raise RuntimeError(error)
    if overdue:
      request, frame = overdue[0]
      missing = [
        name for name in self.cfg.module_ids if name not in frame.acknowledgements
      ]
      message = (
        f"Burst frame #{request} timed out; missing modules: {', '.join(missing)}"
      )
      self._fail(message)
      raise TimeoutError(message)
    for frame in complete_frames:
      self.latest_samples = tuple(
        frame.acknowledgements[name] for name in self.cfg.module_ids
      )
      first = self.latest_samples[0]
      self._record_feedback(
        first.data, (first.request_id, frame.payload["ts"], frame.due_at)
      )
      super().poll()  # Existing JSONL and per-command acknowledgement reporting.
    if self._burst_active:
      with self._lock:
        pending = bool(self._burst_pending)
      if pending or self._clock() < self._burst_end:
        return False
      self._burst_active = False
      print(
        f"Burst playback complete: {self._burst_total} frames. Ready for next batch."
      )
    return super().poll()


class BurstGoalViewer(MqttGoalViewer):
  bridge: BurstMqttBridge

  def __init__(self, *args, bridge: BurstMqttBridge, **kwargs):
    super().__init__(*args, bridge=bridge, **kwargs)
    self._frames: list[list[list[float]]] = []
    self._capture_limit: int | None = None

  def run(self, num_steps: int | None = None) -> None:
    self._capture_limit = num_steps
    # A bounded run must finish queued playback before disconnecting.
    super().run(num_steps=None)

  def is_running(self) -> bool:
    if self._capture_limit is not None and self._step_count >= self._capture_limit:
      if not self._frames and not self.bridge.burst_active:
        return False
    return super().is_running()

  def step_simulation(self) -> None:
    if self._is_paused or self._hardware_fault:
      return
    try:
      if not self.bridge.poll():
        return
      if self._capture_limit is not None and self._step_count >= self._capture_limit:
        return
      # Capture model frames without invoking the parent's per-frame publish.
      KeyboardGoalViewer.step_simulation(self)
      if self.env.unwrapped.episode_length_buf[0].item() == 0:
        raise RuntimeError("Unstable simulation reset; buffered commands discarded")
      self._frames.append(
        module_joint_angles_deg(self.env.unwrapped)[:, 2:].cpu().tolist()
      )
      count = len(self._frames)
      if count == 1 or count % 20 == 0:
        print(f"Recording burst: {count}/{self.bridge.cfg.burst_size} frames")
      at_limit = (
        self._capture_limit is not None and self._step_count >= self._capture_limit
      )
      if count >= self.bridge.cfg.burst_size or at_limit:
        self.bridge.send_burst(self._frames)
        self._frames = []
    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
      self._frames.clear()
      self._stop_on_fault(exc)

  def pause(self) -> None:
    super().pause()
    if self.bridge.burst_active:
      print("Future bursts paused. Commands already queued on the ESPs still finish.")

  def reset_environment(self) -> None:
    if self.bridge.burst_active:
      print("Wait for queued burst playback to finish before resetting the model.")
      return
    self._frames.clear()
    super().reset_environment()

  def close(self) -> None:
    self._frames.clear()
    if self.bridge.burst_active:
      print("Closing prevents new bursts; already queued ESP frames may still execute.")
    super().close()
