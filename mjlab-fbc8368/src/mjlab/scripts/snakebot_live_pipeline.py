"""Prepare future RL frames while timestamped commands play on the real snake."""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread

import tyro

import mjlab
from mjlab.scripts.snakebot_burst import BurstMqttBridge, QueuedFrame
from mjlab.scripts.snakebot_keyboard import (
  KeyboardGoalViewer,
  module_joint_angles_deg,
  run_keyboard,
)
from mjlab.scripts.snakebot_live import LiveSnakebotConfig
from mjlab.scripts.snakebot_mpu_mqtt import MpuSample
from mjlab.scripts.snakebot_mqtt import MqttGoalViewer, servo_payload_data


@dataclass(frozen=True)
class PipelineSnakebotConfig(LiveSnakebotConfig):
  burst_size: int = 100
  """Startup prefill and progress-reporting batch size; production then overlaps playback."""
  burst_frame_interval: float = 0.1
  """100 ms between physical angle sets, independently of simulation capture speed."""
  pipeline_queue_limit: int = 180
  """Maximum sent-but-unconfirmed commands at the slowest ESP (firmware holds 199)."""
  upload_interval: float = 0.01
  """Pace MQTT uploads at 10 ms instead of flooding the broker with whole batches."""
  capture_frame_rate: float = 30.0
  """Maximum display update rate; model capture runs as fast as the machine permits."""
  ack_csv: Path = Path("logs/hardware/pipeline_command_ack.csv")
  ack_status_csv: Path = Path("logs/hardware/pipeline_ack_status.csv")
  all_feedback_jsonl: Path = Path("logs/hardware/pipeline_all_modules.jsonl")


class PipelineMqttBridge(BurstMqttBridge):
  """A CPU-only sender worker; only the viewer thread touches the model/GPU.

  ESP firmware executes a FIFO. A matching execution ACK for request N proves
  that earlier queued requests at that ESP have been consumed. The oldest
  execution watermark across all five modules bounds queue occupancy, even
  when individual earlier ACK packets are lost. Wall time alone never frees
  hardware queue capacity.
  """

  cfg: PipelineSnakebotConfig
  _request_id: int

  def __init__(self, cfg: PipelineSnakebotConfig, **kwargs):
    super().__init__(cfg, **kwargs)
    if not 1 <= cfg.pipeline_queue_limit <= 180:
      raise ValueError("pipeline_queue_limit must be between 1 and 180")
    if (
      not math.isfinite(cfg.upload_interval)
      or not 0 < cfg.upload_interval < cfg.burst_frame_interval
    ):
      raise ValueError("upload_interval must be positive and less than frame interval")
    if not math.isfinite(cfg.capture_frame_rate) or cfg.capture_frame_rate <= 0:
      raise ValueError("capture_frame_rate must be finite and positive")
    milliseconds = cfg.burst_frame_interval * 1000
    if not math.isclose(milliseconds, round(milliseconds), abs_tol=1e-6):
      raise ValueError("Pipeline frame intervals must use whole milliseconds")
    self._frames: deque[dict[str, list[int]]] = deque()
    self._executed: dict[str, int] = {module: 0 for module in cfg.module_ids}
    self._capture_ready = Event()
    self._stop_sender = Event()
    self._sender: Thread | None = None
    self._prefilled = False
    self._producer_finished = False
    self._uploads_paused = False
    self._next_upload = 0.0
    self._next_timestamp: int | None = None
    self._tail_due = 0.0
    self._dry_due: deque[tuple[int, float]] = deque()
    self.captured_frames = 0
    self.published_frames = 0
    self.underruns = 0

  def _occupancy(self) -> int:
    """Called with _lock held; conservative bound for the slowest ESP."""
    return self._request_id - min(self._executed.values())

  @property
  def can_capture(self) -> bool:
    with self._lock:
      return (
        self._capture_ready.is_set()
        and not self._producer_finished
        and not self._uploads_paused
        and self._error is None
        and len(self._frames) < self.cfg.burst_size
        and self._occupancy() + len(self._frames) < 2 * self.cfg.burst_size
      )

  @property
  def drained(self) -> bool:
    with self._lock:
      return (
        not self._frames
        and self._pending is None
        and not self._burst_pending
        and self._clock() >= self._tail_due
      )

  def check_errors(self) -> None:
    with self._lock:
      error = self._error
    if error:
      raise RuntimeError(error)

  def enqueue_frame(self, targets: list[list[float]]) -> None:
    data = servo_payload_data(targets, self.cfg)
    if not self.can_capture:
      raise RuntimeError("Pipeline capture buffer is full or not ready")
    with self._lock:
      self._frames.append(data)
      self.captured_frames += 1
      self._prefilled |= len(self._frames) >= self.cfg.burst_size
      count = self.captured_frames
      queued = self._occupancy()
    batch, frame = divmod(count - 1, self.cfg.burst_size)
    if frame == 0 or (frame + 1) % 20 == 0:
      print(
        f"Pipeline capture batch {batch + 1}: {frame + 1}/{self.cfg.burst_size}; "
        f"{queued} hardware commands unconfirmed"
      )

  def finish_capture(self) -> None:
    with self._lock:
      self._producer_finished = True
      self._prefilled = bool(self._frames) or self._prefilled

  def pause_uploads(self, paused: bool) -> None:
    with self._lock:
      self._uploads_paused = paused

  def reset_buffer(self) -> bool:
    with self._lock:
      if self._pending or self._burst_pending or self._clock() < self._tail_due:
        return False
      self._frames.clear()
      self._prefilled = False
      self._producer_finished = False
      self._next_timestamp = None
    return True

  def _on_feedback(self, client, userdata, msg):
    module = self._topics.get(msg.topic)
    if module is None:
      return
    try:
      data = json.loads(msg.payload.decode())
    except (ValueError, UnicodeDecodeError):
      return
    if not isinstance(data, dict) or type(data.get("request_id")) is not int:
      return
    request_id = data["request_id"]
    with self._lock:
      frame = self._burst_pending.get(request_id)
    if frame is None:
      super()._on_feedback(client, userdata, msg)  # Initial neutral command.
      with self._lock:
        sample = self._samples.get(module)
        if (
          sample is not None
          and sample.request_id == request_id
          and sample.data.get("command_ts") == data.get("command_ts")
        ):
          self._executed[module] = max(self._executed[module], request_id)
      return
    if data.get("esp_id") != module:
      self._fail(f"Pipeline feedback on {msg.topic} has an unexpected ESP identity")
      return
    if (
      type(data.get("command_ts")) is not int
      or data["command_ts"] != frame.payload["ts"]
    ):
      return
    sample = MpuSample(module, request_id, self._clock(), data)
    with self._lock:
      if self._burst_pending.get(request_id) is frame:
        frame.acknowledgements[module] = sample
        self._executed[module] = max(self._executed[module], request_id)

  def service_once(self) -> None:
    """One sender iteration, also used with fake clocks by protocol tests."""
    if self.cfg.dry_run:
      with self._lock:
        while self._dry_due and self._dry_due[0][1] <= self._clock():
          request_id, _ = self._dry_due.popleft()
          self._executed = dict.fromkeys(self.cfg.module_ids, request_id)
    ready = super().poll()  # ACK accounting runs only on this sender thread.
    if not self._capture_ready.is_set():
      if not ready:
        return
      if self.cfg.dry_run:
        self._executed = dict.fromkeys(self.cfg.module_ids, self._request_id)
      self._capture_ready.set()
    now = self._clock()
    with self._lock:
      if (
        self._stop_sender.is_set()
        or self._uploads_paused
        or not self._prefilled
        or not self._frames
        or now < self._next_upload
        or self._occupancy() >= self.cfg.pipeline_queue_limit
      ):
        return
      data = self._frames.popleft()
      wall = self._wall_clock()
      timestamp = self._next_timestamp
      if timestamp is None or timestamp < (wall + self.cfg.command_delay) * 1000:
        if timestamp is not None:
          self.underruns += 1
          print("Pipeline buffer underrun/resume: scheduling a fresh playback lead.")
        timestamp = math.ceil((wall + self.cfg.burst_lead_time) * 1000)
      self._request_id += 1
      payload = {"request_id": self._request_id, "ts": timestamp, "data": data}
      due = now + (timestamp / 1000 - wall)
      if self.cfg.dry_run:
        self._dry_due.append((self._request_id, due))
      else:
        self._burst_pending[self._request_id] = QueuedFrame(payload, due)
      self._next_timestamp = timestamp + round(self.cfg.burst_frame_interval * 1000)
      self._tail_due = due + self.cfg.burst_frame_interval
      self._next_upload = now + self.cfg.upload_interval
      self.published_frames += 1
      published = self.published_frames
    if not self.cfg.dry_run:
      if self._client is None or not self._client.is_connected():
        raise RuntimeError("MQTT is not connected")
      result = self._client.publish(
        self.cfg.topic, json.dumps(payload), qos=0, retain=False
      )
      if result.rc != 0:
        raise RuntimeError(f"Failed to publish pipeline frame #{payload['request_id']}")
    if published == 1 or published % self.cfg.burst_size == 0:
      print(
        f"{'DRY RUN' if self.cfg.dry_run else 'MQTT'} PIPELINE: {published} frames "
        f"uploaded; last request #{payload['request_id']}; "
        f"{self.cfg.burst_frame_interval * 1000:g} ms playback spacing"
      )

  def start_sender(self) -> None:
    if self._sender is not None:
      raise RuntimeError("Pipeline sender already started")

    def work():
      while not self._stop_sender.is_set():
        try:
          self.service_once()
        except (RuntimeError, TimeoutError, ValueError, OSError) as exc:
          self._fail(str(exc))
          return
        self._stop_sender.wait(self.cfg.upload_interval)

    self._sender = Thread(target=work, name="snakebot-pipeline", daemon=True)
    self._sender.start()

  def close(self) -> None:
    self._stop_sender.set()
    if self._sender is not None:
      self._sender.join(timeout=2.0)
    super().close()
    print(
      f"Pipeline stopped: {self.captured_frames} frames captured, "
      f"{self.published_frames} uploaded, {self.underruns} buffer underruns. "
      "Commands already queued on ESPs may still execute."
    )


class PipelineGoalViewer(MqttGoalViewer):
  bridge: PipelineMqttBridge

  def __init__(self, *args, bridge: PipelineMqttBridge, **kwargs):
    super().__init__(*args, bridge=bridge, **kwargs)
    self.frame_rate = bridge.cfg.capture_frame_rate
    self.frame_time = 1.0 / self.frame_rate
    self._next_render = 0.0
    self._capture_limit: int | None = None

  def run(self, num_steps: int | None = None) -> None:
    self._capture_limit = num_steps
    super().run(num_steps=None)  # Keep the sender alive until bounded playback drains.

  def is_running(self) -> bool:
    if self._capture_limit is not None:
      if self._hardware_fault:
        return False
      if self._step_count >= self._capture_limit and self.bridge.drained:
        return False
    return super().is_running()

  def step_simulation(self) -> None:
    if self._is_paused or self._hardware_fault or not self.bridge.can_capture:
      return
    if self._capture_limit is not None and self._step_count >= self._capture_limit:
      return
    try:
      KeyboardGoalViewer.step_simulation(self)
      if self.env.unwrapped.episode_length_buf[0].item() == 0:
        raise RuntimeError("Unstable simulation reset; pipeline production stopped")
      self.bridge.enqueue_frame(
        module_joint_angles_deg(self.env.unwrapped)[:, 2:].cpu().tolist()
      )
      if self._capture_limit is not None and self._step_count >= self._capture_limit:
        self.bridge.finish_capture()
    except (RuntimeError, ValueError, OSError) as exc:
      self._stop_on_fault(exc)

  def tick(self) -> bool:
    self._process_actions()
    if not self._hardware_fault:
      try:
        self.bridge.check_errors()
      except RuntimeError as exc:
        self._stop_on_fault(exc)
    # Generate as fast as possible, not at the simulation's wall-clock playback
    # rate. The sender independently enforces physical execution timestamps.
    if not self._is_paused and not self._hardware_fault and self.bridge.can_capture:
      self.sync_viewer_to_env()
      before = self._step_count
      self.step_simulation()
      self._actual_sim_time += (self._step_count - before) * self.env.unwrapped.step_dt
    now = time.monotonic()
    if now < self._next_render:
      return False
    self._next_render = now + self.frame_time
    with self._render_timer.measure_time():
      self.sync_env_to_viewer()
    self._accumulated_render_time += self._render_timer.measured_time
    self._frame_count += 1
    self._update_fps()
    return True

  def pause(self) -> None:
    was_paused = self._is_paused
    self.bridge.pause_uploads(True)
    super().pause()
    if not was_paused:
      print(
        "Pipeline capture/uploads paused; commands already on the ESPs still execute."
      )

  def resume(self) -> None:
    super().resume()
    if not self._hardware_fault:
      self.bridge.pause_uploads(False)

  def reset_environment(self) -> None:
    if not self.bridge.reset_buffer():
      print("Pause and wait for queued commands/ACKs to drain before resetting.")
      return
    super().reset_environment()


def run_pipeline(cfg: PipelineSnakebotConfig) -> None:
  bridge = PipelineMqttBridge(cfg)

  def make_viewer(env, policy, *args):
    print(
      f"PIPELINED RL: {cfg.burst_size}-frame prefill, "
      f"{cfg.burst_frame_interval * 1000:g} ms physical spacing, "
      f"{cfg.angle_scale:g}x servo scaling.\n"
      "The next frames are prepared and uploaded while earlier frames play.\n"
      f"Hardware queue bound: {cfg.pipeline_queue_limit}; "
      f"pause after {cfg.max_missed_acks} consecutive missing ACKs per ESP.\n"
      "WASD uses the whole snake axis and affects frames not yet captured.\n"
      "Space pauses future capture/uploads; queued motion cannot be cancelled."
    )
    bridge.start()
    bridge.send([[0.0, 0.0]] * 5)
    bridge.start_sender()
    return PipelineGoalViewer(env, policy, *args, bridge=bridge)

  try:
    run_keyboard(cfg, viewer_factory=make_viewer, num_steps=cfg.max_steps)
  finally:
    bridge.close()


def main() -> None:
  run_pipeline(tyro.cli(PipelineSnakebotConfig, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
