"""Rolling queue, overlapping capture and playback, using no real hardware."""

import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from mjlab.scripts.snakebot_keyboard import KeyboardGoalViewer
from mjlab.scripts.snakebot_live_pipeline import (
  PipelineGoalViewer,
  PipelineMqttBridge,
  PipelineSnakebotConfig,
)

TARGETS = [[5.16, -5.16]] * 5


def acknowledge(bridge, payload, modules=None, **overrides):
  for module in modules or bridge.cfg.module_ids:
    packet = {
      "esp_id": module,
      "request_id": payload["request_id"],
      "command_ts": payload["ts"],
      "sensor_valid": False,
      **overrides,
    }
    bridge._on_feedback(
      None,
      None,
      SimpleNamespace(
        topic=f"servos/{module}/feedback", payload=json.dumps(packet).encode()
      ),
    )


def fixture_bridge(tmp_path, **settings):
  cfg = PipelineSnakebotConfig(
    ack_csv=tmp_path / "ack.csv",
    ack_status_csv=tmp_path / "status.csv",
    all_feedback_jsonl=tmp_path / "feedback.jsonl",
    **settings,
  )
  now = [0.0]
  client = MagicMock()
  client.subscribe.return_value = (0, 1)
  client.publish.return_value = SimpleNamespace(rc=0)
  client.is_connected.return_value = True
  factory = MagicMock(return_value=client)
  bridge = PipelineMqttBridge(
    cfg, client_factory=factory, clock=lambda: now[0], wall_clock=lambda: 1000 + now[0]
  )

  def connect(*args):
    client.on_connect(client, None, None, 0, None)
    client.on_subscribe(client, None, 1, [], None)

  client.connect.side_effect = connect
  bridge.start()
  neutral = bridge.send([[0.0, 0.0]] * 5)
  if not cfg.dry_run:
    acknowledge(bridge, neutral)
  now[0] = 1.1
  bridge.service_once()
  now[0] = 1.151
  bridge.service_once()
  assert bridge.can_capture
  return bridge, client, now, factory


def uploaded(client):
  return [json.loads(call.args[1]) for call in client.publish.call_args_list][1:]


def test_next_batch_uploads_while_first_batch_is_still_playing(tmp_path):
  bridge, client, now, _ = fixture_bridge(tmp_path)
  try:
    for _ in range(100):
      bridge.enqueue_frame(TARGETS)
    for _ in range(100):
      bridge.service_once()
      now[0] += 0.011
    first = uploaded(client)
    assert len(first) == 100
    assert now[0] < first[-1]["ts"] / 1000 - 1000
    # No execution ACK is required before the model can prepare frame 101.
    bridge.enqueue_frame(TARGETS)
    bridge.service_once()
    commands = uploaded(client)
    assert len(commands) == 101
    assert commands[-1]["ts"] == first[-1]["ts"] + 100
    assert all(
      b["ts"] - a["ts"] == 100 for a, b in zip(commands[:-1], commands[1:], strict=True)
    )
    assert commands[0]["data"] == {
      module: [105, 75] for module in bridge.cfg.module_ids
    }
    assert bridge.underruns == 0
  finally:
    bridge.close()


def test_slowest_esp_limits_queue_and_old_timestamp_cannot_free_a_slot(tmp_path):
  bridge, client, now, _ = fixture_bridge(tmp_path)
  try:
    for _ in range(100):
      bridge.enqueue_frame(TARGETS)
    while bridge.published_frames < 180:
      bridge.service_once()
      if bridge.can_capture and bridge.captured_frames < 200:
        bridge.enqueue_frame(TARGETS)
      now[0] += 0.011
    assert bridge.captured_frames == 200
    assert not bridge.can_capture
    first = uploaded(client)[0]
    acknowledge(bridge, first, modules=bridge.cfg.module_ids[:4])
    for _ in range(10):
      bridge.service_once()
      now[0] += 0.011
    assert bridge.published_frames == 180
    acknowledge(bridge, first, modules=("ESP_05",), command_ts=first["ts"] - 1)
    bridge.service_once()
    assert bridge.published_frames == 180
    acknowledge(bridge, first, modules=("ESP_05",))
    bridge.service_once()
    assert bridge.published_frames == 181
    assert bridge.can_capture
    assert bridge._occupancy() == 180
  finally:
    bridge.close()


def test_later_execution_ack_frees_fifo_space_without_faking_earlier_ack(tmp_path):
  bridge, client, now, _ = fixture_bridge(tmp_path, burst_size=2)
  try:
    bridge.enqueue_frame(TARGETS)
    bridge.enqueue_frame(TARGETS)
    bridge.service_once()
    now[0] += 0.011
    bridge.service_once()
    commands = uploaded(client)
    others = [module for module in bridge.cfg.module_ids if module != "ESP_03"]
    acknowledge(bridge, commands[0], modules=others)
    acknowledge(bridge, commands[1])
    assert bridge._occupancy() == 0  # FIFO execution passed both commands.
    assert (
      "ESP_03" not in bridge._burst_pending[commands[0]["request_id"]].acknowledgements
    )
    now[0] = 8.0
    bridge.service_once()
    assert bridge.missed_ack_streaks["ESP_03"] == 0  # Miss, then recovery.
    assert "False,1" in bridge.cfg.ack_status_csv.read_text()
  finally:
    bridge.close()


def test_pause_stops_new_uploads_but_still_processes_feedback(tmp_path):
  bridge, client, now, _ = fixture_bridge(tmp_path, burst_size=3)
  try:
    for _ in range(3):
      bridge.enqueue_frame(TARGETS)
    bridge.service_once()
    first = uploaded(client)[0]
    bridge.pause_uploads(True)
    acknowledge(bridge, first)
    now[0] += 2.0
    bridge.service_once()
    assert bridge.published_frames == 1
    assert bridge._last_ack_outcome == first["request_id"]
    assert not bridge.can_capture
    bridge.pause_uploads(False)
    bridge.service_once()
    second = uploaded(client)[1]
    assert second["ts"] >= math.ceil((1000 + now[0] + 1) * 1000)
    assert bridge.underruns == 1
  finally:
    bridge.close()


def test_dry_run_flushes_partial_prefill_and_drains_without_network(tmp_path):
  bridge, client, now, factory = fixture_bridge(tmp_path, dry_run=True)
  try:
    bridge.enqueue_frame(TARGETS)
    bridge.service_once()
    assert bridge.published_frames == 0  # Startup prefill not yet complete.
    bridge.finish_capture()
    bridge.service_once()
    assert bridge.published_frames == 1
    assert not bridge.drained
    now[0] += 1.2
    bridge.service_once()
    assert bridge.drained
    assert bridge._occupancy() == 0
    factory.assert_not_called()
    client.publish.assert_not_called()
    assert not bridge.cfg.ack_csv.exists()
  finally:
    bridge.close()


def test_pipeline_keeps_consecutive_miss_fault_and_prevents_further_production(
  tmp_path,
):
  bridge, client, now, _ = fixture_bridge(tmp_path, burst_size=2, max_missed_acks=2)
  try:
    for _ in range(2):
      bridge.enqueue_frame(TARGETS)
    bridge.finish_capture()
    bridge.service_once()
    now[0] += 0.011
    bridge.service_once()
    others = [module for module in bridge.cfg.module_ids if module != "ESP_03"]
    for command in uploaded(client):
      acknowledge(bridge, command, modules=others)
    now[0] = 8.0
    with pytest.raises(TimeoutError, match="2 consecutive.*ESP_03"):
      bridge.service_once()
    with pytest.raises(RuntimeError):
      bridge.check_errors()
    assert not bridge.can_capture
  finally:
    bridge.close()


def test_sender_worker_records_publish_failure_and_stops(tmp_path):
  bridge, client, _, _ = fixture_bridge(tmp_path, burst_size=1)
  try:
    bridge.enqueue_frame(TARGETS)
    client.publish.return_value = SimpleNamespace(rc=4)
    bridge.start_sender()
    assert bridge._sender is not None
    bridge._sender.join(timeout=1.0)
    assert not bridge._sender.is_alive()
    with pytest.raises(RuntimeError, match="Failed to publish pipeline"):
      bridge.check_errors()
  finally:
    bridge.close()


@pytest.mark.parametrize(
  "settings",
  [
    {"pipeline_queue_limit": 199},
    {"pipeline_queue_limit": 0},
    {"upload_interval": 0.0},
    {"upload_interval": 0.1},
    {"upload_interval": math.nan},
    {"capture_frame_rate": 0.0},
    {"mode": "imu-hybrid"},
    {"burst_frame_interval": 0.1005},
  ],
)
def test_invalid_pipeline_settings_are_rejected(settings):
  with pytest.raises(ValueError):
    PipelineMqttBridge(PipelineSnakebotConfig(**settings))


def test_viewer_captures_while_playback_is_pending_and_finishes_bounded_capture(
  monkeypatch,
):
  bridge = MagicMock()
  bridge.cfg = PipelineSnakebotConfig()
  bridge.can_capture = True
  bridge.poll.return_value = False  # Existing burst viewer would stop capturing.
  env = MagicMock()
  env.unwrapped.episode_length_buf = torch.tensor([1])

  def step(viewer):
    viewer._step_count += 1

  monkeypatch.setattr(KeyboardGoalViewer, "step_simulation", step)
  monkeypatch.setattr(
    "mjlab.scripts.snakebot_live_pipeline.module_joint_angles_deg",
    lambda _: torch.tensor([[0.0, 0.0, 1.0, -1.0]] * 5),
  )
  viewer = PipelineGoalViewer(env, MagicMock(), bridge=bridge)
  viewer._capture_limit = 2
  viewer.step_simulation()
  viewer.step_simulation()
  assert bridge.enqueue_frame.call_count == 2
  bridge.finish_capture.assert_called_once()
  bridge.poll.assert_not_called()  # Only the sender thread resolves feedback.
  viewer.pause()
  bridge.pause_uploads.assert_called_with(True)
  viewer.step_simulation()
  assert bridge.enqueue_frame.call_count == 2


def test_pipeline_fault_pauses_once_and_keeps_the_original_error_visible(
  capsys,
  monkeypatch,
):
  bridge = MagicMock()
  bridge.cfg = PipelineSnakebotConfig()
  bridge.check_errors.side_effect = RuntimeError("ESP_03 missed 10 consecutive ACKs")
  env = MagicMock()
  viewer = PipelineGoalViewer(env, MagicMock(), bridge=bridge)
  monkeypatch.setattr(viewer, "sync_env_to_viewer", MagicMock())
  monkeypatch.setattr(viewer, "sync_viewer_to_env", MagicMock())
  for _ in range(20):
    viewer.tick()
  output = capsys.readouterr().out
  assert output.count("Hardware stream stopped:") == 1
  assert output.count("Pipeline capture/uploads paused;") == 1
  assert "ESP_03 missed 10 consecutive ACKs" in output
  bridge.check_errors.assert_called_once()
  bridge.enqueue_frame.assert_not_called()
  assert viewer._hardware_fault and viewer._is_paused
  viewer.resume()
  assert viewer._is_paused  # An actual fault still requires a restart.
  bridge.pause_uploads.assert_called_once_with(True)


def test_pipeline_repeated_pause_calls_are_silent_until_resumed(capsys):
  bridge = MagicMock()
  bridge.cfg = PipelineSnakebotConfig()
  viewer = PipelineGoalViewer(MagicMock(), MagicMock(), bridge=bridge)
  viewer.pause()
  viewer.pause()
  viewer.pause()
  assert capsys.readouterr().out.count("Pipeline capture/uploads paused;") == 1
  viewer.resume()
  viewer.pause()
  assert capsys.readouterr().out.count("Pipeline capture/uploads paused;") == 1
