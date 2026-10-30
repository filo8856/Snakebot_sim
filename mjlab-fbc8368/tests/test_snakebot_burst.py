"""Scaled hardware targets and timestamped 100-frame bursts; no real hardware."""

import csv
import json
import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from mjlab.scripts.snakebot_burst import BurstGoalViewer, BurstMqttBridge
from mjlab.scripts.snakebot_keyboard import KeyboardGoalViewer
from mjlab.scripts.snakebot_live import LiveSnakebotConfig
from mjlab.scripts.snakebot_mqtt import servo_payload_data


def bridge_fixture(tmp_path, **settings):
  settings.setdefault("burst_size", 100)
  cfg = LiveSnakebotConfig(
    ack_csv=tmp_path / "ack.csv",
    feedback_csv=tmp_path / "imu.csv",
    all_feedback_jsonl=tmp_path / "all.jsonl",
    **settings,
  )
  now = [0.0]
  client = MagicMock()
  client.is_connected.return_value = True
  client.subscribe.return_value = (0, 1)
  client.publish.return_value = SimpleNamespace(rc=0)
  bridge = BurstMqttBridge(
    cfg,
    client_factory=lambda: client,
    clock=lambda: now[0],
    wall_clock=lambda: 1793310000.0 + now[0],
    sleep=lambda _: None,
  )

  def connect(*args):
    client.on_connect(client, None, None, 0, None)
    client.on_subscribe(client, None, 1, [], None)

  client.connect.side_effect = connect
  bridge.start()
  return bridge, client, now


def acknowledge(bridge, payload, modules=None, **overrides):
  for module in modules or bridge.cfg.module_ids:
    data = {
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
        topic=f"servos/{module}/feedback", payload=json.dumps(data).encode()
      ),
    )


def test_scale_changes_offsets_without_scaling_neutral_and_keeps_limits():
  cfg = LiveSnakebotConfig()
  assert cfg.angle_scale == 3 and cfg.burst_size == 1
  assert servo_payload_data([[5.16, -5.16]] * 5, cfg)["ESP_01"] == [105, 75]
  assert servo_payload_data([[0.0, 0.0]] * 5, cfg)["ESP_01"] == [90, 90]
  calibrated = replace(cfg, neutral_angles=(100, 80))
  assert servo_payload_data([[4.0, -4.0]] * 5, calibrated)["ESP_01"] == [112, 68]
  assert servo_payload_data([[100.0, -100.0]] * 5, cfg)["ESP_01"] == [115, 65]


@pytest.mark.parametrize(
  "settings",
  [
    {"angle_scale": 0.0},
    {"angle_scale": math.inf},
    {"angle_scale": math.nan},
    {"burst_size": 181},
    {"burst_lead_time": 0.01},
    {"burst_frame_interval": 0.0},
    {"mode": "imu-hybrid"},
  ],
)
def test_invalid_burst_configuration_is_rejected(settings):
  with pytest.raises(ValueError):
    BurstMqttBridge(LiveSnakebotConfig(**settings))


def test_hundred_frames_upload_before_execution_and_all_acks_complete(tmp_path):
  bridge, client, now = bridge_fixture(tmp_path)
  try:
    frames = [[[5.16, -5.16] for _ in range(5)] for _ in range(100)]
    payloads = bridge.send_burst(frames)
    assert len(payloads) == client.publish.call_count == 100
    assert [p["request_id"] for p in payloads] == list(range(1, 101))
    assert [p["ts"] - payloads[0]["ts"] for p in payloads] == list(range(0, 5000, 50))
    assert payloads[0]["data"] == {
      module: [105, 75] for module in bridge.cfg.module_ids
    }
    assert frames[0][0] == [5.16, -5.16]  # Model angles remain unchanged.
    for call in client.publish.call_args_list:
      assert call.kwargs == {"qos": 0, "retain": False}
      assert len(call.args[1].encode()) + len(call.args[0].encode()) + 7 <= 256
    assert not bridge.poll()
    now[0] = 1.0
    acknowledge(bridge, payloads[0])
    assert not bridge.poll()
    now[0] = 6.0
    for payload in reversed(payloads[1:]):
      acknowledge(bridge, payload)
    assert bridge.poll() and not bridge.burst_active
    with bridge.cfg.ack_csv.open() as stream:
      assert len(list(csv.DictReader(stream))) == 500
    assert len(bridge.cfg.all_feedback_jsonl.read_text().splitlines()) == 500
  finally:
    bridge.close()


def test_early_feedback_cannot_start_overlapping_playback(tmp_path):
  bridge, _, now = bridge_fixture(tmp_path, burst_size=3)
  try:
    payloads = bridge.send_burst([[[0.0, 0.0]] * 5] * 3)
    for payload in payloads:
      acknowledge(bridge, payload)
    assert not bridge.poll()
    with pytest.raises(RuntimeError, match="previous"):
      bridge.send_burst([[[0.0, 0.0]] * 5])
    now[0] = 1.11
    assert bridge.poll()
  finally:
    bridge.close()


def test_feedback_timeout_is_after_frame_execution_and_keeps_missing_id(tmp_path):
  bridge, client, now = bridge_fixture(tmp_path, burst_size=3, feedback_timeout=0.2)
  try:
    payloads = bridge.send_burst([[[0.0, 0.0]] * 5] * 3)
    acknowledge(bridge, payloads[0], modules=bridge.cfg.module_ids[:4])
    now[0] = 0.9
    assert not bridge.poll()  # Commands are still legitimately scheduled in the future.
    now[0] = 1.21
    with pytest.raises(TimeoutError, match="ESP_05"):
      bridge.poll()
    with pytest.raises(RuntimeError):
      bridge.send_burst([[[0.0, 0.0]] * 5])
    assert client.publish.call_count == 3
  finally:
    bridge.close()


def test_invalid_frame_rejects_entire_burst_before_any_publish(tmp_path):
  bridge, client, _ = bridge_fixture(tmp_path, burst_size=3)
  try:
    frames = [[[1.0, 2.0]] * 5, [[2.0, 3.0]] * 5, [[math.nan, 4.0]] * 5]
    with pytest.raises(ValueError):
      bridge.send_burst(frames)
    client.publish.assert_not_called()
  finally:
    bridge.close()


def test_capture_buffers_frames_instead_of_publishing_continuously(monkeypatch):
  cfg = LiveSnakebotConfig(burst_size=3)
  bridge = MagicMock()
  bridge.cfg = cfg
  bridge.poll.return_value = True
  bridge.burst_active = False
  env = MagicMock()
  env.unwrapped.episode_length_buf = torch.tensor([1])

  def step(viewer):
    viewer._step_count += 1

  monkeypatch.setattr(KeyboardGoalViewer, "step_simulation", step)
  monkeypatch.setattr(
    "mjlab.scripts.snakebot_burst.module_joint_angles_deg",
    lambda _: torch.tensor([[0.0, 0.0, 1.0, -1.0]] * 5),
  )
  viewer = BurstGoalViewer(env, MagicMock(), bridge=bridge)
  viewer.step_simulation()
  viewer.step_simulation()
  bridge.send_burst.assert_not_called()
  bridge.send.assert_not_called()
  viewer.step_simulation()
  assert len(bridge.send_burst.call_args.args[0]) == 3
  assert viewer._frames == []
  viewer.pause()
  viewer.step_simulation()
  assert bridge.send_burst.call_count == 1


def test_dry_run_does_not_connect_and_waits_for_scheduled_duration(tmp_path):
  bridge, client, now = bridge_fixture(tmp_path, dry_run=True, burst_size=3)
  try:
    bridge.send_burst([[[5.0, -5.0]] * 5] * 3)
    assert not bridge.poll()
    now[0] = 1.11
    assert bridge.poll()
    client.connect.assert_not_called()
    client.publish.assert_not_called()
  finally:
    bridge.close()
