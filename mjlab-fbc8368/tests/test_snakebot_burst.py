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
from mjlab.scripts.snakebot_live_burst import BurstSnakebotConfig
from mjlab.scripts.snakebot_mqtt import servo_payload_data


def bridge_fixture(tmp_path, **settings):
  settings.setdefault("burst_size", 100)
  cfg = LiveSnakebotConfig(
    ack_csv=tmp_path / "ack.csv",
    ack_status_csv=tmp_path / "ack_status.csv",
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
  burst_cfg = BurstSnakebotConfig()
  assert burst_cfg.angle_scale == 3 and burst_cfg.burst_size == 100
  assert servo_payload_data([[5.16, -5.16]] * 5, cfg)["ESP_01"] == [105, 75]
  assert servo_payload_data([[0.0, 0.0]] * 5, cfg)["ESP_01"] == [90, 90]
  calibrated = replace(cfg, neutral_angles=(100, 80))
  assert servo_payload_data([[4.0, -4.0]] * 5, calibrated)["ESP_01"] == [112, 68]
  assert servo_payload_data([[100.0, -100.0]] * 5, cfg)["ESP_01"] == [115, 65]


def test_ignore_ack_errors_allows_successive_bursts_without_replies(tmp_path):
  bridge, client, now = bridge_fixture(tmp_path, ignore_ack_errors=True)
  try:
    bridge.send([[0.0, 0.0]] * 5)
    now[0] = 1.051
    for _ in range(2):
      payloads = bridge.send_burst([[[5.0, -5.0]] * 5] * 100)
      acknowledge(bridge, payloads[0], esp_id="OTHER")
      assert not bridge.poll()
      now[0] += bridge.cfg.burst_lead_time + 99 * bridge.cfg.burst_frame_interval
      assert bridge.poll()  # No additional five-second ACK wait between bursts.
    assert client.publish.call_count == 201
    assert bridge.latest_samples is None
    assert not bridge.cfg.ack_status_csv.exists()
  finally:
    bridge.close()


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
  bridge, client, now = bridge_fixture(
    tmp_path, burst_size=3, feedback_timeout=0.2, max_missed_acks=1
  )
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


def test_burst_missing_replies_are_counted_in_order_and_later_success_resets(tmp_path):
  bridge, _, now = bridge_fixture(tmp_path, burst_size=3)
  try:
    payloads = bridge.send_burst([[[0.0, 0.0]] * 5] * 3)
    others = [module for module in bridge.cfg.module_ids if module != "ESP_03"]
    acknowledge(bridge, payloads[2], modules=others)
    acknowledge(bridge, payloads[1])
    acknowledge(bridge, payloads[0], modules=others)
    now[0] = 1.2
    assert not bridge.poll()  # Earlier frame still has time to receive its ACK.
    assert bridge.missed_ack_streaks["ESP_03"] == 0
    now[0] = 6.2
    assert bridge.poll()
    # Miss #1, success #2, miss #3: one consecutive miss, not two.
    assert bridge.missed_ack_streaks["ESP_03"] == 1
    with bridge.cfg.ack_csv.open() as stream:
      rows = list(csv.DictReader(stream))
    assert len(rows) == 13  # No fabricated ACKs for either missing packet.
    assert len(bridge.cfg.all_feedback_jsonl.read_text().splitlines()) == 13
    with bridge.cfg.ack_status_csv.open() as stream:
      rows = list(csv.DictReader(stream))
    assert len(rows) == 15
    esp3 = [row for row in rows if row["esp_id"] == "ESP_03"]
    assert [row["consecutive_misses"] for row in esp3] == ["1", "0", "1"]
    assert bridge.poll()  # Repeated polls never count a missing frame twice.
    assert bridge.missed_ack_streaks["ESP_03"] == 1
  finally:
    bridge.close()


def test_tenth_consecutive_burst_miss_stops_even_if_later_frames_already_acked(
  tmp_path,
):
  bridge, client, now = bridge_fixture(tmp_path, burst_size=11)
  try:
    payloads = bridge.send_burst([[[0.0, 0.0]] * 5] * 11)
    others = [module for module in bridge.cfg.module_ids if module != "ESP_03"]
    acknowledge(bridge, payloads[-1])
    for payload in reversed(payloads[:-1]):
      acknowledge(bridge, payload, modules=others)
    now[0] = 7.0
    with pytest.raises(TimeoutError, match="10 consecutive.*ESP_03"):
      bridge.poll()
    assert bridge.missed_ack_streaks["ESP_03"] == 10
    with pytest.raises(RuntimeError):
      bridge.send_burst([[[0.0, 0.0]] * 5])
    assert client.publish.call_count == 11
  finally:
    bridge.close()


def test_late_burst_reply_before_timeout_does_not_count_as_a_miss(tmp_path):
  bridge, _, now = bridge_fixture(tmp_path, burst_size=2)
  try:
    payloads = bridge.send_burst([[[0.0, 0.0]] * 5] * 2)
    acknowledge(bridge, payloads[1])
    now[0] = 1.2
    assert not bridge.poll()
    now[0] = 5.9
    acknowledge(bridge, payloads[0])
    assert bridge.poll()
    assert all(count == 0 for count in bridge.missed_ack_streaks.values())
  finally:
    bridge.close()


def test_ten_isolated_burst_misses_do_not_pause_the_stream(tmp_path):
  bridge, _, now = bridge_fixture(tmp_path, burst_size=20)
  try:
    payloads = bridge.send_burst([[[0.0, 0.0]] * 5] * 20)
    others = [module for module in bridge.cfg.module_ids if module != "ESP_03"]
    for index, payload in enumerate(payloads):
      acknowledge(bridge, payload, modules=others if index % 2 == 0 else None)
    now[0] = 8.0
    assert bridge.poll()
    assert all(count == 0 for count in bridge.missed_ack_streaks.values())
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
