"""Contract tests matching the uploaded five-ESP main.cpp firmware."""

import csv
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mjlab.scripts.snakebot_live import FirmwareMqttBridge, LiveSnakebotConfig


def make_bridge(tmp_path, **kwargs):
  cfg = LiveSnakebotConfig(
    feedback_csv=tmp_path / "head.csv",
    all_feedback_jsonl=tmp_path / "all.jsonl",
    ack_csv=tmp_path / "ack.csv",
    **kwargs,
  )
  now = [1000.0]
  client = MagicMock()
  client.subscribe.return_value = (0, 1)
  client.is_connected.return_value = True
  client.publish.return_value = SimpleNamespace(rc=0)
  bridge = FirmwareMqttBridge(
    cfg,
    client_factory=lambda: client,
    clock=lambda: now[0],
    wall_clock=lambda: now[0] + 1793310000.0,
  )

  def connect(*args):
    client.on_connect(client, None, None, 0, None)
    client.on_subscribe(client, None, 1, [], None)

  client.connect.side_effect = connect
  bridge.start()
  return bridge, client, now


def reply(bridge, command, module_id, **overrides):
  # Fields and units exactly match publishPendingFeedback() in the supplied C++.
  data = {
    "esp_id": module_id,
    "request_id": command["request_id"],
    "command_ts": command["ts"],
    "sensor_valid": True,
    "roll": -7.70,
    "pitch": 6.06,
    "gx": 0.1,
    "gy": -0.2,
    "gz": 0.3,
    **overrides,
  }
  bridge._on_feedback(
    None,
    None,
    SimpleNamespace(
      topic=f"servos/{module_id}/feedback",
      payload=json.dumps(data).encode(),
    ),
  )


def test_uploaded_firmware_contract_all_five_modules(tmp_path):
  bridge, client, now = make_bridge(tmp_path)
  try:
    client.connect.assert_called_once_with("10.220.249.223", 1883, 60)
    client.subscribe.assert_called_once_with("servos/+/feedback")
    assert bridge.cfg.gyro_units == "deg/s"
    assert bridge.cfg.mode == "simulation"
    command = bridge.send([[0.0, 0.0]] * 5)
    assert command["data"] == {f"ESP_{i:02d}": [90, 90] for i in range(1, 6)}
    args, flags = client.publish.call_args
    assert args[0] == "servos/sync_command"
    assert flags == {"qos": 0, "retain": False}
    serialized = args[1]
    parsed = json.loads(serialized)
    assert type(parsed["ts"]) is int and type(parsed["request_id"]) is int
    # Conservative packet allowance includes MQTT header, topic size, and JSON.
    assert len(serialized.encode()) + len(args[0].encode()) + 7 <= 256

    now[0] += 1.05  # ESP executes a timestamped command, then snapshots its MPU.
    for module in ("ESP_05", "ESP_03", "ESP_01", "ESP_04"):
      reply(bridge, command, module)
    assert not bridge.poll()
    reply(bridge, command, "ESP_02")
    assert bridge.poll()
    snapshots = bridge.fresh_samples()
    assert tuple(s.module_id for s in snapshots) == bridge.cfg.module_ids
    assert snapshots[0].data["roll"] == -7.70
    assert len(bridge.cfg.all_feedback_jsonl.read_text().splitlines()) == 5

    # Simulation bottom/top targets map to the same [s2, s1] servo order as Python.
    command = bridge.send([[5.16, -5.16]] * 5)
    assert all(pair == [95, 85] for pair in command["data"].values())
    assert command["request_id"] == 2
    assert command["ts"] > parsed["ts"]
  finally:
    bridge.close()
  client.disconnect.assert_called_once()
  client.loop_stop.assert_called_once()


def test_delayed_reply_from_previous_process_cannot_acknowledge_a_new_request(tmp_path):
  bridge, _, now = make_bridge(tmp_path)
  try:
    command = bridge.send([[0.0, 0.0]] * 5)
    now[0] += 1.05
    for module in bridge.cfg.module_ids:
      reply(bridge, command, module, command_ts=command["ts"] - 10000)
    assert not bridge.poll()
    assert bridge.latest_samples is None
    for module in bridge.cfg.module_ids:
      reply(bridge, command, module)
    assert bridge.poll()
  finally:
    bridge.close()


def test_misidentified_esp_and_missing_module_fail_clearly(tmp_path):
  bridge, _, _ = make_bridge(tmp_path)
  try:
    command = bridge.send([[0.0, 0.0]] * 5)
    reply(bridge, command, "ESP_03", esp_id="ESP_01")
    with pytest.raises(RuntimeError, match="unexpected esp_id"):
      bridge.poll()
  finally:
    bridge.close()
  bridge, _, now = make_bridge(tmp_path)
  try:
    command = bridge.send([[0.0, 0.0]] * 5)
    for module in bridge.cfg.module_ids[:4]:
      reply(bridge, command, module)
    now[0] += 5.1
    with pytest.raises(TimeoutError, match="ESP_05"):
      bridge.poll()
  finally:
    bridge.close()


@pytest.mark.parametrize(
  "settings",
  [
    {"max_steps": 0},
    {"module_ids": ("ESP_01", "ESP_02", "ESP_03", "ESP_04", "OTHER")},
    {"topic": "too-long/" * 40},
  ],
)
def test_incompatible_firmware_profiles_are_rejected(settings):
  with pytest.raises(ValueError):
    FirmwareMqttBridge(LiveSnakebotConfig(**settings))


def test_dry_run_never_opens_network(tmp_path):
  bridge, client, now = make_bridge(tmp_path, dry_run=True)
  try:
    bridge.send([[0.0, 0.0]] * 5)
    now[0] += 1.1
    bridge.send([[-3.2, 4.8]] * 5)
    client.connect.assert_not_called()
    client.publish.assert_not_called()
  finally:
    bridge.close()


def test_simulation_streams_without_any_connected_mpu(tmp_path):
  bridge, client, now = make_bridge(tmp_path)
  try:
    first = bridge.send([[0.0, 0.0]] * 5)
    now[0] += 1.1
    for module in bridge.cfg.module_ids[:4]:
      reply(bridge, first, module, sensor_valid=False)
    assert not bridge.poll()  # Still requires all five command acknowledgements.
    reply(bridge, first, "ESP_05", sensor_valid=False)
    assert bridge.poll()
    with bridge.cfg.ack_csv.open() as stream:
      rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert all(row["sensor_valid"] == "False" for row in rows)
    records = [
      json.loads(line)
      for line in bridge.cfg.all_feedback_jsonl.read_text().splitlines()
    ]
    assert all(row["feedback"]["sensor_valid"] is False for row in records)
    assert not bridge.cfg.feedback_csv.exists()  # No fake valid IMU CSV.
    second = bridge.send([[5.16, -5.16]] * 5)
    assert all(pair == [95, 85] for pair in second["data"].values())
    assert client.publish.call_count == 2
  finally:
    bridge.close()


def test_simulation_acknowledgements_need_no_imu_fields(tmp_path):
  bridge, _, now = make_bridge(tmp_path)
  try:
    command = bridge.send([[0.0, 0.0]] * 5)
    now[0] += 1.1
    for module in bridge.cfg.module_ids:
      data = {
        "esp_id": module,
        "request_id": command["request_id"],
        "command_ts": command["ts"],
        "sensor_valid": False,
      }
      bridge._on_feedback(
        None,
        None,
        SimpleNamespace(
          topic=f"servos/{module}/feedback",
          payload=json.dumps(data).encode(),
        ),
      )
    assert bridge.poll()
  finally:
    bridge.close()


def test_hybrid_mode_still_rejects_invalid_imu(tmp_path):
  bridge, _, _ = make_bridge(tmp_path, mode="imu-hybrid")
  try:
    command = bridge.send([[0.0, 0.0]] * 5)
    reply(bridge, command, "ESP_01", sensor_valid=False)
    with pytest.raises(RuntimeError, match="Invalid IMU feedback from ESP_01"):
      bridge.poll()
  finally:
    bridge.close()
