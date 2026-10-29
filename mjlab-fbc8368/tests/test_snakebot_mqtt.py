"""Hardware protocol tests use a fake MQTT client and never contact a robot."""

import csv
import json
import math
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.snakebot_keyboard import (
  DEFAULT_CHECKPOINT,
  keyboard_env_cfg,
  module_joint_angles_deg,
  snakebot_locomotion_v2_ppo_cfg,
)
from mjlab.scripts.snakebot_mqtt import (
  MqttBridge,
  MqttGoalViewer,
  MqttKeyboardConfig,
  servo_payload_data,
)
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


def connected_bridge(tmp_path, **overrides):
  cfg = MqttKeyboardConfig(feedback_csv=tmp_path / "feedback.csv", **overrides)
  now = [10.0]
  client = MagicMock()
  client.subscribe.return_value = (0, 1)
  client.publish.return_value = SimpleNamespace(rc=0)
  client.is_connected.return_value = True
  bridge = MqttBridge(
    cfg,
    client_factory=lambda: client,
    clock=lambda: now[0],
    wall_clock=lambda: 1000.0 + now[0] - 10.0,
  )

  def connect(*args):
    client.on_connect(client, None, None, 0, None)
    client.on_subscribe(client, None, 1, [SimpleNamespace(is_failure=False)], None)

  client.connect.side_effect = connect
  bridge.start()
  return bridge, client, now


def feedback_message(request_id=1, **overrides):
  data = {
    "request_id": request_id,
    "sensor_valid": True,
    "command_ts": 1001000,
    "roll": 1.0,
    "pitch": -2.0,
    "gx": 0.1,
    "gy": 0.2,
    "gz": 0.3,
    **overrides,
  }
  return SimpleNamespace(payload=json.dumps(data).encode())


def test_payload_maps_all_modules_to_sender_servo_convention():
  targets = [[5.9, -3.9], [-2.2, 1.2], [0.0, 0.0], [50.0, -50.0], [-5.16, 5.16]]
  data = servo_payload_data(targets, MqttKeyboardConfig())
  assert data == {
    "ESP_01": [95, 87],
    "ESP_02": [88, 91],
    "ESP_03": [90, 90],
    "ESP_04": [115, 65],
    "ESP_05": [85, 95],
  }
  calibrated = replace(
    MqttKeyboardConfig(),
    module_ids=("HEAD", "TAIL"),
    neutral_angles=(91, 92, 93, 94),
    servo_directions=(-1, 1, 1, -1),
  )
  assert servo_payload_data(targets, calibrated) == {"HEAD": [86, 89], "TAIL": [91, 93]}


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nonfinite_commands_are_rejected_before_publish(tmp_path, bad_value):
  bridge, client, _ = connected_bridge(tmp_path)
  try:
    with pytest.raises(ValueError, match="Invalid"):
      bridge.send([[bad_value, 0.0]] * 5)
    client.publish.assert_not_called()
  finally:
    bridge.close()


def test_request_feedback_handshake_and_csv(tmp_path):
  bridge, client, now = connected_bridge(tmp_path)
  try:
    client.message_callback_add.assert_called_once_with(
      "servos/ESP_01/feedback", bridge._on_feedback
    )
    payload = bridge.send([[1.2, -2.3]] * 5)
    assert payload["request_id"] == 1 and payload["ts"] == 1001000
    assert len(payload["data"]) == 5
    args, kwargs = client.publish.call_args
    assert args[0] == "servos/sync_command"
    assert json.loads(args[1]) == payload
    assert kwargs == {"qos": 0, "retain": False}
    now[0] += 1.0
    bridge._on_feedback(client, None, feedback_message(request_id=99))
    assert not bridge.poll()
    bridge._on_feedback(client, None, SimpleNamespace(payload=b"not-json"))
    assert not bridge.poll()
    bridge._on_feedback(client, None, feedback_message())
    assert bridge.poll()
    with bridge.cfg.feedback_csv.open() as stream:
      rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["request_id"] == "1" and rows[0]["pitch"] == "-2.0"
    next_payload = bridge.send([[0.0, 0.0]] * 5)
    assert next_payload["request_id"] == 2
    assert (
      next_payload["ts"] == 1001050
    )  # Fresh wall-clock scheduling, not a stale queue.
  finally:
    bridge.close()
  client.disconnect.assert_called_once()
  client.loop_stop.assert_called_once()


@pytest.mark.parametrize(
  "fault", ["timeout", "invalid_sensor", "disconnect", "publish"]
)
def test_hardware_faults_stop_further_commands(tmp_path, fault):
  bridge, client, now = connected_bridge(tmp_path)
  try:
    if fault == "publish":
      client.publish.return_value = SimpleNamespace(rc=4)
      with pytest.raises(RuntimeError, match="Failed to publish"):
        bridge.send([[0.0, 0.0]] * 5)
    else:
      bridge.send([[0.0, 0.0]] * 5)
      if fault == "timeout":
        now[0] += 5.1
      elif fault == "invalid_sensor":
        bridge._on_feedback(client, None, feedback_message(sensor_valid=False))
      else:
        bridge._on_disconnect(client, None, None, 1, None)
      with pytest.raises((RuntimeError, TimeoutError)):
        bridge.poll()
    with pytest.raises(RuntimeError):
      bridge.send([[0.0, 0.0]] * 5)
    assert client.publish.call_count == 1
  finally:
    bridge.close()


def test_dry_run_has_no_network_and_obeys_wall_clock_spacing(tmp_path):
  factory = MagicMock()
  now = [0.0]
  bridge = MqttBridge(
    MqttKeyboardConfig(
      dry_run=True, initial_delay=0.0, feedback_csv=tmp_path / "unused.csv"
    ),
    client_factory=factory,
    clock=lambda: now[0],
    wall_clock=lambda: 100.0,
  )
  bridge.start()
  bridge.send([[0.0, 0.0]] * 5)
  assert not bridge.poll()
  now[0] = 0.05
  assert bridge.poll()
  bridge.send([[1.0, 2.0]] * 5)
  bridge.close()
  factory.assert_not_called()
  assert not bridge.cfg.feedback_csv.exists()


def test_explicit_no_feedback_mode_does_not_wait_or_subscribe(tmp_path):
  bridge, client, now = connected_bridge(
    tmp_path, require_feedback=False, module_ids=("ESP_01",), initial_delay=0.0
  )
  try:
    payload = bridge.send([[3.0, -3.0]] * 5)
    assert payload["data"] == {"ESP_01": [93, 87]}
    client.subscribe.assert_not_called()
    now[0] += 0.05
    assert bridge.poll()
    bridge.send([[0.0, 0.0]] * 5)
    assert client.publish.call_count == 2
  finally:
    bridge.close()


@pytest.mark.parametrize(
  "overrides",
  [
    {"module_ids": ("ESP_01", "ESP_01")},
    {"servo_directions": (1, 0)},
    {"neutral_angles": (0, 90)},
    {"publish_interval": 0.0},
    {"command_delay": -1.0},
  ],
)
def test_invalid_hardware_configuration_is_rejected(overrides):
  with pytest.raises(ValueError):
    MqttBridge(MqttKeyboardConfig(**overrides))


def test_early_feedback_does_not_queue_past_initial_execution_time(tmp_path):
  bridge, client, now = connected_bridge(tmp_path)
  try:
    first = bridge.send([[0.0, 0.0]] * 5)
    bridge._on_feedback(client, None, feedback_message())
    now[0] += 0.1
    assert not bridge.poll()
    now[0] = 11.0
    assert bridge.poll()
    second = bridge.send([[0.0, 0.0]] * 5)
    assert second["ts"] > first["ts"]
  finally:
    bridge.close()


def test_viewer_remains_responsive_while_waiting_and_pauses_on_fault(tmp_path):
  bridge, client, now = connected_bridge(tmp_path)
  try:
    bridge.send([[0.0, 0.0]] * 5)
    env = MagicMock()
    viewer = MqttGoalViewer(env, MagicMock(), bridge=bridge)
    viewer.step_simulation()
    env.step.assert_not_called()
    now[0] += 6.0
    viewer.step_simulation()
    assert viewer._is_paused and viewer._hardware_fault
    viewer.resume()
    assert viewer._is_paused
    assert client.publish.call_count == 1
  finally:
    bridge.close()


@pytest.mark.slow
def test_real_policy_streams_commanded_targets_without_hardware(monkeypatch):
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  raw = ManagerBasedRlEnv(keyboard_env_cfg(), device=device)
  now = [10.0]
  bridge = MqttBridge(MqttKeyboardConfig(dry_run=True), clock=lambda: now[0])
  try:
    agent_cfg = snakebot_locomotion_v2_ppo_cfg()
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = VelocityOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(
      str(DEFAULT_CHECKPOINT),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    send = MagicMock(wraps=bridge.send)
    monkeypatch.setattr(bridge, "send", send)
    viewer = MqttGoalViewer(env, policy, bridge=bridge)
    viewer._safe_key_callback(ord("W"))
    viewer._process_actions()
    viewer.step_simulation()
    assert not viewer._hardware_fault
    assert send.call_count == 1
    targets = send.call_args.args[0]
    torch.testing.assert_close(
      torch.tensor(targets), module_joint_angles_deg(raw)[:, 2:].cpu()
    )
    viewer.pause()
    now[0] += 1.0
    viewer.step_simulation()
    assert send.call_count == 1
  finally:
    bridge.close()
    raw.close()
