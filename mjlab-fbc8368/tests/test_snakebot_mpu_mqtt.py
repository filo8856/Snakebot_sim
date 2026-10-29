"""Hybrid IMU fusion and all-module feedback, without contacting hardware."""

import json
import math
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.snakebot_keyboard import (
  DEFAULT_CHECKPOINT,
  keyboard_env_cfg,
  snakebot_locomotion_v2_ppo_cfg,
)
from mjlab.scripts.snakebot_mpu_mqtt import (
  AllModuleMpuBridge,
  HybridPolicy,
  MpuFusion,
  MpuGoalViewer,
  MpuKeyboardConfig,
  MpuSample,
)
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.lab_api.math import quat_apply

TERMS = (
  "phase_clock",
  "goal_vector",
  "heading_to_goal",
  "joint_pos_hist",
  "joint_vel_hist",
  "action_hist",
  "module_lin_vel_root",
  "module_ang_vel_root",
)
SIZES = (2, 2, 2, 30, 30, 30, 15, 15)


def samples(request_id=1, now=0.0, **values):
  data = dict(
    roll=0.0,
    pitch=0.0,
    gx=0.0,
    gy=0.0,
    gz=0.0,
    sensor_valid=True,
    request_id=request_id,
  )
  data.update(values)
  return tuple(
    MpuSample(f"ESP_{i:02d}", request_id, now, data.copy()) for i in range(1, 6)
  )


def fake_env():
  quat = torch.tensor([[2**-0.5, 0.0, 2**-0.5, 0.0]])
  robot = SimpleNamespace(
    body_names=tuple(f"m{i}_bottom-base-plate-v1" for i in range(1, 6)),
    data=SimpleNamespace(
      body_link_quat_w=quat.unsqueeze(1).repeat(1, 5, 1),
      root_link_quat_w=quat,
      body_link_ang_vel_w=torch.zeros(1, 5, 3),
      body_com_pos_w=torch.zeros(1, 5, 3),
      model=SimpleNamespace(body_mass=torch.ones(1, 5)),
    ),
    indexing=SimpleNamespace(body_ids=list(range(5))),
  )
  return SimpleNamespace(
    scene={"robot": robot},
    observation_manager=SimpleNamespace(
      active_terms={"actor": list(TERMS)},
      group_obs_term_dim={"actor": [(n,) for n in SIZES]},
    ),
    _loco_goal_pos=torch.tensor([[0.6, 0.0]]),
    _loco_prev_dist=torch.tensor([0.6]),
  )


def live_bridge(tmp_path):
  cfg = MpuKeyboardConfig(
    gyro_units="deg/s",
    initial_delay=0.0,
    feedback_csv=tmp_path / "head.csv",
    all_feedback_jsonl=tmp_path / "all.jsonl",
  )
  now = [0.0]
  client = MagicMock()
  client.subscribe.return_value = (0, 1)
  client.publish.return_value = SimpleNamespace(rc=0)
  client.is_connected.return_value = True
  bridge = AllModuleMpuBridge(cfg, client_factory=lambda: client, clock=lambda: now[0])

  def connect(*args):
    client.on_connect(client, None, None, 0, None)
    client.on_subscribe(client, None, 1, [], None)

  client.connect.side_effect = connect
  bridge.start()
  return bridge, client, now


def deliver(bridge, batch):
  for sample in batch:
    msg = SimpleNamespace(
      topic=f"servos/{sample.module_id}/feedback",
      payload=json.dumps(sample.data).encode(),
    )
    bridge._on_feedback(None, None, msg)


def test_all_five_matching_modules_are_required_and_logged(tmp_path):
  bridge, client, now = live_bridge(tmp_path)
  try:
    client.subscribe.assert_called_once_with("servos/+/feedback")
    bridge.send([[0.0, 0.0]] * 5)
    deliver(bridge, samples(request_id=99))
    deliver(bridge, samples()[:4])
    now[0] = 0.1
    assert not bridge.poll()
    assert bridge.latest_samples is None
    deliver(bridge, samples()[4:])
    assert bridge.poll()
    assert tuple(s.module_id for s in bridge.fresh_samples()) == bridge.cfg.module_ids
    records = [
      json.loads(line)
      for line in bridge.cfg.all_feedback_jsonl.read_text().splitlines()
    ]
    assert len(records) == 5
    bridge.poll()
    assert len(bridge.cfg.all_feedback_jsonl.read_text().splitlines()) == 5
    now[0] = 2.0
    with pytest.raises(RuntimeError, match="Stale"):
      bridge.fresh_samples()
  finally:
    bridge.close()


def test_missing_module_and_invalid_sensor_stop_the_stream(tmp_path):
  bridge, _, now = live_bridge(tmp_path)
  try:
    bridge.send([[0.0, 0.0]] * 5)
    deliver(bridge, samples()[:4])
    now[0] = 5.1
    with pytest.raises(TimeoutError, match="ESP_05"):
      bridge.poll()
  finally:
    bridge.close()
  bridge, _, _ = live_bridge(tmp_path)
  try:
    bridge.send([[0.0, 0.0]] * 5)
    deliver(bridge, samples(sensor_valid=False)[3:4])
    with pytest.raises(RuntimeError, match="ESP_04"):
      bridge.poll()
  finally:
    bridge.close()


def test_live_gyro_units_must_be_explicit():
  with pytest.raises(ValueError, match="gyro-units"):
    AllModuleMpuBridge(MpuKeyboardConfig())
  AllModuleMpuBridge(MpuKeyboardConfig(dry_run=True))


@pytest.mark.parametrize("units", [None, "deg/s", "rad/s", "raw"])
def test_dry_run_synthetic_imu_round_trip(units):
  env = fake_env()
  env.scene["robot"].data.body_link_ang_vel_w[:] = torch.tensor([1.0, 2.0, 3.0])
  fusion = MpuFusion(env, MpuKeyboardConfig(dry_run=True, gyro_units=units))
  fusion.update(fusion.synthetic_samples(1, 0.0))
  expected = torch.tensor([[-3.0, 2.0, 1.0]]).repeat(5, 1)
  torch.testing.assert_close(
    fusion.angular_velocity_root, expected, atol=1e-6, rtol=1e-5
  )


@pytest.mark.parametrize(
  "units,gz", [("deg/s", 90.0), ("rad/s", math.pi / 2), ("raw", 90 * 131.0)]
)
def test_mount_calibration_and_gyro_unit_conversion(units, gz):
  env = fake_env()
  cfg = MpuKeyboardConfig(gyro_units=units)
  fusion = MpuFusion(env, cfg)
  fusion.update(samples(gz=gz))
  torch.testing.assert_close(fusion.body_quat, fusion.baseline_body_quat)
  expected = torch.tensor([[-math.pi / 2, 0.0, 0.0]]).repeat(5, 1)
  torch.testing.assert_close(
    fusion.angular_velocity_root, expected, atol=1e-6, rtol=1e-5
  )


def test_tilt_and_yaw_use_real_samples_with_wall_clock_time():
  fusion = MpuFusion(fake_env(), MpuKeyboardConfig(gyro_units="deg/s"))
  fusion.update(samples())
  fusion.update(samples(request_id=2, now=0.1, gz=90.0))
  forward = quat_apply(fusion.body_quat[:1], torch.tensor([[0.0, 0.0, 1.0]]))
  torch.testing.assert_close(
    forward[:, :2],
    torch.tensor([[math.cos(math.pi / 20), math.sin(math.pi / 20)]]),
    atol=1e-6,
    rtol=1e-5,
  )
  old = fusion.body_quat.clone()
  fusion.update(samples(request_id=2, now=0.2, gz=90.0))
  torch.testing.assert_close(
    fusion.body_quat, old
  )  # Same request must not be integrated twice.
  fusion.update(samples(request_id=3, now=0.2, pitch=20.0, yaw=90.0))
  assert not torch.allclose(fusion.body_quat, old)


def test_hybrid_actor_uses_imu_but_retains_model_joint_histories(tmp_path):
  env = fake_env()
  cfg = MpuKeyboardConfig(gyro_units="deg/s")
  bridge = AllModuleMpuBridge(cfg, clock=lambda: 0.0)
  fusion = MpuFusion(env, cfg)
  policy = MagicMock(return_value=torch.zeros(1, 10))
  hybrid = HybridPolicy(env, policy, bridge, fusion)
  obs = TensorDict(
    {
      "actor": torch.arange(126, dtype=torch.float32).reshape(1, 126),
      "critic": torch.ones(1, 182),
    },
    batch_size=[1],
  )
  with pytest.raises(RuntimeError, match="No complete"):
    hybrid(obs)
  policy.assert_not_called()
  bridge.latest_samples = samples(gz=90.0)
  hybrid(obs)
  fused = policy.call_args.args[0]
  torch.testing.assert_close(fused["actor"][:, 6:96], obs["actor"][:, 6:96])
  torch.testing.assert_close(fused["critic"], obs["critic"])
  assert not torch.equal(
    fused["actor"][:, 111:], cast(torch.Tensor, obs["actor"])[:, 111:]
  )
  torch.testing.assert_close(
    obs["actor"], torch.arange(126, dtype=torch.float32).reshape(1, 126)
  )
  # A real head tilt changes the goal observation, without changing model joints.
  bridge.latest_samples = samples(request_id=2, pitch=15.0)
  hybrid(obs)
  tilted = policy.call_args.args[0]
  assert tilted["actor"][0, 2].item() == pytest.approx(
    0.6 * math.cos(math.radians(15)), abs=1e-5
  )


def test_wasd_follows_imu_head_heading_and_resume_refreshes_feedback(tmp_path):
  bridge, _, now = live_bridge(tmp_path)
  try:
    raw = fake_env()
    fusion = MpuFusion(raw, bridge.cfg)
    fusion.update(samples(yaw=0.0))
    fusion.update(samples(request_id=2, now=0.1, yaw=90.0))
    wrapped = MagicMock()
    wrapped.unwrapped = raw
    hybrid = HybridPolicy(raw, MagicMock(), bridge, fusion)
    viewer = MpuGoalViewer(wrapped, hybrid, bridge=bridge, hybrid_policy=hybrid)
    before = raw._loco_goal_pos.clone()
    viewer._safe_key_callback(ord("W"))
    viewer._process_actions()
    torch.testing.assert_close(
      raw._loco_goal_pos - before, torch.tensor([[0.0, 0.1]]), atol=1e-6, rtol=1e-5
    )
    viewer.pause()
    now[0] = 3.0
    viewer.resume()
    assert not viewer._is_paused
    assert fusion._last_times is None
    assert bridge._pending is not None  # Hold positions until fresh MPU replies.
  finally:
    bridge.close()


@pytest.mark.slow
def test_hybrid_observations_work_with_committed_rl_checkpoint():
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  raw = ManagerBasedRlEnv(keyboard_env_cfg(), device=device)
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
    cfg = MpuKeyboardConfig(gyro_units="deg/s")
    bridge = AllModuleMpuBridge(cfg, clock=lambda: 0.0)
    fusion = MpuFusion(raw, cfg)
    hybrid = HybridPolicy(
      raw, runner.get_inference_policy(device=device), bridge, fusion
    )
    bridge.latest_samples = samples(gz=5.0)
    with torch.no_grad():
      action = hybrid(env.get_observations())
      assert action.shape == (1, 10) and torch.isfinite(action).all()
      env.step(action)
      bridge.latest_samples = samples(request_id=2, now=0.05, roll=2.0, gy=3.0)
      action = hybrid(env.get_observations())
      assert torch.isfinite(action).all()
      # Servo angle histories are still maintained by the model's observation manager.
      assert cast(Any, raw)._loco_goal_pos.shape == (1, 2)
  finally:
    raw.close()
