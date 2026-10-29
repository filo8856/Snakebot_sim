"""Keyboard goal routing and real-policy compatibility."""

import math
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.snakebot_keyboard import (
  DEFAULT_CHECKPOINT,
  KeyboardGoalViewer,
  head_planar_axes,
  keyboard_env_cfg,
  module_joint_angles_deg,
  reset_keyboard_goal,
  snakebot_locomotion_v2_ppo_cfg,
)
from mjlab.tasks.locomotion.config.snakebot_v2.env_cfg import _robot_com_xy
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.viewer import NativeMujocoViewer


def make_angle_reporting_env():
  names = [
    f"m{module}_Revolute-{joint}" for module in range(1, 6) for joint in (15, 16)
  ]
  # Include a passive joint and reverse storage order to verify name-based mapping.
  stored_names = ["m1_Revolute-8", *reversed(names)]
  actual = torch.deg2rad(torch.tensor([[999.0, *range(10, 0, -1)]]))
  target = torch.deg2rad(torch.tensor([[888.0, *range(-10, 0)]]))
  robot = SimpleNamespace(
    joint_names=stored_names,
    data=SimpleNamespace(joint_pos=actual, joint_pos_target=target),
  )
  return SimpleNamespace(scene={"robot": robot}, step_dt=0.05)


def test_joint_angle_reporting_maps_named_joints_and_converts_to_degrees(capsys):
  raw = make_angle_reporting_env()
  expected = torch.tensor(
    [
      [1, 2, -1, -2],
      [3, 4, -3, -4],
      [5, 6, -5, -6],
      [7, 8, -7, -8],
      [9, 10, -9, -10],
    ],
    dtype=torch.float32,
  )
  torch.testing.assert_close(module_joint_angles_deg(raw), expected)
  env = MagicMock()
  env.unwrapped = raw
  viewer = KeyboardGoalViewer(env, MagicMock())
  viewer.pause()
  viewer._safe_key_callback(ord("J"))
  assert capsys.readouterr().out == ""  # Callback must not read simulation data.
  viewer._process_actions()
  output = capsys.readouterr().out
  assert "degrees" in output
  assert "Bottom actual" in output and "Bottom target" in output
  assert len(output.strip().splitlines()) == 7  # Two header lines, five module rows.
  assert "-10.00" in output
  assert "999" not in output and "888" not in output


def test_periodic_angle_output_respects_interval_and_pause(monkeypatch, capsys):
  env = MagicMock()
  env.unwrapped = make_angle_reporting_env()
  monkeypatch.setattr(NativeMujocoViewer, "step_simulation", lambda self: None)
  viewer = KeyboardGoalViewer(env, MagicMock(), angle_print_interval=0.5)
  for _ in range(9):
    viewer.step_simulation()
  assert capsys.readouterr().out == ""
  viewer.pause()
  viewer.step_simulation()
  assert capsys.readouterr().out == ""
  viewer.resume()
  for _ in range(12):
    viewer.step_simulation()
  assert capsys.readouterr().out.count("Servo joint angles") == 2


def test_key_events_are_queued_and_goal_edits_reset_progress():
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_quat_w=torch.tensor([[2**-0.5, 0.0, 2**-0.5, 0.0]]),
      body_com_pos_w=torch.tensor([[[2.0, 3.0, 0.0]]]),
      model=SimpleNamespace(body_mass=torch.ones(1, 1)),
    ),
    indexing=SimpleNamespace(body_ids=[0]),
  )
  raw = SimpleNamespace(num_envs=1, device="cpu", scene={"robot": robot})
  reset_keyboard_goal(raw, None, 0.6)
  env = MagicMock()
  env.unwrapped = raw
  viewer = KeyboardGoalViewer(env, MagicMock())
  original = raw._loco_goal_pos.clone()

  viewer._safe_key_callback(ord("W"))
  viewer._safe_key_callback(ord("A"))
  torch.testing.assert_close(raw._loco_goal_pos, original)
  viewer._process_actions()
  torch.testing.assert_close(raw._loco_goal_pos, original + torch.tensor([[0.1, 0.1]]))
  torch.testing.assert_close(
    raw._loco_prev_dist, torch.linalg.vector_norm(torch.tensor([[0.7, 0.1]]), dim=1)
  )
  assert not viewer._show_all_envs  # A is reserved for the manual goal.

  viewer._safe_key_callback(ord("S"))
  viewer._safe_key_callback(ord("D"))
  viewer._process_actions()
  torch.testing.assert_close(raw._loco_goal_pos, original)
  viewer._safe_key_callback(265)  # Up arrow.
  viewer._process_actions()
  torch.testing.assert_close(raw._loco_goal_pos, original + torch.tensor([[0.1, 0.0]]))
  viewer._safe_key_callback(ord("T"))
  viewer._process_actions()
  torch.testing.assert_close(raw._loco_goal_pos, original)


@pytest.mark.parametrize("heading", [0.0, math.pi / 2, math.pi, -math.pi / 2])
def test_target_motion_tracks_head_heading(heading):
  # Rz(heading) * Ry(pi/2): local +Z follows the desired ground heading.
  c, s = math.cos(heading / 2) * 2**-0.5, math.sin(heading / 2) * 2**-0.5
  raw = SimpleNamespace(
    scene={
      "robot": SimpleNamespace(
        data=SimpleNamespace(root_link_quat_w=torch.tensor([[c, -s, c, s]]))
      )
    },
    _loco_goal_pos=torch.zeros(1, 2),
    _loco_prev_dist=torch.zeros(1),
  )
  robot = raw.scene["robot"]
  robot.data.body_com_pos_w = torch.zeros(1, 1, 3)
  robot.data.model = SimpleNamespace(body_mass=torch.ones(1, 1))
  robot.indexing = SimpleNamespace(body_ids=[0])
  raw.num_envs, raw.device = 1, "cpu"
  env = MagicMock()
  env.unwrapped = raw
  viewer = KeyboardGoalViewer(env, MagicMock())
  forward = torch.tensor([[math.cos(heading), math.sin(heading)]])
  left = torch.tensor([[-math.sin(heading), math.cos(heading)]])
  for key, expected in (("W", forward), ("S", -forward), ("A", left), ("D", -left)):
    raw._loco_goal_pos.zero_()
    viewer._safe_key_callback(ord(key))
    viewer._process_actions()
    torch.testing.assert_close(raw._loco_goal_pos, expected * 0.1, atol=1e-6, rtol=1e-5)
  viewer._safe_key_callback(ord("T"))
  viewer._process_actions()
  torch.testing.assert_close(raw._loco_goal_pos, forward * 0.6, atol=1e-6, rtol=1e-5)

  # Changing the heading after construction changes the next command.
  robot.data.root_link_quat_w[:] = torch.tensor([[0.5, -0.5, 0.5, 0.5]])
  before = raw._loco_goal_pos.clone()
  viewer._safe_key_callback(ord("W"))
  viewer._process_actions()
  torch.testing.assert_close(
    raw._loco_goal_pos - before, torch.tensor([[0.0, 0.1]]), atol=1e-6, rtol=1e-5
  )


@pytest.mark.parametrize("tilt", [math.pi / 4, 0.0])
def test_head_axes_stay_horizontal_and_normalized(tilt):
  raw = SimpleNamespace(
    scene={
      "robot": SimpleNamespace(
        data=SimpleNamespace(
          root_link_quat_w=torch.tensor(
            [[math.cos(tilt / 2), 0.0, math.sin(tilt / 2), 0.0]]
          )
        )
      )
    }
  )
  forward, left = head_planar_axes(raw)
  torch.testing.assert_close(forward, torch.tensor([[1.0, 0.0]]))
  torch.testing.assert_close(left, torch.tensor([[0.0, 1.0]]))


def test_manual_goal_does_not_terminate_at_arrival_or_distance():
  cfg = keyboard_env_cfg()
  assert cfg.scene.num_envs == 1
  assert "goal_reached" not in cfg.terminations
  assert "too_far" not in cfg.terminations
  assert "unstable_motion" in cfg.terminations
  assert "nan_state" in cfg.terminations
  assert not cfg.observations["actor"].enable_corruption
  assert cfg.events["sample_goal"].func is reset_keyboard_goal


@pytest.mark.slow
def test_committed_policy_follows_edited_goal_without_arrival_reset():
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
    policy = runner.get_inference_policy(device=device)
    viewer = KeyboardGoalViewer(env, policy)
    viewer._safe_key_callback(ord("W"))
    viewer._process_actions()
    goal_pos: torch.Tensor = cast(Any, raw)._loco_goal_pos
    moved_goal = goal_pos.clone()
    with torch.no_grad():
      for _ in range(5):
        obs = env.get_observations()
        assert obs["actor"].shape == (1, 126)
        actions = policy(obs)
        assert actions.shape == (1, 10)
        assert torch.isfinite(actions).all()
        env.step(actions)
      torch.testing.assert_close(goal_pos, moved_goal)
      reported = module_joint_angles_deg(raw)
      assert reported.shape == (5, 4)
      assert torch.isfinite(reported).all()
      # Commands are the scaled position targets, not raw policy outputs.
      assert reported[:, 2:].abs().max() <= math.degrees(0.09) + 1e-4

      # Arrival used to reset the robot and sample a new random target.
      goal_pos[:] = _robot_com_xy(raw.scene["robot"])
      reached_goal = goal_pos.clone()
      _, _, dones, _ = env.step(policy(env.get_observations()))
      assert not dones.any()
      torch.testing.assert_close(goal_pos, reached_goal)
  finally:
    raw.close()
