"""Right-stick mapping and stale/disconnected input; no controller or robot needed."""

import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from mjlab.scripts.snakebot_gamepad import RightStickReader, right_stick_direction
from mjlab.scripts.snakebot_live_pipeline import (
  PipelineGoalViewer,
  PipelineSnakebotConfig,
)


@pytest.mark.parametrize(
  ("rx", "ry", "expected"),
  [
    (0, 32767, (1.0, 0.0)),
    (0, -32768, (-1.0, 0.0)),
    (-32768, 0, (0.0, 1.0)),
    (32767, 0, (0.0, -1.0)),
    (0, 0, (0.0, 0.0)),
    (4000, 4000, (0.0, 0.0)),
  ],
)
def test_right_stick_matches_wasd_and_ignores_deadzone(rx, ry, expected):
  assert right_stick_direction(rx, ry, 0.25) == pytest.approx(expected)


def test_stick_is_proportional_and_diagonal_speed_is_bounded():
  forward, left = right_stick_direction(0, 16384, 0.25)
  assert forward == pytest.approx((16384 / 32767 - 0.25) / 0.75)
  assert left == 0
  forward, left = right_stick_direction(32767, 32767, 0.25)
  assert forward > 0 and left < 0
  assert math.hypot(forward, left) == pytest.approx(1.0)


def test_reader_ignores_stale_disconnected_and_invalid_input(capsys):
  now = [1.0]
  reader = RightStickReader(clock=lambda: now[0])
  packet = json.dumps({"connected": True, "index": 0, "rx": 100, "ry": 200})
  for _ in range(10):
    reader.ingest(packet)
  assert reader.read() == (100, 200)
  assert capsys.readouterr().out.count("Gamepad connected:") == 1
  now[0] += 0.251
  assert reader.read() is None
  reader.ingest('{"connected": true, "rx": 99999, "ry": 0}')
  reader.ingest("not JSON")
  assert reader.read() is None
  reader.ingest(packet)
  assert reader.read() == (100, 200)
  reader.ingest('{"connected": false}')
  assert reader.read() is None
  reader.ingest(packet)
  assert reader.read() == (
    100,
    200,
  )  # Hot reconnect does not need a simulation restart.


def test_right_stick_moves_in_whole_body_axes_with_wall_time_and_no_large_jump(
  monkeypatch,
):
  positions = torch.tensor(
    [[[0, 2, 0], [0, 1, 0], [0, 0, 0], [0, -1, 0], [0, -2, 0]]], dtype=torch.float32
  )
  robot = SimpleNamespace(
    body_names=[f"m{i}_bottom-base-plate-v1" for i in range(1, 6)],
    indexing=SimpleNamespace(body_ids=list(range(5))),
    data=SimpleNamespace(
      body_link_pos_w=positions,
      body_com_pos_w=positions,
      model=SimpleNamespace(body_mass=torch.ones(1, 5)),
    ),
  )
  raw = SimpleNamespace(
    scene={"robot": robot},
    _loco_goal_pos=torch.zeros(1, 2),
    _loco_prev_dist=torch.zeros(1),
  )
  env = MagicMock()
  env.unwrapped = raw
  bridge = MagicMock()
  bridge.cfg = PipelineSnakebotConfig()
  viewer = PipelineGoalViewer(env, MagicMock(), bridge=bridge)
  reader = MagicMock()
  viewer._gamepad = reader
  now = [1.0]
  monkeypatch.setattr(
    "mjlab.scripts.snakebot_live_pipeline.time.monotonic", lambda: now[0]
  )
  viewer._last_gamepad_tick = 0.95
  reader.read.return_value = (0, 32767)
  viewer._update_gamepad()
  torch.testing.assert_close(raw._loco_goal_pos, torch.tensor([[0.0, 0.02]]))
  now[0] += 0.05
  reader.read.return_value = (32767, 0)
  viewer._update_gamepad()
  torch.testing.assert_close(raw._loco_goal_pos, torch.tensor([[0.02, 0.02]]))
  torch.testing.assert_close(raw._loco_prev_dist, raw._loco_goal_pos.norm(dim=1))
  now[0] += 10.0  # A delayed GUI tick must not integrate ten seconds of input.
  viewer._update_gamepad()
  torch.testing.assert_close(raw._loco_goal_pos, torch.tensor([[0.06, 0.02]]))
  reader.read.return_value = None
  now[0] += 0.05
  viewer._update_gamepad()
  torch.testing.assert_close(raw._loco_goal_pos, torch.tensor([[0.06, 0.02]]))
  reader.read.return_value = (0, 32767)
  viewer._hardware_fault = True
  now[0] += 0.05
  viewer._update_gamepad()
  torch.testing.assert_close(raw._loco_goal_pos, torch.tensor([[0.06, 0.02]]))
