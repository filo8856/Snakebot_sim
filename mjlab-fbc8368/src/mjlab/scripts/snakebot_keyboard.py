"""Move a Snakebot v2 goal with WASD while a trained policy follows it."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.locomotion.config.snakebot_v2.env_cfg import (
  _robot_com_xy,
  snakebot_locomotion_v2_flat_cfg,
)
from mjlab.tasks.locomotion.config.snakebot_v2.rl_cfg import (
  snakebot_locomotion_v2_ppo_cfg,
)
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import ViewerAction

DEFAULT_CHECKPOINT = (
  Path(__file__).resolve().parents[3] / "checkpoints/snakebot_v2/model_1200.pt"
)


def configure_wsl_gpu_rendering(device: str) -> bool:
  """Select WSLg's hardware renderer before the first OpenGL context is made.

  CUDA physics can work while Mesa silently renders through llvmpipe on CPU.
  Limit this workaround to WSL with its D3D12 runtime and respect explicit
  driver/software-rendering choices. No system or shell settings are changed.
  """
  if (
    not device.startswith("cuda")
    or not Path("/dev/dxg").exists()
    or not Path("/usr/lib/wsl/lib/libd3d12.so").is_file()
    or any(
      name in os.environ
      for name in (
        "GALLIUM_DRIVER",
        "LIBGL_ALWAYS_SOFTWARE",
        "MESA_LOADER_DRIVER_OVERRIDE",
      )
    )
  ):
    return False
  os.environ["GALLIUM_DRIVER"] = "d3d12"
  adapter = os.environ.setdefault(
    "MESA_D3D12_DEFAULT_ADAPTER_NAME", torch.cuda.get_device_name(device)
  )
  print(f"WSLg hardware rendering: D3D12 on {adapter}")
  return True


def snake_planar_axes(env) -> tuple[torch.Tensor, torch.Tensor]:
  """Return ground-plane forward/left from the whole snake's tail-to-head axis."""
  robot = env.scene["robot"]
  body_ids = [
    robot.body_names.index(f"m{module}_bottom-base-plate-v1") for module in range(1, 6)
  ]
  positions = robot.data.body_link_pos_w[:, body_ids, :2]
  forward_xy = positions[:, 0] - positions[:, -1]
  # If head and tail overlap in XY, fit a direction to the ordered module
  # positions. Plate orientation is never used as the forward reference.
  weights = positions.new_tensor([2.0, 1.0, 0.0, -1.0, -2.0])
  centered = positions - positions.mean(dim=1, keepdim=True)
  fitted = (centered * weights[None, :, None]).sum(dim=1)
  length = torch.linalg.vector_norm(forward_xy, dim=-1, keepdim=True)
  forward_xy = torch.where(length > 1e-6, forward_xy, fitted)
  length = torch.linalg.vector_norm(forward_xy, dim=-1, keepdim=True)
  # A collapsed/vertical snake has no horizontal body axis. Use world +X
  # until its module positions define one, keeping both axes finite and unit.
  fallback = forward_xy.new_tensor([1.0, 0.0]).expand_as(forward_xy)
  forward_xy = torch.where(length > 1e-6, forward_xy / length.clamp_min(1e-6), fallback)
  left_xy = torch.stack((-forward_xy[:, 1], forward_xy[:, 0]), dim=-1)
  return forward_xy, left_xy


def reset_keyboard_goal(env, env_ids, distance: float) -> None:
  """Place the goal ahead of the COM along the whole snake's horizontal axis."""
  # Reset events run after writing joint/root positions and before the normal
  # forward pass. Refresh body positions before deriving the new body axis.
  env.sim.forward()
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  com_xy = _robot_com_xy(env.scene["robot"])
  if not hasattr(env, "_loco_goal_pos"):
    env._loco_goal_pos = com_xy.clone()
    env._loco_prev_dist = torch.zeros(env.num_envs, device=env.device)
  forward_xy, _ = snake_planar_axes(env)
  env._loco_goal_pos[env_ids] = com_xy[env_ids] + distance * forward_xy[env_ids]
  env._loco_prev_dist[env_ids] = distance


def keyboard_env_cfg(initial_distance: float = 0.6) -> ManagerBasedRlEnvCfg:
  cfg = snakebot_locomotion_v2_flat_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.events["sample_goal"].func = reset_keyboard_goal
  cfg.events["sample_goal"].params = {"distance": initial_distance}
  # A manual goal persists after arrival and can be moved beyond the original
  # training radius. Keep instability and non-finite-state recovery enabled.
  cfg.terminations.pop("goal_reached")
  cfg.terminations.pop("too_far")
  return cfg


def module_joint_angles_deg(env) -> torch.Tensor:
  """Return rows for modules 1-5: actual bottom/top, target bottom/top."""
  robot = env.scene["robot"]
  names = [
    f"m{module}_Revolute-{joint}" for module in range(1, 6) for joint in (15, 16)
  ]
  joint_ids = [robot.joint_names.index(name) for name in names]
  actual = robot.data.joint_pos[0, joint_ids].reshape(5, 2)
  target = robot.data.joint_pos_target[0, joint_ids].reshape(5, 2)
  return torch.rad2deg(torch.cat((actual, target), dim=1)).detach()


def print_module_joint_angles(env) -> None:
  print("\nServo joint angles (degrees, relative to joint zero)")
  print("Module | Bottom actual | Top actual | Bottom target | Top target")
  for module, angles in enumerate(module_joint_angles_deg(env).cpu().tolist(), start=1):
    bottom, top, bottom_target, top_target = angles
    print(
      f"{module:6d} | {bottom:13.2f} | {top:10.2f} | "
      f"{bottom_target:13.2f} | {top_target:10.2f}"
    )


class KeyboardGoalViewer(NativeMujocoViewer):
  """Queue keyboard input on the viewer thread; move goals on the main thread."""

  # GLFW uses uppercase letter codes. Arrow keys are convenient aliases.
  DIRECTIONS = {
    ord("W"): (1.0, 0.0),
    ord("S"): (-1.0, 0.0),
    ord("A"): (0.0, 1.0),
    ord("D"): (0.0, -1.0),
    265: (1.0, 0.0),
    264: (-1.0, 0.0),
    263: (0.0, 1.0),
    262: (0.0, -1.0),
  }

  def __init__(
    self,
    env,
    policy,
    goal_step: float = 0.1,
    initial_distance=0.6,
    angle_print_interval: float = 0.0,
  ):
    super().__init__(env, policy)
    self.goal_step = goal_step
    self.initial_distance = initial_distance
    self.angle_print_interval = angle_print_interval
    self._angle_elapsed = 0.0

  def _safe_key_callback(self, key: int) -> None:
    if key in self.DIRECTIONS:
      self.request_action("MOVE_GOAL", ("move_goal", self.DIRECTIONS[key]))
    elif key == ord("T"):
      self.request_action("RECENTER_GOAL", ("recenter_goal", None))
    elif key == ord("J"):
      self.request_action("PRINT_JOINT_ANGLES", ("print_joint_angles", None))
    else:
      super()._safe_key_callback(key)

  def _handle_custom_action(self, action: ViewerAction, payload: Any) -> bool:
    if action != ViewerAction.CUSTOM or not isinstance(payload, tuple):
      return super()._handle_custom_action(action, payload)
    name, direction = payload
    env = self.env.unwrapped
    if name == "print_joint_angles":
      print_module_joint_angles(env)
      return True
    elif name == "move_goal":
      with torch.no_grad():
        forward_xy, left_xy = snake_planar_axes(env)
        delta = self.goal_step * (direction[0] * forward_xy + direction[1] * left_xy)
        env._loco_goal_pos += delta
        # Goal edits are not physical progress and must not spike the reward.
        env._loco_prev_dist[:] = torch.linalg.vector_norm(
          env._loco_goal_pos - _robot_com_xy(env.scene["robot"]), dim=1
        )
    elif name == "recenter_goal":
      with torch.no_grad():
        reset_keyboard_goal(env, None, self.initial_distance)
    else:
      return super()._handle_custom_action(action, payload)
    x, y = env._loco_goal_pos[0].tolist()
    print(f"Target: X={x:.2f} m, Y={y:.2f} m")
    return True

  def step_simulation(self) -> None:
    if self._is_paused:
      return
    super().step_simulation()
    if self.angle_print_interval > 0:
      self._angle_elapsed += self.env.unwrapped.step_dt
      if self._angle_elapsed + 1e-9 >= self.angle_print_interval:
        print_module_joint_angles(self.env.unwrapped)
        periods = math.floor((self._angle_elapsed + 1e-9) / self.angle_print_interval)
        self._angle_elapsed = max(
          0.0, self._angle_elapsed - periods * self.angle_print_interval
        )

  def reset_environment(self) -> None:
    super().reset_environment()
    self._angle_elapsed = 0.0


@dataclass(frozen=True)
class KeyboardConfig:
  checkpoint_file: Path = DEFAULT_CHECKPOINT
  """Snakebot v2 checkpoint; defaults to the committed model_1200.pt."""
  goal_step: float = 0.1
  """Target movement in meters per key press."""
  initial_distance: float = 0.6
  """Initial/recentered goal distance ahead of COM along the whole-body axis."""
  angle_print_interval: float = 0.0
  """Print all module angles every N simulated seconds; zero disables streaming."""
  device: str | None = None
  gpu_rendering: bool = True
  """Select hardware OpenGL rendering on WSLg when CUDA is used."""


def run_keyboard(
  cfg: KeyboardConfig,
  viewer_factory: Callable[..., KeyboardGoalViewer] = KeyboardGoalViewer,
  *,
  num_steps: int | None = None,
) -> None:
  for name, value in (
    ("goal_step", cfg.goal_step),
    ("initial_distance", cfg.initial_distance),
  ):
    if not math.isfinite(value) or value <= 0:
      raise ValueError(f"{name} must be finite and greater than zero")
  if not cfg.checkpoint_file.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_file}")
  if not math.isfinite(cfg.angle_print_interval) or cfg.angle_print_interval < 0:
    raise ValueError("angle_print_interval must be finite and non-negative")

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  if cfg.gpu_rendering:
    configure_wsl_gpu_rendering(device)
  agent_cfg = snakebot_locomotion_v2_ppo_cfg()
  raw_env = ManagerBasedRlEnv(keyboard_env_cfg(cfg.initial_distance), device=device)
  try:
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = VelocityOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(
      str(cfg.checkpoint_file),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    print(
      "Click the MuJoCo window to focus keyboard input.\n"
      f"W/S: target forward/back | A/D: target left/right | step: {cfg.goal_step:g} m\n"
      "Arrow keys: same directions | T: recenter target  | Enter: reset robot\n"
      "J: print actual and target joint angles for all five modules (degrees)\n"
      "Space: pause/resume | Close the window to quit.\n"
      "Directions follow the whole snake's tail-to-head axis on the ground."
    )
    viewer_factory(
      env, policy, cfg.goal_step, cfg.initial_distance, cfg.angle_print_interval
    ).run(num_steps=num_steps)
  finally:
    raw_env.close()


def main() -> None:
  run_keyboard(tyro.cli(KeyboardConfig, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
