"""Experimental IMU/model hybrid: real module IMUs, simulated joint state."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.scripts.snakebot_keyboard import _robot_com_xy, run_keyboard
from mjlab.scripts.snakebot_mqtt import (
  MqttBridge,
  MqttGoalViewer,
  MqttKeyboardConfig,
)
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_from_euler_xyz,
  quat_mul,
)
from mjlab.viewer.base import ViewerAction


@dataclass(frozen=True)
class MpuKeyboardConfig(MqttKeyboardConfig):
  feedback_topic: str = "servos/+/feedback"
  feedback_csv: Path = Path("logs/hardware/mpu_hybrid_head.csv")
  all_feedback_jsonl: Path = Path("logs/hardware/mpu_hybrid_all.jsonl")
  gyro_units: Literal["deg/s", "rad/s", "raw"] | None = None
  """Required for live operation; match gx/gy/gz units in ESP firmware."""
  raw_gyro_sensitivity: float = 131.0
  """Raw counts per degree/second, only for --gyro-units raw (MPU range specific)."""
  max_imu_age: float = 1.0
  """Maximum age of any module's IMU sample at policy inference, seconds."""


@dataclass(frozen=True)
class MpuSample:
  module_id: str
  request_id: int
  received_at: float
  data: dict


class AllModuleMpuBridge(MqttBridge):
  cfg: MpuKeyboardConfig

  def __init__(self, cfg: MpuKeyboardConfig, **kwargs):
    super().__init__(cfg, **kwargs)
    if len(cfg.module_ids) != 5 or not cfg.require_feedback:
      raise ValueError(
        "The hybrid policy requires five module IDs and feedback enabled"
      )
    if not cfg.dry_run and cfg.gyro_units is None:
      raise ValueError(
        "Set --gyro-units deg/s, rad/s, or raw to match the ESP firmware"
      )
    if cfg.feedback_topic != "servos/+/feedback":
      raise ValueError("Use servos/+/feedback to receive all module feedback topics")
    if not math.isfinite(cfg.max_imu_age) or cfg.max_imu_age <= 0:
      raise ValueError("max_imu_age must be finite and positive")
    if not math.isfinite(cfg.raw_gyro_sensitivity) or cfg.raw_gyro_sensitivity <= 0:
      raise ValueError("raw_gyro_sensitivity must be finite and positive")
    self.latest_samples: tuple[MpuSample, ...] | None = None
    self.last_targets = [[0.0, 0.0] for _ in range(5)]
    self._samples: dict[str, MpuSample] = {}
    self._logged_request = 0
    self._topics = {f"servos/{name}/feedback": name for name in cfg.module_ids}

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
    with self._lock:
      pending = self._pending
    if pending is None or data["request_id"] != pending[0]:
      return
    fields = ("roll", "pitch", "gx", "gy", "gz")
    if "yaw" in data:
      fields += ("yaw",)
    if data.get("sensor_valid") is not True or any(
      type(data.get(field)) not in (float, int) or not math.isfinite(data[field])
      for field in fields
    ):
      self._fail(f"Invalid IMU feedback from {module} for request #{pending[0]}")
      return
    sample = MpuSample(module, data["request_id"], self._clock(), data)
    with self._lock:
      if self._pending is None or self._pending[0] != sample.request_id:
        return
      self._samples[module] = sample
      if len(self._samples) == 5:
        self.latest_samples = tuple(self._samples[name] for name in self.cfg.module_ids)
        # Reuse the existing transport's timeout, rate limit and head CSV logging.
        # It becomes ready only after every selected module has replied.
        self._feedback = self.latest_samples[0].data

  def poll(self) -> bool:
    try:
      ready = super().poll()
    except TimeoutError as exc:
      with self._lock:
        missing = [name for name in self.cfg.module_ids if name not in self._samples]
      raise TimeoutError(f"{exc}; missing modules: {', '.join(missing)}") from exc
    samples = self.latest_samples
    if samples is not None and samples[0].request_id != self._logged_request:
      path = self.cfg.all_feedback_jsonl
      path.parent.mkdir(parents=True, exist_ok=True)
      with path.open("a", encoding="utf-8") as stream:
        for sample in samples:
          stream.write(
            json.dumps(
              {
                "module_id": sample.module_id,
                "received_monotonic_s": sample.received_at,
                "feedback": sample.data,
              }
            )
            + "\n"
          )
      self._logged_request = samples[0].request_id
    return ready

  def send(self, targets_deg: list[list[float]]) -> dict:
    if not self.poll():
      raise RuntimeError("Waiting for all module feedback or the next command time")
    with self._lock:
      self._samples.clear()
    payload = super().send(targets_deg)
    self.last_targets = [pair.copy() for pair in targets_deg]
    return payload

  def fresh_samples(self) -> tuple[MpuSample, ...]:
    samples = self.latest_samples
    if samples is None:
      raise RuntimeError("No complete five-module IMU snapshot is available")
    stale = [
      s.module_id
      for s in samples
      if self._clock() - s.received_at > self.cfg.max_imu_age
    ]
    if stale:
      raise RuntimeError(f"Stale IMU readings: {', '.join(stale)}")
    return samples


class MpuFusion:
  """Calibrate sensor mounts once; fuse measured tilt/gyro into policy inputs."""

  def __init__(self, env, cfg: MpuKeyboardConfig):
    self.cfg = cfg
    self.robot = env.scene["robot"]
    names = [f"m{i}_bottom-base-plate-v1" for i in range(1, 6)]
    self.body_ids = [self.robot.body_names.index(name) for name in names]
    self.baseline_body_quat = self.robot.data.body_link_quat_w[0, self.body_ids].clone()
    self.mount_quat: torch.Tensor | None = None
    self.body_quat = self.baseline_body_quat.clone()
    self.angular_velocity_root = self.body_quat.new_zeros(5, 3)
    self._yaw = self.body_quat.new_zeros(5)
    self._last_times: list[float] | None = None
    self._request_id: int | None = None

  def update(self, samples: tuple[MpuSample, ...]) -> None:
    if len(samples) != 5 or tuple(s.module_id for s in samples) != self.cfg.module_ids:
      raise ValueError("IMU snapshot must be ordered by the five configured module IDs")
    if len({s.request_id for s in samples}) != 1:
      raise ValueError("IMU samples must belong to the same command request")
    request_id = samples[0].request_id
    if request_id == self._request_id:
      return
    values = self.body_quat.new_tensor(
      [[s.data[key] for key in ("roll", "pitch", "gx", "gy", "gz")] for s in samples]
    )
    if not torch.isfinite(values).all():
      raise ValueError("Non-finite IMU input")
    roll, pitch = torch.deg2rad(values[:, 0]), torch.deg2rad(values[:, 1])
    gyro = values[:, 2:].clone()
    if self.cfg.gyro_units == "raw":
      gyro = torch.deg2rad(gyro / self.cfg.raw_gyro_sensitivity)
    elif self.cfg.gyro_units == "deg/s":
      gyro = torch.deg2rad(gyro)
    times = [s.received_at for s in samples]
    for i, sample in enumerate(samples):
      if "yaw" in sample.data:
        self._yaw[i] = math.radians(sample.data["yaw"])
      elif self._last_times is not None:
        dt = times[i] - self._last_times[i]
        if dt < 0 or dt > self.cfg.max_imu_age:
          raise ValueError("IMU sample gap is too large for gyro-based yaw integration")
        # Body gyro -> Euler yaw rate; simple gz integration is wrong when tilted.
        cosine = torch.cos(pitch[i])
        if cosine.abs() < 0.05:
          raise ValueError("IMU pitch is too close to vertical for yaw estimation")
        yaw_rate = (
          gyro[i, 1] * torch.sin(roll[i]) + gyro[i, 2] * torch.cos(roll[i])
        ) / cosine
        self._yaw[i] += yaw_rate * dt
    imu_quat = quat_from_euler_xyz(roll, pitch, self._yaw)
    if self.mount_quat is None:
      # At startup the physical robot should match the model's neutral pose.
      # q_WB = q_WI * q_IB: infer the fixed IMU-to-module mount rotations.
      self.mount_quat = quat_mul(quat_conjugate(imu_quat), self.baseline_body_quat)
    self.body_quat = quat_mul(imu_quat, self.mount_quat)
    gyro_world = quat_apply(imu_quat, gyro)
    root_quat = self.body_quat[0:1].expand(5, -1)
    self.angular_velocity_root = quat_apply_inverse(root_quat, gyro_world).clamp(
      -50, 50
    )
    self._last_times, self._request_id = times, request_id

  def synthetic_samples(self, request_id: int, now: float) -> tuple[MpuSample, ...]:
    """Network-free dry-run data, explicitly synthesized from the model."""
    body_quat = self.robot.data.body_link_quat_w[0, self.body_ids]
    imu_quat = quat_mul(body_quat, quat_conjugate(self.baseline_body_quat))
    roll, pitch, yaw = euler_xyz_from_quat(imu_quat, wrap_to_2pi=False)
    gyro_world = self.robot.data.body_link_ang_vel_w[0, self.body_ids]
    gyro = quat_apply_inverse(imu_quat, gyro_world)
    if self.cfg.gyro_units == "deg/s":
      gyro = torch.rad2deg(gyro)
    elif self.cfg.gyro_units == "raw":
      gyro = torch.rad2deg(gyro) * self.cfg.raw_gyro_sensitivity
    values = torch.cat(
      (torch.rad2deg(torch.stack((roll, pitch, yaw), dim=1)), gyro), dim=1
    )
    samples = []
    for name, row in zip(self.cfg.module_ids, values.cpu().tolist(), strict=True):
      data = dict(zip(("roll", "pitch", "yaw", "gx", "gy", "gz"), row, strict=True))
      data.update(sensor_valid=True, request_id=request_id)
      samples.append(MpuSample(name, request_id, now, data))
    return tuple(samples)


class HybridPolicy:
  """Replace only IMU-derived terms; retain model joint/translation estimates."""

  def __init__(self, env, policy, bridge: AllModuleMpuBridge, fusion: MpuFusion):
    self.env, self.policy, self.bridge, self.fusion = env, policy, bridge, fusion
    manager = env.observation_manager
    self.slices = {}
    offset = 0
    for name, shape in zip(
      manager.active_terms["actor"], manager.group_obs_term_dim["actor"], strict=True
    ):
      size = math.prod(shape)
      self.slices[name] = slice(offset, offset + size)
      offset += size
    if offset != 126:
      raise ValueError("This hybrid adapter requires the 126-input Snakebot v2 actor")

  def sync(self) -> None:
    if self.bridge.cfg.dry_run:
      samples = self.fusion.synthetic_samples(
        self.bridge._request_id, self.bridge._clock()
      )
    else:
      samples = self.bridge.fresh_samples()
    self.fusion.update(samples)

  def __call__(self, obs):
    self.sync()
    fused_obs = obs.clone()
    actor = fused_obs["actor"]
    root_quat = self.fusion.body_quat[0:1]
    robot = self.env.scene["robot"]
    # Translation and joint histories remain simulation estimates, as requested.
    delta = self.env._loco_goal_pos - _robot_com_xy(robot)
    delta3 = torch.cat((delta, torch.zeros_like(delta[:, :1])), dim=1)
    goal_body = quat_apply_inverse(root_quat, delta3)
    goal = torch.stack((goal_body[:, 2], -goal_body[:, 1]), dim=1).clamp(-3, 3)
    heading = torch.atan2(goal[:, 1], goal[:, 0])
    actor[:, self.slices["goal_vector"]] = goal
    actor[:, self.slices["heading_to_goal"]] = torch.stack(
      (heading.sin(), heading.cos()), dim=1
    )
    actor[:, self.slices["module_ang_vel_root"]] = (
      self.fusion.angular_velocity_root.flatten().unsqueeze(0)
    )
    # Express model-estimated linear velocity in the same fused head frame.
    velocity_slice = self.slices["module_lin_vel_root"]
    sim_velocity = actor[:, velocity_slice].reshape(5, 3)
    world_velocity = quat_apply(robot.data.root_link_quat_w.expand(5, -1), sim_velocity)
    actor[:, velocity_slice] = (
      quat_apply_inverse(root_quat.expand(5, -1), world_velocity)
      .clamp(-20, 20)
      .reshape(1, 15)
    )
    return self.policy(fused_obs)


class MpuGoalViewer(MqttGoalViewer):
  bridge: AllModuleMpuBridge

  def __init__(self, *args, hybrid_policy: HybridPolicy, **kwargs):
    super().__init__(*args, **kwargs)
    self.hybrid_policy = hybrid_policy

  def tick(self) -> bool:
    if not self._hardware_fault:
      try:
        ready = self.bridge.poll()
        # Preserve responsiveness while waiting for the next five-module batch.
        if (
          ready
          and not self._is_paused
          and (self.bridge.latest_samples is not None or self.bridge.cfg.dry_run)
        ):
          self.hybrid_policy.sync()
      except (RuntimeError, ValueError, TimeoutError, OSError) as exc:
        self._stop_on_fault(exc)
    return super().tick()

  def resume(self) -> None:
    if not self._hardware_fault and not self.bridge.cfg.dry_run:
      try:
        # Command-triggered firmware sends no new samples while paused. Request
        # a fresh batch by holding the last commanded servo positions.
        if self.bridge.poll():
          self.hybrid_policy.fusion._last_times = None
          self.bridge.send(self.bridge.last_targets)
      except (RuntimeError, ValueError, TimeoutError, OSError) as exc:
        self._stop_on_fault(exc)
        return
    super().resume()

  def _handle_custom_action(self, action, payload) -> bool:
    if action != ViewerAction.CUSTOM or not isinstance(payload, tuple):
      return super()._handle_custom_action(action, payload)
    name, direction = payload
    if name not in ("move_goal", "recenter_goal"):
      return super()._handle_custom_action(action, payload)
    if self.hybrid_policy.fusion.mount_quat is None:
      print("Waiting for the first complete five-module IMU calibration batch.")
      return True
    env = self.env.unwrapped
    quat = self.hybrid_policy.fusion.body_quat[0:1]
    forward = quat_apply(quat, quat.new_tensor([[0.0, 0.0, 1.0]]))[:, :2]
    if torch.linalg.vector_norm(forward) < 1e-6:
      return True
    forward = torch.nn.functional.normalize(forward, dim=-1)
    left = torch.stack((-forward[:, 1], forward[:, 0]), dim=1)
    with torch.no_grad():
      if name == "move_goal":
        env._loco_goal_pos += self.goal_step * (
          direction[0] * forward + direction[1] * left
        )
      else:
        env._loco_goal_pos[:] = (
          _robot_com_xy(env.scene["robot"]) + self.initial_distance * forward
        )
      env._loco_prev_dist[:] = torch.linalg.vector_norm(
        env._loco_goal_pos - _robot_com_xy(env.scene["robot"]), dim=1
      )
    print(f"Hybrid target (model XY): {env._loco_goal_pos[0].tolist()}")
    return True


def run_mpu_mqtt(cfg: MpuKeyboardConfig) -> None:
  bridge = AllModuleMpuBridge(cfg)

  def make_viewer(env, policy, *args):
    print(
      "EXPERIMENTAL HYBRID: real module IMUs + model servo/translation estimates.\n"
      "Start with the physical robot in the model's neutral pose for IMU calibration.\n"
      "No physical position tracking is available; goal distance remains estimated."
    )
    if cfg.dry_run:
      print(
        "DRY RUN uses synthetic model IMUs; no broker connection or hardware motion."
      )
    fusion = MpuFusion(env.unwrapped, cfg)
    hybrid = HybridPolicy(env.unwrapped, policy, bridge, fusion)
    bridge.start()
    bridge.send([[0.0, 0.0]] * 5)  # Neutral hold elicits the first IMU snapshot.
    return MpuGoalViewer(env, hybrid, *args, bridge=bridge, hybrid_policy=hybrid)

  try:
    run_keyboard(cfg, viewer_factory=make_viewer)
  finally:
    bridge.close()


def main() -> None:
  run_mpu_mqtt(tyro.cli(MpuKeyboardConfig, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
