# Snakebot v2 goal-reaching in mjlab

This is the authoritative guide for the current Snakebot policy in this repository.

> [!IMPORTANT]
> Use `Mjlab-Locomotion-Flat-Snakebot-v2` for current training and playback.
> The evaluated `model_1200.pt` belongs to the `snakebot_locomotion_v2`
> experiment.

## Current task

- Task ID: `Mjlab-Locomotion-Flat-Snakebot-v2`
- Experiment directory: `snakebot_locomotion_v2/`
- Purpose: navigate to COM-relative XY goals

The v2 environment starts from `make_velocity_env_cfg()` only to reuse common
scene defaults. It replaces the robot, commands, observations, rewards, events,
and terminations with goal-reaching behavior. It does not load the velocity
policy.

## Authoritative robot and scene

The v2 task is registered in
`src/mjlab/tasks/locomotion/config/snakebot_v2/__init__.py` and constructed by
`snakebot_locomotion_v2_flat_cfg()`.

The scene is composed as follows:

1. Flat plane terrain, with no terrain generator or height scanner.
2. One Snakebot entity created by `get_snakebot_robot_cfg()`.
3. The robot loader reads:
   `src/mjlab/asset_zoo/robots/snakebot/xmls/chain_5.xml`.
4. Meshes are loaded from the adjacent `xmls/meshes/` directory.

The robot has five modules and ten position-controlled joints:
`m[1-5]_Revolute-15` and `m[1-5]_Revolute-16`.

## Setup

From outside the repository:

```bash
git clone --recurse-submodules https://github.com/vimarsh244/Snakebot_sim.git
cd Snakebot_sim/mjlab-fbc8368
uv sync
```

For an existing checkout:

```bash
git submodule update --init --recursive
cd mjlab-fbc8368
uv sync
```

The project environment is `mjlab-fbc8368/.venv`. Prefer `uv run ...` so the
correct interpreter and dependencies are selected automatically.

Requirements:

- Python 3.10–3.13.
- NVIDIA GPU and CUDA for practical training.
- Node 20.19+ for the mjswan frontend toolchain.

Confirm the registered tasks:

```bash
uv run list_envs --keyword Snakebot
```

## Current v2 environment

### Goal behavior

- Goals are sampled relative to the full-robot center of mass, not the head.
- Directions are balanced across all four XY quadrants.
- Goal-distance curriculum: `0.50`–`3.00 m`.
- Initial curriculum maximum: `0.65 m`.
- Curriculum reaches the full range after `2,000,000` environment steps.
- Goal completion/episode stopping radius: `0.13 m` from robot COM.
- Training episode limit: `75 s`.
- Control frequency: `20 Hz`.

### Simulation

| Setting | v2 value |
| --- | ---: |
| Physics timestep | `0.0025 s` |
| Control decimation | `20` |
| Control timestep | `0.05 s` |
| Solver | Newton |
| Integrator | implicitfast |
| Solver iterations | `120` |
| Line-search iterations | `40` |
| `impratio` | `12.0` |
| CCD iterations | `120` |
| `njmax` | `12000` |
| `nconmax` | `1200` |

### Policy interface

The actor consumes 126 values and produces ten joint-position actions.

| Actor observation | Size |
| --- | ---: |
| Phase clock | 2 |
| Goal vector in body frame | 2 |
| Heading to goal | 2 |
| Joint-position history, 3 frames | 30 |
| Joint-velocity history, 3 frames | 30 |
| Action history, 3 frames | 30 |
| Five module linear velocities in root frame | 15 |
| Five module angular velocities in root frame | 15 |
| **Total** | **126** |

The critic uses 182 values: the actor observations plus goal distance, module
positions/velocities, and joint efforts. Actions use a `0.10 rad` scale and PPO
clips them to `0.9`.

### Main reward weights

| Reward | Weight |
| --- | ---: |
| Progress toward goal | `33.0` |
| Root velocity toward goal | `9.0` |
| COM velocity toward goal | `4.0` |
| Distance shaping | `4.5` |
| Goal reached bonus | `66.0` |
| Alive bonus | `0.3` |
| Stagnation | `-0.5` |
| Lateral slip | `-0.06` |
| Vertical velocity | `-0.08` |
| Yaw rate | `-0.06` |
| Action smoothness | `-0.015` |
| Control cost | `-0.003` |
| Joint-position limits | `-0.002` |

Training includes friction, inertia, joint, encoder, PD-gain, effort-limit, and
gentle post-settle push randomization. Play mode disables observation corruption
and those randomization events.

## Train v2

Run commands from `mjlab-fbc8368/`.

Single GPU:

```bash
uv run train Mjlab-Locomotion-Flat-Snakebot-v2 \
  --env.scene.num-envs 2048
```

Choose a GPU or adjust parallelism:

```bash
uv run train Mjlab-Locomotion-Flat-Snakebot-v2 \
  --env.scene.num-envs 1024 \
  --gpu-ids 1

uv run train Mjlab-Locomotion-Flat-Snakebot-v2 \
  --env.scene.num-envs 4096 \
  --gpu-ids 0 1
```

Training logs go to:

```text
logs/rsl_rl/snakebot_locomotion_v2/<timestamp>/
```

Weights & Biases uses project `mjlab`. Authenticate once with `wandb login`.

Resume the latest matching local v2 checkpoint:

```bash
uv run train Mjlab-Locomotion-Flat-Snakebot-v2 --agent.resume
```

Resume a W&B checkpoint:

```bash
uv run train Mjlab-Locomotion-Flat-Snakebot-v2 \
  --agent.resume \
  --wandb-run-path ENTITY/mjlab/RUN_ID
```

## Known working checkpoints

Two curated v2 checkpoints are committed:

- `checkpoints/snakebot_v2/model_800.pt`
- `checkpoints/snakebot_v2/model_1200.pt` — used for the committed evaluation

Both come from source run `2026-03-10_19-34-58`. TensorBoard recorded mean
rewards of `328.034119` at iteration 800 and `312.155579` at
iteration 1200.

The source run's associated ONNX export is:

```text
logs/rsl_rl/snakebot_locomotion_v2/2026-03-10_19-34-58/2026-03-10_19-34-58.onnx
```

Evidence that it is v2:

- Captured agent config says `experiment_name: snakebot_locomotion_v2`.
- ONNX input is `obs[1,126]`; output is `actions[1,10]`.
- ONNX observation names match the v2 phase, goal, history, and module-velocity terms.
- The checkpoint safely loads with `iter=1200`.
- Ten committed goal rollouts reached their targets.

The run continued through iteration 1292 and was then manually interrupted.
Iteration 1200 is the last saved checkpoint. It was trained with a `0.10 m`
reach threshold; current code uses `0.13 m`, which does not change the
observation or action architecture. The checkpoint was re-evaluated successfully
with the current stopping radius.

Training-generated checkpoints, ONNX files, W&B state, and logs remain
Git-ignored by default. The two curated `.pt` copies above are committed and
available in a fresh clone. The evaluation report and videos are committed at
`docs/evaluations/snakebot_v2_model_1200/`.

## Play the v2 checkpoint

Native viewer:

```bash
uv run play Mjlab-Locomotion-Flat-Snakebot-v2 \
  --checkpoint-file checkpoints/snakebot_v2/model_1200.pt
```

Viser web viewer:

```bash
uv run play Mjlab-Locomotion-Flat-Snakebot-v2 \
  --viewer viser \
  --checkpoint-file checkpoints/snakebot_v2/model_1200.pt
```

The goal is displayed as a red debug sphere.

Record a trained rollout:

```bash
MUJOCO_GL=egl uv run play Mjlab-Locomotion-Flat-Snakebot-v2 \
  --checkpoint-file checkpoints/snakebot_v2/model_1200.pt \
  --video \
  --video-length 400
```

Videos are written beneath the selected run in `videos/play/`. Close the viewer
after the requested recording finishes.

## Keyboard-controlled target

Use the existing trained policy with an interactive target in the native
MuJoCo viewer. From `mjlab-fbc8368/` in WSL/Linux with a graphical display:

```bash
uv run python -m mjlab.scripts.snakebot_keyboard
```

This defaults to the committed `checkpoints/snakebot_v2/model_1200.pt`.
Click the viewer window to focus keyboard input.

| Key | Action |
| --- | --- |
| W / Up | Move target forward along the head's heading by 0.10 m |
| S / Down | Move target backward along the head's heading by 0.10 m |
| A / Left | Move target left relative to the head by 0.10 m |
| D / Right | Move target right relative to the head by 0.10 m |
| T | Put the target 0.60 m ahead of COM along the head's heading |
| Enter | Reset the robot and target |
| Space | Pause/resume |

Directions follow the head module's current heading, projected onto the
ground plane, and do not rotate with the camera. Each key
press moves the red target; the policy supplies the joint commands. The
target remains where you put it after arrival, and the robot does not reset
when it reaches the target or when you move it farther away. Instability
and non-finite-state recovery remain enabled; a robot reset also resets
the target.

Change the step size or use another compatible v2 checkpoint:

```bash
uv run python -m mjlab.scripts.snakebot_keyboard --goal-step 0.2 \
  --checkpoint-file checkpoints/snakebot_v2/model_800.pt
```

No retraining is required for this demo. The existing evaluation covers
fixed goals; tracking a repeatedly moving goal and holding position after
arrival have not been trained or evaluated as separate behaviors.

## mjswan browser viewer

Initialize the pinned fork first:

```bash
git submodule update --init --recursive
```

mjswan consumes ONNX, so a committed `.pt` checkpoint alone is not enough.
Use the source-run checkpoint and its neighboring local ONNX explicitly, or
export/copy an ONNX file next to a checkpoint:

```bash
uv run play Mjlab-Locomotion-Flat-Snakebot-v2 \
  --viewer mjswan \
  --checkpoint-file logs/rsl_rl/snakebot_locomotion_v2/2026-03-10_19-34-58/model_1200.pt
```

Or launch directly from ONNX:

```bash
uv run snakebot_mjswan \
  --onnx-file logs/rsl_rl/snakebot_locomotion_v2/2026-03-10_19-34-58/2026-03-10_19-34-58.onnx
```

Running `--viewer mjswan` without a checkpoint auto-selects the best local v2
ONNX using TensorBoard mean reward. It cannot auto-select anything in a fresh
clone until local v2 ONNX exports are present.

## Evaluation

The committed model-1200 evaluation contains four 0.60 m cardinal goals and six
random far goals from 1.57 m to 2.81 m. All ten reached the `0.13 m` stopping
radius within 24.5 seconds.

See:

- [Evaluation report](docs/evaluations/snakebot_v2_model_1200/README.md)
- [Embedded rollout gallery](README.md#snakebot-v2-goal-reaching)

## File map

```text
src/mjlab/asset_zoo/robots/snakebot/
├── snakebot_constants.py
└── xmls/
    ├── chain_5.xml
    └── meshes/

src/mjlab/tasks/locomotion/config/snakebot_v2/
    ├── __init__.py
    ├── env_cfg.py
    ├── goal_pose_command.py
    ├── rl_cfg.py
    └── snake_locomotion_mdp.py

checkpoints/snakebot_v2/                  # committed model 800 and model 1200
third_party/mjswan/                       # pinned browser-viewer fork
docs/evaluations/snakebot_v2_model_1200/  # committed results and recordings
```

## Troubleshooting

- If MuJoCo Warp runs out of constraint capacity, reduce
  `--env.scene.num-envs`.
- For headless video rendering, set `MUJOCO_GL=egl`.
- If mjswan dependencies or sources are missing, run
  `git submodule update --init --recursive`.
- If playback reports an observation-size mismatch, confirm the task ID is
  exactly `Mjlab-Locomotion-Flat-Snakebot-v2` and the checkpoint comes from
  `snakebot_locomotion_v2/`.
- If no local checkpoint is available, use a W&B run path or copy the ignored
  run directory into `logs/rsl_rl/snakebot_locomotion_v2/`.
