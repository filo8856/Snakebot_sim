# Snakebot v2 goal-reaching in mjlab

This is the authoritative guide for the current Snakebot policy in this repository.

> [!IMPORTANT]
> Use `Mjlab-Locomotion-Flat-Snakebot-v2` for current training and playback.
> The evaluated `model_1200.pt` belongs to the `snakebot_locomotion_v2`
> experiment.

## Current task

For the physical five-module robot, use the
[live-system launcher](#five-module-live-system-uploaded-esp-firmware).

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
| J | Print actual and commanded bottom/top servo angles for all five modules |

Directions follow the head module's current heading, projected onto the
ground plane, and do not rotate with the camera. Each key
press moves the red target; the policy supplies the joint commands. The
target remains where you put it after arrival, and the robot does not reset
when it reaches the target or when you move it farther away. Instability
and non-finite-state recovery remain enabled; a robot reset also resets
the target.

Press **J** for a terminal snapshot of all ten servo joints in degrees,
including actual angles and the most recently applied target angles. It
also works while paused. Bottom is `mN_Revolute-15`; top is
`mN_Revolute-16`. Angles are relative to each joint's model zero, rather
than absolute module orientation or calibrated hardware servo angles.

To print the angle table continuously every 0.5 simulated seconds:

```bash
uv run python -m mjlab.scripts.snakebot_keyboard --angle-print-interval 0.5
```

Change the step size or use another compatible v2 checkpoint:

```bash
uv run python -m mjlab.scripts.snakebot_keyboard --goal-step 0.2 \
  --checkpoint-file checkpoints/snakebot_v2/model_800.pt
```

No retraining is required for this demo. The existing evaluation covers
fixed goals; tracking a repeatedly moving goal and holding position after
arrival have not been trained or evaluated as separate behaviors.

## Five-module live system (uploaded ESP firmware)

This launcher matches the supplied ESP32 `main.cpp` and the confirmed broker
at **10.220.249.223:1883**. It runs the v2 policy in MuJoCo, lets WASD move
the target relative to the head, and sends the resulting ten servo targets
to the real robot. The uploaded ESP firmware can remain installed.

From the existing working WSL environment:

```bash
cd ~/Snakebot_sim/mjlab-fbc8368
uv pip install -r requirements-hardware.txt
uv run --no-sync python -m mjlab.scripts.snakebot_live
```

Use this launcher as the active command sender. Click the MuJoCo window:

| Input | Effect |
| --- | --- |
| W / Up | Move target forward relative to the model's head |
| S / Down | Move target backward |
| A / Left | Move target left |
| D / Right | Move target right |
| Space | Pause/resume new servo commands |
| J | Print actual model angles and commanded targets in degrees |
| T | Place target ahead of the head |
| Enter | Reset the simulation |
| Close window | Exit and disconnect MQTT |

Each key press moves the target 0.10 m by default. The robot is initially
sent a neutral hold (90°, 90° with the default calibration). Each outgoing
message contains `request_id`, a future Unix-millisecond `ts`, and all five
`ESP_01`–`ESP_05` angle pairs. Simulation mode now records and uploads bursts
of **100 complete frames**, with ten servo angles in each frame. That is
1,000 individual servo-angle values per burst. All five ESPs must acknowledge
the initial neutral hold before recording starts.

During recording the model runs normally and WASD affects the captured
trajectory; the physical robot holds its previous targets. Once 100 frames
are collected, the sender uploads 100 standard firmware-compatible MQTT
messages in a short burst. Their timestamps are 50 ms apart, giving about
five seconds of physical playback at 20 Hz. This removes per-frame network
round-trip delays from the playback timing. The sender waits for playback
and matching acknowledgements before starting the next batch. Capture time
depends on simulation throughput and can exceed five wall-clock seconds.

The 200-slot ESP queue has 199 usable entries; burst size is limited to 180,
and only one burst is outstanding. Feedback deadlines are measured from each
frame's scheduled execution time, not from the upload time.

**WASD changes during playback affect the next batch. Space prevents future
bursts; it cannot cancel a burst already queued on the ESPs. Closing the viewer
also leaves already queued frames to finish.** The supplied firmware has no
queue-cancel command. Reset is unavailable during queued playback. Partial
unsent buffers are discarded on reset, simulation instability, or window close.

Defaults and tuning:

```bash
# Default: 3x hardware angle offsets, 100 frames per burst.
uv run --no-sync python -m mjlab.scripts.snakebot_live --angle-scale 3 --burst-size 100
# Smaller batches reduce recording and control latency.
uv run --no-sync python -m mjlab.scripts.snakebot_live --angle-scale 3 --burst-size 20
# Original continuous, unscaled behavior.
uv run --no-sync python -m mjlab.scripts.snakebot_live --angle-scale 1 --burst-size 1
```
Simulation mode requires no MPU connection: `sensor_valid: false` is accepted
as part of an otherwise correctly identified command acknowledgement. Every
angle and policy observation comes from the simulation. Hybrid mode continues
to require valid IMUs. Terminal output confirms when all five ESPs have acknowledged a command.
Missing feedback identifies the missing module and pauses the stream.

The first model module (head) maps to ESP_01 and the tail maps to ESP_05.
The first angle is the bottom servo / GPIO32, and the second is the top
servo / GPIO33, following the supplied Python sender's `[s2, s1]` order.
Each model joint target becomes `90 + int(target_degrees * angle_scale)`,
within the existing 65–115° range. The default **3x** gain changes the policy's
roughly +/-5.16° targets to approximately +/-15.48° (integer hardware commands
usually 75–105°). Neutral stays 90°. Scaling is applied only to hardware
commands; it does not change the trained policy or model action scale.
Servo offsets/directions can be calibrated using
the launcher options. Reorder the five `--module-ids` if the physical
head-to-tail numbering differs.

The supplied firmware publishes roll/pitch in degrees and bias-corrected
`gx`, `gy`, `gz` in **degrees/second**, so the launcher has that unit configured.
Each ESP must have its own `THIS_ESP_ID` and publish `esp_id`, `request_id`,
`command_ts`, `sensor_valid`, `roll`, `pitch`, `gx`, `gy`, and `gz` on
`servos/<ESP_ID>/feedback`. MCU feedback is captured immediately after
`servo.write()`; it acknowledges issuing the command, not measured servo
position or completion of the mechanical movement. Feedback is logged to
`logs/hardware/live_all_modules.jsonl`, preserving the true `sensor_valid` flags.
Simulation command acknowledgements are saved to `logs/hardware/live_command_ack.csv`;
hybrid head IMU readings use `logs/hardware/live_head.csv`.

The old Python sender points to 10.49.2.223 and its active feedback function
addresses only ESP_01. This launcher uses the confirmed firmware address and
addresses all five modules. Both the laptop and ESPs must share the broker
and Unix time basis used by the firmware's NTP scheduling. If all five modules
time out, check their MQTT and time-sync messages on Serial rather than assuming
the queued timestamp has already executed.

Preview without a broker connection or hardware commands:

```bash
uv run --no-sync python -m mjlab.scripts.snakebot_live --dry-run True
# Optional bounded preview: eight policy steps, then close/disconnect.
uv run --no-sync python -m mjlab.scripts.snakebot_live --dry-run True --max-steps 8
```

The default `--mode simulation` mirrors the simulation in batches and logs all physical
IMUs. To feed those IMUs into the RL actor while still using model servo and
position estimates, select the separate hybrid behavior:

```bash
uv run --no-sync python -m mjlab.scripts.snakebot_live --mode imu-hybrid
```

Hybrid mode retains continuous per-frame feedback, regardless of burst size,
because its policy needs current IMU measurements. The configurable hardware
angle scale also applies in hybrid mode. In hybrid mode, WASD uses IMU-estimated
head orientation. Begin in the model's
neutral pose for alignment. Physical position remains unmeasured; the
[hybrid limitations and data sources](#imumodel-hybrid-controller-separate-version)
still apply. Existing `snakebot_keyboard`, `snakebot_mqtt`, and
`snakebot_mpu_mqtt` entry points remain available.

## Live MQTT hardware mirror

`snakebot_mqtt` runs the same head-relative WASD simulation and sends its
commanded servo targets to the physical robot using the supplied ESP sender's
MQTT initialization, timestamped JSON messages, and request/feedback protocol.
Run from the existing working `mjlab-fbc8368/` environment:

```bash
uv pip install -r requirements-hardware.txt
uv run --no-sync python -m mjlab.scripts.snakebot_mqtt
```

The live command connects to `10.49.2.223:1883` and begins sending when the
simulation starts. Click the MuJoCo window and use WASD/arrow keys as before.
Space pauses/resumes new commands, J prints angles, and closing the window
disconnects MQTT. To inspect the exact outgoing messages without connecting
to any broker or moving hardware:

```bash
uv run --no-sync python -m mjlab.scripts.snakebot_mqtt --dry-run True
```

The default outgoing topic is `servos/sync_command`. Each payload contains
`request_id`, Unix-millisecond `ts`, and a `data` dictionary containing
`ESP_01` through `ESP_05`, mapped to simulation modules 1 through 5. Each pair
is **[bottom servo, top servo]**, matching the old sender's [s2, s1] order.
The pasted sender's active feedback function addressed only ESP_01; this
bridge sends all five pairs while still awaiting feedback from ESP_01.

Hardware angles are `90 + int(simulated_target_degrees)`, clamped to 65–115°,
following the supplied sender's neutral position, truncation, and limits.
The model's targets are already servo angles, so the old rolling-joint
geometric conversion is not repeated. Actual simulated angles are not sent.
The actor is updated at most 20 times per wall-clock second. The first command
is scheduled 1 second ahead; subsequent commands use a fresh timestamp 0.05
seconds ahead. The ESP clocks must use the same Unix time basis as the sender.

Matching `request_id` feedback is expected on `servos/ESP_01/feedback` with
`sensor_valid`, `roll`, `pitch`, `gx`, `gy`, and `gz`, as in the provided code.
Only one command is outstanding. While waiting for feedback, the simulation
does not advance its policy; rendering and WASD input remain responsive.
Feedback slower than 20 Hz therefore slows the joint sequence rather than
building a future-command queue. Valid feedback is saved to
`logs/hardware/mpu_feedback.csv`. Missing feedback after 5 seconds, invalid
sensor readings, disconnection, or an unstable simulation stops new commands
and pauses the demo. Restart it after resolving the fault.

Override the broker or reproduce the original single-module command:

```bash
uv run --no-sync python -m mjlab.scripts.snakebot_mqtt --broker 192.168.1.20
uv run --no-sync python -m mjlab.scripts.snakebot_mqtt --module-ids "('ESP_01',)"
```

`--neutral-angles "(90,90)"` and `--servo-directions "(1,1)"` accept a shared
bottom/top pair or ten values for individual servo calibration. Module IDs
can also be replaced in order using `--module-ids`. To use firmware without
request/feedback support, explicitly pass `--require-feedback False`.
This disables the feedback handshake and its sensor/timeout checks.

This mirrors simulated motion. MPU feedback is logged and used to acknowledge
commands; it does not replace the simulation's state observations. WASD still
uses the simulated head orientation. Pausing or disconnecting stops new
publications; any already scheduled command may still execute, and servo
holding behavior remains controlled by the ESP firmware.

## IMU/model hybrid controller (separate version)

The original `snakebot_mqtt` mirror remains available. The separate
`snakebot_mpu_mqtt` version feeds IMU measurements from **all five physical
modules** into the existing v2 actor while retaining simulated servo state,
as required when the servos have no angle feedback.

| Actor information | Source in this version |
| --- | --- |
| Module angular velocities | Five real module gyros, converted into the head frame |
| Head orientation for goal direction and WASD | Measured head roll/pitch and reported or gyro-estimated yaw |
| Servo angles and joint velocities, with history | Simulated model |
| Robot COM position and module linear velocities | Simulated model |
| Phase clock and previous actions | Controller/model |

This is an **experimental hybrid**, not a measured physical-position
controller. Without position tracking, the robot's distance to the target
remains a model estimate. The simulation display also remains the model state;
it does not reconstruct the physical robot's full pose from the IMUs.

Run from the working environment, choosing gyro units to match the firmware:

```bash
# If gx/gy/gz are in degrees/second:
uv run --no-sync python -m mjlab.scripts.snakebot_mpu_mqtt --gyro-units deg/s

# If gx/gy/gz are already in radians/second:
uv run --no-sync python -m mjlab.scripts.snakebot_mpu_mqtt --gyro-units rad/s

# If gx/gy/gz are raw MPU counts (example: +/-250 deg/s range):
uv run --no-sync python -m mjlab.scripts.snakebot_mpu_mqtt \
  --gyro-units raw --raw-gyro-sensitivity 131
```

No gyro unit is assumed for live operation. The raw sensitivity must match
the configured MPU full-scale range. Roll/pitch and optional yaw must be in
degrees using a consistent XYZ roll/pitch/yaw and sensor-axis convention.

Start with the physical robot matching the model's neutral pose. The script
first sends a neutral servo hold to elicit IMU replies, then aligns each
sensor orientation to its corresponding model module using the first complete
feedback batch. This initial alignment assumes consistent IMU axis/yaw
references across modules; verify the sensor mounting and firmware convention.
Calibration is local to each run. If yaw is absent, the controller integrates
gyro-derived yaw rate between feedback batches, which accumulates drift.

The firmware on each ESP must publish to its own topic:
`servos/ESP_01/feedback` through `servos/ESP_05/feedback`. Each reply must contain:

```json
{"request_id": 1, "sensor_valid": true, "roll": -7.7, "pitch": 6.06,
 "gx": 0.1, "gy": 0.2, "gz": 0.3}
```

`request_id` must match the command sent to all five modules. An optional
`yaw` field supplies heading directly instead of gyro integration. The next
policy step waits for every module from the same request; missing, invalid,
or stale data stops the stream rather than silently replacing a module with
simulated IMU data. `--max-imu-age` defaults to 1 second. All raw module
feedback is recorded to `logs/hardware/mpu_hybrid_all.jsonl`; the head summary
is also saved to `logs/hardware/mpu_hybrid_head.csv`.

WASD and arrows use the IMU-estimated head heading. T places the estimated
target ahead of that heading. Space pauses/resumes; on resume, the controller
holds the last servo targets to request fresh IMU readings. Gyro yaw integration
does not extrapolate across a long pause, so motion during a pause cannot be
recovered without reported yaw. Enter resets the model, not the physical robot.

For a network-free demonstration of the hybrid code path:

```bash
uv run --no-sync python -m mjlab.scripts.snakebot_mpu_mqtt --dry-run True
```

Dry-run mode explicitly synthesizes IMU packets from the model. It exercises
the adapter and RL inference without connecting to the broker or moving hardware.

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
