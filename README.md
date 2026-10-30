# Snakebot v2

Use `Mjlab-Locomotion-Flat-Snakebot-v2` with the evaluated [`model_1200.pt`](mjlab-fbc8368/checkpoints/snakebot_v2/model_1200.pt) goal-reaching policy.<br>
[Guide](mjlab-fbc8368/SNAKEBOT.md) · [Checkpoints](mjlab-fbc8368/checkpoints/snakebot_v2/) · [Evaluation](mjlab-fbc8368/docs/evaluations/snakebot_v2_model_1200/README.md)<br>
Live five-module WASD control: from `mjlab-fbc8368/`, run
`uv run --no-sync python -m mjlab.scripts.snakebot_live`.
[Live setup and controls](mjlab-fbc8368/SNAKEBOT.md#five-module-live-system-uploaded-esp-firmware).
For overlapping RL capture and physical playback, use
`uv run --no-sync python -m mjlab.scripts.snakebot_live_pipeline`.
[Pipelined playback](mjlab-fbc8368/SNAKEBOT.md#pipelined-rl-playback-separate-version).

Far goals: 2.81 m and 2.67 m. Click either preview for the full video.<br>
[![2.81 m far-goal rollout](mjlab-fbc8368/docs/evaluations/snakebot_v2_model_1200/far_random_01.gif)](mjlab-fbc8368/docs/evaluations/snakebot_v2_model_1200/far_random_01.mp4) [![2.67 m far-goal rollout](mjlab-fbc8368/docs/evaluations/snakebot_v2_model_1200/far_random_02.gif)](mjlab-fbc8368/docs/evaluations/snakebot_v2_model_1200/far_random_02.mp4)
