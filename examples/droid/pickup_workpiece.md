# `pickup_workpiece` Fine-Tuning

This recipe uses the local real-robot dataset at `data/panda_raw_episodes_grasp_workpiece_necessary_h5` to fine-tune an OpenPI DROID-style policy for a Franka `pickup_workpiece` task.

## 1. Convert raw H5 episodes to LeRobot

Run from the repo root:

```bash
cd openpi
uv run examples/droid/convert_droid_h5_to_lerobot.py \
  --raw-dir ../data/panda_raw_episodes_grasp_workpiece_necessary_h5 \
  --repo-id local/pickup_workpiece \
  --task "pick up the workpiece" \
  --overwrite
```

This writes a local LeRobot dataset under `$LEROBOT_HOME/local/pickup_workpiece`.

## 2. Fine-tune with pretraining DROID norm stats

This is the best first run when the robot/action space matches Franka+DROID conventions.

```bash
cd openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_fast_pickup_workpiece \
  --exp-name=pickup_workpiece_ft \
  --overwrite
```

You can also train the non-FAST model:

```bash
cd openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_pickup_workpiece \
  --exp-name=pickup_workpiece_ft_pi0 \
  --overwrite
```

## 3. Optional: recompute norm stats from your own dataset

If the reused DROID normalization stats underperform, compute fresh stats from the local dataset and train with the local-stats config:

```bash
cd openpi
uv run scripts/compute_norm_stats.py --config-name pi0_fast_pickup_workpiece_local_stats

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_fast_pickup_workpiece_local_stats \
  --exp-name=pickup_workpiece_ft_local_stats \
  --overwrite
```

## 4. Serve the fine-tuned checkpoint

Example for a checkpoint saved at step `20000`:

```bash
cd openpi
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi0_fast_pickup_workpiece \
  --policy.dir=checkpoints/pi0_fast_pickup_workpiece/pickup_workpiece_ft/20000
```

If you trained with `pi0_fast_pickup_workpiece_local_stats`, use that same config name when serving.

## 5. Use the fine-tuned policy with DSRL real-robot training

Once the policy server is up, point `examples/scripts/run_real.sh` at the server by filling:

- `remote_host`
- `remote_port`
- `LEFT_CAMERA_ID`
- `RIGHT_CAMERA_ID`
- `WRIST_CAMERA_ID`

Then run:

```bash
bash examples/scripts/run_real.sh
```

That gives you the same online DSRL loop as the top-level `README.md`, but now with a task-specific fine-tuned OpenPI policy as the backbone.
