uv run examples/droid/convert_droid_h5_to_lerobot.py \
  --raw-dir ../data/panda_raw_episodes_grasp_workpiece_necessary_h5 \
  --repo-id local/pickup_workpiece \
  --task "pickup workpiece" \
  --image-writer-processes 4 \
  --image-writer-threads 8 \
  --prefetch-episodes 8 \
  --overwrite
