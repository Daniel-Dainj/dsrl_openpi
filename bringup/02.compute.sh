CUDA_VISIBLE_DEVICES=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/compute_norm_stats.py \
  --config-name pi0_pickup_workpiece_local_stats
