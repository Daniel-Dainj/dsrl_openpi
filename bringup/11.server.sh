# CUDA_VISIBLE_DEVICES=1 \
# XLA_PYTHON_CLIENT_PREALLOCATE=false \
# uv run scripts/serve_policy.py policy:checkpoint \
#   --policy.config="pi0_base" \
#   --policy.dir="gs://openpi-assets/checkpoints/pi0_base"

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config="pi0_pickup_workpiece_local_stats" \
  --policy.dir="checkpoints/pi0_pickup_workpiece_local_stats/pickup_workpiece_ft_pi0_local_stats/9000"
