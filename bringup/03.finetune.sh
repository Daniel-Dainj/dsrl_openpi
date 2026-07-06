CUDA_VISIBLE_DEVICES=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_pickup_workpiece_local_stats \
  --exp-name=pickup_workpiece_ft_pi0_local_stats \
  --overwrite
