export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

export MLP_WORKER_GPU=8
export MLP_WORKER_NUM=$WORLD_SIZE
export MLP_ROLE_INDEX=$RANK
export MLP_WORKER_0_HOST=$MASTER_ADDR
export MLP_WORKER_0_PORT=$MASTER_PORT
export FORCE_TORCHRUN=1

export WANDB_API_KEY=bd07082949adc2cf88e0397dd793961b6bd646ac

cd /mnt/data/wangxiaokun/LLaMA-Factory
llamafactory-cli train /mnt/data/wangxiaokun/LLaMA-Factory/config_files/sft/qwen25_instr.yaml
