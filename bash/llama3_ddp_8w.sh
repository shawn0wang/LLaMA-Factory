export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO


export FORCE_TORCHRUN=1
export NNODES=$WORLD_SIZE
export RANK=$RANK
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WANDB_API_KEY=bd07082949adc2cf88e0397dd793961b6bd646ac

cd /mnt/data/wangxiaokun/LLaMA-Factory
llamafactory-cli train /mnt/data/wangxiaokun/LLaMA-Factory/config_files/sft/llama31-8b-code_8w.yaml



pip install pyext
cd /mnt/data/wangxiaokun/O1/code/code-eval/LiveCodeBench
export MODEL_DIR=/mnt/data/wangxiaokun/O1/model/LF/llama3-8b-opencoder
export MODEL_REPR=llama3-8b-opencoder
python -m lcb_runner.runner.main \
--model /mnt/data/wangxiaokun/O1/model/LF/llama3-8b-opencoder \
--scenario codegeneration
sleep 10
for file_path in $(find "$OUTPUT_DIR"/output/"$MODEL_REPR" -type f -name "*.json"); do
    # 检查是否找到 .json 文件
    if [[ -f "$file_path" ]]; then
        echo "Processing file: $file_path"
        # 执行 Python 命令，传入找到的 .json 文件路径
        python -m lcb_runner.runner.custom_evaluator --custom_output_file "$file_path"
    else
        echo "No .json files found in $OUTPUT_DIR or its subdirectories"
    fi
done
