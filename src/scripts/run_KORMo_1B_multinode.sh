#!/bin/bash

#SBATCH --job-name=KORMo-1.3B
#SBATCH -D . 
#SBATCH --output=_slog/%j-%x.out
#SBATCH --error=_slog/%j-%x.error
#SBATCH --nodelist=compute-st-kait-gpu-[1-16]
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-gpu=24

######################
### Set environment ###
######################
export WANDB_API_KEY=1ae2094f132efb6dcc2baa7ebe367afc8aaed912
export WANDB_PROJECT='KORMo-1.3B'

export GPUS_PER_NODE=8
export NODE_RANK=$SLURM_NODEID
export BASE_DIR="${BASE_DIR:-/fsx/KORMo}"
source /fsx/KORMo/.venv_mj_kormo_250624/bin/activate

head_node=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
head_node_ip=$head_node
echo "== Head node: ${head_node} (${head_node_ip}) =="

export LAUNCHER="accelerate launch \
  --config_file ${BASE_DIR:-/fsx/KORMo}/src/scripts/configs/accelerate_multinode.yaml \
  --num_processes 128 \
  --num_machines 16 \
  --machine_rank $NODE_RANK \
  --rdzv_backend c10d \
  --main_process_ip $head_node_ip \
  --main_process_port 29500"

export SCRIPT="${BASE_DIR}/src/kormo/train/train_kormo_1B.py"
export SCRIPT_ARGS="--config ${BASE_DIR}/src/scripts/configs/KORMo_1B.yaml"

srun $LAUNCHER $SCRIPT $SCRIPT_ARGS
