export WANDB_API_KEY=1ae2094f132efb6dcc2baa7ebe367afc8aaed912
export WANDB_PROJECT='KORMo-1.3B'
export BASE_DIR="${BASE_DIR:-/home/work/mlp/KORMo}"

accelerate launch \
  --config_file ${BASE_DIR}/src/scripts/configs/accelerate_singlenode.yaml \
  --num_processes 8 \
  --num_machines 1 \
  ${BASE_DIR}/src/kormo/train/train_kormo_1B.py --config ${BASE_DIR}/src/scripts/configs/KORMo_1B.yaml
