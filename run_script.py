import sys
import subprocess
import argparse
import utils

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='roberta-base')
parser.add_argument('--ctr_model', default='DCNv2')
parser.add_argument('--fusion', default='add')
parser.add_argument('--num_prompt', default=10)
parser.add_argument('--num_gpu')
parser.add_argument('--total_bs')
parser.add_argument('--mode')
parser.add_argument('--seed', default=42)
args = parser.parse_args()

# Target py file
TARGET_PY_FILE = 'run.py'

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID = 15637

# The dataset name
DATASET = 'ml-1m'

# The model to be used, as well as the possible fusion with CTR models
MODEL = args.model
CTR_MODEL_NAME = args.ctr_model
MODEL_FUSION = args.fusion
CTR_MODEL_PARAM = utils.get_ctr_model_param(CTR_MODEL_NAME)
NUM_PROMPT = args.num_prompt

# Hyper-param grid search
NUM_GPU = int(args.num_gpu)
BATCH_SIZE_PER_DEVICE = 256
TOTAL_BATCH_SIZE = int(args.total_bs)
BATCH_SIZE_PER_DEVICE = min(BATCH_SIZE_PER_DEVICE, TOTAL_BATCH_SIZE)
ACCUMULATION = TOTAL_BATCH_SIZE // NUM_GPU // BATCH_SIZE_PER_DEVICE
assert ACCUMULATION > 0
if NUM_GPU > 1:
    PREFIX = f'python -m torch.distributed.launch --nproc_per_node {NUM_GPU} --master_port {PORT_ID} {TARGET_PY_FILE}'.split(' ')
else:
    PREFIX = ['python', TARGET_PY_FILE]
for LR_SCHED in ['constant', 'linear']:
    for LR in ['1e-3']:
        for EPOCH in [10, 20]:
            for WARMUP in [0]:
                FILE_PATH = f'{LR_SCHED}-BS{TOTAL_BATCH_SIZE}-LR{LR}-EPOCH{EPOCH}-WARM{WARMUP}-{args.seed}'
                subprocess.run(PREFIX + [
                    f'--model_name_or_path={MODEL}',
                    f'--train_file=./data/{DATASET}/proc_data/train.csv',
                    f'--test_file=./data/{DATASET}/proc_data/test.csv',
                    f'--meta_data_dir=./data/{DATASET}/proc_data/{DATASET}-meta.json',
                    f'--h5_data_dir=./data/{DATASET}/proc_data/{DATASET}.h5',
                    f'--output_dir=result/{DATASET}/{MODEL}/{CTR_MODEL_NAME}/{MODEL_FUSION}/{FILE_PATH}',
                    f'--logging_dir=runs/{DATASET}/{MODEL}/{CTR_MODEL_NAME}/{MODEL_FUSION}/{FILE_PATH}',
                    f'--report_to=tensorboard',
                    f'--remove_unused_columns=False',
                    f'--shuffle_fields=False',
                    f'--warmup_ratio={WARMUP}',
                    f'--num_train_epochs={EPOCH}',
                    f'--lr_scheduler_type={LR_SCHED}',
                    f'--per_device_train_batch_size={BATCH_SIZE_PER_DEVICE}',
                    f'--per_device_eval_batch_size={BATCH_SIZE_PER_DEVICE}',
                    f'--gradient_accumulation_steps={ACCUMULATION}',
                    f'--learning_rate={LR}',
                    f'--ctr_learning_rate=1e-3',
                    f'--weight_decay=0',
                    f'--max_seq_length=512',
                    f'--evaluation_strategy=epoch',
                    f'--save_strategy=epoch',
                    f'--save_total_limit=2',
                    f'--patience=3',
                    f'--label_names=labels',
                    f'--logging_steps=100',
                    f'--metric_for_best_model=auc',
                    f'--load_best_model_at_end',
                    f'--overwrite_output_dir',
                    f'--do_train',
                    f'--do_eval',
                    f'--{args.mode}',
                    f'--seed={args.seed}',
                    f'--fp16',
                    f'--ctr_model_name={CTR_MODEL_NAME}',
                    f'--model_fusion={MODEL_FUSION}',
                    f'--num_prompt={NUM_PROMPT}',
                ] + [f'--{n}={p}' for n, p in CTR_MODEL_PARAM.items() if n != 'sub_output_dir'])
                exit(0)