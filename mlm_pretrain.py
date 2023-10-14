import sys
import subprocess
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='roberta-base')
parser.add_argument('--extend_vocab', default='none')
args = parser.parse_args()

# Target py file
TARGET_PY_FILE = 'run.py'

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID = 15637

# The dataset name
DATASET = 'ML-1M'

# The model to be used
MODEL = args.model

# Vocabulary Extension
EXTEND_VOCAB = args.extend_vocab

# Hyper-param grid search
NUM_GPU = 1
BATCH_SIZE_PER_DEVICE = 16
TOTAL_BATCH_SIZE = 1024
ACCUMULATION = TOTAL_BATCH_SIZE // NUM_GPU // BATCH_SIZE_PER_DEVICE
assert ACCUMULATION > 0
if NUM_GPU > 1:
    PREFIX = f'python -m torch.distributed.launch --nproc_per_node {NUM_GPU} --master_port {PORT_ID} {TARGET_PY_FILE}'.split(' ')
else:
    PREFIX = ['python', TARGET_PY_FILE]
for LR in ['5e-5']:
    for EPOCH in [10, 20, 30]:
        for WARMUP in [0.1, 0.05]:
            FILE_PATH = f'mlm_pretrain/BS{TOTAL_BATCH_SIZE}-LR{LR}-EPOCH{EPOCH}-WARM{WARMUP}-VOCAB{EXTEND_VOCAB}'
            subprocess.run(PREFIX + [
                f'--model_name_or_path={MODEL}',
                f'--train_file=./data/{DATASET}/proc_data/train.csv',
                f'--test_file=./data/{DATASET}/proc_data/test.csv',
                f'--subword_init_file=./data/{DATASET}/proc_data/roberta-base-subword-init.csv',
                f'--output_dir=result/{DATASET}/{MODEL}/{FILE_PATH}',
                f'--logging_dir=runs/{DATASET}/{MODEL}/{FILE_PATH}',
                f'--report_to=tensorboard',
                f'--remove_unused_columns=False',
                f'--extend_vocab={EXTEND_VOCAB}',
                f'--shuffle_fields=False',
                f'--warmup_ratio={WARMUP}',
                f'--num_train_epochs={EPOCH}',
                f'--per_device_train_batch_size={BATCH_SIZE_PER_DEVICE}',
                f'--per_device_eval_batch_size={BATCH_SIZE_PER_DEVICE}',
                f'--gradient_accumulation_steps={ACCUMULATION}',
                f'--learning_rate={LR}',
                f'--weight_decay=0',
                f'--max_seq_length=512',
                f'--evaluation_strategy=no',
                f'--save_strategy=no',
                f'--save_total_limit=2',
                f'--patience=3',
                f'--label_names=labels',
                f'--logging_steps=100',
                # f'--load_best_model_at_end',
                f'--pooler_type=cls',
                f'--mlp_only_train',
                f'--overwrite_output_dir',
                f'--do_train',
                # f'--do_eval',
                f'--do_mlm_only',
                f'--fp16',
            ])
            