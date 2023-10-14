import sys, os
import subprocess
import argparse
import utils

# Parse arguments
args = utils.parse()

# Target py file
TARGET_PY_FILE = f'{args.root}/run.py'

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID = 15636

# The dataset name
if args.dataset == "ml-1m":
    DATASET_SHORT = "ML"
elif args.dataset == "BookCrossing":
    DATASET_SHORT = "BX"
elif args.dataset == "AZ-Toys":
    DATASET_SHORT = "AZT"
elif args.dataset == "GoodReads":
    DATASET_SHORT = "GD"
elif args.dataset == "GoodReads-4M":
    DATASET_SHORT = "GD4M"
DATA_DIR_DICT = utils.get_data_dir(args.root, args.dataset)

# The model to be used, as well as the possible fusion with CTR models
CTR_MODEL_PARAM = utils.get_ctr_model_param(args.ctr_model, args.dataset)

# Hyper-param grid search
NUM_GPU = int(args.num_gpu)
BATCH_SIZE_PER_DEVICE = int(args.bs)
TOTAL_BATCH_SIZE = int(args.total_bs)
ACCUMULATION = TOTAL_BATCH_SIZE // NUM_GPU // BATCH_SIZE_PER_DEVICE
assert ACCUMULATION > 0
if NUM_GPU > 1:
    PREFIX = f'python -m torch.distributed.launch --nproc_per_node {NUM_GPU} --master_port {PORT_ID} {TARGET_PY_FILE}'.split(' ')
else:
    PREFIX = ['python', TARGET_PY_FILE]
for args.ptSched in ["cosine", "linear"] if args.ptSched is None else [args.ptSched]:
    for args.ptWarm in [0.05, 0.1] if args.ptWarm is None else [args.ptWarm]:
        for ptBS in [1024]:
            load_from_path = f"{args.root}/result/{args.dataset}/{args.model}/{args.ctr_model}/mlm_pretrain/BS1024-LR5e-5-EPOCH20-WARM{args.ptWarm}-{args.ptSched}-{args.num_prompt}/pytorch_model.bin"
            # load_from_path = f"{args.root}/{args.dataset}/{args.ctr_model}/prefix/{args.ptSched}-BS{ptBS}-LR5e-5-EPOCH{args.ptEpoch}-WARM{args.ptWarm}-{args.num_prompt}/pytorch_model.bin"
            # assert os.path.exists(load_from_path), load_from_path
            print(load_from_path)
            if not os.path.exists(load_from_path):
                print(load_from_path)
                continue
            for args.warm in [args.warm]:
                for args.pooler in ["avg", "avg_first_last", "avg_top2", "cls"] if args.fusion != "ctr_only" else ["None"]:
                    for args.sched in ['cosine', "linear"]:
                        for args.wd in [0, 0.1, 0.5]:
                            for args.prompt_usage in ['no'] if args.fusion in ["prefix", "ctr_only"] else ["no", "avg", "cat"]:
                                for args.lr in [args.lr]:
                                    if args.dataset == "ml-1m":
                                        EPOCHS = [4]
                                    elif args.dataset == "AZ-Toys":
                                        EPOCHS = [2, 3]
                                    elif args.dataset == "BookCrossing":
                                        EPOCHS = [2, 3]
                                    elif args.dataset == "GoodReads":
                                        EPOCHS = [2, 3]
                                    for args.epoch in EPOCHS:
                                        FILE_PATH = f'pt[{args.ptEpoch}-{ptBS}-{args.ptSched}-{args.ptWarm}-{args.num_prompt}]-{args.prompt_usage}-{args.sched}-BS{TOTAL_BATCH_SIZE}-LR{args.lr}-EPOCH{args.epoch}-WARM{args.warm}-WD{args.wd}-{args.pooler}-{args.freeze_nlp}-{args.freeze_ctr}'
                                        if args.weighted_add == "True":
                                            FILE_PATH += f"-{args.weighted_add}"
                                        subprocess.run(PREFIX + [
                                            f'--pooler_type={args.pooler}',
                                            f'--weighted_add={args.weighted_add}',
                                            f'--freeze_nlp={args.freeze_nlp}',
                                            f'--freeze_ctr={args.freeze_ctr}',
                                            f'--model_name_or_path={args.model}',
                                            f'--output_dir={args.root}/result/{args.dataset}/{args.model}/{args.ctr_model}/{args.fusion}/{FILE_PATH}',
                                            f'--logging_dir={args.root}/result/{args.dataset}/{args.model}/{args.ctr_model}/{args.fusion}/{FILE_PATH}',
                                            f'--report_to=tensorboard',
                                            f'--remove_unused_columns=False',
                                            f'--shuffle_fields=False',
                                            f'--warmup_ratio={args.warm}',
                                            f'--num_train_epochs={args.epoch}',
                                            f'--lr_scheduler_type={args.sched}',
                                            f'--per_device_train_batch_size={BATCH_SIZE_PER_DEVICE}',
                                            f'--per_device_eval_batch_size={BATCH_SIZE_PER_DEVICE}',
                                            f'--gradient_accumulation_steps={ACCUMULATION}',
                                            f'--learning_rate={args.lr}',
                                            f'--ctr_learning_rate={args.lr}',
                                            f'--weight_decay={args.wd}',
                                            f'--max_seq_length=512',
                                            f'--evaluation_strategy=epoch',
                                            f'--save_strategy=epoch',
                                            f'--save_total_limit=2',
                                            f'--patience=2',
                                            f'--label_names=labels',
                                            f'--logging_steps=1000',
                                            f'--metric_for_best_model=auc',
                                            f'--load_best_model_at_end',
                                            f'--overwrite_output_dir',
                                            f'--do_train',
                                            f'--do_eval',
                                            f'--do_ctr',
                                            f'--seed={args.seed}',
                                            f'--fp16',
                                            f'--ctr_model_name={args.ctr_model}',
                                            f'--model_fusion={args.fusion}',
                                            f'--num_prompt={args.num_prompt}',
                                            f'--load_from_path={load_from_path}',
                                            f'--prompt_usage={args.prompt_usage}',
                                        ] + [f'--{n}={p}' for n, p in CTR_MODEL_PARAM.items() if n != 'sub_output_dir']
                                        + [f'--{n}={p}' for n, p in DATA_DIR_DICT.items()])
                                        