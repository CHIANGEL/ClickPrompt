import logging
import math
import os
import pathlib, shutil
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import torch
import collections
import random

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel,
    RobertaForMaskedLM,
    EarlyStoppingCallback
)
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, is_torch_available, is_torch_tpu_available

from models import PLM4CTR_Model
from trainers import PLM4CTRTrainer
from dataset import load_csv_as_df, PLM4CTRDataset, PLM4CTRDataCollator
from utils import compute_metrics, vocabulary_extension
from arguments import ModelArguments, DataTrainingArguments, OurTrainingArguments, CTRModelArguments, Config

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, CTRModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, ctr_model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, ctr_model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if (
        os.path.exists(os.path.join(training_args.output_dir, "train_results.txt"))
        and training_args.do_train
    ):
        print("Job already finished, quit")
        exit(0)

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )
    
    # Sanity check
    if model_args.do_ctr:
        assert not model_args.do_mlm_only, "Should not do CTR and MLM at the same time."
    elif model_args.do_mlm_only:
        assert not model_args.do_ctr, "Should not do CTR when use --do_mlm_only, use --do_mlm instead."

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    # Load datasets
    train_df = load_csv_as_df(data_args.train_file, training_args.extend_vocab)
    test_df = load_csv_as_df(data_args.test_file, training_args.extend_vocab)
    if "ml-1m" in data_args.train_file.lower():
        extension_fields = ["User ID", "Movie ID"]
    else:
        extension_fields = []
    tokenizer = vocabulary_extension(
        tokenizer=tokenizer, 
        df_list=[train_df, test_df], 
        extension_method=training_args.extend_vocab,
        extension_fields=extension_fields
    )
    # Originally, len(train_df) : len(test_df) = 9 : 1
    # Therefore, 8/9 * len(train_df) : 1/9 * len(train_df) : len(test_df) = 8 : 1 : 1
    # The validation set is also used in mlm pretraining.
    total_datasets = {
        "train": PLM4CTRDataset.from_pandas(train_df[:len(train_df) // 9 * 8] if not model_args.do_mlm_only else train_df),
        "valid": PLM4CTRDataset.from_pandas(train_df[len(train_df) // 9 * 8:]),
        "test": PLM4CTRDataset.from_pandas(test_df),
    }
    for split_name in ["train", "valid", "test"]:
        total_datasets[split_name]._post_setups(
            tokenizer=tokenizer,
            shuffle_fields=training_args.shuffle_fields,
            meta_data_dir=data_args.meta_data_dir,
            h5_data_dir=data_args.h5_data_dir,
            mode=split_name,
            model_fusion=model_args.model_fusion,
            do_mlm_only=model_args.do_mlm_only,
        )
        logger.info(total_datasets[split_name])

    # Build config for CTR model
    model_config_dict = ctr_model_args.to_dict()
    model_config_dict["input_size"] = total_datasets["train"].input_size
    model_config_dict["num_fields"] = total_datasets["train"].num_fields
    model_config_dict["device"] = training_args.device
    model_config_dict["n_gpu"] = training_args.n_gpu
    model_config_dict["model_fusion"] = model_args.model_fusion
    ctr_config = Config.from_dict(model_config_dict)

    # Load model
    model = PLM4CTR_Model(
        ctr_config=ctr_config,
        model_args=model_args,
    )
    
    # Data collator
    data_collator = PLM4CTRDataCollator(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
        model_fusion=model_args.model_fusion,
        do_ctr=model_args.do_ctr,
        do_mlm_only=model_args.do_mlm_only,
        mlm_probability=data_args.mlm_probability,
    )
    
    # Trainer
    trainer = PLM4CTRTrainer(
        model=model,
        args=training_args,
        train_dataset=total_datasets["train"],
        eval_dataset=total_datasets["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if model_args.do_ctr else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.patience)] if training_args.do_eval else None
    )
    trainer.model_args = model_args
    assert not trainer.args.remove_unused_columns # IMPOERTANT! Otherwise, the tabualr columns will be removed.

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        if model_args.do_ctr:
            test_outputs = trainer.evaluate(
                eval_dataset=total_datasets["test"],
                metric_key_prefix="test",
            )
        
        trainer.save_model() # Save the tokenizer too for easy upload.
        if trainer.is_world_process_zero():
            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model.
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
            
            # Write to `train_results.txt`.
            with open(os.path.join(training_args.logging_dir, "train_results.txt"), "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
                if model_args.do_ctr:
                    logger.info(f"  test_auc = {test_outputs['test_auc']}")
                    logger.info(f"  test_loss = {test_outputs['test_loss']}")
                    writer.write(f"test_auc = {test_outputs['test_auc']}\n")
                    writer.write(f"test_loss = {test_outputs['test_loss']}\n")
                    if model_args.weighted_add:
                        logger.info(f"  alpha = {model.alpha.item()}")
                        writer.write(f"alpha = {model.alpha.item()}\n")

            # Only need `train_results.txt`
            if model_args.do_ctr:
                for x in os.listdir(training_args.output_dir):
                    if x != "train_results.txt":
                        path = os.path.join(training_args.output_dir, x)
                        if pathlib.Path(path).is_file():
                            os.remove(path)
                        elif pathlib.Path(path).is_dir():
                            shutil.rmtree(path)
    else:
        if model_args.do_ctr:
            test_outputs = trainer.evaluate(
                eval_dataset=total_datasets["test"],
                metric_key_prefix="test",
            )
            logger.info(test_outputs)


if __name__ == "__main__":
    main()
