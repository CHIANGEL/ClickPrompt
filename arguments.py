import logging
import math
import os
import copy
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
    EarlyStoppingCallback
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, is_torch_available, is_torch_tpu_available

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # Our arguments
    do_ctr: bool = field(
        default=False,
        metadata={
            "help": "Whether to do CTR training, instead of CL training."
        }
    )
    do_mlm_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to do MLM pretraining only."
        }
    )
    model_fusion: str = field(
        default="no",
        metadata={
            "help": "The method to fuse CTR models and language models for prediction."
        }
    )
    weighted_add: bool = field(
        default=False,
        metadata={
            "help": "Whether apply weighted sum for `prefix_add`/`prompt_add`/`add`"
        }
    )
    init_weighted_alpha: float = field(
        default=0.5,
        metadata={
            "help": "The initial alpha for weighted add"
        }
    )
    update_alpha: bool = field(
        default=True,
        metadata={
            "help": "Whether to update the alpha for weighted_add"
        }
    )
    load_from_path: str = field(
        default="no",
        metadata={
            "help": "The model parameters to be loaded from local files, e.g., pytorch_model.bin."
        }
    )
    freeze_ctr: bool = field(
        default=False,
        metadata={
            "help": "Whether to freeze the CTR model."
        }
    )
    freeze_nlp: bool = field(
        default=False,
        metadata={
            "help": "Whether to freeze the NLP model."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, avg_last)."
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv or .tsv)."}
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The testing data file (.txt or .csv or .tsv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )
    
    # Data arguments for CTR data
    meta_data_dir: str=field(
        default="./data/AZ-Fashion-CTR-meta.json",
        metadata={
            "help": "The tabular metedata path"
        },
    )
    h5_data_dir :str=field(
        default='./data/AZ-Fashion-CTR.h5',
        metadata={
            "help": "The tabular data path"
        }
    )


@dataclass
class OurTrainingArguments(TrainingArguments):
    shuffle_fields: bool = field(
        default=False, 
        metadata={"help": "Whether to shuffle the fields for lossless augmentation"}
    )
    patience: int = field(
        default=3, 
        metadata={"help": "The patience for early stoppint strategy"}
    )
    extend_vocab: str = field(
        default='none', 
        metadata={"help": "The method to extend the vocabulary. Default to `None` indicating no extension."}
    )
    ctr_learning_rate: float = field(
        default=1e-3, 
        metadata={"help": "The learning rate for CTR model part, which could be different from the original language model part."}
    )


    @cached_property
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


@dataclass
class CTRModelArguments:
    ctr_model_name: str

    embed_size: int = field(default=32)
    embed_dropout_rate: float = field(default=0.0)
    hidden_size: int = field(default=128)
    num_hidden_layers: int = field(default=1)
    hidden_act: str = field(default='relu')
    hidden_dropout_rate: float = field(default=0.0)
    num_attn_heads: int = field(default=1)
    attn_probs_dropout_rate: float = field(default=0.1)
    intermediate_size: int = field(default=128)
    norm_first: bool = field(default=False)
    layer_norm_eps: float = field(default=1e-12)
    res_conn: bool = field(default=False)
    output_dim: int = field(default=1)
    num_cross_layers: int = field(default=1)
    share_embedding: bool = field(default=False)
    channels: str = field(default='14,16,18,20')
    kernel_heights: str = field(default='7,7,7,7')
    pooling_sizes: str = field(default='2,2,2,2')
    recombined_channels: str = field(default='3,3,3,3')
    conv_act: str = field(default='tanh')
    reduction_ratio: int = field(default=3)
    bilinear_type: str = field(default='field_interaction')
    reuse_graph_layer: bool = field(default=False)
    attn_scale: bool = field(default=False)
    use_lr: bool = field(default=False)
    attn_size: int = field(default=40)
    num_attn_layers: int = field(default=2)
    cin_layer_units: str = field(default='50,50')
    product_type: str = field(default='inner')
    outer_product_kernel_type: str = field(default='mat')
    dnn_size: int = field(default=1000, metadata={'help': "The size of each dnn layer"})
    num_dnn_layers: int = field(default=0, metadata={"help": "The number of dnn layers"})
    dnn_act: str = field(default='relu', metadata={'help': "The activation function for dnn layers"})
    dnn_drop: float = field(default=0.0, metadata={'help': "The dropout for dnn layers"})

    num_prompt: int = field(
        default=0,
        metadata={
            'help': "The number of soft prompt tokens to be generated by CTR model."
        },
    )
    prompt_usage: str = field(
        default='no',
        metadata={
            'help': "How to use pretrained prompt layers for CTR prediction"
        },
    )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def save(self, save_directory):
        assert os.path.isdir(save_directory), f"not a directory: {save_directory}"
        output_config_file = os.path.join(save_directory, 'config.json')
        self.to_json_file(output_config_file)

    @classmethod
    def load(cls, load_directory):
        output_config_file = os.path.join(load_directory, 'config.json')
        config_dict = cls.from_json_file(output_config_file)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict):
        config = cls(**config_dict)
        return config

    @classmethod
    def from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_string(self):
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())