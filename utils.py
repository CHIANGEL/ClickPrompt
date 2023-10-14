import logging
from typing import Optional, Union, List, Dict, Tuple, Any
from sklearn.metrics import log_loss, roc_auc_score
import pandas as pd
import torch.nn as nn

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="./")
    parser.add_argument('--model', default=f'models/roberta-base/')
    parser.add_argument('--ctr_model', default='DeepFM')
    parser.add_argument('--fusion', default='prompt')
    parser.add_argument('--num_prompt', default=10)
    parser.add_argument('--prompt_usage', default="no")
    parser.add_argument('--num_gpu')
    parser.add_argument('--total_bs')
    parser.add_argument('--bs')
    parser.add_argument('--wd')
    parser.add_argument('--lr', default="1e-3")
    parser.add_argument('--warm', default=None)
    parser.add_argument('--sched', default=None)
    parser.add_argument('--dataset')
    parser.add_argument('--epoch')
    parser.add_argument('--pooler')
    parser.add_argument('--ptEpoch')
    parser.add_argument('--ptSched', default=None)
    parser.add_argument('--ptWarm', default=None)
    parser.add_argument('--freeze_nlp', default="False")
    parser.add_argument('--freeze_ctr', default="False")
    parser.add_argument('--weighted_add', default="False")
    parser.add_argument('--update_alpha', default="True")
    parser.add_argument('--init_weighted_alpha', default=0.5)
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()
    return args


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight)


def load_from_target_dict(model, target_state_dict):
    state_dict = model.state_dict()
    for k, v in target_state_dict.items():
        if k in state_dict and state_dict[k].shape == target_state_dict[k].shape:
            state_dict[k] = v
            logger.info(f"Load tensor: {k}, {v.shape}")
        else:
            logger.info(f"Unmatched tensor: {k}, {v.shape}")
    model.load_state_dict(state_dict)


def compute_metrics(eval_output):
    labels = eval_output.label_ids
    logits = eval_output.predictions
    auc = roc_auc_score(labels, logits)
    # Log loss metric is equal to the eval loss
    return {
        'auc': auc,
    }


def vocabulary_extension(
    tokenizer: PreTrainedTokenizerBase, 
    df_list: List, 
    extension_method: str,
    extension_fields: List,
):
    if extension_method in ['none', 'prefix_none']:
        pass
    elif extension_method == 'raw':
        df = pd.concat(df_list)
        for field in extension_fields:
            tokenizer.add_tokens(list(set(list(df[field]))))
            # Sanity check
            for v in set(list(df[field])):
                assert v in tokenizer.tokenize(f"{field} is {v}.") or v.lower() in tokenizer.tokenize(f"{field} is {v}."), tokenizer.tokenize(f"{field} is {v}.")
    else:
        raise NotImplementedError
    return tokenizer


def get_data_dir(data_root: str, dataset_name: str):
    if dataset_name == "ml-1m-new":
        data_dir_dict = {
            "train_file": f'{data_root}/data/{dataset_name}/proc_data/train.parquet.gz',
            "test_file": f'{data_root}/data/{dataset_name}/proc_data/test.parquet.gz',
            "meta_data_dir": f'{data_root}/data/{dataset_name}/proc_data/ctr-meta.json',
            "h5_data_dir": f'{data_root}/data/{dataset_name}/proc_data/ctr.h5',
        }
    elif dataset_name == "ml-1m":
        data_dir_dict = {
            "train_file": f'{data_root}/data/{dataset_name}/proc_data/train.csv',
            "test_file": f'{data_root}/data/{dataset_name}/proc_data/test.csv',
            "meta_data_dir": f'{data_root}/data/{dataset_name}/proc_data/ctr-meta.json',
            "h5_data_dir": f'{data_root}/data/{dataset_name}/proc_data/ctr.h5',
        }
    elif dataset_name == "BookCrossing":
        data_dir_dict = {
            "train_file": f'{data_root}/data/{dataset_name}/proc_data/train.tsv',
            "test_file": f'{data_root}/data/{dataset_name}/proc_data/test.tsv',
            "meta_data_dir": f'{data_root}/data/{dataset_name}/proc_data/ctr-meta.json',
            "h5_data_dir": f'{data_root}/data/{dataset_name}/proc_data/ctr.h5',
        }
    elif dataset_name == "AZ-Toys":
        data_dir_dict = {
            "train_file": f'{data_root}/data/{dataset_name}/proc_data/train.parquet.gz',
            "test_file": f'{data_root}/data/{dataset_name}/proc_data/test.parquet.gz',
            "meta_data_dir": f'{data_root}/data/{dataset_name}/proc_data/ctr-meta.json',
            "h5_data_dir": f'{data_root}/data/{dataset_name}/proc_data/ctr.h5',
        }
    elif dataset_name == "GoodReads":
        data_dir_dict = {
            "train_file": f'{data_root}/data/{dataset_name}/proc_data/train.parquet.gz',
            "test_file": f'{data_root}/data/{dataset_name}/proc_data/test.parquet.gz',
            "meta_data_dir": f'{data_root}/data/{dataset_name}/proc_data/ctr-meta.json',
            "h5_data_dir": f'{data_root}/data/{dataset_name}/proc_data/ctr.h5',
        }
    elif dataset_name == "GoodReads-4M":
        data_dir_dict = {
            "train_file": f'{data_root}/data/GoodReads/proc_data/train-4M.parquet.gz',
            "test_file": f'{data_root}/data/GoodReads/proc_data/test-4M.parquet.gz',
            "meta_data_dir": f'{data_root}/data/GoodReads/proc_data/ctr-meta.json',
            "h5_data_dir": f'{data_root}/data/GoodReads/proc_data/ctr-4M.h5',
        }
    else:
        raise NotImplementedError
    return data_dir_dict


def get_ctr_model_param(ctr_model_name: str, dataset_name: str):
    assert dataset_name in ["ml-1m-new", "ml-1m", "BookCrossing", "GoodReads", "GoodReads-4M", "AZ-Toys"]
    if ctr_model_name == 'DNN':
        if dataset_name == "ml-1m-new":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
        if dataset_name == "ml-1m":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
        if dataset_name == "BookCrossing":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
        if dataset_name == "AZ-Toys":
            embed_size = 16
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
        if dataset_name == "GoodReads" or dataset_name == "GoodReads-4M":
            embed_size = 16
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
        param_dict = {
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'hidden_dropout_rate': dropout,
            'num_hidden_layers': num_hidden_layers,
            'sub_output_dir': f'_emb{embed_size}_hs{hidden_size}_hl{num_hidden_layers}_drop{dropout}/',
        }
    elif ctr_model_name == 'FM':
        if dataset_name == "ml-1m":
            embed_size = 32
            use_lr = False
        if dataset_name == "BookCrossing":
            embed_size = 32
            use_lr = False
        if dataset_name == "AZ-Toys":
            embed_size = 16
            use_lr = False
        if dataset_name == "GoodReads" or dataset_name == "GoodReads-4M":
            embed_size = 16
            use_lr = True
        param_dict = {
            'embed_size': embed_size,
            'use_lr': use_lr,
            'sub_output_dir': f'_emb{embed_size}_lr{use_lr}/',
        }
    elif ctr_model_name == 'DeepFM':
        if dataset_name == "ml-1m-new":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
            use_lr = True
        if dataset_name == "ml-1m":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
            use_lr = True
        if dataset_name == "AZ-Toys":
            embed_size = 16
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
            use_lr = False
        if dataset_name == "BookCrossing":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 4
            use_lr = False
        if dataset_name == "GoodReads" or dataset_name == "GoodReads-4M":
            embed_size = 16
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
            use_lr = False
        param_dict = {
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'hidden_dropout_rate': dropout,
            'num_hidden_layers': num_hidden_layers,
            'use_lr': use_lr,
            'sub_output_dir': f'_emb{embed_size}_hs{hidden_size}_hl{num_hidden_layers}_drop{dropout}_lr{use_lr}/',
        }
    elif ctr_model_name == 'xDeepFM':
        if dataset_name == "ml-1m-new":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 3
            cin_layer_units = '25-25-25'
        if dataset_name == "ml-1m":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 3
            cin_layer_units = '25-25-25'
        if dataset_name == "AZ-Toys":
            embed_size = 16
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 3
            cin_layer_units = '25-25-25'
        if dataset_name == "BookCrossing":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
            cin_layer_units = '25-25-25'
        if dataset_name == "GoodReads" or dataset_name == "GoodReads-4M":
            embed_size = 16
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
            cin_layer_units = '25-25-25'
        param_dict = {
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'hidden_dropout_rate': dropout,
            'num_hidden_layers': num_hidden_layers,
            'cin_layer_units': cin_layer_units,
            'sub_output_dir': f'_emb{embed_size}_hs{hidden_size}_hl{num_hidden_layers}_drop{dropout}_cin{cin_layer_units}/',
        }
    elif ctr_model_name == 'IPNN':
        if dataset_name == "ml-1m-new":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
        if dataset_name == "ml-1m":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
        if dataset_name == "AZ-Toys":
            embed_size = 16
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
        if dataset_name == "BookCrossing":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
        if dataset_name == "GoodReads" or dataset_name == "GoodReads-4M":
            embed_size = 16
            dropout = 0
            hidden_size = 512
            num_hidden_layers = 6
        param_dict = {
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'hidden_dropout_rate': dropout,
            'num_hidden_layers': num_hidden_layers,
            'sub_output_dir': f'_emb{embed_size}_hs{hidden_size}_hl{num_hidden_layers}_drop{dropout}/',
        }
    elif ctr_model_name == 'DCN':
        if dataset_name == "ml-1m":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 4
            num_cross_layers = 4
        if dataset_name == "AZ-Toys":
            embed_size = 16
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 3
            num_cross_layers = 3
        if dataset_name == "BookCrossing":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 4
            num_cross_layers = 4
        if dataset_name == "GoodReads" or dataset_name == "GoodReads-4M":
            embed_size = 16
            dropout = 0
            hidden_size = 512
            num_hidden_layers = 4
            num_cross_layers = 4
        param_dict = {
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'hidden_dropout_rate': dropout,
            'num_hidden_layers': num_hidden_layers,
            'num_cross_layers': num_cross_layers,
            'sub_output_dir': f'_emb{embed_size}_hs{hidden_size}_hl{num_hidden_layers}_cl{num_cross_layers}_drop{dropout}/',
        }
    elif ctr_model_name == 'DCNv2':
        if dataset_name == "ml-1m-new":
            embed_size = 32
            dropout = 0.1
            hidden_size = 256
            num_hidden_layers = 6
            num_cross_layers = 6
        if dataset_name == "ml-1m":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
            num_cross_layers = 6
        if dataset_name == "AZ-Toys":
            embed_size = 16
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 3
            num_cross_layers = 3
        if dataset_name == "BookCrossing":
            embed_size = 32
            dropout = 0
            hidden_size = 256
            num_hidden_layers = 6
            num_cross_layers = 6
        if dataset_name == "GoodReads" or dataset_name == "GoodReads-4M":
            embed_size = 16
            dropout = 0
            hidden_size = 512
            num_hidden_layers = 6
            num_cross_layers = 6
        param_dict = {
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'hidden_dropout_rate': dropout,
            'num_hidden_layers': num_hidden_layers,
            'num_cross_layers': num_cross_layers,
            'sub_output_dir': f'_emb{embed_size}_hs{hidden_size}_hl{num_hidden_layers}_cl{num_cross_layers}_drop{dropout}/',
        }
    elif ctr_model_name == 'FiBiNet':
        if dataset_name == "ml-1m-new":
            embed_size = 32
            hidden_size = 256
            num_hidden_layers = 2
            dropout = 0
            reduction_ratio = 3
            bilinear_type = 'field_interaction'
        if dataset_name == "ml-1m":
            embed_size = 32
            hidden_size = 256
            num_hidden_layers = 2
            dropout = 0
            reduction_ratio = 3
            bilinear_type = 'field_interaction'
        if dataset_name == "BookCrossing":
            embed_size = 32
            hidden_size = 256
            num_hidden_layers = 2
            dropout = 0
            reduction_ratio = 3
            bilinear_type = 'field_interaction'
        if dataset_name == "GoodReads" or dataset_name == "GoodReads-4M":
            embed_size = 16
            hidden_size = 256
            num_hidden_layers = 2
            dropout = 0
            reduction_ratio = 3
            bilinear_type = 'field_interaction'
        param_dict = {
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'hidden_dropout_rate': dropout,
            'num_hidden_layers': num_hidden_layers,
            'reduction_ratio': reduction_ratio,
            'bilinear_type': bilinear_type,
            'sub_output_dir': f'_emb{embed_size}_hs{hidden_size}_hl{num_hidden_layers}_drop{dropout}_rr{reduction_ratio}_{bilinear_type}/',
        }
    elif ctr_model_name == 'FiGNN':
        if dataset_name == "ml-1m-new":
            embed_size = 32
            num_hidden_layers = 3
            dropout = 0
            res_conn = True
            reuse_graph_layer = True
        if dataset_name == "ml-1m":
            embed_size = 32
            num_hidden_layers = 3
            dropout = 0
            res_conn = True
            reuse_graph_layer = True
        if dataset_name == "AZ-Toys":
            embed_size = 16
            num_hidden_layers = 3
            dropout = 0
            res_conn = True
            reuse_graph_layer = True
        if dataset_name == "BookCrossing":
            embed_size = 32
            num_hidden_layers = 6
            dropout = 0
            res_conn = True
            reuse_graph_layer = True
        if dataset_name == "GoodReads" or dataset_name == "GoodReads-4M":
            embed_size = 16
            num_hidden_layers = 3
            dropout = 0
            res_conn = True
            reuse_graph_layer = True
        param_dict = {
            'embed_size': embed_size,
            'hidden_dropout_rate': dropout,
            'num_hidden_layers': num_hidden_layers,
            'res_conn': res_conn,
            'reuse_graph_layer': reuse_graph_layer,
            'sub_output_dir': f'_emb{embed_size}_hl{num_hidden_layers}_drop{dropout}_reuse{reuse_graph_layer}/',
        }
    elif ctr_model_name == 'FGCNN': # We cut down the embedding size since FGCNN maintains two separate embedding tables.
        if dataset_name == "ml-1m-new":
            embed_size = 16
            hidden_size = 256
            num_hidden_layers = 3
            dropout = 0
            channels = '6|8|10|12'
            kernel_heights = '7|7|7|7'
            pooling_sizes = '2|2|2|2'
            recombined_channels = '3|3|3|3'
            share_embedding = False
        if dataset_name == "ml-1m":
            embed_size = 16
            hidden_size = 256
            num_hidden_layers = 3
            dropout = 0
            channels = '6|8|10|12'
            kernel_heights = '7|7|7|7'
            pooling_sizes = '2|2|2|2'
            recombined_channels = '3|3|3|3'
            share_embedding = False
        if dataset_name == "AZ-Toys":
            embed_size = 8
            hidden_size = 256
            num_hidden_layers = 3
            dropout = 0
            channels = '6|8|10|12'
            kernel_heights = '7|7|7|7'
            pooling_sizes = '2|2|2|2'
            recombined_channels = '3|3|3|3'
            share_embedding = False
        if dataset_name == "BookCrossing":
            embed_size = 16
            hidden_size = 256
            num_hidden_layers = 3
            dropout = 0
            channels = '6|8|10|12'
            kernel_heights = '7|7|7|7'
            pooling_sizes = '2|2|2|2'
            recombined_channels = '3|3|3|3'
            share_embedding = False
        if dataset_name == "GoodReads" or dataset_name == "GoodReads-4M":
            embed_size = 8
            hidden_size = 256
            num_hidden_layers = 3
            dropout = 0
            channels = '6|8|10|12'
            kernel_heights = '7|7|7|7'
            pooling_sizes = '2|2|2|2'
            recombined_channels = '3|3|3|3'
            share_embedding = False
        param_dict = {
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_hidden_layers,
            'hidden_dropout_rate': dropout,
            'channels': channels,
            'kernel_heights': kernel_heights,
            'pooling_sizes': pooling_sizes,
            'recombined_channels': recombined_channels,
            'share_embedding': share_embedding,
            'sub_output_dir': f'_emb{embed_size}_hs{hidden_size}_hl{num_hidden_layers}_channels{channels}_REchannels{recombined_channels}/',
        }
    elif ctr_model_name == 'AutoInt':
        if dataset_name == "ml-1m-new":
            embed_size = 32
            num_attn_layers = 6
            num_attn_heads = 1
            attn_size = 32
            attn_probs_dropout_rate = 0
            dnn_size = 256
            num_dnn_layers = 6
            res_conn = True
            attn_scale = False
        if dataset_name == "ml-1m":
            embed_size = 32
            num_attn_layers = 6
            num_attn_heads = 1
            attn_size = 32
            attn_probs_dropout_rate = 0
            dnn_size = 256
            num_dnn_layers = 6
            res_conn = True
            attn_scale = False
        if dataset_name == "AZ-Toys":
            embed_size = 16
            num_attn_layers = 6
            num_attn_heads = 1
            attn_size = 32
            attn_probs_dropout_rate = 0
            dnn_size = 256
            num_dnn_layers = 6
            res_conn = True
            attn_scale = False
        if dataset_name == "BookCrossing":
            embed_size = 32
            num_attn_layers = 6
            num_attn_heads = 1
            attn_size = 32
            attn_probs_dropout_rate = 0
            dnn_size = 256
            num_dnn_layers = 6
            res_conn = True
            attn_scale = False
        if dataset_name == "GoodReads" or dataset_name == "GoodReads-4M":
            embed_size = 16
            num_attn_layers = 6
            num_attn_heads = 1
            attn_size = 32
            attn_probs_dropout_rate = 0
            dnn_size = 256
            num_dnn_layers = 6
            res_conn = True
            attn_scale = False
        param_dict = {
            'embed_size': embed_size,
            'num_attn_layers': num_attn_layers,
            'num_attn_heads': num_attn_heads,
            'attn_size': attn_size,
            'attn_probs_dropout_rate': attn_probs_dropout_rate,
            'dnn_size': dnn_size,
            'num_dnn_layers': num_dnn_layers,
            'res_conn': res_conn,
            'attn_scale': attn_scale,
            'sub_output_dir': f'_emb{embed_size}_As{attn_size}_Al{num_attn_layers}_Ah{num_attn_heads}_Ad{attn_probs_dropout_rate}_Ds{dnn_size}_Dl{num_dnn_layers}_{res_conn}_{attn_scale}/',
        }
    elif ctr_model_name == "Transformer":
        if dataset_name == "ml-1m-new":
            embed_size = 32
            hidden_size = embed_size
            num_hidden_layers = 2
            num_attn_heads = 1
            hidden_dropout_rate = 0
            norm_first = True
            intermediate_size = 4 * embed_size
        if dataset_name == "ml-1m":
            embed_size = 32
            hidden_size = embed_size
            num_hidden_layers = 2
            num_attn_heads = 1
            hidden_dropout_rate = 0
            norm_first = True
            intermediate_size = 4 * embed_size
        if dataset_name == "AZ-Toys":
            embed_size = 16
            hidden_size = embed_size
            num_hidden_layers = 2
            num_attn_heads = 1
            hidden_dropout_rate = 0
            norm_first = True
            intermediate_size = 4 * embed_size
        if dataset_name == "BookCrossing":
            embed_size = 32
            hidden_size = embed_size
            num_hidden_layers = 3
            num_attn_heads = 2
            hidden_dropout_rate = 0
            norm_first = True
            intermediate_size = 4 * embed_size
        param_dict = {
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_hidden_layers,
            'num_attn_heads': num_attn_heads,
            'hidden_dropout_rate': hidden_dropout_rate,
            'norm_first': norm_first,
            'intermediate_size': intermediate_size,
            'sub_output_dir': f'_emb{embed_size}_l{num_hidden_layers}_h{num_attn_heads}_d{hidden_dropout_rate}_ffn{intermediate_size}/',
        }
    else:
        raise NotImplementedError
    return param_dict