import json
import h5py
import random
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Tuple, Any

import torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase


def load_csv_as_df(file_path, extend_vocab="none"):
    if 'ml-1m-new' in file_path.lower():
        dataset = df_books = pd.read_parquet(file_path)
        fields = ['User ID', 'Gender', 'Age', 'Job', 'Zipcode', "Movie ID", "Movie title", "Movie genre", "labels"]
        dataset = dataset[fields]
    elif 'ml-1m' in file_path.lower():
        dataset = pd.read_csv(file_path, dtype={'Zipcode': 'str'})
        dataset['labels'] = dataset['Label']
        dataset['Film genre'] = dataset['First genre']
        fields = ["User ID", "Gender", "Age", "Job", "Zipcode", "Movie ID", "Title", "Film genre", "labels"]
        dataset = dataset[fields]
        if extend_vocab in ["prefix_none", "raw"]:
            dataset['User ID'] = dataset['User ID'].map(lambda x: f'U{x}')
            dataset['Movie ID'] = dataset['Movie ID'].map(lambda x: f'M{x}')
    elif 'bookcrossing' in file_path.lower():
        dataset = pd.read_csv(file_path, dtype={"labels": int, "User ID": str}, sep="\t")
        fields = ['User ID', 'Location', 'Age', 'ISBN', 'Book title', "Author", "Publication year", "Publisher", "labels"]
        dataset = dataset[fields]
    elif 'az-toys' in file_path.lower():
        dataset = pd.read_parquet(file_path)
        fields = ["User ID", "Item ID", "Category", "Title", "Brand", "labels"]
        dataset = dataset[fields]
    elif 'goodreads' in file_path.lower():
        dataset = df_books = pd.read_parquet(file_path)
        fields = ["User ID","Book ID", "Book title", "Book genres", "Average rating", "Number of book reviews", "Author ID", "Author name", "Number of pages", "eBook flag", "Format", "Publisher", "Publication year", "Work ID", "Media type", "labels"]
        dataset = dataset[fields]
    else:
        raise NotImplementedError
    
    return dataset


class PLM4CTRDataset(Dataset):
    """ PLM4CTR Dataset
    The PLM4CTRDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the templating step.
    """
    def _post_setups(
        self, 
        tokenizer, 
        shuffle_fields: bool,
        meta_data_dir: str, 
        h5_data_dir: str,
        mode: str,
        model_fusion: str,
        do_mlm_only: str,
    ):
        """ Set up the parameters
        Args:
            tokenizer: Tokenizer from HuggingFace
            shuffle_fields: Whether to shuffle the fields for lossless augmentation
            meta_data_dir: The data path for meta CTR data
            h5_data_dir: The data path for CTR data
            mode: `train`/`test`
            model_fusion: Method to fuse CTR & NLP model for prediction
            do_mlm_only: Whether to do MLM pretraining only
        """
        self.tokenizer = tokenizer
        self.shuffle_fields = shuffle_fields
        self.meta_data_dir = meta_data_dir
        self.h5_data_dir = h5_data_dir
        self.mode = mode
        self.model_fusion = model_fusion
        self.do_mlm_only = do_mlm_only
        
        self.get_meta_data()
        self.get_h5_data(mode)
    
    def get_meta_data(self):
        meta_data = json.load(open(self.meta_data_dir, 'r'))
        self.field_names = meta_data['field_names']
        self.feature_count = meta_data['feature_count']
        self.feature_dict = meta_data['feature_dict']
        self.feature_offset = meta_data['feature_offset']
        self.num_fields = len(self.field_names)
        self.input_size = sum(self.feature_count)
       
    def get_h5_data(self, mode):
        assert mode in ["train", "valid", "test"]
        with h5py.File(self.h5_data_dir, 'r') as f:
            mode_name = mode if mode != "valid" else "train"
            self.ctr_X = f[f"{mode_name} data"][:]
            self.ctr_Y = f[f"{mode_name} label"][:]
        if mode == "train" and not self.do_mlm_only: # The validation set is also used for mlm pretraining.
            self.ctr_X = self.ctr_X[:len(self.ctr_X) // 9 * 8]
            self.ctr_Y = self.ctr_Y[:len(self.ctr_Y) // 9 * 8]
        if mode == "valid":
            self.ctr_X = self.ctr_X[len(self.ctr_X) // 9 * 8:]
            self.ctr_Y = self.ctr_Y[len(self.ctr_Y) // 9 * 8:]
        offset = np.array(self.feature_offset).reshape(1, self.num_fields)
        assert self.__len__() == len(self.ctr_X)
        assert self.__len__() == len(self.ctr_Y)
        self.ctr_X += offset

    def _getitem(self, key: Union[int, slice, str], decoded: bool = True, **kwargs) -> Union[Dict, List]:
        """ Get Item from Tabular Data
        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        if self.model_fusion == "ctr_only":
            return self.ctr_X[key], self.ctr_Y[key]
        
        row = self._data.fast_slice(key, 1)

        shuffle_fields = list(row.column_names)
        shuffle_fields.remove("labels")
        if self.shuffle_fields:
            random.shuffle(shuffle_fields)

        shuffled_text = " ".join(
            ["%s is %s." % (field, str(row[field].to_pylist()[0]).strip()) for field in shuffle_fields]
        )

        return self.ctr_X[key], self.ctr_Y[key], shuffled_text


@dataclass
class PLM4CTRDataCollator(DataCollatorWithPadding):
    """ PLM4CTR Data Collator
    Support different training process: do_ctr, do_mlm_only
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    model_fusion: str = "prompt"
    do_ctr: bool = False
    do_mlm_only: bool = False
    mlm_probability: float = 0.15

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.model_fusion != "ctr_only":
            # Tokenize the text
            batch = self.tokenizer(
                [f[2] for f in features],
                padding=self.padding,
                truncation=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            if self.do_mlm_only:
                batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])
        else:
            batch = {}
        
        if self.model_fusion != "nlp_only":
            batch["ctr_input_ids"] = torch.tensor(np.array([f[0] for f in features])).long()
        
        batch["labels"] = torch.tensor(np.array([f[1] for f in features])).long()
        batch["do_ctr"] = self.do_ctr
        batch["do_mlm_only"] = self.do_mlm_only

        return batch
    
    def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels