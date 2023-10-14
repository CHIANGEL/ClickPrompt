import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import transformers
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
from transformers import RobertaTokenizer, RobertaForMaskedLM, AutoModel, AutoConfig
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead, BertForPreTraining
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

from ctr_base.ctr_models import BaseModel
from utils import weight_init, load_from_target_dict

logger = logging.getLogger(__name__)


class PLM4CTR_Model(torch.nn.Module):
    def __init__(self, ctr_config, *model_args, **model_kargs):
        super().__init__()
        
        self.ctr_config = ctr_config
        self.model_args = model_kargs["model_args"]
        
        logger.info(f"CTR model fusion mode: {self.model_args.model_fusion}")
        self.init_nlp_model()
        self.init_ctr_model()
        if self.model_args.load_from_path != "no":
            logger.info(f"Load model parameters from {self.model_args.load_from_path}")
            target_state_dict = torch.load(self.model_args.load_from_path)
            load_from_target_dict(self, target_state_dict)
        
        if hasattr(self, "ctr_model") and self.model_args.freeze_ctr:
            logger.info("Freeze the CTR model.")
            for param in self.ctr_model.parameters():
                param.requires_grad = False
        if hasattr(self, "nlp_encoder") and self.model_args.freeze_nlp:
            logger.info("Freeze the NLP model.")
            for param in self.nlp_encoder.parameters():
                param.requires_grad = False
        
        logger.info(f"weighted_add: {self.model_args.weighted_add}")
        if self.model_args.weighted_add:
            logger.info(f"update_alpha: {self.model_args.update_alpha}")
            logger.info(f"init_weighted_alpha: {self.model_args.init_weighted_alpha}")
            self.alpha = nn.Parameter(torch.FloatTensor(1).fill_(self.model_args.init_weighted_alpha), requires_grad=self.model_args.update_alpha)
    
    def init_nlp_model(self):
        self.nlp_config = AutoConfig.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        if self.model_args.model_fusion == "ctr_only":
            logger.info("Do NOT fuse with NLP models, use CTR models only.")
            return
        logger.info(f"Instantiate NLP model: {self.model_args.model_name_or_path}.")
        
        # self.nlp_config.hidden_dropout_prob = 0
        # self.nlp_config.attention_probs_dropout_prob = 0
        # self.nlp_encoder = AutoModel.from_config(self.nlp_config, add_pooling_layer=False)
        # pretrained_model = AutoModel.from_pretrained(self.model_args.model_name_or_path, add_pooling_layer=False)
        # self.nlp_encoder.load_state_dict(pretrained_model.state_dict())
        
        self.nlp_encoder = AutoModel.from_pretrained(self.model_args.model_name_or_path, add_pooling_layer=False)
        if self.model_args.do_ctr:
            self.nlp_head = nn.Sequential(
                nn.Linear(self.nlp_config.hidden_size, self.nlp_config.hidden_size),
                nn.Tanh(),
                nn.Linear(self.nlp_config.hidden_size, 1),
            )
            self.nlp_head.apply(weight_init)
            logger.info(f"Pooler type: {self.model_args.pooler_type}")
        elif self.model_args.do_mlm_only:
            if "roberta" in self.model_args.model_name_or_path:
                self.lm_head = RobertaLMHead(self.nlp_config)
                pretrained_model = RobertaForMaskedLM.from_pretrained(self.model_args.model_name_or_path)
                self.lm_head.load_state_dict(pretrained_model.lm_head.state_dict())
            elif "bert-base-uncased" in self.model_args.model_name_or_path:
                self.lm_head = BertLMPredictionHead(self.nlp_config)
                pretrained_model = BertForPreTraining.from_pretrained(self.model_args.model_name_or_path)
                self.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
            elif "TinyBERT_General_4L_312D" in self.model_args.model_name_or_path:
                self.lm_head = BertLMPredictionHead(self.nlp_config)
            else:
                raise NotImplementedError
    
    def init_ctr_model(self):
        if self.model_args.model_fusion == "nlp_only":
            logger.info("Do NOT fuse with CTR models, use NLP models only.")
            return
        logger.info(f"Instantiate CTR model: {self.ctr_config.ctr_model_name}.")
        setattr(self.ctr_config, "nlp_embed_size", self.nlp_config.hidden_size)
        setattr(self.ctr_config, "nlp_num_layers", self.nlp_config.num_hidden_layers)
        self.ctr_model = BaseModel.from_config(self.ctr_config)
        self.ctr_model.apply(weight_init)
    
    def pool_nlp_outputs(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states
        if self.model_args.pooler_type == "cls":
            return last_hidden[:, 0]
        elif self.model_args.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.model_args.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.model_args.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError
    
    def forward(self,
        input_ids=None,
        ctr_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        do_ctr=False,
        do_mlm_only=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if do_ctr:
            return self.ctr_forward(
                input_ids=input_ids,
                ctr_input_ids=ctr_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif do_mlm_only:
            return self.mlm_forward(
                input_ids=mlm_input_ids,
                ctr_input_ids=ctr_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=mlm_labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    def ctr_forward(self,
        input_ids=None,
        ctr_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        if self.model_args.model_fusion in ["nlp_only", "ctr_only", "add"]:
            # NLP logits
            if self.model_args.model_fusion != "ctr_only":
                nlp_outputs = self.nlp_encoder(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=False,
                    return_dict=True,
                )
                pooler_outputs = self.pool_nlp_outputs(attention_mask, nlp_outputs)
                nlp_logits = self.nlp_head(pooler_outputs)
            
            # CTR logits
            if self.model_args.model_fusion != "nlp_only":
                ctr_logits = self.ctr_model(ctr_input_ids)["logits"]
            
            # Compute logits
            if self.model_args.model_fusion == "add":
                if self.model_args.weighted_add:
                    logits = self.alpha * nlp_logits + (1 - self.alpha) * ctr_logits
                else:
                    logits = nlp_logits + ctr_logits
            elif self.model_args.model_fusion == "nlp_only":
                logits = nlp_logits
            elif self.model_args.model_fusion == "ctr_only":
                logits = ctr_logits
        elif self.model_args.model_fusion in ["prompt", "prompt_add"]:
            # Concatenate the prompts to the input embeddings
            inputs_embeds = self.nlp_encoder.get_input_embeddings()(input_ids)
            _, prompts = self.ctr_model.generate_prompt(ctr_input_ids)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
            
            # Extend the attention mask for prompts
            batch_size = inputs_embeds.shape[0]
            attention_mask = torch.cat([
                torch.full((batch_size, self.ctr_config.num_prompt), 1).to(inputs_embeds.device), 
                attention_mask,
            ], dim=1)
            
            # NLP forward with `inputs_embeds` instead of `input_ids`.
            nlp_outputs = self.nlp_encoder(
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )
            pooler_outputs = self.pool_nlp_outputs(attention_mask, nlp_outputs)
            logits = self.nlp_head(pooler_outputs)
            
            # Add CTR logits if needed
            if self.model_args.model_fusion == "prompt_add":
                if self.model_args.weighted_add:
                    logits = self.alpha * logits + (1 - self.alpha) * self.ctr_model(ctr_input_ids)["logits"]
                else:
                    logits += self.ctr_model(ctr_input_ids)["logits"]
        elif self.model_args.model_fusion in ["prefix", "prefix_add"]:
            batch_size = input_ids.shape[0]
            _, prompts = self.ctr_model.generate_prompt(ctr_input_ids)
            past_key_values = prompts.view(
                batch_size,
                self.ctr_model.config.num_prompt,
                self.nlp_config.num_hidden_layers * 2,
                self.nlp_config.num_attention_heads,
                self.nlp_config.hidden_size // self.nlp_config.num_attention_heads, 
            )
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            prefix_attention_mask = torch.ones(batch_size, self.ctr_config.num_prompt, device=input_ids.device)
            
            # NLP forward with `past_key_values`
            nlp_outputs = self.nlp_encoder(
                input_ids=input_ids,
                attention_mask=torch.cat((prefix_attention_mask, attention_mask), dim=1),
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                output_attentions=output_attentions,
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True,
            )
            pooler_outputs = self.pool_nlp_outputs(attention_mask, nlp_outputs)
            logits = self.nlp_head(pooler_outputs)
            
            # Add CTR logits if needed
            if self.model_args.model_fusion == "prefix_add":
                if self.model_args.weighted_add:
                    logits = self.alpha * logits + (1 - self.alpha) * self.ctr_model(ctr_input_ids)["logits"]
                else:
                    logits += self.ctr_model(ctr_input_ids)["logits"]
        else:
            raise NotImplementedError
        
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.flatten(), labels.float()) # Sigmoid funciton is combined in nn.BCEWithLogitsLoss().

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=torch.sigmoid(logits), # We do sigmoid here for metric computation during evaluation.
        )
    
    def mlm_forward(self,
        input_ids=None,
        ctr_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        assert self.model_args.model_fusion in ["prompt", "nlp_only", "prefix"]
        
        if self.model_args.model_fusion == "prefix":
            batch_size = input_ids.shape[0]
            _, prompts = self.ctr_model.generate_prompt(ctr_input_ids)
            past_key_values = prompts.view(
                batch_size,
                self.ctr_model.config.num_prompt,
                self.nlp_config.num_hidden_layers * 2,
                self.nlp_config.num_attention_heads,
                self.nlp_config.hidden_size // self.nlp_config.num_attention_heads, 
            )
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            prefix_attention_mask = torch.ones(batch_size, self.ctr_config.num_prompt, device=input_ids.device)
            
            # NLP forward with `past_key_values`
            nlp_outputs = self.nlp_encoder(
                input_ids=input_ids,
                attention_mask=torch.cat((prefix_attention_mask, attention_mask), dim=1),
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                output_attentions=output_attentions,
                past_key_values=past_key_values,
                output_hidden_states=False,
                return_dict=True,
            )
        else:
            inputs_embeds = self.nlp_encoder.get_input_embeddings()(input_ids)
            
            if self.model_args.model_fusion == "prompt":
                # Concatenate the prompts to the input embeddings
                _, prompts = self.ctr_model.generate_prompt(ctr_input_ids)
                inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
                
                # Extend the attention mask for prompts
                batch_size = inputs_embeds.shape[0]
                attention_mask = torch.cat([
                    torch.full((batch_size, self.ctr_config.num_prompt), 1).to(inputs_embeds.device), 
                    attention_mask,
                ], dim=1)
                
                # Extend the mlm_labels for prompts
                batch_size = inputs_embeds.shape[0]
                labels = torch.cat([
                    torch.full((batch_size, self.ctr_config.num_prompt), -100).to(inputs_embeds.device), 
                    labels,
                ], dim=1)
            
            # NLP forward with `inputs_embeds` instead of `input_ids`.
            nlp_outputs = self.nlp_encoder(
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=False,
                return_dict=True,
            )
        
        # Calculate loss for MLM
        loss_fct = nn.CrossEntropyLoss()
        labels = labels.view(-1, labels.size(-1))
        logits = self.lm_head(nlp_outputs.last_hidden_state)
        loss = loss_fct(logits.view(-1, self.nlp_config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            # logits=logits, # NOTE: softmax is not applied to logtis here.
        )
