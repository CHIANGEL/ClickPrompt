from responses import target
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import logging
from typing import Dict, Optional, Tuple
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import math
import copy
import sys

sys.path.append("..")
from arguments import Config
from ctr_base.layers import Embeddings, InnerProductLayer, OuterProductLayer, MLPBlock, get_act, \
    ProductLayer, CrossNet, CrossNetV2, FGCNNBlock, SqueezeExtractionLayer, \
    BilinearInteractionLayer, FiGNNBlock, AttentionalPrediction, SelfAttention, \
    CIN, MultiHeadSelfAttention, InnerProductInteraction

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, ctr_model_name='BaseModel', config: Config=None):
        super(BaseModel, self).__init__()
        self.ctr_model_name = ctr_model_name
        self.config = config

    @classmethod
    def from_config(cls, config: Config):
        model_name_lower = config.ctr_model_name.lower()
        if model_name_lower == 'lr':
            model_class = LR
        elif model_name_lower == 'fm':
            model_class = FM
        elif model_name_lower == 'dnn':
            model_class = DNN
        elif model_name_lower == 'deepfm':
            model_class = DeepFM
        elif model_name_lower == 'xdeepfm':
            model_class = xDeepFM
        elif model_name_lower == 'ipnn':
            model_class = IPNN
        elif model_name_lower == 'dcn':
            model_class = DCN
        elif model_name_lower == 'dcnv2':
            model_class = DCNV2
        elif model_name_lower == 'fgcnn':
            model_class = FGCNN
        elif model_name_lower == 'fibinet':
            model_class = FiBiNet
        elif model_name_lower == 'fignn':
            model_class = FiGNN
        elif model_name_lower == 'autoint':
            model_class = AutoInt
        elif model_name_lower == 'transformer':
            model_class = Transformer
        else:
            raise NotImplementedError
        model = model_class(config)
        return model

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters():
            if not count_embedding and 'embedding' in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logger.info(f'total number of parameters: {total_params}')

    def get_outputs(self, logits, labels=None):
        outputs = {
            'logits': logits,
        }
        if labels is not None:
            if self.config.output_dim > 1:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view((-1, self.config.output_dim)), labels.long())
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.float())
            outputs['loss'] = loss

        return outputs
    
    def create_heads(self):
        assert self.config.model_fusion in ["add", "concat", "prompt", "ctr_only", "prompt_add", "prefix", "prefix_add"]
        if self.config.model_fusion == "concat":
            pass
        if self.config.model_fusion in ["prompt", "prompt_add"] or self.config.prompt_usage != "no":
            assert self.config.num_prompt > 0, f"`num_prompt` should be greater than 0, but got {self.config.num_prompt}."
            self.prompt_layers = nn.ModuleList([nn.Linear(self.final_dim, self.config.nlp_embed_size) for _ in range(self.config.num_prompt)])
        if self.config.model_fusion in ["prefix", "prefix_add"]:
            self.prompt_layers = nn.Sequential(
                nn.Linear(self.final_dim, self.config.nlp_embed_size),
                nn.Tanh(),
                nn.Linear(self.config.nlp_embed_size, self.config.num_prompt * self.config.nlp_embed_size * self.config.nlp_num_layers * 2),
            )
        if self.config.model_fusion in ["ctr_only", "add", "prompt_add", "prefix_add"]:
            logger.info(f"CTR model prompt usage: {self.config.prompt_usage}.")
            if self.config.prompt_usage == "no":
                if self.ctr_model_name == "FiGNN":
                    self.fc_out = AttentionalPrediction(self.config)
                elif self.ctr_model_name == "DeepFM":
                    self.fc_out = nn.Linear(self.config.hidden_size, self.config.output_dim)
                else:
                    self.fc_out = nn.Linear(self.final_dim, self.config.output_dim)
            else:
                if self.config.prompt_usage == 'avg':
                    self.fc_out = nn.Linear(self.config.nlp_embed_size, self.config.output_dim)
                elif self.config.prompt_usage == 'cat':
                    if self.config.model_fusion in ["prefix_add"]:
                        input_dim = self.config.num_prompt * self.config.nlp_embed_size * self.config.nlp_num_layers * 2
                    else:
                        input_dim = self.config.num_prompt * self.config.nlp_embed_size
                    self.fc_out = nn.Linear(input_dim, self.config.output_dim)
                elif self.config.prompt_usage == 'moe':
                    self.gate = nn.Sequential(
                        nn.Linear(self.final_dim, self.config.num_prompt),
                        nn.Softmax(dim=1)
                    )
                    self.fc_out = nn.Linear(self.config.nlp_embed_size, self.config.output_dim)
                elif self.config.prompt_usage == 'attn':
                    self.query_vec = nn.Linear(self.config.nlp_embed_size, 1)
                    self.fc_out = nn.Linear(self.config.nlp_embed_size, self.config.output_dim)
                else:
                    raise NotImplementedError
        
    def generate_prompt(self, input_ids):
        final_output = self.get_final_representation(input_ids)
        if isinstance(final_output, tuple):
            final_output = torch.cat(final_output, dim=1)
        if self.config.model_fusion in ["prefix", "prefix_add"]:
            prompts = self.prompt_layers(final_output)
        else:
            prompts = []
            for fc in self.prompt_layers:
                prompts.append(fc(final_output))
            prompts = torch.stack(prompts, dim=1)
        return final_output, prompts
        
    def comput_logits_from_prompt(self, input_ids):
        final_output, prompts = self.generate_prompt(input_ids)
        if self.config.prompt_usage == "avg":
            if self.config.model_fusion in ["prefix_add"]:
                prompts = prompts.view(
                    input_ids.shape[0],
                    self.config.num_prompt * self.config.nlp_num_layers * 2,
                    self.config.nlp_embed_size, 
                )
            logits = self.fc_out(prompts.mean(dim=1))
        elif self.config.prompt_usage == "cat":
            logits = self.fc_out(prompts.view(input_ids.shape[0], -1))
        elif self.config.prompt_usage == "moe":
            gate_value = self.gate(final_output).unsqueeze(-1)
            logits = self.fc_out(torch.sum(gate_value * prompts, dim=1))
        elif self.config.prompt_usage == "attn":
            attn_scores = self.query_vec(prompts).squeeze(-1)
            attn_scores = F.softmax(attn_scores, dim=1).unsqueeze(-1)
            logits = self.fc_out(torch.sum(attn_scores * prompts, dim=1))
        return logits, prompts


class LR(BaseModel):
    def __init__(self, config: Config):
        super().__init__(ctr_model_name='LR', config=config)
        self.embed_w = nn.Embedding(config.input_size, embedding_dim=1)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input_ids, labels=None):
        wx = self.embed_w(input_ids)
        logits = wx.sum(dim=1) + self.bias
        outputs = self.get_outputs(logits, labels)
        return outputs


class FM(BaseModel):
    def __init__(self, config: Config):
        super().__init__(ctr_model_name='FM', config=config)
        
        self.embed = Embeddings(config)
        if config.use_lr:
            self.lr_layer = LR(config)
        self.ip_layer = InnerProductLayer(num_fields=config.num_fields)

    def forward(self, input_ids, labels=None):
        feat_embed = self.embed(input_ids)
        logits = self.ip_layer(feat_embed)
        if self.config.use_lr:
            lr_logits = self.lr_layer(input_ids)['logits']
            logits += lr_logits
        outputs = self.get_outputs(logits, labels)
        return outputs


class DNN(BaseModel):
    def __init__(self, config: Config):
        super(DNN, self).__init__(ctr_model_name='DNN', config=config)
        
        self.embed = Embeddings(config)
        self.dnn = MLPBlock(input_dim=config.embed_size * config.num_fields,
                            hidden_size=config.hidden_size,
                            num_hidden_layers=config.num_hidden_layers,
                            hidden_dropout_rate=config.hidden_dropout_rate,
                            hidden_act=config.hidden_act)
        self.final_dim = config.hidden_size
        self.create_heads()

    def get_final_representation(self, input_ids):
        feat_embed = self.embed(input_ids)
        final_output = self.dnn(torch.flatten(feat_embed, 1))
        return final_output

    def forward(self, input_ids, labels=None):
        if self.config.prompt_usage == "no":
            final_output = self.get_final_representation(input_ids)
            logits = self.fc_out(final_output)
        else:
            logits, _ = self.comput_logits_from_prompt(input_ids)
        outputs = self.get_outputs(logits, labels)
        return outputs


class DeepFM(BaseModel):
    def __init__(self, config: Config):
        super(DeepFM, self).__init__(ctr_model_name='DeepFM', config=config)
        
        self.embed = Embeddings(config)
        if config.use_lr:
            self.lr_layer = LR(config)
        self.dnn = MLPBlock(input_dim=config.num_fields * config.embed_size,
                            hidden_size=config.hidden_size,
                            num_hidden_layers=config.num_hidden_layers,
                            hidden_dropout_rate=config.hidden_dropout_rate,
                            hidden_act=config.hidden_act)
        self.ip_layer = InnerProductLayer(num_fields=config.num_fields)
        self.final_dim = config.hidden_size
        if self.config.model_fusion in ["prompt", "prompt_add", "prefix", "prefix_add"] or self.config.prompt_usage != "no":
            self.final_dim += 1
        self.create_heads()

    def get_final_representation(self, input_ids):
        feat_embed = self.embed(input_ids)
        dnn_vec = self.dnn(feat_embed.flatten(start_dim=1))
        fm_logits = self.ip_layer(feat_embed)
        if self.config.use_lr:
            lr_logits = self.lr_layer(input_ids)['logits']
            fm_logits += lr_logits
        return dnn_vec, fm_logits

    def forward(self, input_ids, labels=None):
        if self.config.prompt_usage == "no":
            dnn_vec, fm_logits = self.get_final_representation(input_ids)
            logits = self.fc_out(dnn_vec)
            logits += fm_logits
        else:
            logits, _ = self.comput_logits_from_prompt(input_ids)
        outputs = self.get_outputs(logits, labels)
        return outputs
        

class xDeepFM(BaseModel):
    def __init__(self, config: Config):
        super(xDeepFM, self).__init__(ctr_model_name='xDeepFM', config=config)

        self.embed = Embeddings(config)
        input_dim = config.num_fields * config.embed_size
        cin_layer_units = [int(c) for c in config.cin_layer_units.split('-')]
        self.cin = CIN(config.num_fields, cin_layer_units)
        self.final_dim = sum(cin_layer_units)
        if config.num_hidden_layers > 0:
            self.dnn = MLPBlock(input_dim=input_dim,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                hidden_dropout_rate=config.hidden_dropout_rate,
                                hidden_act=config.hidden_act)
            self.final_dim += config.hidden_size
        self.create_heads()
    
    def get_final_representation(self, input_ids):
        feat_embed = self.embed(input_ids)
        final_vec = self.cin(feat_embed)
        if self.config.num_hidden_layers > 0:
            dnn_vec = self.dnn(feat_embed.flatten(start_dim=1))
            final_vec = torch.cat([final_vec, dnn_vec], dim=1)
        return final_vec

    def forward(self, input_ids, labels=None):
        if self.config.prompt_usage == "no":
            final_vec = self.get_final_representation(input_ids)
            logits = self.fc_out(final_vec)
        else:
            logits, _ = self.comput_logits_from_prompt(input_ids)
        outputs = self.get_outputs(logits, labels)
        return outputs


class IPNN(BaseModel):
    def __init__(self, config: Config):
        super(IPNN, self).__init__(ctr_model_name='IPNN', config=config)
        
        self.embed = Embeddings(config)
        self.inner_product_layer = InnerProductInteraction(config.num_fields, output="inner_product")
        dnn_input_dim = config.num_fields * (config.num_fields - 1) // 2 + config.num_fields * config.embed_size
        self.dnn = MLPBlock(input_dim=dnn_input_dim,
                            hidden_size=config.hidden_size,
                            num_hidden_layers=config.num_hidden_layers,
                            hidden_dropout_rate=config.hidden_dropout_rate,
                            hidden_act=config.hidden_act)
        self.final_dim = config.hidden_size
        self.create_heads()

    def get_final_representation(self, input_ids):
        feat_embed = self.embed(input_ids)
        inner_products = self.inner_product_layer(feat_embed)
        dense_input = torch.cat([feat_embed.flatten(start_dim=1), inner_products], dim=1)
        final_output = self.dnn(dense_input)
        return final_output

    def forward(self, input_ids, labels=None):
        if self.config.prompt_usage == "no":
            final_output = self.get_final_representation(input_ids)
            logits = self.fc_out(final_output)
        else:
            logits, _ = self.comput_logits_from_prompt(input_ids)
        outputs = self.get_outputs(logits, labels)
        return outputs


class DCN(BaseModel):
    def __init__(self, config: Config):
        super(DCN, self).__init__(ctr_model_name='DCN', config=config)
        
        self.config = config
        self.embed = Embeddings(config)
        input_dim = config.num_fields * config.embed_size
        self.cross_net = CrossNet(input_dim, config.num_cross_layers)
        self.final_dim = input_dim
        if config.num_hidden_layers > 0:
            self.parallel_dnn = MLPBlock(input_dim=input_dim,
                                        hidden_size=config.hidden_size,
                                        num_hidden_layers=config.num_hidden_layers,
                                        hidden_dropout_rate=config.hidden_dropout_rate,
                                        hidden_act=config.hidden_act)
            self.final_dim += config.hidden_size
        self.create_heads()
    
    def get_final_representation(self, input_ids):
        feat_embed = self.embed(input_ids).flatten(start_dim=1)
        final_output = self.cross_net(feat_embed)
        if self.config.num_hidden_layers > 0:
            dnn_output = self.parallel_dnn(feat_embed)
            final_output = torch.cat([final_output, dnn_output], dim=-1)
        return final_output

    def forward(self, input_ids, labels=None):
        if self.config.prompt_usage == "no":
            final_output = self.get_final_representation(input_ids)
            logits = self.fc_out(final_output)
        else:
            logits, _ = self.comput_logits_from_prompt(input_ids)
        outputs = self.get_outputs(logits, labels)
        return outputs


class DCNV2(BaseModel):
    def __init__(self, config: Config):
        super(DCNV2, self).__init__(ctr_model_name='DCNV2', config=config)
        
        self.config = config
        self.embed = Embeddings(config)
        input_dim = config.num_fields * config.embed_size
        self.cross_net = CrossNetV2(input_dim, config.num_cross_layers)
        self.final_dim = input_dim
        if config.num_hidden_layers > 0:
            self.parallel_dnn = MLPBlock(input_dim=input_dim,
                                        hidden_size=config.hidden_size,
                                        num_hidden_layers=config.num_hidden_layers,
                                        hidden_dropout_rate=config.hidden_dropout_rate,
                                        hidden_act=config.hidden_act)
            self.final_dim += config.hidden_size
        self.create_heads()
    
    def get_final_representation(self, input_ids):
        feat_embed = self.embed(input_ids).flatten(start_dim=1)
        final_output = self.cross_net(feat_embed)
        if self.config.num_hidden_layers > 0:
            dnn_output = self.parallel_dnn(feat_embed)
            final_output = torch.cat([final_output, dnn_output], dim=-1)
        return final_output

    def forward(self, input_ids, labels=None):
        if self.config.prompt_usage == "no":
            final_output = self.get_final_representation(input_ids)
            logits = self.fc_out(final_output)
        else:
            logits, _ = self.comput_logits_from_prompt(input_ids)
        outputs = self.get_outputs(logits, labels)
        return outputs
        

class FGCNN(BaseModel):
    def __init__(self, config: Config):
        super(FGCNN, self).__init__(ctr_model_name='fgcnn', config=config)
        
        self.embed = Embeddings(config)
        if not self.config.share_embedding:
            self.fg_embed = Embeddings(config)
        channels = [int(c) for c in config.channels.split('|')]
        kernel_heights = [int(c) for c in config.kernel_heights.split('|')]
        pooling_sizes = [int(c) for c in config.pooling_sizes.split('|')]
        recombined_channels = [int(c) for c in config.recombined_channels.split('|')]
        self.fgcnn_layer = FGCNNBlock(
            config.num_fields,
            config.embed_size,
            channels=channels,
            kernel_heights=kernel_heights,
            pooling_sizes=pooling_sizes,
            recombined_channels=recombined_channels,
            activation=config.conv_act,
            batch_norm=True
        )
        self.final_dim, total_features = self.compute_input_dim(config.embed_size,
                                                                config.num_fields,
                                                                channels,
                                                                pooling_sizes,
                                                                recombined_channels)
        self.ip_layer = InnerProductLayer(total_features, output='inner_product')
        if config.num_hidden_layers > 0:
            self.dnn = MLPBlock(input_dim=self.final_dim,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                hidden_dropout_rate=config.hidden_dropout_rate,
                                hidden_act=config.hidden_act)
            self.final_dim = config.hidden_size
        self.create_heads()

    def compute_input_dim(self,
                          embedding_dim,
                          num_fields,
                          channels,
                          pooling_sizes,
                          recombined_channels):
        total_features = num_fields
        input_height = num_fields
        for i in range(len(channels)):
            input_height = int(math.ceil(input_height / pooling_sizes[i]))
            total_features += input_height * recombined_channels[i]
        final_dim = int(total_features * (total_features - 1) / 2) \
                  + total_features * embedding_dim
        return final_dim, total_features

    def get_final_representation(self, input_ids):
        feat_embed = self.embed(input_ids)
        if not self.config.share_embedding:
            feat_embed2 = self.fg_embed(input_ids)
        else:
            feat_embed2 = feat_embed
        conv_in = torch.unsqueeze(feat_embed2, 1)
        new_feat_embed = self.fgcnn_layer(conv_in)
        combined_feat_embed = torch.cat([feat_embed, new_feat_embed], dim=1)
        ip_vec = self.ip_layer(combined_feat_embed)
        final_output = torch.cat([combined_feat_embed.flatten(start_dim=1), ip_vec], dim=1)
        if self.config.num_hidden_layers > 0:
            final_output = self.dnn(final_output)
        return final_output
        
    def forward(self, input_ids, labels=None):
        if self.config.prompt_usage == "no":
            final_output = self.get_final_representation(input_ids)
            logits = self.fc_out(final_output)
        else:
            logits, _ = self.comput_logits_from_prompt(input_ids)
        outputs = self.get_outputs(logits, labels)
        return outputs


class FiBiNet(BaseModel):
    def __init__(self, config: Config):
        super(FiBiNet, self).__init__(ctr_model_name='FiBiNet', config=config)
        
        self.embed = Embeddings(config)
        self.senet_layer = SqueezeExtractionLayer(config)
        self.bilinear_layer = BilinearInteractionLayer(config)
        self.final_dim = config.num_fields * (config.num_fields - 1) * config.embed_size
        if config.num_hidden_layers > 0:
            self.dnn = MLPBlock(input_dim=self.final_dim,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                hidden_dropout_rate=config.hidden_dropout_rate,
                                hidden_act=config.hidden_act)
            self.final_dim = config.hidden_size
        self.create_heads()
    
    def get_final_representation(self, input_ids):
        feat_embed = self.embed(input_ids)
        senet_embed = self.senet_layer(feat_embed)
        bilinear_p = self.bilinear_layer(feat_embed)
        bilinear_q = self.bilinear_layer(senet_embed)
        final_output = torch.flatten(torch.cat([bilinear_p, bilinear_q], dim=1), start_dim=1)
        if self.config.num_hidden_layers > 0:
            final_output = self.dnn(final_output)
        return final_output

    def forward(self, input_ids, labels=None):
        if self.config.prompt_usage == "no":
            final_output = self.get_final_representation(input_ids)
            logits = self.fc_out(final_output)
        else:
            logits, _ = self.comput_logits_from_prompt(input_ids)
        outputs = self.get_outputs(logits, labels)
        return outputs


class FiGNN(BaseModel):
    def __init__(self, config: Config):
        super(FiGNN, self).__init__(ctr_model_name='FiGNN', config=config)
        
        self.embed = Embeddings(config)
        self.fignn = FiGNNBlock(config)
        self.final_dim = config.num_fields * config.embed_size
        self.create_heads()

    def get_final_representation(self,input_ids):
        feat_embed = self.embed(input_ids)
        final_output = self.fignn(feat_embed).flatten(start_dim=1)
        return final_output

    def forward(self, input_ids, labels=None):
        if self.config.prompt_usage == "no":
            final_output = self.get_final_representation(input_ids)
            final_output = final_output.view(-1, self.config.num_fields, self.config.embed_size)
            logits = self.fc_out(final_output)
        else:
            logits, _ = self.comput_logits_from_prompt(input_ids)
        outputs = self.get_outputs(logits, labels)
        return outputs


class AutoInt(BaseModel):
    def __init__(self, config: Config):
        super(AutoInt, self).__init__(ctr_model_name='AutoInt', config=config)

        self.embed = Embeddings(config)
        self.self_attention = nn.Sequential(
            *[MultiHeadSelfAttention(config.embed_size if i == 0 else config.num_attn_heads * config.attn_size,
                                     attention_dim=config.attn_size, 
                                     num_heads=config.num_attn_heads, 
                                     dropout_rate=config.attn_probs_dropout_rate, 
                                     use_residual=config.res_conn, 
                                     use_scale=config.attn_scale, 
                                     layer_norm=False,
                                     align_to='output') 
             for i in range(config.num_attn_layers)])
        self.final_dim = config.num_fields * config.attn_size * config.num_attn_heads
        if config.num_dnn_layers > 0:
            self.dnn = MLPBlock(input_dim=config.num_fields * config.embed_size,
                                hidden_size=config.dnn_size,
                                num_hidden_layers=config.num_dnn_layers,
                                hidden_dropout_rate=config.dnn_drop,
                                hidden_act=config.dnn_act)
            self.final_dim += config.dnn_size
        self.create_heads()
    
    def get_final_representation(self, input_ids):
        feat_embed = self.embed(input_ids)
        attention_out = self.self_attention(feat_embed)
        final_output = torch.flatten(attention_out, start_dim=1)
        if self.config.num_dnn_layers > 0:
            dnn_output = self.dnn(feat_embed.flatten(start_dim=1))
            final_output = torch.cat([final_output, dnn_output], dim=-1)
        return final_output
            
    def forward(self, input_ids, labels=None):
        if self.config.prompt_usage == "no":
            final_output = self.get_final_representation(input_ids)
            logits = self.fc_out(final_output)
        else:
            logits, _ = self.comput_logits_from_prompt(input_ids)
        outputs = self.get_outputs(logits, labels)
        return outputs


class Transformer(BaseModel):
    def __init__(self, config: Config):
        super(Transformer, self).__init__(ctr_model_name='trans', config=config)
        
        self.embed = Embeddings(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attn_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_rate,
            activation=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=config.norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_hidden_layers)
        self.final_dim = config.num_fields * config.embed_size
        self.create_heads()
    
    def get_final_representation(self, input_ids):
        feat_embed = self.embed(input_ids)
        enc_output = self.encoder(feat_embed)
        final_output = enc_output.flatten(start_dim=1)
        return final_output

    def forward(self, input_ids, labels=None):
        if self.config.prompt_usage == "no":
            final_output = self.get_final_representation(input_ids)
            logits = self.fc_out(final_output)
        else:
            logits, _ = self.comput_logits_from_prompt(input_ids)
        outputs = self.get_outputs(logits, labels)
        return outputs
        