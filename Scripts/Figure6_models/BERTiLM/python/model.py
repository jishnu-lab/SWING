"""
Create Hugging Face RoBERTa transformer mode.
REQUIRES: vocab_size              - an integer; the number of tokens in the vocabulary
          max_position_embeddings - an integer; the maximum number of tokens in one sample/sentence/example
          hidden_size             - an integer; the dimensions of the hidden layers (default: 512)
          num_hidden_layers       - an integer; the number of hidden layers (default: 8)
          num_attention_heads     - an integer; the number of attention heads (default: 8)
          hidden_act              - a string; the hidden layer activation function (default: gelu)
          hidden_dropout_prob     - a float; the probability of dropout in the hidden layers (default: 0.4)
          attention_dropout_prob  - a float; the probability of dropout in the attention layers (default: 0.4)
          classifier_dropout_prob - a float; the probability of dropout in the classification layers (default: 0.4)
          position_embedding_type - a string; the type of embedding to use for the position of tokens (default: relative_key)
          
MODIFIES: none
RETURNS:  model - a Roberta transformer for masked language modelling
"""

from __future__ import annotations
from transformers import RobertaConfig, RobertaForMaskedLM

def build_pretrain_model(vocab_size:              int,
                         max_position_embeddings: int,
                         hidden_size:             int = 512,
                         num_hidden_layers:       int = 8,
                         num_attention_heads:     int = 8,
                         hidden_act:              str = 'gelu',
                         hidden_dropout_prob:     float = 0.4,
                         attention_dropout_prob:  float = 0.4,
                         classifier_dropout:      float = 0.4,
                         position_embedding_type: str = 'relative_key'):

    config = RobertaConfig(vocab_size              = vocab_size,
                           max_position_embeddings = max_position_embeddings,
                           hidden_size             = hidden_size,
                           num_hidden_layers       = num_hidden_layers,
                           num_attention_heads     = num_attention_heads,
                           hidden_act              = hidden_act,
                           hidden_dropout_prob     = hidden_dropout_prob,
                           attention_dropout_prob  = attention_dropout_prob,
                           classifier_dropout      = classifier_dropout,
                           position_embedding_type = position_embedding_type)
    
    model = RobertaForMaskedLM(config)
    
    return model
    
