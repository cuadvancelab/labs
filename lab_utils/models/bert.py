#!/usr/bin/env python3

import torch
import torch.nn as nn

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer # I want to use hugging face
#from pytorch_pretrained_bert.modeling import BertModel, BertConfig
#from pytorch_pretrained_bert.tokenization import BertTokenizer 

import heapq
import numpy as np
import sys

class HateBert(nn.Module):
  def __init__(self, args):
    super(HateBert, self).__init__()
    self.args = args
    configuration = BertConfig()
    configuration.output_attentions = True
    self.bert = BertForSequenceClassification.from_pretrained(args.bert_model, config = configuration)
    # freeze for fine tuning
    if args.fine_tune == True:
      #import IPython; IPython.embed(); exit(1)
      for param in self.bert.bert.parameters():
        param.requires_grad = False

  def forward(self, input_ids, attention_mask):
    #mask = (input_ids != 0).float()
    logits, atts = self.bert(input_ids, attention_mask)
    return logits, atts
