#!/usr/bin/env python3

import argparse
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from lab_utils.data.helpers import get_data_loaders
from lab_utils.models import get_model
from lab_utils.utils.logger import create_logger
from lab_utils.utils.utils import *

def get_criterion(args):
  criterion = nn.CrossEntropyLoss()
  return criterion

def get_optimizer(model, args):

  optimizer = optim.AdamW(
    model.parameters(),
    lr = args.lr,
  )

  return optimizer


def get_scheduler(optimizer, args):
  return optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'max', patience = args.lr_patience, verbose = True, factor = args.lr_factor
  )


def model_eval(i_epoch, data, model, args, criterion, store_preds = False):
  with torch.no_grad():
    losses, preds, tgts = [], [], []
    for batch in data:
      loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch)
      losses.append(loss.item())

      pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

      preds.append(pred)
      tgt = tgt.cpu().detach().numpy()
      tgts.append(tgt)

  metrics = {"loss": np.mean(losses)}

  tgts = [l for sl in tgts for l in sl]
  preds = [l for sl in preds for l in sl]
  metrics["acc"] = accuracy_score(tgts, preds)
  
  metrics["precision"] = precision_score(tgts, preds)
  metrics["recall"] = recall_score(tgts, preds)
  metrics["f1"] = f1_score(tgts, preds)

  if store_preds:
    store_preds_to_disk(tgts, preds, args)

  return metrics


def model_forward(i_epoch, model, args, criterion, batch):
  text, attention_mask, tgt = batch

  text, attention_mask, tgt = text.cuda(), attention_mask.cuda(), tgt.cuda()
  out, atts = model(text, attention_mask)

  #print(out.shape)
  #print(text.shape)
  loss = criterion(out, tgt)
  return loss, out, tgt
