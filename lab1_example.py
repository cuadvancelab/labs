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
from lab_utils.train_utils import *

def get_args(parser):
  parser.add_argument("--name", type=str, default='combined_agnostic') # giving the same name for tuning.
  parser.add_argument("--batch_sz", type=int, default = 64)
  parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
  parser.add_argument("--data_path", type=str, default="/home/nvishwa/pytorch_stuff/HateSpeech/NAACL/dataset/combined")
  parser.add_argument("--lr", type=float, default=1e-4) # normal one
  parser.add_argument("--lr_factor", type=float, default=0.5)
  parser.add_argument("--lr_patience", type=int, default=2)
  parser.add_argument("--max_epochs", type=int, default=50)
  parser.add_argument("--n_workers", type=int, default=12)
  parser.add_argument("--patience", type=int, default=10)
  parser.add_argument("--savedir", type=str, default="/home/nvishwa/pytorch_stuff/HateSpeech/NAACL/savedir/")
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--weight_classes", type=int, default=1)
  parser.add_argument("--model", type=str, default='hate_bert')
  parser.add_argument("--max_len", type = int, default = 128)
  parser.add_argument("--train", type = bool, default = True)
  parser.add_argument("--phase", type = str, default = 'train')
  parser.add_argument("--sample_weights", type = list, default = [47287, 11000]) # agnostic
  parser.add_argument("--fine_tune", type = bool, default = False)

def train(args):

  set_seed(args.seed)
  args.savedir = os.path.join(args.savedir, args.name)
  os.makedirs(args.savedir, exist_ok = True)

  train_loader, val_loader, test_loader = get_data_loaders(args)

  model = get_model(args)
  criterion = get_criterion(args)
  optimizer = get_optimizer(model, args)
  scheduler = get_scheduler(optimizer, args)

  logger = create_logger('%s/logfile.log' % args.savedir, args)
  logger.info(model)
  model.cuda()

  torch.save(args, os.path.join(args.savedir, 'args.pt'))

  start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

  if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
    checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
    start_epoch = checkpoint["epoch"]
    n_no_improve = checkpoint["n_no_improve"]
    best_metric = checkpoint["best_metric"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

  logger.info("Training..")
  for i_epoch in range(start_epoch, args.max_epochs):
    train_losses = []
    model.train()
    optimizer.zero_grad()

    for batch in tqdm(train_loader, total=len(train_loader)):
      loss, _, _ = model_forward(i_epoch, model, args, criterion, batch)

      train_losses.append(loss.item())
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    model.eval()
    metrics = model_eval(i_epoch, val_loader, model, args, criterion)
    logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
    log_metrics("Val", metrics, args, logger)

    tuning_metric = (metrics["acc"])

    scheduler.step(tuning_metric)
    is_improvement = tuning_metric > best_metric
    if is_improvement:
      best_metric = tuning_metric
      n_no_improve = 0
    else:
      n_no_improve += 1

    save_checkpoint(
      {
        "epoch": i_epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "n_no_improve": n_no_improve,
        "best_metric": best_metric,
      },
      is_improvement,
      args.savedir,
    )

    if n_no_improve >= args.patience:
      logger.info("No improvement. Breaking out of loop.")
      break

  # Test best model
  load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
  model.eval()
  test_metrics = model_eval(
    np.inf, test_loader, model, args, criterion, store_preds=True
  )
  log_metrics(f"Test - test", test_metrics, args, logger)


def cli_main():
  parser = argparse.ArgumentParser(description = 'Train Models')
  get_args(parser)
  args, remaining_args = parser.parse_known_args()
  assert remaining_args == [], remaining_args
  train(args)


if __name__ == "__main__":
  import warnings

  warnings.filterwarnings("ignore")

  cli_main()

