#!/usr/bin/python3

import numpy as np
import json
import random
from tqdm import tqdm
from os import path, makedirs

import torch

import chainer
from chainer.serializers import load_npz as load_model
from chainer.serializers import save_npz as save_model

import opts
import models
from dataset import StoryDataset
from utils.vocabulary import Vocabulary

def get_huffman_tree(params):
  if "huff_tree_loc" in params:
    with open(params["huff_tree_loc"], 'rb') as f:
      huff_tree = pickle.load(f)
  else:
    vocab_size = params["n_vocab"]
    soseos_counts_estim = [40114695 for i in range(2)]
    vocab = Vocabulary.load("/hdd/data/nlp/raw/unzipped/ff15_book/vocabs/final_vocab") 
    sorted_vocab = sorted(vocab.items(), key = lambda x: x[1], reverse = True)
    sorted_counts = [x[1] for x in sorted_vocab]
    cutoff_counts = sorted_counts[0:vocab_size]  
    oov_counts = [sum(sorted_counts[vocab_size:])]
    print("#words: {0}".format(len(sorted_vocab)))
    print("cutoff oov = {0}".format(sorted_vocab[vocab_size]))
    print("oov words right after cutoff:")
    print([x[0] for x in sorted_vocab[vocab_size:vocab_size + 50]])
    print("randomly sampled oov words:")
    print(random.sample([x[0] for x in sorted_vocab[vocab_size:]], 50))
    oov_percent = (100.0 * oov_counts[0]) / sum(cutoff_counts)
    print("oov % = {0:.5f}".format(oov_percent))
    all_counts = soseos_counts_estim + cutoff_counts + oov_counts
    params["vocab_counts"] = all_counts
    as_hash = {i: v for (i, v) in enumerate(all_counts)}
    huff_tree = chainer.links.BinaryHierarchicalSoftmax.create_huffman_tree(as_hash) 
  return huff_tree

def read_params(args):
  with open(args.params.strip(), 'r') as f:
    params = json.loads(f.read())
  return params

def load_unspecified_params(params):
  params["huffman_tree"] = get_huffman_tree(params)
  return params

def make_and_fill_output_dir(args, params):
  out_dir = args.out.rstrip()
  if not path.exists(out_dir):
    makedirs(out_dir)
  save_params(path.join(out_dir, "params_run.json"), params)

def save_params(fname, params):
  with open(fname, 'w') as f:
    json.dump(params, f)

def gen_model(params, resume_from = None):
  model = models.model_hash[params["arch"]](params)    
  if resume_from:
    print('Loading model from', args.resume_from)
    load_model(args.resume_from, model)
  return model

def setup_optimizer(model):
  optimizer = torch.optim.Adam(model.parameters())
  return optimizer

def load_data(data_dir_str):
  data_dir = data_dir_str.rstrip()
  datasets = {sub_name: StoryDataset(path.join(data_dir, sub_name)) 
      for sub_name in ["train", "dev"]}
  dataset_stats = {sub_name: 
    json.loads(open(path.join(data_dir, "{0}_stats.json".format(sub_name)), 'r').read())
      for sub_name in ["train", "dev"]
  }
  return datasets, dataset_stats

def run_epoch(model, optimizer, datasets, dataset_stats, batch_size, bptt_len, epoch_num):
  train_iter = datasets["train"].get_wordxy_batch_iter(batch_size)
  dev_iter = datasets["dev"].get_wordxy_batch_iter(batch_size)

#  n_words_train = 2369091072
#  n_words_dev = 573618777

  n_words_train = dataset_stats["train"]["words"]
  n_words_dev = dataset_stats["dev"]["words"]

  if epoch_num == 1 and False: #and False: # first epoch!
    dev_epoch(
      model = model,
      data_iter = dev_iter,
      desc = "Val Epoch: {0}".format(0),
      data_iter_len = n_words_dev // batch_size,
    )
    dev_iter = datasets["dev"].get_wordxy_batch_iter(batch_size)

  train_epoch(
    model = model,
    optimizer = optimizer,
    data_iter = train_iter,
    desc = "Train Epoch: {0}".format(epoch_num),
    data_iter_len = n_words_train // batch_size,
    bptt_len = bptt_len
  )

  dev_epoch(
    model = model,
    data_iter = dev_iter,
    desc = "Val Epoch: {0}".format(epoch_num),
    data_iter_len = n_words_dev // batch_size,
  )


def train_epoch(model, optimizer, data_iter, desc, data_iter_len, bptt_len):
  p_bar = tqdm(
    iterable = data_iter,
    desc = desc,
    total = data_iter_len,
    smoothing=0,
    dynamic_ncols = True
  )

  loss = 0.0
  epoch_loss = 0.0
  model.init_state(train = True)
  model.zero_grad()

  for i, batch in enumerate(p_bar):
#  for batch in p_bar:
    try:
      if i % bptt_len == 0 and i != 0:
        loss = loss / bptt_len
        loss.backward()
        optimizer.step()
        model.stop_bptt()
        model.zero_grad()
        epoch_loss += loss.data[0]
        ep_loss_norm = epoch_loss/(i/bptt_len)
        p_bar.set_postfix(ls=loss.data[0], ep_ls = ep_loss_norm, px = 2 ** loss.data[0], ep_px = 2 ** ep_loss_norm)
        loss = 0.0
      loss += model(batch)
      i += 1
    except Exception as exc:
      import pdb; pdb.set_trace()
      print("Error")

def dev_epoch(model, data_iter, desc, data_iter_len):
  p_bar = tqdm(
    iterable = data_iter,
    desc = desc,
    total = data_iter_len,
    dynamic_ncols = True
  )

  loss = 0.0
  model.init_state(train = False)
  for i, batch in enumerate(p_bar):
    try:
      loss += model(batch, train=False).data
      ep_ls = loss[0]/(i + 1)
      p_bar.set_postfix(ep_ls=ep_ls, ep_px = 2 ** ep_ls)
    except Exception as exc:
      import pdb; pdb.set_trace()
      print("Error")   
  print("")

def main():
  parser = opts.get_parser()
  args = parser.parse_args()
  params = read_params(args)
  make_and_fill_output_dir(args, params)
  load_unspecified_params(params)

  # Initialize model and move it to gpu
  torch.cuda.set_device(args.gpu)
  model = gen_model(params, args.resume_from)
  model.cuda(args.gpu)
  model.init_params()

  # Load the datasets and setup optimizer
  datasets, dataset_stats = load_data(args.data)  
  optimizer = setup_optimizer(model)

  n_epochs = int(args.n_epochs)
  for i in range(1, n_epochs + 1):
    run_epoch(
      model = model,
      optimizer = optimizer,
      datasets = datasets,
      dataset_stats = dataset_stats,
      batch_size = params["batch_size"],
      bptt_len = params["bptt_len"],
      epoch_num = i
    )
#    save_model(path.join(args.out, "snapshot_{0}".format(i)), model)
    
if __name__ == '__main__':
  main()
