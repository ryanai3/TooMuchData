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
#from dataset import CharAnnoDataset
from story import StoryDataset
from utils.vocabulary import Vocabulary

from itertools import islice

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
#    print("#words: {0}".format(len(sorted_vocab)))
#    print("cutoff oov = {0}".format(sorted_vocab[vocab_size]))
#    print("oov words right after cutoff:")
#    print([x[0] for x in sorted_vocab[vocab_size:vocab_size + 50]])
#    print("randomly sampled oov words:")
#    print(random.sample([x[0] for x in sorted_vocab[vocab_size:]], 50))
    oov_percent = (100.0 * oov_counts[0]) / sum(cutoff_counts)
#    print("oov % = {0:.5f}".format(oov_percent))
    all_counts = soseos_counts_estim + cutoff_counts + oov_counts
    params["vocab_counts"] = all_counts
    as_hash = {i: v for (i, v) in enumerate(all_counts)}
    huff_tree = chainer.links.BinaryHierarchicalSoftmax.create_huffman_tree(as_hash) 
    print("loaded huffman tree")
  return huff_tree

def read_params(args):
  with open(args.params.strip(), 'r') as f:
    params = json.loads(f.read())
  print(params)
  return params

def load_unspecified_params(params):
  params["huffman_tree"] = get_huffman_tree(params)
  return params

def make_and_fill_output_dir(args, params):
  out_dir = args.out.rstrip()
  if not path.exists(out_dir):
    makedirs(out_dir)
  save_params(path.join(out_dir, "params_run.json"), params)
  return out_dir

def save_params(fname, params):
  with open(fname, 'w') as f:
    json.dump(params, f)

def gen_model(params, args):
  model = models.model_hash[params["arch"]](params)
  resume_from = args.resume_from
  if resume_from:
    print('Loading model from', args.resume_from)
    model.load_state_dict(torch.load(resume_from.strip()))
  else:
    model.init_params()
  return model

optim_dict = {
  'adam': torch.optim.Adam,
  'sgd': torch.optim.SGD,
  'adadelta': torch.optim.Adadelta
}
def setup_optimizer(model, params):
  optim_args = {}
  try:
    optim_args = params['optim_args']
  except:
    pass
  optimizer = optim_dict[params['optim']](model.parameters(), **optim_args)
#  optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay =0.01)
#  optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.1)
#  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
#  optimizer = torch.optim.Adam(model.parameters(), weight_decay = 0.01)
#  optimizer = torch.optim.Rprop(model.parameters())
#  optimizer = torch.optim.Adam(model.parameters())
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

def run_epoch(model, optimizer, datasets, dataset_stats, out_dir, batch_size, bptt_len, epoch_num):
  train_iter = datasets["train"].get_batched_iter(batch_size)
  dev_iter = datasets["dev"].get_batched_iter(batch_size)
#  n_words_train = 2369091072
#  n_words_dev = 573618777

  n_words_train = dataset_stats["train"]["words"]
  n_words_dev = dataset_stats["dev"]["words"]

  if epoch_num == 1 and False: #and False: # first epoch!
    dev_iter = datasets["dev"].get_batched_iter(batch_size)
    dev_epoch(
      model = model,
      data_iter = dev_iter,
      desc = "Val Epoch: {0}".format(0),
      data_iter_len = n_words_dev // batch_size,
    )

  train_epoch(
    model = model,
    optimizer = optimizer,
    data_iter = train_iter,
    desc = "Train Epoch: {0}".format(epoch_num),
    data_iter_len = n_words_train // batch_size,
    dev_dataset = datasets["dev"],
    bptt_len = bptt_len,
    out_dir = out_dir
  )

  torch.save(model.state_dict(), path.join(out_dir, "snapshot_epoch_{0}".format(epoch_num)))
  dev_epoch(
    model = model,
    data_iter = dev_iter,
    desc = "Val Epoch: {0}".format(epoch_num),
    data_iter_len = n_words_dev // batch_size,
  )


def train_epoch(model, optimizer, data_iter, desc, data_iter_len, dev_dataset, bptt_len, out_dir):#, dev_iter):
  p_bar = tqdm(
    iterable = data_iter,
    desc = desc,
    total = data_iter_len,
    smoothing=0,
    dynamic_ncols = True,
    position=0
  )

  loss = 0.0
  epoch_loss = 0.0
  rsv_ls = 0.0 # randomly sampled val
  ce_100 = torch.zeros(100)
  px_100 = torch.zeros(100)
  model.init_state(train = True)
  model.train()
  model.zero_grad()

  for i, batch in enumerate(p_bar):
    if i % bptt_len == 0 and i != 0:
      loss = loss / bptt_len
      loss.backward()
      optimizer.step()
      model.stop_bptt()
      model.zero_grad()
      dloss = loss.data[0]
      ce_100[i//bptt_len % 100] = dloss
      curr_ce_100 = ce_100.sum()/min(i//bptt_len, 100)
      epoch_loss += dloss
      ep_loss_norm = epoch_loss/(i/bptt_len)
      p_bar.set_postfix(
        ls =loss.data[0], 
        ls_ep = "%.3f" % ep_loss_norm, 
        px = "%.3f" % (2.0 ** loss.data[0]), 
        px_ep = "%.3f" % (2.0 ** ep_loss_norm), 
        ls_100 = "%.3f" % curr_ce_100, 
        px_100 = "%.3f" % (2.0 ** curr_ce_100), 
        vls="%.3f" % rsv_ls, 
        vpx= "%.3f" % (2.0 ** rsv_ls)
      )
      loss = 0.0
      if (i // bptt_len) % ((20 * 1000) // bptt_len) == 0:
        torch.save(model.state_dict(), path.join(out_dir, "snapshot_epoch_{0}_{1}".format(desc, i)))
        try:
          model.eval()
          rsv_ls = dev_epoch(model, islice(dev_iter, 40 * 1000), desc + "_minidev", 40 * 1000, position=1)
          print("")
        except:
          dev_iter = dev_dataset.get_batched_iter(data_iter.batch_size)
        model.init_state(train = True)
        model.train()
        model.zero_grad()
    loss += model(batch)
    del batch
    i += 1
      #    except Exception as exc:
      #      import pdb; pdb.set_trace()
      #      print("Error")
      #
def dev_epoch(model, data_iter, desc, data_iter_len, position=0):
  p_bar = tqdm(
    iterable = data_iter,
    desc = desc,
    total = data_iter_len,
    dynamic_ncols = True,
    position=position
  )

  loss = 0.0
  ep_ls = 0.0
  model.init_state(train = False)
  model.eval()
  for i, batch in enumerate(p_bar):
    try:
      loss += model(batch).data
      ep_ls = loss[0]/(i + 1)
      p_bar.set_postfix(ep_ls=ep_ls, ep_px = "%.3f" % (2.0 ** ep_ls))
    except Exception as exc:
      import pdb; pdb.set_trace()
      print("Error")   
  print("")
  return ep_ls

def main():
  parser = opts.get_parser()
  args = parser.parse_args()
  params = read_params(args)
  out_dir = make_and_fill_output_dir(args, params)
  load_unspecified_params(params)

  # Initialize model and move it to gpu
  torch.cuda.set_device(args.gpu)
  model = gen_model(params, args)
  print(model)
  model.cuda(args.gpu)

  # Load the datasets and setup optimizer
  datasets, dataset_stats = load_data(args.data)  
  optimizer = setup_optimizer(model, params)
  
  if args.eval:
    dev_epoch(
      model = model,
      data_iter = datasets["dev"].get_batched_iter(params["batch_size"]),
      desc = "Evaluation:",
      data_iter_len = dataset_stats["dev"]["words"] // params["batch_size"]
    )
  else:
    n_epochs = int(args.n_epochs)
    for i in range(1, n_epochs + 1):
      run_epoch(
        model = model,
        optimizer = optimizer,
        datasets = datasets,
        dataset_stats = dataset_stats,
        out_dir = out_dir,
        batch_size = params["batch_size"],
        bptt_len = params["bptt_len"],
        epoch_num = i
      )
#    save_model(path.join(args.out, "snapshot_{0}".format(i)), model)
    
if __name__ == '__main__':
  main()
