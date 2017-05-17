#!/usr/bin/python
import argparse
import fileinput
from os import path
from functools import partial
import numpy as np
import sys

from utils.vocabulary import Vocabulary
from utils import SOS, EOS

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-v', '--vocab_prefix', help = 'prefix for vocab files')
  parser.add_argument('-o', '--output_dir', help = 'output directory for npy files')
  parser.add_argument('-soseos', action='store_true', help = 'include start of sentence, end of sentence tokens')
  args = parser.parse_args()
  return args

def line2arr_no_soseos(vocab, line):
  return np.array([vocab[word] for word in line.split(" ")], dtype=np.int32)

def line2arr_with_soseos(vocab, line):
  return np.array(
          [SOS] + [vocab[word] + 2 for word in line.split(" ")] + [EOS],
          dtype=np.int32
         )

def main():
  args = get_args()

  vocab = Vocabulary.load(args.vocab_prefix.strip())
  output_dir = path.realpath(args.output_dir.strip())
  if args.soseos:
    line2arr = partial(line2arr_with_soseos, vocab)
    print("sos-eos!!!")
  else:
    line2arr = partial(line2arr_no_soseos, vocab)

  counter = -1
  for line in sys.stdin.readlines():
    counter += 1
#    print(counter)
#    sys.stdout.flush()
#    counter += 1
    if counter % 100 == 0:
      print(counter)
      sys.stdout.flush()
    fname = line.strip()
#    try:
    lines = open(fname, 'r').readlines()
    stripped = map(lambda x: x.strip(), lines)
    non_empty = filter(lambda x: x != "", stripped)
    file_arr = [line2arr(line) for line in non_empty ]
    np_arr = np.array(file_arr)
    try:
      new_fname = path.join(output_dir, path.split(fname)[1].replace(".tok", ".npy"))
      np.save(new_fname, np_arr)
    except:
      print("errored out on: {0}".format(fname))
#    except:
#      print("errored out processing: {0}".format(fname))

if __name__ == '__main__':
  main()
