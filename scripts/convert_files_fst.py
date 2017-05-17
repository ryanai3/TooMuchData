#!/usr/bin/python
import fileinput
from datrie import BaseTrie
import string
import sys
import numpy as np

word2int = BaseTrie.load("/tmp/full_vocab.tvi")
#fname = sys.argv[1].rstrip()
maxint = (2**31) - 1 # we use int32 in numpy so

def word2intsafe(word):
  try:
    return word2int[word]
  except:
    return maxint

def line2arr(line):
  return np.array([word2intsafe(word.rstrip()) for word in line.split(" ")])

for in_line in sys.stdin.readlines():
  fname = in_line.rstrip()
  try:
    file_arr = [line2arr(line) for line in open(fname, 'r').readlines()]
    np_arr = np.array(file_arr)

    try:
      new_fname = fname.replace(".tok", "")
      if new_fname != fname:
        np.save(new_fname, np_arr)
        print("saved!: {0}".format(new_fname))
        sys.stdout.flush()
      else:
        print("error saving: {0}".format(new_fname))
    except ex:
      print("errored out on: {0}".format(fname))
  except:
    print("failed read on: {0}".format(fname))
