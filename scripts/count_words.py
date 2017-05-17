#!/usr/bin/python
import argparse
import fileinput
from os import path
from functools import partial
import numpy as np
import sys

from utils.vocabulary import Vocabulary
from utils import SOS, EOS

def main():
  n_words = 0
  words_in_story = 0
  counter = 0
  for line in sys.stdin.readlines():
    counter += 1
    fname = line.rstrip()
    story = np.load(fname)
    words_in_story = sum([len(sentence) for sentence in story])
    n_words += words_in_story
    print("{0} {1}".format(words_in_story, fname))

    if counter % 100 == 0:
      sys.stdout.flush()

if __name__ == '__main__':
  main()
