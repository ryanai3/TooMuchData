#!/usr/bin/python3

import argparse
import fileinput
import pickle

from utils.vocabulary import Vocabulary

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input_files', help ='file containing list of input files')
  parser.add_argument('-o', '--output_prefix', help='prefix for output files', required=True)
  args = parser.parse_args()
  return args

def main():
  word2cnt = Vocabulary()
  args = get_args()
  i = -1
  for line in open(args.input_files.strip(), 'r').readlines():
#  for line in fileinput.input():
    i += 1
    if i % 100 == 0:
      print(i)
      if i % 1000 == 0:
        word2cnt.save(args.output_prefix.strip() + "_partial")
    with open(line.rstrip(), 'r') as f:
      for line in f.readlines():
        for word in line.strip().split():
          word2cnt.observe_word(word)
  word2cnt.save(args.output_prefix.strip())
#  pickle.dump(word2cnt, "./output.hash")

def main2():
  vocabs = [Vocabulary.load("./vocabs/v{0}".format(i)) for i in range(8)]
  print("loaded vocabs!")
  master_vocab = Vocabulary.merge_vocabularies(vocabs)
  master_vocab.save("./vocabs/final_vocab")
  import pdb; pdb.set_trace()

def main3():
  master_vocab = Vocabulary.load("./vocabs/final_vocab")
  offsets = master_vocab.get_offsets()
  with open("./vocabs/offsets.pkl", 'wb') as out_f:
    pickle.dump(offsets, out_f)
  import pdb; pdb.set_trace()
  print(32)


if __name__ == '__main__':
  main3()
