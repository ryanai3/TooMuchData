#!/usr/bin/python3

import argparse
import models

def get_parser():
  archs = models.model_hash
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', help='Path to data dir')
  parser.add_argument('--arch', '-a', choices = archs.keys(), default='ModelX', help='Model type')
  parser.add_argument('--batch_size', '-B', type=int, default=128, help='Learning minibatch size')
  parser.add_argument('--n_epochs', '-E', type=int, default=30, help='Number of epochs to train')
  parser.add_argument('--gpu', '-g', type=int, default=-1)
  parser.add_argument('--resume_from', help='Resume training from given file', default=False)
  parser.add_argument('--loaderjob', '-j', type=int, help='Number of parallel data loading processes')
  parser.add_argument('--params', '-p', help='path to parameter file')
  parser.add_argument('--out', '-O', default='result', help='Output directory')
  parser.add_argument('--eval', action='store_true')
  return parser

