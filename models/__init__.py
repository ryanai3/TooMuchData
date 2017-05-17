#!/usr/bin/python3

import sys
from os import path

sys.path.append(path.join(path.dirname(__file__), "../"))
from . import gru_base, torch_gru_base, actor_gru_clean, simple_agru_hstate, actor_autoencode_gru, am_gru
model_hash = {
#  'LSTM_baseline': lstm_base.LSTM_Baseline
  'gru_baseline': gru_base.GRU_Baseline,
  't_gru_baseline': torch_gru_base.GRU_Baseline,
  'actor_gru': actor_gru_clean.ActorGRU,
  'actor_gru_hstate': simple_agru_hstate.ActorGRU,
  'actor_autoencode_gru': actor_autoencode_gru.ActorGRU,
  'am_gru': am_gru.ActorGRU
}


kernel_dir = path.join(path.dirname(__file__), "kernels")
