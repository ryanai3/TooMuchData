#!/usr/bin/python3

default_loc_str = "/hdd/data/nlp/raw/unzipped/ff15_book/{0}/{1}.{0}"

import torch
from torch.utils.data import Dataset, DataLoader
from dataset import FromFileListDataset
from os import path

import numpy as np

def str2idxpair(s):
 return [[int(el) for el in pair.split(":")] for pair in s.split(" ")]

def read_listify(obj):
  if type(obj) is str:
    with open(obj, 'r') as f:
      res = [line.strip() for line in f.readlines()]
      res = [line for line in res if line]
  else:
    res = obj
  return res

class StoryDataset():
  def __init__(self, id_list, loc_str=default_loc_str):
    self.ids = read_listify(id_list)    
    self.loc_str = loc_str
  
  @classmethod
  def from_file_list(cls, file_list, to_replace_type="npy"):
    fname_list = read_listify(file_list)
    repl_0 = "/" + to_replace_type + "/"
    repl_1 = "." + to_replace_type 
    loc_str = path.dirname(fname_list[0].replace(repl_0, "{0}")) + "/{1}.{0}"
    ids = [fname.replace(repl_0, "").replace(repl1, "") for fname in fname_list]
    return cls(loc_str, id_list) 

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, idx):
    return Story(self.ids[idx], self.loc_str)

  def get_dataloader(self, shuffle=True, num_workers=4):
    return DataLoader(
      dataset = self,
      collate_fn = lambda x: x[0],
      shuffle=shuffle,
      num_workers=num_workers
    )

  def get_batched_iter(self, batch_size, **dataloader_args):
    loader = self.get_dataloader(**dataloader_args)
    return StoryBatchIter(loader, batch_size)
  
class Story():
  def __init__(self, id_str, loc_str=default_loc_str):
    subf_locs = Story.get_subf_locs(id_str, loc_str)
    self.id_str = id_str
    self.preprocess_and_populate_metadata(*Story.load_from_files(**subf_locs))

  @staticmethod
  def get_subf_locs(id_str, loc_str):
    return {'{0}_fname'.format(k): loc_str.format(v, id_str)
        for k, v in {'np': 'npy', 'ch': 'chars', 'tk': 'tok'}.items()
    }

  @staticmethod
  def load_from_files(np_fname, ch_fname, tk_fname):
    np_data = np.load(np_fname)
    with open(ch_fname, 'r') as ch_f:
      ch_lines = [line.strip() for line in ch_f.readlines()] + ['']
    with open(tk_fname, 'r') as tk_f:
      tk_lines = [line.strip() for line in tk_f.readlines()]
    res_np = []
    res_ch = []
    n_i = 0
    for ch_line, tk_line in zip(ch_lines, tk_lines):
      if tk_line != '':
        res_np.append(np_data[n_i])
        n_i += 1
        res_ch.append([-1 for i in range(len(res_np[-1]))])
        if ch_line != '' and ch_line[0] == '<':
          idxs = str2idxpair(ch_line.split("> ")[1])
          for idx, a_id in idxs:
            try:
              res_ch[-1][idx + 1] = a_id #handle existence of SOS token
            except:
              pass
    return (res_np, res_ch)

  def preprocess_and_populate_metadata(self, np_sens, actor_mentions):
    self.sentence_lens = [len(sentence) for sentence in np_sens]
    self.sentence_starts = np.cumsum([0] + self.sentence_lens)[:-1]
    self.words = np.concatenate(np_sens)
    self.actor_mentions = np.concatenate(actor_mentions)
    self.n_actors = max(self.actor_mentions) + 1 

  def __iter__(self):
    return StoryIter(self)

class StoryIter():

  def __init__(self, story):
    self.story = story
    self.sentence_lens = story.sentence_lens
    self.sentence_starts = story.sentence_starts
    self.id_str = story.id_str
    self.s_i = 0 # current sentence index
    self.w_off = 0 # current word offset in sentence
    self.whole_sentence_mentions = []

  def __next__(self):
    # Need to worry about:
    # current word
    # current mentioned actor
    # if sentence ends and -> actors mentioned in whole sentence
    # if story ends and -> metadata of next story (happens in call to next StoryIter)
    res = {'sentence_end': 0, 'last_in_story': 0, 'sams': -1, 'id': self.id_str, 'sams': -1}
    # handle sentence end case
    if self.w_off == self.sentence_lens[self.s_i] - 1:
      res['sentence_end'] = 1
      # add observed actors if there was only one
      if self.whole_sentence_mentions and len(self.whole_sentence_mentions) == 1:
        res['sams'] = self.whole_sentence_mentions[0]
        self.whole_sentence_mentions = []
      # handle story end case
      if self.s_i == len(self.sentence_lens) - 1:
        res['last_in_story'] = 1

    idx_in = self.sentence_starts[self.s_i] + self.w_off

    # if we're starting a new story, provide metadata
    if idx_in == 0:
      res.update({
        'n_actors': self.story.n_actors,
      })
    
    # handle current word, actor mention, bookkeeping
    res['word'] = self.story.words[idx_in]
    actor_mention = self.story.actor_mentions[idx_in]
#    if self.use_actor_token:
    if not (res['word'] == 0 or res['word'] == 1):
      res['word'] += 1
    if actor_mention != -1:
      res['word'] = 2
    if actor_mention not in self.whole_sentence_mentions and actor_mention != -1:
      self.whole_sentence_mentions.append(actor_mention)
    res['actor_id'] = actor_mention
    self.w_off += 1
    if self.w_off == self.sentence_lens[self.s_i]:
      self.s_i += 1
      self.w_off = 0
    return res

def ld2dl(ld):
  dl = {}
  for d in ld:
    for k, v in d.items():
      try:
        dl[k].append(v)
      except KeyError:
        dl[k] = [v]
  return dl

class StoryBatchIter():
  
  def __init__(self, story_loader, batch_size):
    self.story_source = iter(story_loader)
    self.batch_size = batch_size
    self.curr_stories = [iter(next(self.story_source)) for i in range(self.batch_size)]
    self.last = self.get_next()

  def __iter__(self):
    return self

  def get_next(self):
    res = [next(story) for story in self.curr_stories]
    for i in range(self.batch_size):
      # handle stitching together at story ends
      if res[i]['last_in_story'] == 1:
        new_story = iter(next(self.story_source))
#        next_elem = next(new_story)
#        next_elem['story_end'] = 1
#        res[i] = next_elem
        self.curr_stories[i] = new_story
    as_dl = ld2dl(res)
    as_dl['word'] = torch.from_numpy(np.array(as_dl['word'])).long()
    as_dl['actor_id'] = torch.from_numpy(np.array(as_dl['actor_id'])).long()
    as_dl['batch_actor_locs'] = (as_dl['actor_id'] != -1).nonzero().squeeze()
    as_dl['sams'] = torch.from_numpy(np.array(as_dl['sams'])).long()
    as_dl['sam_i'] = (as_dl['sams'] != -1).nonzero().squeeze()
    as_dl['sentence_end'] = torch.ByteTensor(as_dl['sentence_end'])
    as_dl['sentence_end_locs'] = as_dl['sentence_end'].nonzero().squeeze()
    as_dl['last_in_story'] = torch.ByteTensor(as_dl['last_in_story'])
    as_dl['story_end'] = as_dl['last_in_story'].nonzero().squeeze()
    return as_dl

  def __next__(self):
    x = self.last
    y = self.get_next()
    res = {
      'wX':            x['word'],
      'wY':            y['word'],
      'am_idX':        x['actor_id'],
      'am_idY':        y['actor_id'],
      'am_locX':       x['batch_actor_locs'],
      'am_locY':       y['batch_actor_locs'],
      'sam_loc':       x['sam_i'],
      'sam_id':        x['sams'],
      'sentence_end':  x['sentence_end'],
      'se_loc':        x['sentence_end_locs'],
      'story_end':     x['story_end'],
      'id':            x['id']
    }
    self.last = y
    return res

from utils.vocabulary import Vocabulary

def main():
  ds = StoryDataset("./splits/10pct_ids/train")
  it = ds.get_batched_iter(4096)
  res = [next(it) for i in range(1000)]
#  for i in range(1000):
#    if sum(res[i]['am_idX']) != -4:
#      print(i)
#  vocab = Vocabulary.load("/hdd/data/nlp/raw/unzipped/ff15_book/vocabs/final_vocab")
#  sorted_vocab = sorted(vocab.items(), key = lambda x: x[1], reverse=True)
#  full_vocab = ["<SOS>", "<EOS>"] + list(map(lambda x: x[0], sorted_vocab))
#  import pdb; pdb.set_trace()
#  print(32)

if __name__ == '__main__':
  main()
