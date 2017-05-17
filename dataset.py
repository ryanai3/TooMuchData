#!/usr/bin/python3
import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import numpy as np
import json

import bisect
from bisect import bisect_left

class FromFileListDataset():
  def __init__(self, file_list):
    if type(file_list) is str:
      with open(file_list, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        self.fnames = [line for line in lines if line]
    else:
      self.fnames = file_list

class StoryDataset(FromFileListDataset):
  def __init__(self, file_list, shuffle=True, num_workers=0): #32):
    super(StoryDataset, self).__init__(file_list)
    self.np_dataset = NPDataset(self.fnames)
    self.data_loader = DataLoader(
        dataset = self.np_dataset,
        batch_size = 1, #200,
        shuffle = shuffle,
        num_workers = num_workers,
        collate_fn = lambda x: x[0]
    )
#    itr = iter(self.data_loader)
#    for i in range(4000):
#      throw_away = next(itr)
#      print(i * 1000)

  def get_sentence_batch_iter(self, batch_size):
    return SentenceBatchIterator(self.data_loader, batch_size)

  def get_word_batch_iter(self, batch_size):
    return WordBatchIterator(self.data_loader, batch_size)
 
  def get_wordxy_batch_iter(self, batch_size):
    return WordXYBatchIterator(self.data_loader, batch_size)

class CharAnnoDataset(FromFileListDataset):
  def __init__(self, file_list, shuffle=True, num_workers=0):
    super(CharAnnoDataset, self).__init__(file_list)
    self.dataset = NPCharDataset(self.fnames)
    self.data_loader = DataLoader(
        dataset = self.dataset,
        batch_size = 1,
        shuffle = shuffle,
        num_workers = num_workers,
        collate_fn = lambda x: x[0]
    )

  def get_word_level_iter(self, batch_size):
    return WordCharBatchIterator(self.data_loader, batch_size)

def load_char_file(fname):
  count = 0
  with open(fname, 'r') as f:
    lines = [line.rstrip() for line in f.readlines()]
  res = []
  for i, line in enumerate(lines):
    if line == '':
      res.append([])
    else:
      try:
        to_app = [[int(el) for el in pair.split(":")] for pair in 
          line.split("> ")[1].split()
        ]
        res.append(to_app)      
      except Exception as exc:
#        print("{0} :: {1}".format(res[-1],line))
        res.append([]) # give up. TODO: Fix proper sentence segmentation output in char files + regen char files
        count += 1
#        with open(fname.replace("/chars/", "/tok/").replace(".chars", ".tok"), 'r') as tok_f:
#          rlines = tok_f.readlines()
#          try:
#            print(rlines[i-1].rstrip())
#          except:
#            pass
#          print(rlines[i].rstrip())
#        print("\n")
#  if count != 0:
#    print("{0}: {1}".format(fname, count))
#    global g_count
#    g_count += 1
#    print(g_count)
  return res

#TODO: THIS IS DISGUSTING:
# We're doing live surgery on the stored numpy file + the stored char mention file as we read it to correct
# newline issues (sentences that span multiple "paragraphs").
# Long term solution is to change the way we save .tok's, so that if a token has 
# whitespace_after != S (space), and instead some sequence of N's: (e.g. NNN), to output 3 <newl> tokens (or a "new paragraph" token (e.g. <newp>) ?? - so that each line in a .tok is a full sentence (with maybe paragraph breaks in the middle))
# ^^ above soln subject to change. 
# for the time being: we're simply taking sentences that have paragraph breaks in the middle
# and intelligently stitching them together

def str2idxpair(s):
 return [[int(el) for el in pair.split(":")] for pair in s.split(" ")]

def flatten1(x):
  return [j for i in x for j in i]

def load_pair(np_fname, ch_fname):
  npy = np.load(np_fname)
  with open(ch_fname, 'r') as f:
    lines = [line.strip() for line in f.readlines()] + ['']
  tok_fname = ch_fname.replace("/chars/","/tok/").replace(".chars", ".tok")
  with open(tok_fname, 'r') as tok_f:
    tok_lines = [line.strip() for line in tok_f.readlines()]
  res_npy = []
  res_chr = []
  n_i = 0
  for line, tok_line in zip(lines, tok_lines): 
    if tok_line != '':
      res_npy.append(npy[n_i])
      n_i += 1
      if line != '' and line[0] == '<':
        res_chr.append([-1 for i in range(len(res_npy[-1]))])
        idxs = str2idxpair(line.split("> ")[1])
        for idx, a_id in idxs:
          try:
            res_chr[-1][idx] = a_id
          except:
            pass
      else:
        res_chr.append([-1 for i in range(len(res_npy[-1]))])
  return (res_npy, res_chr)

def aload_pair(np_fname, ch_fname):
  count = 0
  npy = np.load(np_fname)
  with open(ch_fname, 'r') as f:
    lines = [line.strip() for line in f.readlines()]
  tok_fname = ch_fname.replace("/chars/","/tok/").replace(".chars", ".tok")
  with open(tok_fname, 'r') as tok_f:
    tok_lines = [line.strip() for line in tok_f.readlines()]
  res_npy = []
  res_chr = []
  j = 0
  n_j = 0
  if lines[j] == '':
    while (j < len(lines) and lines[j] == ''):
      if tok_lines[j] != '':
        print(tok_lines[j])
        res_npy.append(npy[j])
        res_chr.append([])
        n_j += 1
      j += 1
    if j >= len(lines):
      return res_npy, res_chr
  # now we're assured it looks like
  # <~~~> blah
  # ...
  # def run_for_set
  temp_npy = []
  temp_chr  = []
  print("{0} =?= {1} || {2}".format(len(npy), len(lines), len(tok_lines)))
  i = j
  n_i = n_j
  while (i < len(lines)):
    line = lines[i]
    if line == '':
      if tok_lines[i] != '':
        temp_npy.append(npy[n_i])
      temp_chr.append([])
    elif line[0] == "<":
      if len(temp_npy) != 0:
        res_npy[-1] = np.hstack([res_npy[-1][:-1]] + [el[1:-1] for el in temp_npy[:-1]] + [temp_npy[-1][1:]])
        res_chr[-1].extend(temp_chr)
        temp_npy = []
        temp_chr = []
      try:
        res_npy.append(npy[n_i])
      except:
        import pdb; pdb.set_trace()
      res_chr.append(str2idxpair(line.split("> ")[1]))           
    else: #hit charstuff
      temp_npy.append(npy[n_i])
      temp_chr.append(str2idxpair(line))
      res_npy[-1] = np.hstack([res_npy[-1][:-1]] + [el[1:-1] for el in temp_npy[:-1]] + [temp_npy[-1][1:]])
      res_chr[-1].extend(temp_chr)
      temp_npy = []
      temp_chr = []
    if tok_lines[i] != '':
      n_i -= 1
    i += 1
  #handle end
  if len(temp_npy) != 0:
    res_npy[-1] = np.hstack([res_npy[-1][:-1]] + [el[1:-1] for el in temp_npy[:-1]] + [temp_npy[-1][1:]])
    res_chr[-1].extend(temp_chr)
  return res_npy, res_chr   

class NPCharDataset(Dataset):
  def __init__(self, fnames):
    char_fnames =[fname.replace("/npy/", "/chars/").replace(".npy", ".chars") for fname in fnames]
    self.fname_tups = list(zip(fnames, char_fnames))

  def __len__(self):
    return len(self.fname_tups)

  def __getitem__(self, idx):
    return load_pair(*self.fname_tups[idx])

class NPDataset(Dataset):

  def __init__(self, fnames):
    self.fnames = fnames

  def __len__(self):
    return len(self.fnames)

  def __getitem__(self, idx):
    return np.load(self.fnames[idx])

class SentenceBatchIterator():
  
  def __init__(self, story_dataloader, batch_size):
    self.story_iter= iter(story_dataloader)
    self.batch_size = batch_size
    self.current_stories = [iter(self.story_iter.__next__()) for i in range(self.batch_size)]

  def __iter__(self):
    return self

  # returns a tuple of:
  # (sentences, [idxs of stories that just ended])
  def __next__(self):
    res = []
    stop_idxs = []
    for i in range(self.batch_size):
      try:
        next_sentence = self.current_stories[i].__next__()
      except StopIteration:
        try:
          self.current_stories[i] = iter(self.story_iter.__next__())
        except StopIteration:
          raise StopIteration
        next_sentence = self.current_stories[i].__next__()
        stop_idxs.append(i)
      res.append(next_sentence.astype(np.int32))
    return (res, stop_idxs)

class WordBatchIterator():

  def __init__(self, story_dataloader, batch_size):
    self.story_iter = iter(story_dataloader)
    self.batch_size = batch_size
    self.current_stories = [np.hstack(self.story_iter.__next__()) for i in range(self.batch_size)]
    self.curr_idxs = [0 for i in range(batch_size)]

  def __iter__(self):
    return self

  def __next__(self):
    res = []
    stop_idxs = []
    for i in range(self.batch_size):
      try:
        next_word = self.current_stories[i][self.curr_idxs[i]]
        self.curr_idxs[i] += 1
      except StopIteration:
        self.curr_idxs[i] = 1
        try:
          self.current_stories[i] = np.hstack(self.story_iter.__next__())
        except StopIteration:
          raise StopIteration
        next_word - self.current_stories[i][0]
        stop_idxs.append(i)
      res.append(next_word)
    return (np.array(res, dtype=np.int32), stop_idxs) 

class WordXYBatchIterator():

  def __init__(self, story_dataloader, batch_size):
#    self.story_iter = iter(tqdm(
#      iterable = story_dataloader, 
#      desc = "Story progress",
#      total = len(story_dataloader),
#      mininterval = 1.0,
#      maxinterval = 10.0
#    ))
    self.story_iter = iter(story_dataloader) 
    self.batch_size = batch_size
    self.current_stories = []
    for i in range(self.batch_size):
      self.current_stories.append(np.hstack(next(self.story_iter)))
      if i % 100 == 0:
        print(i)
    self.current_stories = [np.hstack(next(self.story_iter)) for i in range(self.batch_size)]
    self.curr_idxs = [0 for i in range(batch_size)]
    self.story_count = 0
    self.last_words = self.get_next()[0]

  def __iter__(self):
    return self

  def get_next(self):
    res = []
    stop_idxs = []
    for i in range(self.batch_size):
      try:
        next_word = self.current_stories[i][self.curr_idxs[i]]
        self.curr_idxs[i] += 1
      except IndexError:
        self.curr_idxs[i] = 1
        try:
          self.current_stories[i] = np.hstack(next(self.story_iter))
          self.story_count += 1
#          if self.story_count % 100 == 0:
#            print("Story #{0}".format(self.story_count))
        except StopIteration:
          raise StopIteration
        next_word = self.current_stories[i][0]
        stop_idxs.append(i)
      res.append(next_word)
    return (np.array(res, dtype=np.int32), stop_idxs) 

  def __next__(self):
    resY, stop_idxs = self.get_next()
    resX = self.last_words
    # no try/except handling here assumes all stories have at least 2 words
    for i in stop_idxs:
      resX[i] = resY[i]
      resY[i] = self.current_stories[i][self.curr_idxs[i]]
      self.curr_idxs[i] += 1
    self.last_words = resY
    return ((resX, resY), stop_idxs)

class WordCharIterator():
  def __init__(self, sentences, actor_mentions):
    self.sentences = sentences
    self.am_sentences = actor_mentions
    self.sentence_idx = 0
    self.word_idx = 0

  def __iter__(self):
    return self

  def __next__(self):
    try:
      sentence = self.sentences[self.sentence_idx]
      am_sentence = self.am_sentences[self.sentence_idx]
    except:
      raise StopIteration
    try:
      res = (sentence[self.word_idx], am_sentence[self.word_idx])
      self.word_idx += 1
      return res
    except IndexError:
      self.sentence_idx += 1
      self.word_idx = 0
      raise IndexError

class WordCharBatchIterator():

  def __init__(self, story_dataloader, batch_size):
    self.story_iter = iter(story_dataloader) 
    self.batch_size = batch_size
    self.curr_wc_iters = [iter(WordCharIterator(*next(self.story_iter))) for i in range(self.batch_size)]
    self.ams_in_sentence = [[] for i in range(self.batch_size)]
    self.curr_idxs = [0 for i in range(batch_size)]
    self.story_count = 0
    self.use_actor_token = True #False # this flag is a hack until the new dataset code is up and running
    self.last = self.get_next()
 
  def __iter__(self):
    return self

  def finish_story(self, story_stop_idxs, i):
    story_stop_idxs.append(i)
    self.curr_wc_iters[i] = iter(WordCharIterator(*next(self.story_iter)))
    next_word, am = next(self.curr_wc_iters[i])
    return next_word, am

  def get_next(self):
    res_words = []
    res_am_batch_i = []
    res_am_id = []
    res_sam_batch_i = []
    actor_sentence_mentions = []
    sentence_stop_idxs = []
    story_stop_idxs = []
    for i in range(self.batch_size):
      try:
        next_word, am = next(self.curr_wc_iters[i])
      except IndexError:
        sentence_stop_idxs.append(i)
        actors_mentioned = self.ams_in_sentence[i]
        self.ams_in_sentence[i] = []
        #if actors_mentioned:
        if len(actors_mentioned) == 1:
          res_sam_batch_i.append(i)
          actor_sentence_mentions.append(actors_mentioned[0])
        try:
          next_word, am = next(self.curr_wc_iters[i]) 
        except StopIteration:
          next_word, am = self.finish_story(story_stop_idxs, i)
      except StopIteration:
        try:
          next_word, am = self.finish_story(story_stop_idxs, i)
        except StopIteration: #all out of stories!
          raise StopIteration
      if self.use_actor_token:
        if next_word != 0 and next_word != 1:
          next_word += 1
        if am != -1:
          next_word = 2
      res_words.append(next_word)
      if am != -1:
        res_am_batch_i.append(i)
        res_am_id.append(am)
        if am not in self.ams_in_sentence[i]:
          self.ams_in_sentence[i].append(am)
    return (
      np.array(res_words, dtype=np.int32), 
      res_am_batch_i, 
      res_am_id, 
      res_sam_batch_i,
      actor_sentence_mentions, 
      sentence_stop_idxs, 
      story_stop_idxs
    )  

  def __next__(self):
    resY, am_batch_iY, am_idY, sam_batch_i, actor_sentence_mentions, sentence_stop_idxs, story_stop_idxs = self.get_next()
    resX, am_batch_iX, am_idX, sam_batch_i_, _, sentence_stop_idxs_, _ = self.last
    for i in story_stop_idxs:
      resX[i] = resY[i]
      # assume first token is always <sos> and that can never be an actor mention
      # also assume last token of previous story is always <eos> and that can never be an actor mention...
      # so we don't have to worry about it!
      # sentence_stop_idxs is not on, so don't worry about it
    self.last = resY, am_batch_iY, am_idY, sam_batch_i, actor_sentence_mentions, sentence_stop_idxs, story_stop_idxs
    
    return (
      # x's: input word, which examples have an actor mentioned, and their id
      (resX, am_batch_iX, am_idX),
      # y's: next word, which examples have an actor mentioned in the next word, and their id
      (resY, am_batch_iY, am_idY), 
      # sentences for which X is <eos>, and the actors mentioned in each sentence - for updating
      (sam_batch_i, actor_sentence_mentions),
      # idxs for which the story ends :'(
      (sentence_stop_idxs, story_stop_idxs)
    )
    
  def lnexxt(self):
    resY, stop_idxs = self.get_next()
    resX = self.last_words
    # no try/except handling here assumes all stories have at least 2 words
    for i in stop_idxs:
      resX[i] = resY[i]
      resY[i] = self.current_stories[i][self.curr_idxs[i]]
      self.curr_idxs[i] += 1
    self.last_words = resY
    return ((resX, resY), stop_idxs)


def main():
  ds = CharAnnoDataset("./splits/tiny/train")
  w_it = ds.get_word_level_iter(4)

if __name__ == '__main__':
  main()
