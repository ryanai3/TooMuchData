from datrie import Trie
import string
import sys
import pickle

class Vocabulary():
  def __init__(self, trie=Trie(string.printable), untrieable={}):
    self.trie = trie
    self.untrieable = untrieable
    self.max_index = 2 ** 28

  @classmethod
  def load(cls, prefix):
    trie = Trie.load(prefix + ".trie")
    with open(prefix + ".utrie", 'rb') as out_f:
      untrieable = pickle.load(out_f)
    obj = cls(trie, untrieable)
    return obj

  def __contains__(self, word):
    return (word in self.trie) or (word in self.untrieable)

  def __getitem__(self, word):
    try:
      return self.trie[word]
    except:
      if word in self.untrieable:
        return self.untrieable[word]
      else:
        return self.max_index

  def __setitem__(self, k, v):
    try:
      self.trie[k] = v
    except:
      self.untrieable[k] = v

  def __len__(self):
    return len(self.trie) + len(self.untrieable)

  def items(self):
    return self.trie.items() + list(self.untrieable.items())

  def observe_word(self, word, to_add=1):
    try:
      if word not in self.trie:
        self.trie[word] = 0
      self.trie[word] += to_add
    except:
      if word not in self.untrieable:
        self.untrieable[word] = 0
      self.untrieable[word] += to_add

  def save(self, prefix):
    self.trie.save(prefix + ".trie")
    with open(prefix + ".utrie", 'wb') as out_f:
      pickle.dump(self.untrieable, out_f) 

  def __iadd__(self, other_vocab):
    for (i, (word, count)) in enumerate(other_vocab.items()):
      if i % 100 == 0:
        print(i)
      self.observe_word(word, count)
    return self

  def count_to_index(self, n_special_tokens=2):
    tvi = Vocabulary()
    tvi.n_special_tokens = n_special_tokens
    sorted_words = sorted(self.items(), key = lambda x: x[1], reverse=True)
    for (idx, (word, count)) in enumerate(sorted_words):
      tvi[word] = idx + n_special_tokens
    return tvi

  @staticmethod
  def merge_vocabularies(vocabs):
    s_vocabs = sorted(vocabs, key = lambda x: len(x), reverse=True)
    to_ret = s_vocabs[0]
    for to_add in s_vocabs[1:]:
      print("adding vocab!")
      to_ret += to_add
    return to_ret

  # should only be called on a tvi
  def get_offsets(self):
    srt = sorted(self.items(), key = lambda x: x[1], reverse=True)
    res = {}
    last_count = 0
    for (idx, (word, count)) in enumerate(srt):
      if count != last_count:
        last_count = count
        res[count] = idx + 1
    return res
 
