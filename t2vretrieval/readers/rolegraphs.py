import os
import json
import numpy as np
import h5py
import collections
import torch

import t2vretrieval.readers.mpdata

def extract_video_classes(dataset_file, video_captions_keys, name, percent=0.25):
  # dataset_file -> dict containing the annotations
  # video_captions_keys -> list of caption keys for a given video (likely in a format {video_id}_{caption_id})
  # name -> name of the column (noun_classes, verb_class, etc)
  # percent -> each video has C captions, we want to keep the classes appearing in >=percent*C captions
  #all_vcs = [self.dataset_file[__k]["annotations"][0]["verb_class"] for __k in video_captions_keys]
  all_classes = [dataset_file[__k]["annotations"][0][name] for __k in video_captions_keys]
  count = {}
  for vcs in all_classes:
    for c in vcs:
      if c not in count.keys():
        count[c] = 0
      count[c] += 1

  kept_classes = []
  for cl, cn in count.items():
    if cn >= percent * len(video_captions_keys):
      kept_classes.append(cl)

  return kept_classes

ROLES = ['V', 'ARG1', 'ARG0', 'ARG2', 'ARG3', 'ARG4',
 'ARGM-LOC', 'ARGM-MNR', 'ARGM-TMP', 'ARGM-DIR', 'ARGM-ADV', 
 'ARGM-PRP', 'ARGM-PRD', 'ARGM-COM', 'ARGM-MOD', 'NOUN']

class RoleGraphDataset(t2vretrieval.readers.mpdata.MPDataset):
  def __init__(self, name_file, attn_ft_files, word2int_file,
    max_words_in_sent, num_verbs, num_nouns, ref_caption_file, ref_graph_file, 
    max_attn_len=20, load_video_first=False, is_train=False, _logger=None,
               dataset_file='', is_test=False, dname="", rel_mat_path="", threshold_pos=1):
    if _logger is None:
      self.print_fn = print
    else:
      from torch.utils.tensorboard import SummaryWriter
      self.print_fn = _logger.info if not isinstance(_logger, SummaryWriter) else print

    self.max_words_in_sent = max_words_in_sent
    self.is_train = is_train
    self.attn_ft_files = attn_ft_files
    self.max_attn_len = max_attn_len
    self.load_video_first = load_video_first

    if dname == "charades":
        self.names = np.load(name_file, allow_pickle=True)
    else:
        self.names = np.load(name_file)
    self.word2int = json.load(open(word2int_file))

    self.num_videos = len(self.names)
    self.print_fn('num_videos %d' % (self.num_videos))
    self.dname = dname
    self.print_fn('working on dataset: %s' % (self.dname))

    if ref_caption_file is None:
      self.ref_captions = None
    else:
      self.ref_captions = json.load(open(ref_caption_file))
      self.captions = list()
      self.pair_idxs = []
      for i, name in enumerate(self.names):
        if dname == "charades":
          valid_names = [n for n in self.ref_captions if n.startswith(name)]
          iter_on = []
          for n in valid_names:
            iter_on.extend(self.ref_captions[n])
        else:
          iter_on = self.ref_captions[name]
        for j, sent in enumerate(iter_on):
          self.captions.append(sent)
          self.pair_idxs.append((i, j))

      # for val/test here we may also load the relevance matrix if we want to compute nDCG&mAP
      if is_test or not is_train:  # test or validation
        print(f"rel mat path {rel_mat_path}")
        import pandas
        if is_test:
          assert rel_mat_path != ""
        if rel_mat_path != "":
          if "hdf5" in rel_mat_path:
            import h5py
            with h5py.File(rel_mat_path, "r") as f:
              self.relevance_matrix = np.array(f["rel_mat"])
          else:
            self.relevance_matrix = pandas.read_pickle(rel_mat_path)

        if is_test:
          if dname == "epic":
            print("reading epic100 unique caps")
            self.captions = pandas.read_csv("annotation/epic100RET/EPIC_100_retrieval_test_sentence.csv")['narration'].values
#             self.captions = pandas.read_csv("annotation/epic100RET/EPIC_100_retrieval_train.csv")['narration'].values
          if dname == "msr-vtt-1kA":
            self.captions = list()
            self.pair_idxs = []
            self.ref_captions = json.load(open("annotation/msr-vttRET/ref_captions_msrvtt_val_1kA.json"))
            tmp = json.load(open("annotation/msr-vttRET/caps_msrvtt_1kA_wrong.json"))
            for i, name in enumerate(self.names):
              if name.replace(".mp4", "") in tmp:
                self.captions.append(tmp[name.replace(".mp4", "")])
                self.pair_idxs.append((i, self.ref_captions[name].index(tmp[name.replace(".mp4", "")])))
              else:
                #print(i, name)
                self.captions.append(self.ref_captions[name][0])
                self.pair_idxs.append((i, 0))

      self.num_pairs = len(self.pair_idxs)
      self.print_fn('captions size %d' % self.num_pairs)

    if self.load_video_first:
      self.all_attn_fts, self.all_attn_lens = [], []
      for name in self.names:
        attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
        attn_fts, attn_len = self.pad_or_trim_feature(attn_fts, self.max_attn_len, trim_type='select')
        self.all_attn_fts.append(attn_fts)
        self.all_attn_lens.append(attn_len)
      self.all_attn_fts = np.array(self.all_attn_fts)
      self.all_attn_lens = np.array(self.all_attn_lens)

    self.num_verbs = num_verbs
    self.num_nouns = num_nouns
    
    self.role2int = {}
    for i, role in enumerate(ROLES):
      self.role2int[role] = i
      self.role2int['C-%s'%role] = i
      self.role2int['R-%s'%role] = i

    self.ref_graphs = json.load(open(ref_graph_file))

    if is_train:
      print(f"using a threshold of {threshold_pos} [0, 1] to distinguish positives (>= thr) from negatives (< thr)")
      self.threshold_pos = threshold_pos
      assert 0 < threshold_pos and threshold_pos <= 1

    self.is_test = is_test
    self.dataset_file_path = dataset_file
    if dataset_file != '':
      self.dataset_file = json.load(open(dataset_file, "r"))['database']
      if 'msvd' in dataset_file or 'charades' in dataset_file:
        self.video_cap_keys = {}
        for vidkey_senkey in self.dataset_file.keys():
          vidkey = "_".join(vidkey_senkey.split("_")[:-1])
          if vidkey not in self.video_cap_keys:
            self.video_cap_keys[vidkey] = []
          self.video_cap_keys[vidkey].append(vidkey_senkey)
          
        self.video_captions_verb_classes = {}
        self.video_captions_noun_classes = {}
        for vk, vcks in self.video_cap_keys.items():
          self.video_captions_verb_classes[vk] = extract_video_classes(self.dataset_file, vcks, "verb_class")
          self.video_captions_noun_classes[vk] = extract_video_classes(self.dataset_file, vcks, "noun_classes")
        print("computed video->[captions] classes")
  
      if 'msrvtt' in dataset_file:
        self.video_cap_keys = {}
        tmp = {}
        for vidkey_senkey in self.dataset_file.keys():
          vidkey = vidkey_senkey.split("_")[0]
          if vidkey not in self.video_cap_keys:
            self.video_cap_keys[vidkey] = []
            tmp[vidkey] = []
          tmp[vidkey].append(vidkey_senkey)
        for vidkey in tmp.keys():
          self.video_cap_keys[vidkey] = sorted(tmp[vidkey], key=lambda vk: int(vk.replace("video", "")))

    if 'msrvtt' in dataset_file:
      self.video_captions_verb_classes = {}
      self.video_captions_noun_classes = {}
      for vk, vcks in self.video_cap_keys.items():
        self.video_captions_verb_classes[vk] = extract_video_classes(self.dataset_file, vcks, "verb_class")
        self.video_captions_noun_classes[vk] = extract_video_classes(self.dataset_file, vcks, "noun_classes")
      print("computed video->[captions] classes")

  def load_attn_ft_by_name(self, name, attn_ft_files):
    attn_fts = []
    for i, attn_ft_file in enumerate(attn_ft_files):
      with h5py.File(attn_ft_file, 'r') as f:
        key = name.replace('/', '_')
        attn_ft = f[key][...]
        attn_fts.append(attn_ft)
    #attn_fts = np.concatenate([attn_ft for attn_ft in attn_fts], axis=-1)
    attn_fts = [attn_ft for attn_ft in attn_fts]
    lens = [len(a) for a in attn_fts]
    if not all([l == lens[0] for l in lens]):  # not all share the same length
        min_len = min(lens)
        attn_fts = [attn_ft[np.linspace(0, len(attn_ft)-1, min_len, dtype=int)] for attn_ft in attn_fts]
    
    attn_fts = np.concatenate(attn_fts, axis=-1)
    return attn_fts

  def pad_or_trim_feature(self, attn_ft, max_attn_len, trim_type='top'):
    if len(attn_ft.shape) == 2:
      seq_len, dim_ft = attn_ft.shape
    else:
      sqz, seq_len, dim_ft = attn_ft.shape
      assert sqz == 1
      attn_ft = attn_ft.squeeze(0)
    attn_len = min(seq_len, max_attn_len)

    # pad
    if seq_len < max_attn_len:
      new_ft = np.zeros((max_attn_len, dim_ft), np.float32)
      new_ft[:seq_len] = attn_ft
    # trim
    else:
      if trim_type == 'top':
        new_ft = attn_ft[:max_attn_len]
      elif trim_type == 'select':
        idxs = np.round(np.linspace(0, seq_len-1, max_attn_len)).astype(np.int32)
        new_ft = attn_ft[idxs]
    return new_ft, attn_len

  def get_caption_outs(self, out, sent, graph):
    graph_nodes, graph_edges = graph
    #print(graph)

    verb_node2idxs, noun_node2idxs = {}, {}
    edges = []
    out['node_roles'] = np.zeros((self.num_verbs + self.num_nouns, ), np.int32)

    # root node
    sent_ids, sent_len = self.process_sent(sent, self.max_words_in_sent)
    out['sent_ids'] = sent_ids
    out['sent_lens'] = sent_len

    # graph: add verb nodes
    node_idx = 1
    out['verb_masks'] = np.zeros((self.num_verbs, self.max_words_in_sent), np.bool)
    for knode, vnode in graph_nodes.items():
      k = node_idx - 1
      if k >= self.num_verbs:
        break
      if vnode['role'] == 'V' and np.min(vnode['spans']) < self.max_words_in_sent:
        verb_node2idxs[knode] = node_idx
        for widx in vnode['spans']:
          if widx < self.max_words_in_sent:
            out['verb_masks'][k][widx] = True
        out['node_roles'][node_idx - 1] = self.role2int['V']
        # add root to verb edge
        edges.append((0, node_idx))
        node_idx += 1
        
    # graph: add noun nodes
    node_idx = 1 + self.num_verbs
    out['noun_masks'] = np.zeros((self.num_nouns, self.max_words_in_sent), np.bool)
    for knode, vnode in graph_nodes.items():
      k = node_idx - self.num_verbs - 1
      if k >= self.num_nouns:
          break
      if vnode['role'] not in ['ROOT', 'V'] and np.min(vnode['spans']) < self.max_words_in_sent:
        noun_node2idxs[knode] = node_idx
        for widx in vnode['spans']:
          if widx < self.max_words_in_sent:
            out['noun_masks'][k][widx] = True
        out['node_roles'][node_idx - 1] = self.role2int.get(vnode['role'], self.role2int['NOUN'])
        node_idx += 1

    # graph: add verb_node to noun_node edges
    for e in graph_edges:
      if e[0] in verb_node2idxs and e[1] in noun_node2idxs:
        edges.append((verb_node2idxs[e[0]], noun_node2idxs[e[1]]))
        edges.append((noun_node2idxs[e[1]], verb_node2idxs[e[0]]))

    num_nodes = 1 + self.num_verbs + self.num_nouns
    rel_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src_nodeidx, tgt_nodeidx in edges:
      rel_matrix[tgt_nodeidx, src_nodeidx] = 1
    # row norm
    for i in range(num_nodes):
      s = np.sum(rel_matrix[i])
      if s > 0:
        rel_matrix[i] /= s

    out['rel_edges'] = rel_matrix
    return out

  def __getitem__(self, idx):
    out = {}
    if self.is_train:
      video_idx, cap_idx = self.pair_idxs[idx]
      name = self.names[video_idx]
      if self.dname == "charades":
        sent = self.ref_captions[f"{name}_{cap_idx}"][0]
      else:
        sent = self.ref_captions[name][cap_idx]
      out = self.get_caption_outs(out, sent, self.ref_graphs[sent])
    else:
      video_idx = idx
      name = self.names[idx]
    
    if self.load_video_first:
      attn_fts, attn_len = self.all_attn_fts[video_idx], self.all_attn_lens[video_idx]
    else:
      attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
      attn_fts, attn_len = self.pad_or_trim_feature(attn_fts, self.max_attn_len, trim_type='select')
    
    out['names'] = name
    out['attn_fts'] = attn_fts
    out['attn_lens'] = attn_len
    if self.is_train:
      out['threshold_pos'] = self.threshold_pos

    if self.dataset_file_path != '' and self.is_train:
      if self.dname == "msr-vtt":
        video_name = name.replace(".mp4", "")
        _keys = self.video_cap_keys[video_name]
        #sorted([_k for _k in self.dataset_file.keys() if name.replace(".mp4", "") in _k],
        #              key=lambda vk: int(vk.replace("video", "")))
        _key = _keys[cap_idx]

        out['video_noun_classes'] = self.video_captions_noun_classes[video_name]
        out['video_verb_classes'] = self.video_captions_verb_classes[video_name]
      elif self.dname in ["vatex", "msvd", "charades"]:
        _key = f'{name}_{cap_idx}'
        out['video_noun_classes'] = self.video_captions_noun_classes[name]
        out['video_verb_classes'] = self.video_captions_verb_classes[name]
      else:
        _key = name
      nc_str = self.dataset_file[_key]["annotations"][0]["noun_classes"]
      if isinstance(nc_str, str):
        nc_str = list(map(int, nc_str.replace("[", "").replace("]", "").split(",")))
      out['noun_classes'] = nc_str
      out['verb_class'] = self.dataset_file[_key]["annotations"][0]["verb_class"]

    return out

  def iterate_over_captions(self, batch_size):
    # the sentence order is the same as self.captions
    for s in range(0, len(self.captions), batch_size):
      e = s + batch_size
      data = []
      for sent in self.captions[s: e]:
        out = self.get_caption_outs({}, sent, self.ref_graphs[sent])
        data.append(out)
      outs = collate_graph_fn(data)
      yield outs


class Word2VecRoleGraphDataset(RoleGraphDataset):  
    def __init__(self, name_file, attn_ft_files, word2int_file,
                 max_words_in_sent, num_verbs, num_nouns, ref_caption_file, ref_graph_file, 
                 max_attn_len=20, load_video_first=False, is_train=False, _logger=None,
                 dataset_file='', is_test=False, dname="", rel_mat_path="", threshold_pos=1):
        super().__init__(name_file, attn_ft_files, word2int_file,
                 max_words_in_sent, num_verbs, num_nouns, ref_caption_file, ref_graph_file, 
                 max_attn_len, load_video_first, is_train, _logger,
                 dataset_file, is_test, dname, rel_mat_path, threshold_pos)
        from gensim.models.keyedvectors import KeyedVectors
        # import gensim.downloader as api
        # self.wv = api.load('word2vec-google-news-300')
        from gensim.models.keyedvectors import KeyedVectors
        self.wv = KeyedVectors.load_word2vec_format("../everything_at_once/data/GoogleNews-vectors-negative300.bin", binary=True)
        
    def process_sent(self, sent, max_words):
        tokens = [self.wv[w] for w in sent.split() if w in self.wv]  # else np.random.randn((300)).astype('f') 
        # # add BOS, EOS?
        # tokens = [BOS] + tokens + [EOS]
        tokens = tokens[:max_words]
        tokens_len = len(tokens)
        tokens = np.array(tokens + [np.zeros((300))] * (max_words - tokens_len), dtype=np.float32)
        return tokens, tokens_len

class Word2VecNoGraphsDataset(Word2VecRoleGraphDataset):
    def __getitem__(self, idx):
        out = {}
        if self.is_train:
          video_idx, cap_idx = self.pair_idxs[idx]
          name = self.names[video_idx]
          sent = self.ref_captions[name][cap_idx]
          sent_ids, sent_len = self.process_sent(sent, self.max_words_in_sent)
          out['sent_ids'] = sent_ids
          out['sent_lens'] = sent_len
        else:
          video_idx = idx
          name = self.names[idx]

        if self.load_video_first:
          attn_fts, attn_len = self.all_attn_fts[video_idx], self.all_attn_lens[video_idx]
        else:
          attn_fts = self.load_attn_ft_by_name(name, self.attn_ft_files)
          attn_fts, attn_len = self.pad_or_trim_feature(attn_fts, self.max_attn_len, trim_type='select')

        out['names'] = name
        out['attn_fts'] = attn_fts
        out['attn_lens'] = attn_len
        if self.is_train:
          out['threshold_pos'] = self.threshold_pos

        if self.dataset_file_path != '' and self.is_train:
          if self.dname == "msr-vtt":
            video_name = name.replace(".mp4", "")
            _keys = self.video_cap_keys[video_name]
            #sorted([_k for _k in self.dataset_file.keys() if name.replace(".mp4", "") in _k],
            #              key=lambda vk: int(vk.replace("video", "")))
            _key = _keys[cap_idx]

            out['video_noun_classes'] = self.video_captions_noun_classes[video_name]
            out['video_verb_classes'] = self.video_captions_verb_classes[video_name]
          elif self.dname == "vatex":
            _key = f'{name}_{cap_idx}'
            out['video_noun_classes'] = self.video_captions_noun_classes[name]
            out['video_verb_classes'] = self.video_captions_verb_classes[name]
          else:
            _key = name
          nc_str = self.dataset_file[_key]["annotations"][0]["noun_classes"]
          if isinstance(nc_str, str):
            nc_str = list(map(int, nc_str.replace("[", "").replace("]", "").split(",")))
          out['noun_classes'] = nc_str
          out['verb_class'] = self.dataset_file[_key]["annotations"][0]["verb_class"]

        return out
    
    def iterate_over_captions(self, batch_size):
        # the sentence order is the same as self.captions
        for s in range(0, len(self.captions), batch_size):
          e = s + batch_size
          data = []
          for sent in self.captions[s: e]:
            out = {}
            sent_ids, sent_len = self.process_sent(sent, self.max_words_in_sent)
            out['sent_ids'] = sent_ids
            out['sent_lens'] = sent_len
            data.append(out)
          outs = collate_graph_fn(data)
          yield outs
    
    def load_attn_ft_by_name(self, name, attn_ft_files):
        attn_fts = []
        for i, attn_ft_file in enumerate(attn_ft_files):
          with h5py.File(attn_ft_file, 'r') as f:
            key = name.replace('/', '_')
            attn_ft = f[key][...]
            attn_fts.append(attn_ft)
        #attn_fts = np.concatenate([attn_ft for attn_ft in attn_fts], axis=-1)
        attn_fts = [attn_ft for attn_ft in attn_fts]
        lens = [len(a) for a in attn_fts]
        if not all([l == lens[0] for l in lens]):  # not all share the same length
            #min_len = min(lens)
            #attn_fts = [attn_ft[np.linspace(0, len(attn_ft)-1, min_len, dtype=int)] for attn_ft in attn_fts]
            max_len = 48 #max(lens)
            tmp = []
            for attn_ft in attn_fts:
                if len(attn_ft) < max_len:
                    attn_ft = torch.nn.functional.interpolate(torch.from_numpy(attn_ft).float().permute(1,0).unsqueeze(0), 
                                                              size=max_len, 
                                                              mode='nearest').squeeze(0).permute(1,0).numpy()
                attn_ft = torch.nn.functional.normalize(torch.from_numpy(attn_ft[:max_len]), dim=1)
                #else:
                #    new_ft = attn_ft[np.linspace(0, len(attn_ft)-1, max_len, dtype=int)]
                tmp.append(attn_ft)
        #attn_fts = np.concatenate(attn_fts, axis=-1)
        attn_fts = np.concatenate(tmp, axis=-1)
        return attn_fts

def collate_graph_fn(data):
  outs = {}
  for key in ['names', 'attn_fts', 'attn_lens', 'sent_ids', 'sent_lens',
              'verb_masks', 'noun_masks', 'node_roles', 'rel_edges']:
    if key in data[0]:
      outs[key] = [x[key] for x in data]

  batch_size = len(data)

  # reduce attn_lens
  if 'attn_fts' in outs:
    max_len = np.max(outs['attn_lens'])
    outs['attn_fts'] = np.stack(outs['attn_fts'], 0)[:, :max_len]

  # reduce caption_ids lens
  if 'sent_lens' in outs:
    max_cap_len = np.max(outs['sent_lens'])
    outs['sent_ids'] = np.array(outs['sent_ids'])[:, :max_cap_len]
    if 'verb_masks' in outs:
        outs['verb_masks'] = np.array(outs['verb_masks'])[:, :, :max_cap_len]
        outs['noun_masks'] = np.array(outs['noun_masks'])[:, :, :max_cap_len]

  if 'noun_classes' in data[0]:
    outs['noun_classes'] = [x['noun_classes'] for x in data]
    outs['verb_class'] = [x['verb_class'] for x in data]

  if 'threshold_pos' in data[0]:
    outs["threshold_pos"] = data[0]["threshold_pos"]

  if 'video_noun_classes' in data[0]:
    outs["video_noun_classes"] = [x['video_noun_classes'] for x in data]
    outs["video_verb_classes"] = [x['video_verb_classes'] for x in data]

  return outs
