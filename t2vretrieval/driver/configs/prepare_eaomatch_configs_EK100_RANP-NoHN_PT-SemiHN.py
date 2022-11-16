import os
import sys
import argparse
import numpy as np
import json

import t2vretrieval.models.mlmatch
import t2vretrieval.models.everything_at_once_model

from t2vretrieval.models.mlmatch import VISENC, TXTENC
from t2vretrieval.models.everything_at_once_model import FUSENC

def prepare_match_model(root_dir):
  anno_dir = os.path.join(root_dir, 'annotation', 'epic100RET')
  attn_ft_dir = os.path.join(root_dir, 'ordered_feature', 'epic100SA')
  split_dir = os.path.join(root_dir, 'public_split_epic100')
  res_dir = os.path.join(root_dir, 'results', 'RET.released')
  
  attn_ft_names = ['TBN.pth']
  num_words = len(np.load(os.path.join(anno_dir, 'int2word_epic100.npy')))

  model_cfg = t2vretrieval.models.everything_at_once_model.EverythingAtOnceModelConfig()
  model_cfg.threshold_pos = 0.15
  model_cfg.mine_hard_positives = True
  model_cfg.margin_pos = 0.2
  model_cfg.use_word2vec = True

  # PARAMS
  dim_embed = 4096  # to be divided by 3*num_heads
  max_frames_in_video = 20
  max_words_in_sent = 30

  fusenc_cfg = model_cfg.subcfgs[FUSENC]
  fusenc_cfg.max_frames_in_video = max_frames_in_video 
  fusenc_cfg.max_words_in_sent = max_words_in_sent
  fusenc_cfg.projection_dim = 6144
  fusenc_cfg.use_positional_embed = False
  fusenc_cfg.individual_projections = True
  fusenc_cfg.projection = "gated"
  fusenc_cfg.dim_embed = dim_embed
  fusenc_cfg.fusion_params = {
      "embed_dim": dim_embed,
      "depth": 1,
      "num_heads": 64, 
      "mlp_ratio": 1.,
      "qkv_bias": True,
      "drop_rate": 0.,
      "attn_drop_rate": 0.,
      "drop_path_rate": 0., 
      "norm_layer": None,
      "act_layer": None,
      "use_cls_token": False
  }

  model_cfg.max_frames_in_video = max_frames_in_video 
  model_cfg.max_words_in_sent = max_words_in_sent
  model_cfg.num_verbs = 4
  model_cfg.num_nouns = 6

  model_cfg.attn_fusion = 'embed' # sim, embed
  model_cfg.simattn_sigma = 4
  model_cfg.margin = 0.2
  model_cfg.loss_direction = 'bi'

  model_cfg.num_epoch = 50
  model_cfg.max_violation = False #False
  model_cfg.hard_topk = 1 #3
  model_cfg.loss_weights = None #[1, 0.2, 0.2, 0.2]

  model_cfg.trn_batch_size = 64
  model_cfg.tst_batch_size = 300
  model_cfg.monitor_iter = 1000
  model_cfg.summary_iter = 1000

  visenc_cfg = model_cfg.subcfgs[VISENC]
  visenc_cfg.dim_fts = [3072]
  visenc_cfg.dim_embed = dim_embed 
  visenc_cfg.dropout = 0.2
  visenc_cfg.share_enc = False

  txtenc_cfg = model_cfg.subcfgs[TXTENC]
  txtenc_cfg.num_words = num_words
  txtenc_cfg.dim_word = 300 
  txtenc_cfg.fix_word_embed = True
  txtenc_cfg.dim_embed = dim_embed 

  txtenc_name = 'fix' if txtenc_cfg.fix_word_embed else ''

  output_dir = os.path.join(res_dir, 'eaomatch', 
    'ek100_RANP-NoHN_PT-SemiHN_m%smPos%sthr%s.vis.%s.txt.%s.%d.%dheads.%s%sProj%d.MLPr%.1f.loss.%s.af.%s.%d%s.w2v.init.%dep'%
    (model_cfg.margin, model_cfg.margin_pos, model_cfg.threshold_pos, '-'.join(attn_ft_names),
      txtenc_name, 
      visenc_cfg.dim_embed, 
      fusenc_cfg.fusion_params['num_heads'],
      'PosEm.' if fusenc_cfg.use_positional_embed else '',
      'Ind' if fusenc_cfg.individual_projections else 'One',
      fusenc_cfg.projection_dim,
      fusenc_cfg.fusion_params['mlp_ratio'],
      model_cfg.loss_direction, 
      model_cfg.attn_fusion, model_cfg.simattn_sigma,
      '.4loss' if model_cfg.loss_weights is not None else '',
      model_cfg.num_epoch
      )
    )
  print(output_dir)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  model_cfg.save(os.path.join(output_dir, 'model.json'))

  path_cfg = {
    'output_dir': output_dir,
    'attn_ft_files': {},
    'name_file': {},
    'word2int_file': os.path.join(anno_dir, 'word2int_epic100.json'),
    'int2word_file': os.path.join(anno_dir, 'int2word_epic100.npy'),
    'ref_caption_file': {},
    'dataset_file': {},
    'ref_graph_file': {},
    'val_relevance_file': os.path.join(anno_dir, "validation_rel_mat_epic100_iciap.pkl"),
    'relevance_file': os.path.join(anno_dir, "epic100_rel_mat.pkl")
  }
  for setname in ['trn', 'val', 'tst']:
    path_cfg['attn_ft_files'][setname] = [
      os.path.join(attn_ft_dir, ft_name, 'EPIC100_TBN_feats_SA_%s.hdf5'%setname) for ft_name in attn_ft_names
    ]
    path_cfg['name_file'][setname] = os.path.join(split_dir, '%s_names.npy'%setname)
    path_cfg['ref_caption_file'][setname] = os.path.join(anno_dir, 'ref_captions.json')
    if setname == 'trn':
      path_cfg['dataset_file'][setname] = os.path.join(anno_dir, 'epic100_{}.json'.format('train' if setname=='trn' else 'test'))
    path_cfg['ref_graph_file'][setname] = os.path.join(anno_dir, 'sent2rolegraph.augment.json')
    
  with open(os.path.join(output_dir, 'path.json'), 'w') as f:
    json.dump(path_cfg, f, indent=2)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('root_dir')
  opts = parser.parse_args()

  prepare_match_model(opts.root_dir)
  
