import os
import sys
import argparse
import json
import time

import t2vretrieval.models.everything_at_once_model
import t2vretrieval.readers.rolegraphs

import torch.utils.data.dataloader as dataloader
import framework.run_utils
import framework.logbase


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('model_cfg_file')
  parser.add_argument('path_cfg_file')
  parser.add_argument('--is_train', default=False, action='store_true')
  parser.add_argument('--resume_file', default=None)
  parser.add_argument('--eval_set')
  parser.add_argument('--num_workers', type=int, default=0)
  parser.add_argument('--load_video_first', action='store_true', default=False)
  opts = parser.parse_args()

  path_cfg = framework.run_utils.gen_common_pathcfg(
    opts.path_cfg_file, is_train=opts.is_train)
  if path_cfg.log_file is not None:
    _logger = framework.logbase.set_logger(path_cfg.log_file, 'trn_%f'%time.time())
  else:
    _logger = None
 
  model_cfg = t2vretrieval.models.everything_at_once_model.EverythingAtOnceModelConfig()
  model_fn = t2vretrieval.models.everything_at_once_model.EverythingAtOnceModel
  dataset_fn = t2vretrieval.readers.rolegraphs.RoleGraphDataset
  
  model_cfg.load(opts.model_cfg_file)
  if hasattr(model_cfg, "use_word2vec") and model_cfg.use_word2vec:
    dataset_fn = t2vretrieval.readers.rolegraphs.Word2VecRoleGraphDataset
  if "1kA" in path_cfg.name_file['val']:
    dataset_fn = t2vretrieval.readers.rolegraphs.Word2VecNoGraphsDataset
    
  if hasattr(model_cfg, "mine_hard_positives") and model_cfg.mine_hard_positives:
    model_fn = t2vretrieval.models.everything_at_once_model.EverythingAtOnceRANPModel
  if hasattr(model_cfg, "use_normsoftmax") and model_cfg.use_normsoftmax:
    model_fn = t2vretrieval.models.everything_at_once_model.EverythingAtOnceNormSoftModel
  if hasattr(model_cfg, "use_word2vec") and model_cfg.use_word2vec:
    model_fn = t2vretrieval.models.everything_at_once_model.EverythingAtOnceWord2VecModel
    if hasattr(model_cfg, "threshold_pos") and model_cfg.threshold_pos < 1.:
        model_fn = t2vretrieval.models.everything_at_once_model.EverythingAtOnceRANWord2VecModel
  if hasattr(model_cfg, "use_normsoftmax") and model_cfg.use_normsoftmax and hasattr(model_cfg, "use_word2vec") and model_cfg.use_word2vec:
    model_fn = t2vretrieval.models.everything_at_once_model.EverythingAtOnceNormSoftW2VModel
  if hasattr(model_cfg, "mine_hard_positives") and model_cfg.mine_hard_positives and hasattr(model_cfg, "use_word2vec") and model_cfg.use_word2vec:
    print("Using RANP with word2vec-GN text")
    model_fn = t2vretrieval.models.everything_at_once_model.EverythingAtOnceRANPWord2VecModel
  if hasattr(model_cfg, "use_normsoftmax_ranp") and model_cfg.use_normsoftmax_ranp and hasattr(model_cfg, "use_word2vec") and model_cfg.use_word2vec:
    model_fn = t2vretrieval.models.everything_at_once_model.EverythingAtOnceNormSoftRANPModel
  if hasattr(model_cfg, "mine_hard_positives") and model_cfg.mine_hard_positives and hasattr(model_cfg, "use_word2vec") and model_cfg.use_word2vec and hasattr(model_cfg, "rap_only") and model_cfg.rap_only:
    model_fn = t2vretrieval.models.everything_at_once_model.EverythingAtOnceRAPOnlyW2VModel
  _model = model_fn(model_cfg, _logger=_logger)
  
  collate_fn = t2vretrieval.readers.rolegraphs.collate_graph_fn

  if "epic100" in path_cfg.int2word_file:
    dname = "epic"
  elif "msr-vtt" in path_cfg.int2word_file:
    dname = "msr-vtt"
    if "1kA" in path_cfg.name_file['val']:
        dname = "msr-vtt-1kA"
  elif "msvd" in path_cfg.int2word_file:
    dname = "msvd"
  elif "charades" in path_cfg.int2word_file:
    dname = "charades"
  else:
    assert False, f'can not recognize dataset name from {path_cfg.int2word_file}'

  if opts.is_train:
    model_cfg.save(os.path.join(path_cfg.log_dir, 'model.cfg'))
    path_cfg.save(os.path.join(path_cfg.log_dir, 'path.cfg'))
    json.dump(vars(opts), open(os.path.join(path_cfg.log_dir, 'opts.cfg'), 'w'), indent=2)

    trn_dataset = dataset_fn(path_cfg.name_file['trn'], 
      path_cfg.attn_ft_files['trn'], path_cfg.word2int_file, model_cfg.max_words_in_sent, 
      model_cfg.num_verbs, model_cfg.num_nouns,
      path_cfg.ref_caption_file['trn'], path_cfg.ref_graph_file['trn'],
      is_train=True, _logger=_logger, 
      max_attn_len=model_cfg.max_frames_in_video, load_video_first=opts.load_video_first,
                             dataset_file=path_cfg.dataset_file['trn'],
                             dname=dname,
                             threshold_pos=model_cfg.threshold_pos if hasattr(model_cfg, "threshold_pos") else 1.)
    trn_reader = dataloader.DataLoader(trn_dataset, batch_size=model_cfg.trn_batch_size, 
      shuffle=True, collate_fn=collate_fn, num_workers=opts.num_workers)
    val_dataset = dataset_fn(path_cfg.name_file['val'], 
      path_cfg.attn_ft_files['val'], path_cfg.word2int_file, model_cfg.max_words_in_sent, 
      model_cfg.num_verbs, model_cfg.num_nouns,
      path_cfg.ref_caption_file['val'], path_cfg.ref_graph_file['val'],
      is_train=False, _logger=_logger, 
      max_attn_len=model_cfg.max_frames_in_video, load_video_first=opts.load_video_first,
                             dname=dname,
                             rel_mat_path=path_cfg.val_relevance_file if hasattr(path_cfg, "val_relevance_file") else "")
    val_reader = dataloader.DataLoader(val_dataset, batch_size=model_cfg.tst_batch_size, 
      shuffle=False, collate_fn=collate_fn, num_workers=opts.num_workers)

    _model.train(trn_reader, val_reader, path_cfg.model_dir, path_cfg.log_dir,
      resume_file=opts.resume_file)

  else:
    tst_dataset = dataset_fn(path_cfg.name_file[opts.eval_set],
      path_cfg.attn_ft_files[opts.eval_set], path_cfg.word2int_file, model_cfg.max_words_in_sent, 
      model_cfg.num_verbs, model_cfg.num_nouns,
      path_cfg.ref_caption_file[opts.eval_set], path_cfg.ref_graph_file[opts.eval_set],
      is_train=False, _logger=_logger, 
      max_attn_len=model_cfg.max_frames_in_video, load_video_first=opts.load_video_first,
                             is_test=True, dname=dname, rel_mat_path=path_cfg.relevance_file)
    tst_reader = dataloader.DataLoader(tst_dataset, batch_size=model_cfg.tst_batch_size,
      shuffle=False, collate_fn=collate_fn, num_workers=opts.num_workers)

    model_str_scores = []
    is_first_eval = True
    if opts.resume_file is None:
      model_files = framework.run_utils.find_best_val_models(path_cfg.log_dir, path_cfg.model_dir)
    else:
      model_files = {'predefined': opts.resume_file}

    for measure_name, model_file in model_files.items():
      if 'predefined' not in measure_name and 'rsum' not in measure_name:
        continue
      set_pred_dir = os.path.join(path_cfg.pred_dir, opts.eval_set)
      if not os.path.exists(set_pred_dir):
        os.makedirs(set_pred_dir)
      tst_pred_file = os.path.join(set_pred_dir, 
        os.path.splitext(os.path.basename(model_file))[0]+'.npy')
      
      scores = _model.test(tst_reader, tst_pred_file, tst_model_file=model_file)
      if scores is not None:
        if is_first_eval:
          score_names = scores.keys()
          model_str_scores.append(','.join(score_names))
          is_first_eval = False
          print(model_str_scores[-1])
        str_scores = [measure_name, os.path.basename(model_file)]
        for score_name in score_names:
          str_scores.append('%.2f'%(scores[score_name]))
        str_scores = ','.join(str_scores)
        print(str_scores)
        model_str_scores.append(str_scores)

    if len(model_str_scores) > 0:
      score_log_file = os.path.join(path_cfg.pred_dir, opts.eval_set, 'scores.csv')
      with open(score_log_file, 'w') as f:
        for str_scores in model_str_scores:
          print(str_scores, file=f)


if __name__ == '__main__':
  main()

