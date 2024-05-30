For all experiments, first ''run'' the .py config file to create experiment directory, etc; then call the driver .py passing the experiment directory.

Using HGR on MSR-VTT:
- To perform the Semi pretraining: 
  1. run ```python t2vretrieval/driver/configs/prepare_mlmatch_configs_MSRVTT_EAO-fts_R152_Rx101_val-nDCG_SemiHN.py .``` -> obtain directory $m (using m=directory)
  2. run ```python t2vretrieval/driver/multilevel_match.py $m/model.json $m/path.json --is_train --load_video_first --resume_file glove_checkpoint```
  3. in my experiments I used the last checkpoint (epoch.49.th) as a starting checkpoint for the additional finetunings (i.e. Semi+All and Semi+Hard use this last checkpoint as the --resume_file parameter)

- All with RANP config file: prepare_mlmatch_configs_MSRVTT_EAO-fts_R152_Rx101_val-nDCG_RANP-NoHN.py
- Semi+All with RANP config file: prepare_mlmatch_configs_MSRVTT_EAO-fts_R152_Rx101_val-nDCG_RANP-NoHN_SemiHN-PT.py
- Semi+Hard with RANP config file: prepare_mlmatch_configs_MSRVTT_EAO-fts_R152_Rx101_val-nDCG_RANP_SemiHN-PT.py
- Hard with RANP config file: prepare_mlmatch_configs_MSRVTT_EAO-fts_R152_Rx101_val-nDCG_thrPos_hardPos.py
- All config file: prepare_mlmatch_configs_MSRVTT_EAO-fts_R152_Rx101_val-nDCG_NoHN.py
- Semi+All config file: prepare_mlmatch_configs_MSRVTT_EAO-fts_R152_Rx101_val-nDCG_NoHN_SemiHN-PT.py
- Semi+Hard config file: prepare_mlmatch_configs_MSRVTT_EAO-fts_R152_Rx101_val-nDCG_SemiHN-PT.py
- Hard config file: prepare_mlmatch_configs_MSRVTT_EAO-fts_R152_Rx101_val-nDCG.py

Using HGR on EPIC-Kitchens-100:
- Semi pretraining: prepare_mlmatch_configs_EK100_fixMargin_TBN_SemiHN.py

- All with RANP config file: prepare_mlmatch_configs_EK100_TBN_thrPos_hardPos_NoHN.py
- Semi+All with RANP config file: prepare_mlmatch_configs_EK100_TBN_thrPos_hardPos_NoHN_SemiHN-PT.py
- Semi+Hard with RANP config file: prepare_mlmatch_configs_EK100_TBN_thrPos_hardPos_SemiHN-PT.py
- Hard with RANP config file: prepare_mlmatch_configs_EK100_TBN_thrPos_hardPos.py
- All config file: prepare_mlmatch_configs_EK100_fixMargin_TBN_NoHN.py
- Semi+All config file: prepare_mlmatch_configs_EK100_fixMargin_TBN_NoHN_SemiHN-PT.py
- Semi+Hard config file: prepare_mlmatch_configs_EK100_fixMargin_TBN_SemiHN-PT.py
- Hard config file: prepare_mlmatch_configs_EK100_fixMargin_TBN.py

Using HGR on MSVD:
- Example config file: prepare_mlmatch_configs_MSVD_baseline.py
- Example with RANP config file: prepare_mlmatch_configs_MSVD_TripletRANP.py

Using HGR on Charades:
- Example config file: prepare_mlmatch_configs_Charades_baseNoHN.py
- Example with RANP config file: prepare_mlmatch_configs_Charades_TripletRANP_NoHN.py

Using EAO on MSR-VTT:
- Zero shot: 
  1. run ```python t2vretrieval/driver/configs/prepare_eaomatch_configs_MSRVTT_base_testPT.py .``` -> obtain directory $m
  2. run ```python t2vretrieval/driver/eao_match.py.py $m/model.json $m/path.json --eval_set tst --resume_file PROVA_MSR_PT_noFT.th```
- NCE-RANP config file: prepare_eaomatch_configs_MSRVTT_base_R152_Rx101_EAO-fts_NormSoftmaxRANP.py
- NCE-RANP with PT config file: prepare_eaomatch_configs_MSRVTT_base_R152_Rx101_EAO-fts_NormSoftmaxRANP_EAO-PT.py  (also resume HT100M-PT while training)
- Semi pretraining: prepare_eaomatch_configs_MSRVTT_base_EAO-fts_EAO-PT_SemiHardNeg.py

- All with RANP config file: prepare_eaomatch_configs_MSRVTT_RANP_EAO-fts_noHardNeg.py
- Semi+All with RANP config file: prepare_eaomatch_configs_MSRVTT_RANP_NoHN_EAO-fts_SemiHN-PT.py
- Semi+Hard with RANP config file: prepare_eaomatch_configs_MSRVTT_RANP_EAO-fts_SemiHN-PT.py
- All config file: prepare_eaomatch_configs_MSRVTT_baseNoHN_EAO-fts_SemiHN-PT.py
- Semi+All config file: prepare_eaomatch_configs_MSRVTT_base_R152_Rx101_EAO-fts_NoHN_SemiHN-PT.py
- Semi+Hard config file: prepare_eaomatch_configs_MSRVTT_baseHN_EAO-fts_SemiHN-PT.py

Using EAO on EPIC-Kitchens-100:
- NCE-RANP config file: prepare_eaomatch_configs_EK100_base_NormSoftmax.py
- NCE-RANP with RANP config file: prepare_eaomatch_configs_EK100_base_NormSoftmaxRANP.py
- Semi pretraining: prepare_eaomatch_configs_EK100_base_SemiHardNeg.py

- All with RANP config file: prepare_eaomatch_configs_EK100_RANP_noHardNeg.py
- Semi+All with RANP config file: prepare_eaomatch_configs_EK100_RANP-NoHN_PT-SemiHN.py
- Semi+Hard with RANP config file: prepare_eaomatch_configs_EK100_RANP_PT-SemiHN.py
- All config file: prepare_eaomatch_configs_EK100_base_noHardNeg.py
- Semi+All config file: prepare_eaomatch_configs_EK100_base-NoHN_PT-SemiHN.py
- Semi+Hard config file: prepare_eaomatch_configs_EK100_base_PT-SemiHN.py

Using EAO on MSVD:
- Semi pretraining: prepare_eaomatch_configs_MSVD_baseSemiHN.py
- Semi+Hard config file: prepare_eaomatch_configs_MSVD_baseHN_SemiHNPT.py
- Semi+Hard with RANP config file: prepare_eaomatch_configs_MSVD_RANPHN_SemiHNPT.py

Using EAO on Charades:
- Example Semi config file: prepare_eaomatch_configs_Charades_baseSemiHN.py
- Example Semi+Hard config file: prepare_eaomatch_configs_Charades_baseHN_SemiHNPT.py
- Example Semi+Hard with RANP config file: prepare_eaomatch_configs_Charades_RANPHN_SemiHNPT.py
