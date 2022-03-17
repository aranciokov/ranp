# Learning video retrieval models with relevance-aware online mining
In this repo, we provide code and pretrained models for the paper ["**Learning video retrieval models with relevance-aware online mining**"](https://arxiv.org/abs/2203.08688) which has been accepted for presentation at the 21st International Conference on Image Analysis and Processing (ICIAP).

#### Python environment
Requirements: python 3, allennlp 2.8.0, h5py 3.6.0, pandas 1.3.5, spacy 2.3.5, torch 1.7.0 (also tested with 1.8)
```
# clone the repository
cd ranp
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

#### Data
- Features: 
    - TBN **EPIC-Kitchens-100** [**features**]() from [JPoSE's repo](https://github.com/mwray/Joint-Part-of-Speech-Embeddings). 
    - ResNet-152 **MSR-VTT** [**features**](https://drive.google.com/file/d/1MrViy6BPGG0xFiss0dxLmmSscYB1N-CY/view?usp=sharing) from [HGR's repo](https://github.com/cshizhe/hgr_v2t).
- Additional:
    - pre-extracted annotations for [EPIC-Kitchens-100](https://drive.google.com/file/d/1sZmbyAiOmclYSP0CZk6WRhOHqEMEF4Ej/view?usp=sharing) and [MSR-VTT](https://drive.google.com/file/d/19tahPCjOEQmfdU250qdJYufj5lGDNVm4/view?usp=sharing)
    - split folders for [EPIC-Kitchens-100](https://drive.google.com/file/d/1eYxzyCb2Jl0oeHP_y2awZhTTNz5th7X2/view?usp=sharing) and [MSR-VTT](https://drive.google.com/file/d/14CQ_6o9WN-bsl0Zx6CkCb1i3Jovj6bgi/view?usp=sharing)
    - GloVe checkpoints for [EPIC-Kitchens-100](https://drive.google.com/file/d/1q7viOUp_kByPc3-y8PIZw1A7BZcLdtAD/view?usp=sharing) and [MSR-VTT](https://drive.google.com/file/d/1UNiU-J_cRrnU1yRfRj6I7QeWwdHOvIPX/view?usp=sharing)

#### Training
To launch a training, first select a configuration file (e.g. ``prepare_mlmatch_configs_EK100_TBN_thrPos_hardPos.py``) and execute the following:

``python t2vretrieval/driver/configs/prepare_mlmatch_configs_EK100_TBN_thrPos_hardPos.py .``

This will return a folder name (where config, models, logs, etc will be saved). Let that folder be ``$resdir``. Then, execute the following to start a training:

``python t2vretrieval/driver/multilevel_match.py $resdir/model.json $resdir/path.json --is_train --load_video_first --resume_file glove_checkpoint_path``

#### Evaluating
To automatically check for the best checkpoint (after a training run):

``python t2vretrieval/driver/multilevel_match.py $resdir/model.json $resdir/path.json --eval_set tst``

To resume one of the checkpoints provided:

``python t2vretrieval/driver/multilevel_match.py $resdir/model.json $resdir/path.json --eval_set tst --resume_file checkpoint.th``

#### Pretrained models
*On EPIC-Kitchens-100:*
- Baseline model: [(35.9 nDCG, 39.5 mAP)](https://drive.google.com/file/d/1uIiUVQhrfI3GBXmNpr8jQNNI6NEWPqdU/view?usp=sharing) 
- With RAN: [thr=0.15 (48.4 nDCG, 46.5 mAP)](https://drive.google.com/file/d/1E33_C89waxqIogWJGO3a6q7eMiO_gjO9/view?usp=sharing)
- With **RANP**: [thr=0.15 (58.8 nDCG, 47.2 mAP)](https://drive.google.com/file/d/1TrT38HclugJ_l49tvfr0AbW0Pg5wrSMF/view?usp=sharing)

*On MSR-VTT:*
- Baseline model: [(25.3 nDCG)](https://drive.google.com/file/d/1gUzgtU1GTVPM6rZJO51YcTJqZXzkBFVO/view?usp=sharing) 
- With RAN: [thr=0.10 (28.7 nDCG)](https://drive.google.com/file/d/1_SKJ7K49AocmJ-IQbDlvwgI3q-jCzK0o/view?usp=sharing)
- With **RANP**: [thr=0.10 (31.1 nDCG)](https://drive.google.com/file/d/1A09yEM0PO49N_1XFdnMdzFE57k54Lz70/view?usp=sharing)

#### Acknowledgements
We thank the authors of 
 [Chen et al. (CVPR, 2020)](https://arxiv.org/abs/2003.00392) ([github](https://github.com/cshizhe/hgr_v2t)),
 [Wray et al. (ICCV, 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wray_Fine-Grained_Action_Retrieval_Through_Multiple_Parts-of-Speech_Embeddings_ICCV_2019_paper.pdf) ([github](https://github.com/mwray/Joint-Part-of-Speech-Embeddings)),
 [Wray et al. (CVPR, 2021)](https://arxiv.org/abs/2103.10095) ([github](https://github.com/mwray/Semantic-Video-Retrieval))
 for the release of their codebases. 

## Citations
If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:
```text
@article{falcon2022ranp,
  title={Learning video retrieval models with relevance-aware online mining},
  author={Falcon, Alex and Serra, Giuseppe and Lanz, Oswald},
  journal={ICIAP},
  year={2022}
}
```

## License

MIT License
