# Relevance-aware online mining for a more semantic video retrieval
In this repo, we provide code and pretrained models for the paper **Relevance-aware online mining for a more semantic video retrieval**, which was submitted to *IEEE Transactions on Multimedia*. The code also covers the implementation of a preliminary version of this work, called ["**Learning video retrieval models with relevance-aware online mining**"](https://arxiv.org/abs/2203.08688), which was accepted for presentation at the 21st International Conference on Image Analysis and Processing (ICIAP).

#### Python environment
Requirements: python 3, allennlp 2.8.0, h5py 3.6.0, pandas 1.3.5, spacy 2.3.5, torch 1.7.0 (also tested with 1.8)
```
# clone the repository
cd ranp
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

#### Data
- Features: 
    - **EPIC-Kitchens-100**: TBN [**features**](https://drive.google.com/file/d/16_WXNg2aziVBsWjc1_egE4YjnJ_aKbrM/view?usp=sharing) 
    - **MSR-VTT**: ResNet-152 [**features**](https://drive.google.com/file/d/16l_oFh9fknJkCHYo15CmiaE18NGHZJ8a/view?usp=sharing) and 3D-ResNeXt-101 [**features**](https://drive.google.com/file/d/1jYKnLu4XQvsdlIkcBz1iP0FYkbkYLbuW/view?usp=sharing)

- Additional:
    - pre-extracted annotations for [EPIC-Kitchens-100](https://drive.google.com/file/d/1sZmbyAiOmclYSP0CZk6WRhOHqEMEF4Ej/view?usp=sharing) and [MSR-VTT](https://drive.google.com/file/d/19tahPCjOEQmfdU250qdJYufj5lGDNVm4/view?usp=sharing)
    - split folders for [EPIC-Kitchens-100](https://drive.google.com/file/d/1eYxzyCb2Jl0oeHP_y2awZhTTNz5th7X2/view?usp=sharing) and [MSR-VTT](https://drive.google.com/file/d/14CQ_6o9WN-bsl0Zx6CkCb1i3Jovj6bgi/view?usp=sharing)
    - GloVe checkpoints for [EPIC-Kitchens-100](https://drive.google.com/file/d/1q7viOUp_kByPc3-y8PIZw1A7BZcLdtAD/view?usp=sharing) and [MSR-VTT](https://drive.google.com/file/d/1UNiU-J_cRrnU1yRfRj6I7QeWwdHOvIPX/view?usp=sharing)
    - HowTo100M [weights]() for EAO

#### Training
To launch a training, first select a configuration file (e.g. ``prepare_mlmatch_configs_EK100_TBN_thrPos_hardPos.py``) and execute the following:

``python t2vretrieval/driver/configs/prepare_mlmatch_configs_EK100_TBN_thrPos_hardPos.py .``

This will return a folder name (where config, models, logs, etc will be saved). Let that folder be ``$resdir``. Then, execute the following to start a training:

``python t2vretrieval/driver/multilevel_match.py $resdir/model.json $resdir/path.json --is_train --load_video_first --resume_file glove_checkpoint_path``

Replace ``multilevel_match.py`` with ``eao_match.py`` to use Everything-at-once (txt-vid version) in place of HGR.

#### Evaluating
To automatically check for the best checkpoint (after a training run):

``python t2vretrieval/driver/multilevel_match.py $resdir/model.json $resdir/path.json --eval_set tst``

To resume one of the checkpoints provided:

``python t2vretrieval/driver/multilevel_match.py $resdir/model.json $resdir/path.json --eval_set tst --resume_file checkpoint.th``

#### Pretrained models
*On EPIC-Kitchens-100:*
- HGR: [(35.9 nDCG, 39.5 mAP)](https://drive.google.com/file/d/1uIiUVQhrfI3GBXmNpr8jQNNI6NEWPqdU/view?usp=sharing) 
- HGR with **RANP**: [thr=0.15 (58.8 nDCG, 47.2 mAP)](https://drive.google.com/file/d/1TrT38HclugJ_l49tvfr0AbW0Pg5wrSMF/view?usp=sharing)
- EAO: [(34.5 nDCG, 35.0 mAP)](https://drive.google.com/file/d/1APeQO1tj4ErzH2AvCbRFN1wR6WsHNmp-/view?usp=sharing)
- EAO with RANP: [thr=0.10 (59.5 nDCG, 45.1 mAP)](https://drive.google.com/file/d/1AokXrQh5wvy655Jf-6zYL-btDtrO-jxI/view?usp=sharing)

*On MSR-VTT:*
- HGR: [(26.7 nDCG)](https://drive.google.com/file/d/1a7dtZsDoAoxoO3Zi0FL7yr5nkopUC08r/view?usp=sharing) 
- HGR with **RANP**: [thr=0.10 (35.4 nDCG)](https://drive.google.com/file/d/14PK9lUoZVGK0Jv8YuhA4iY7Yaa0cR9zZ/view?usp=sharing)
- EAO: [(24.8 nDCG)](https://drive.google.com/file/d/1biopwaPo8UExBs47PnAQJVqnOqCbRnwo/view?usp=sharing)
- EAO with RANP: [thr=0.10 (34.4 nDCG)](https://drive.google.com/file/d/19s5lI4kBdI6XHwae__B-rUhjrTF8vz7o/view?usp=sharing)
- EAO with RANP (+HowTo100M PT): [thr=0.10 (35.6 nDCG)](https://drive.google.com/file/d/1u44zv7PZGHiF6vwS2WYp2oJY4uKmnnoT/view?usp=sharing)

#### Acknowledgements
We thank the authors of 
 [Chen et al. (CVPR, 2020)](https://arxiv.org/abs/2003.00392) ([github](https://github.com/cshizhe/hgr_v2t)),
 [Wray et al. (ICCV, 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wray_Fine-Grained_Action_Retrieval_Through_Multiple_Parts-of-Speech_Embeddings_ICCV_2019_paper.pdf) ([github](https://github.com/mwray/Joint-Part-of-Speech-Embeddings)),
 [Wray et al. (CVPR, 2021)](https://arxiv.org/abs/2103.10095) ([github](https://github.com/mwray/Semantic-Video-Retrieval)),
 [Shvetsova et al. (CVPR, 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Shvetsova_Everything_at_Once_-_Multi-Modal_Fusion_Transformer_for_Video_Retrieval_CVPR_2022_paper.html) ([github](https://github.com/ninatu/everything_at_once))
 for the release of their codebases. 

## Citations
If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:
```text
@inproceedings{falcon2022learning,
  title={Learning video retrieval models with relevance-aware online mining},
  author={Falcon, Alex and Serra, Giuseppe and Lanz, Oswald},
  booktitle={International Conference on Image Analysis and Processing},
  pages={182--194},
  year={2022},
  organization={Springer}
}
```

## License

MIT License
