# Relevance-aware online mining for a more semantic video retrieval
In this repo, we provide code and pretrained models for the paper [**Improving semantic video retrieval models by training with a relevance-aware online mining strategy** (coming soon)](), which has been accepted at **Computer Vision and Image Understanding**! The code also covers the implementation of a preliminary version of this work, called ["**Learning video retrieval models with relevance-aware online mining**"](https://arxiv.org/abs/2203.08688), which was accepted for presentation at the 21st International Conference on Image Analysis and Processing (ICIAP).

#### Python environment
Requirements: python 3, allennlp 2.8.0, h5py 3.6.0, pandas 1.3.5, spacy 2.3.5, torch 1.7.0 (also tested with 1.8)
```
# clone the repository
cd ranp
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

#### Data
- All the used data has been packed in the following [**Zenodo repository**]()! It contains:
    - Pre-extracted features for EPIC-Kitchens-100 (TBN), MSR-VTT (ResNet-152, 3D-ResNeXt-101), Charades (ViT), MSVD (ViT)
    - Split folders for train/val/test
    - GloVe checkpoints
    - HowTo100M weights for EAO

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
- Pretrained models have also been added to the [**Zenodo repository**]()! It contains:
    *On EPIC-Kitchens-100:*
    - HGR (35.9 nDCG, 39.5 mAP)
    - HGR with **RANP**, thr=0.15 (58.8 nDCG, 47.2 mAP)
    - EAO (34.5 nDCG, 35.0 mAP)
    - EAO with **RANP**, thr=0.10 (59.5 nDCG, 45.1 mAP)

    *On MSR-VTT:*
    - HGR (26.7 nDCG)
    - HGR with **RANP**, thr=0.10 (35.4 nDCG)
    - EAO (24.8 nDCG)
    - EAO with **RANP**, thr=0.10 (34.4 nDCG)
    - EAO with **RANP** (+HowTo100M PT), thr=0.10 (35.6 nDCG)
    
    *Charades:*
    - HGR ()
    - HGR with **RANP** ()
    - EAO ()
    - EAO with **RANP** ()
    
    *MSVD:*
    - HGR ()
    - HGR with **RANP** ()
    - EAO ()
    - EAO with **RANP** ()

#### Acknowledgements
This work was supported by MUR Progetti di Ricerca di Rilevante Interesse Nazionale (PRIN) 2022 (project code 2022YTE579), by the Department Strategic Plan (PSD) of the University of Udine – Interdepartmental Project on Artificial Intelligence (2020-25), and by Startup Grant IN2814 (2021-24) of University of Bolzano. We gratefully acknowledge the support from Amazon AWS Machine Learning Research Awards (MLRA) and NVIDIA AI Technology Centre (NVAITC), EMEA. We acknowledge the CINECA award under the ISCRA initiative, which provided  computing resources for this work.

We also thank the authors of 
 [Chen et al. (CVPR, 2020)](https://arxiv.org/abs/2003.00392) ([github](https://github.com/cshizhe/hgr_v2t)),
 [Wray et al. (ICCV, 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wray_Fine-Grained_Action_Retrieval_Through_Multiple_Parts-of-Speech_Embeddings_ICCV_2019_paper.pdf) ([github](https://github.com/mwray/Joint-Part-of-Speech-Embeddings)),
 [Wray et al. (CVPR, 2021)](https://arxiv.org/abs/2103.10095) ([github](https://github.com/mwray/Semantic-Video-Retrieval)),
 [Shvetsova et al. (CVPR, 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Shvetsova_Everything_at_Once_-_Multi-Modal_Fusion_Transformer_for_Video_Retrieval_CVPR_2022_paper.html) ([github](https://github.com/ninatu/everything_at_once))
 for the release of their codebases. 

## Citations
If you use this code as part of any published research, we'd really appreciate it if you could cite the following papers:
```text
@article{falcon2024improving,
  title={Improving semantic video retrieval models by training with a relevance-aware online mining strategy},
  author={Falcon, Alex and Serra, Giuseppe and Lanz, Oswald},
  journal={Computer Vision and Image Understanding},
  pages={},
  year={2024}
}
```

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
