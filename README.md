# CancerEmo

This is the code for our EMNLP 2020 paper [Canceremo: A Dataset for Fine-Grained Emotion Detection](https://www.aclweb.org/anthology/2020.emnlp-main.715/). The dataset is available at https://drive.google.com/file/d/1kYJaCYBL1B8dCJYojp34W2C6smUfYPvv/view?usp=sharing. If you use this dataset, please cite our paper:

```bibtex
@inproceedings{sosea-caragea-2020-canceremo,
    title = "{C}ancer{E}mo: A Dataset for Fine-Grained Emotion Detection",
    author = "Sosea, Tiberiu  and
      Caragea, Cornelia",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.715",
    doi = "10.18653/v1/2020.emnlp-main.715",
    pages = "8892--8904",
}
```
## Abstract

Emotions are an important element of human nature, often affecting the overall wellbeing of a person. Therefore, it is no surprise that the health domain is a valuable area of interest for emotion detection, as it can provide medical staff or caregivers with essential information about patients. However, progress on this task has been hampered by the absence of large labeled datasets. To this end, we introduce CancerEmo, an emotion dataset created from an online health community and annotated with eight fine-grained emotions. We perform a comprehensive analysis of these emotions and develop deep learning models on the newly created dataset. Our best BERT model achieves an average F1 of 71%, which we improve further using domain-specific pre-training.

## Baselines

We provide the code to train and do hyperparameter tuning for our main BERT baseline. First, configure the environment:

```
$ conda create --name CancerEmo python=3.8
$ conda activate CancerEmo
$ pip install -r requirements.txt
```

Then, the code can be run using:

```
TOKENIZERS_PARALLELISM=false python main.py \
    --data_dir <data_path> \
    --emotion <emotion> \
    --tuning_savedir <savedir> \
    --experiment_name <model_savedir> \
    --repetitions 10 \
    --tuning 1 \
    --tuning_trials 20 \
    --api_key_wandb <your_wandb_key>
```

`emotion` can take one of the following values [`Sadness`, `Joy`, `Fear`, `Anger`, `Surprise`, `Disgust`, `Trust`, `Anticipation`]

## Pre-training using MLM

We also provide a short notebook at `mlm/csn_mlm.ipynb` to train BERT using mlm on additional data. You only need to create a file with a sentence per line, and fill its path in the notebook. After training, you can use the checkpoint in the main code. Unfortunately, we cannot provide the CSN forum, but one has permissions to crawl it.
