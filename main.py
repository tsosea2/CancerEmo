"""
Author: Tiberiu Sosea (tsosea2@uic.edu)
Code for CancerEmo: A Dataset for Fine-grained Emotion Detection.
"""

import ray
from ray import tune
# from ray.tune.integration.wandb import wandb_mixin
from ray.tune.examples.pbt_transformers import utils

import torch
from torch import nn
from torch.utils.data import Dataset

import transformers
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer
from transformers import BertModel, BertConfig
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.file_utils import is_torch_tpu_available

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import wandb
import json
import argparse
import pandas as pd
import random
import logging
import os
from os import path

import numpy as np

from typing import Dict, Optional, Tuple
from pathlib import Path

API_KEY_LEN = 40

def simple_bert():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.classifier = torch.nn.Linear(768, 2)
    model.num_labels = 2
    return model

def bert_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

model_mapping = {
    'bert-base-uncased' : simple_bert,
}

tokenizer_mapping = {
    'bert-base-uncased' : bert_tokenizer,
}

# To add additional models, just follow the steps from simple_bert()

def comp_metrics(eval_pred):
    metrics = {}
    model_predictions = []

    for i, elem in enumerate(eval_pred.predictions):
        model_predictions.append(np.argmax(elem))

    metrics['accuracy'] = accuracy_score(eval_pred.label_ids,
                                         model_predictions)

    metrics['recall'] = recall_score(eval_pred.label_ids, model_predictions)
    metrics['precision'] = precision_score(
        eval_pred.label_ids, model_predictions)
    metrics['f1'] = f1_score(eval_pred.label_ids, model_predictions)

    return metrics

def my_compute_obj(metrics: Dict[str, float]) -> float:
    return metrics.pop("eval_f1", None)

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)

def read_split(split_directory, emotion):
    split_dir = Path(split_directory)
    df = pd.read_csv(Path(split_dir) / (emotion + '.csv'))

    df_train = df[df['Split'] == 0]
    df_dev = df[df['Split'] == 1]
    df_test = df[df['Split'] == 2]

    zipped_train = list(
        zip(df_train['Sentence'].tolist(), df_train[emotion].tolist()))
    zipped_dev = list(
        zip(df_dev['Sentence'].tolist(), df_dev[emotion].tolist()))
    zipped_test = list(
        zip(df_test['Sentence'].tolist(), df_test[emotion].tolist()))

    random.shuffle(zipped_train)
    random.shuffle(zipped_dev)
    random.shuffle(zipped_test)

    unzipped_train = list(zip(*zipped_train))
    unzipped_dev = list(zip(*zipped_dev))
    unzipped_test = list(zip(*zipped_test))

    return list(unzipped_train[0]), list(unzipped_train[1]), \
        list(unzipped_dev[0]), list(unzipped_dev[1]), \
        list(unzipped_test[0]), list(unzipped_test[1])


def my_tuning_metrics(trial) -> Dict[str, float]:
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "num_train_epochs": tune.choice(list(range(1, 6))),
        "per_device_train_batch_size": tune.choice([4, 8, 16]),
        "gradient_accumulation_steps": tune.choice([2]),
    }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--api_key_wandb", type=str)
    parser.add_argument("--data_dir",
                        type=str,
                        help="The root directory of the data")
    parser.add_argument("--emotion",
                        type=str,
                        help="The emotion type (e.g., Anger, Joy, etc)")
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--model", type=str, default='bert-base-uncased')
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--tokenizer", type=str, default='bert-base-uncased')
    parser.add_argument("--tuning", type=int, default=1)
    parser.add_argument("--tuning_savedir", type=str,
                        default='tuning_experiment_params')
    parser.add_argument("--tuning_trials", type=int, default=20)

    args = parser.parse_args()

    if len(args.api_key_wandb) == API_KEY_LEN:
        wandb.login(key=args.api_key_wandb)

    X_train, y_train, X_val, y_val, X_test, y_test = read_split(
        args.data_dir, args.emotion)

    tokenizer = tokenizer_mapping[args.tokenizer]()

    if not path.exists('logs'):
        os.mkdir('logs')

    if not path.exists(args.tuning_savedir):
        os.mkdir(args.tuning_savedir)

    tuner_save_path = os.path.join(args.tuning_savedir, args.experiment_name)

    if not path.exists(tuner_save_path):
        os.mkdir(tuner_save_path)

    if args.tuning == 1:
        train_encodings = tokenizer(X_train, truncation=True, padding=True)
        val_encodings = tokenizer(X_val, truncation=True, padding=True)
        test_encodings = tokenizer(X_test, truncation=True, padding=True)

        train_dataset = EmotionDataset(train_encodings, y_train)
        validation_dataset = EmotionDataset(val_encodings, y_val)
        test_dataset = EmotionDataset(test_encodings, y_test)

        training_args = TrainingArguments(
            output_dir=args.tuning_savedir,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs/' + args.experiment_name,
            logging_steps=100,
            evaluation_strategy='epoch',
        )

        trainer = Trainer(args=training_args,
                          train_dataset=train_dataset,
                          eval_dataset=validation_dataset,
                          compute_metrics=comp_metrics,
                          model_init=model_mapping[args.model])

        best_metrics = trainer.hyperparameter_search(
            compute_objective=my_compute_obj,
            hp_space=my_tuning_metrics,
            n_trials=args.tuning_trials,
            direction='maximize')

        best_metrics.hyperparameters['score'] = best_metrics.objective
        best_metrics = best_metrics.hyperparameters

        with open(os.path.join(tuner_save_path, 'params.json'), 'w') as f:
            f.write(json.dumps(best_metrics))
    else:
        with open(os.path.join(tuner_save_path, 'params.json')) as f:
            best_metrics = json.load(f)

    train_encodings = tokenizer(X_train + X_val, truncation=True, padding=True)
    test_encodings = tokenizer(X_test, truncation=True, padding=True)

    train_dataset = EmotionDataset(train_encodings, y_train + y_val)
    test_dataset = EmotionDataset(test_encodings, y_test)

    f1s = []
    recalls = []
    precisions = []

    for j in range(args.repetitions):
        training_args = TrainingArguments(
            output_dir='./results' + '/' + args.experiment_name,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs/' + args.experiment_name,
            logging_steps=100,
            evaluation_strategy='epoch',
            learning_rate=best_metrics['learning_rate'],
            num_train_epochs=best_metrics['num_train_epochs'],
            per_device_train_batch_size=best_metrics['per_device_train_batch_size'],
            gradient_accumulation_steps=best_metrics['gradient_accumulation_steps'],
        )

        trainer = Trainer(args=training_args,
                          train_dataset=train_dataset,
                          eval_dataset=test_dataset,
                          compute_metrics=comp_metrics,
                          model=model_mapping[args.model]())

        trainer.train()

        results = trainer.evaluate(test_dataset)
        precisions.append(results['eval_precision'])
        recalls.append(results['eval_recall'])
        f1s.append(results['eval_f1'])

    print("Final F1:" + str(np.mean(np.array(f1s))) + '\n')
    print("Final precision:" + str(np.mean(np.array(precisions))) + '\n')
    print("Final recall:" + str(np.mean(np.array(recalls))) + '\n')

if __name__ == "__main__":
    main()
