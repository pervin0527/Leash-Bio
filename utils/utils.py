import os
import yaml
import json
import joblib
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

from data.dataset import (num_class_data, unique_per_columns, get_protein_sequences, 
                          load_protein_sequences_from_file, save_protein_ctd_to_parquet, preprocess_molecule)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def prepare_dataset(train_parq_path, test_parq_path, output_path, uniprot_dict, radius, dim, debug=False):
    num_b0, num_b1 = num_class_data(train_parq_path)
    num_molecules, target_proteins = unique_per_columns(train_parq_path, output_path)
    target_pt_seqs = get_protein_sequences(uniprot_dict, output_path)
    
    protein_sequences = load_protein_sequences_from_file(f"{output_path}/protein_sequence.json")
    save_protein_ctd_to_parquet(protein_sequences, output_path)
    
    preprocess_molecule(train_parq_path,
                        f"{output_path}/ctd.parquet",
                        output_path,
                        "train_added",
                        radius=radius,
                        dim=dim,
                        debug=debug)

    preprocess_molecule(test_parq_path,
                        f"{output_path}/ctd.parquet",
                        output_path,
                        "test_added",
                        radius=radius,
                        dim=dim,
                        debug=debug)


def save_normalizers(preprocessor, save_path):
    joblib.dump(preprocessor, save_path)


def load_normalizers(load_path):
    return joblib.load(load_path)


def load_models(weights_dir):
    models = []
    for i in range(len(os.listdir(weights_dir))):
        model_path = os.path.join(weights_dir, f'model_{i}.txt')
        model = lgb.Booster(model_file=model_path)
        models.append(model)
    return models


def load_top_k_models(performance_file, k):
    with open(performance_file, 'r') as f:
        model_performance = json.load(f)
    
    top_k_models_info = sorted(model_performance, key=lambda x: x['roc_auc'], reverse=True)[:k]
    models = []
    for model_info in top_k_models_info:
        model = lgb.Booster(model_file=model_info['model_path'])
        models.append(model)
    
    return models


def predict_with_models(models, data):
    preds = []
    for model in models:
        pred = model.predict(data, num_iteration=model.best_iteration)
        preds.append(pred)
    avg_pred = np.mean(preds, axis=0)
    return avg_pred