import os
import gc
import yaml
import json
import joblib
import duckdb
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

from datetime import datetime
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from data.dataset import preprocess_data

from data.dataset import (num_class_data, unique_per_columns, get_protein_sequences, 
                          load_protein_sequences_from_file, save_protein_ctd_to_parquet)


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


def parameter_search(parquet_path, ctd_path, limit, radius, dim, save_dir):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = os.path.join(save_dir, timestamp)
    print(f"Save directory: {save_dir}")

    utils_dir = os.path.join(save_dir, 'utils')
    weights_dir = os.path.join(save_dir, 'weights')
    os.makedirs(utils_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    performance_file = os.path.join(utils_dir, 'model_performance.json')
    if not os.path.exists(performance_file):
        with open(performance_file, 'w') as f:
            json.dump([], f)

    con = duckdb.connect()
    try:
        query_0 = f"""
            SELECT molecule_smiles, protein_name, binds
            FROM parquet_scan('{parquet_path}')
            WHERE binds = 0
            ORDER BY random()
            LIMIT {limit}
        """
        query_1 = f"""
            SELECT molecule_smiles, protein_name, binds
            FROM parquet_scan('{parquet_path}')
            WHERE binds = 1
            ORDER BY random()
            LIMIT {limit}
        """
        data_0 = con.query(query_0).df()
        data_1 = con.query(query_1).df()
    finally:
        con.close()

    if data_1.empty:
        con = duckdb.connect()
        try:
            random_data_1 = con.query(f"""
                SELECT molecule_smiles, protein_name, binds
                FROM parquet_scan('{parquet_path}')
                WHERE binds = 1
                ORDER BY random()
                LIMIT {limit}
            """).df()
            data = pd.concat([data_0, random_data_1])
        finally:
            con.close()
    else:
        data = pd.concat([data_0, data_1])

    data = data.sample(frac=1).reset_index(drop=True)
    binds_0_count = data[data['binds'] == 0].shape[0]
    binds_1_count = data[data['binds'] == 1].shape[0]
    print(f"Dataset shape: {data.shape}, binds=0 count: {binds_0_count}, binds=1 count: {binds_1_count}")

    # 데이터 전처리
    smiles_list = data['molecule_smiles'].tolist()
    data = preprocess_data(data, smiles_list, ctd_path, save_dir, radius, dim)

    # 데이터 분할
    features = data.drop(columns=['id', 'molecule_smiles', 'binds']).astype('float32')
    targets = data['binds'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    print(f"Train data shape: {X_train.shape}, {len(y_train)}")
    print(f"Test data shape: {X_test.shape}, {len(y_test)}")

    param_dist = {
        'num_leaves': sp_randint(20, 100),  # num_leaves 범위를 증가
        'learning_rate': sp_uniform(0.01, 0.1),
        'feature_fraction': sp_uniform(0.7, 0.3),
        'max_depth': sp_randint(3, 15),  # max_depth 범위를 증가
        'min_child_samples': sp_randint(5, 50),  # min_child_samples 추가 및 범위 설정
        'min_split_gain': sp_uniform(0.0, 0.1),  # min_split_gain 추가 및 범위 설정
        'reg_alpha': sp_uniform(0, 0.1),
        'reg_lambda': sp_uniform(0, 0.1)
    }

    lgb_estimator = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        metric='binary_logloss',
        n_estimators=100,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
    )

    print("Starting RandomizedSearchCV for hyperparameter tuning...")
    rs = RandomizedSearchCV(
        estimator=lgb_estimator,
        param_distributions=param_dist,
        n_iter=3,
        scoring='roc_auc',
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    rs.fit(X_train, y_train)
    print("Hyperparameter tuning completed.")

    best_params = rs.best_params_
    print(f"Best Parameters: {best_params}")

    print("Starting model training with best parameters...")
    lgb_estimator.set_params(**best_params)
    lgb_estimator.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_names=['valid_0'],
        early_stopping_rounds=10,
        verbose=10
    )
    print("Model training completed.")

    y_pred = lgb_estimator.predict(X_test)
    y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]

    accuracy = accuracy_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred)

    model_info = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'best_params': best_params,
        'model_path': os.path.join(weights_dir, 'best_model.txt')
    }

    lgb_estimator.booster_.save_model(model_info['model_path'])
    print(f"Best Model Saved.\n")

    with open(performance_file, 'r') as f:
        model_performance = json.load(f)
    model_performance.append(model_info)

    with open(performance_file, 'w') as f:
        json.dump(model_performance, f, indent=4)

    del data, X_train, X_test, y_train, y_test
    gc.collect()

    print("Training process completed.")