import os
import gc
import json
import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from data.dataset import preprocess_data
from util.utils import load_config


def train(cfg):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = os.path.join(cfg["save_dir"], timestamp)
    print(save_dir)
    
    utils_dir = os.path.join(save_dir, 'utils')
    weights_dir = os.path.join(save_dir, 'weights')
    os.makedirs(utils_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    performance_file = os.path.join(utils_dir, 'model_performance.json')
    if not os.path.exists(performance_file):
        with open(performance_file, 'w') as f:
            json.dump([], f)

    offset_0 = 0
    offset_1 = 0
    for i in range(cfg['n_iter']):
        con = duckdb.connect()
        try:
            query_0 = f"""
                SELECT id, molecule_smiles, protein_name, binds
                FROM parquet_scan('{cfg['train_parquet']}')
                WHERE binds = 0
                ORDER BY random()
                LIMIT {cfg['limit']} 
                OFFSET {offset_0}
            """
            query_1 = f"""
                SELECT id, molecule_smiles, protein_name, binds
                FROM parquet_scan('{cfg['train_parquet']}')
                WHERE binds = 1
                ORDER BY random()
                LIMIT {cfg['limit']} 
                OFFSET {offset_1}
            """
            data_0 = con.query(query_0).df()
            data_1 = con.query(query_1).df()
        finally:
            con.close()

        if data_1.empty:
            con = duckdb.connect()
            try:
                random_data_1 = con.query(f"""
                    SELECT id, molecule_smiles, protein_name, binds
                    FROM parquet_scan('{cfg['train_parquet']}')
                    WHERE binds = 1
                    ORDER BY random()
                    LIMIT {cfg['limit']}
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

        smiles_list = data['molecule_smiles'].tolist()
        data = preprocess_data(data, 
                               smiles_list, 
                               f"{cfg['data_dir']}/ctd.parquet", 
                               save_dir, 
                               cfg['radius'], 
                               cfg['vec_dim'], 
                               is_train=True, 
                               important_features=['SMR_VSA4', 'SlogP_VSA1', 'fr_phenol', 'NumSaturatedCarbocycles', 'fr_Ar_NH'])
        features = data.drop(columns=['id', 'molecule_smiles', 'binds'])
        targets = data['binds'].tolist()

        features = features.astype('float32')
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.01, random_state=42)
        print(f"Train data shape : {X_train.shape}, {len(y_train)}")
        print(f"Test data shape : {X_test.shape}, {len(y_test)}")

        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': cfg['num_leaves'],
            'learning_rate': cfg['learning_rate'],
            'feature_fraction': cfg['feature_frac'],
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'l2_regularization' : cfg['l2_lambda'],
            'l1_regularization' : cfg['l1_lambda'],
            'drop_rate' : cfg['drop_prob']
        }

        evals_result = {}
        model = lgb.train(
            params,
            train_data,
            num_boost_round=cfg['num_boost'],
            valid_sets=[test_data],
            callbacks=[
                lgb.log_evaluation(10),
                lgb.early_stopping(10),
                lgb.record_evaluation(evals_result)
            ]
        )

        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]

        accuracy = accuracy_score(y_test, y_pred_binary)
        roc_auc = roc_auc_score(y_test, y_pred)

        print(f"Iteration {i+1}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        model_info = {
            'iteration': i + 1,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'model_path': os.path.join(weights_dir, f'model_{i}.txt')
        }

        model.save_model(model_info['model_path'])
        print(f"{i+1}th Model Saved.\n")

        with open(performance_file, 'r') as f:
            model_performance = json.load(f)
        
        model_performance.append(model_info)
        
        with open(performance_file, 'w') as f:
            json.dump(model_performance, f, indent=4)

        offset_0 += cfg['limit']
        offset_1 += cfg['limit']
        
        del data, X_train, X_test, y_train, y_test, train_data, test_data
        gc.collect()


if __name__ == "__main__":
    config_path = 'config.yaml'
    cfg = load_config(config_path)

    train(cfg)