import os
import gc
import json
import duckdb
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
from utils.utils import load_config, prepare_dataset


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
            SELECT id, molecule_smiles, protein_name, binds
            FROM parquet_scan('{parquet_path}')
            WHERE binds = 0
            ORDER BY random()
            LIMIT {limit}
        """
        query_1 = f"""
            SELECT id, molecule_smiles, protein_name, binds
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
                SELECT id, molecule_smiles, protein_name, binds
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


def train(parquet_path, ctd_path, limit, radius, dim, n_iter, save_dir, nbr=100, lr=0.01, ff=1.0, n_leaves=31):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = os.path.join(save_dir, timestamp)
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
    for i in range(n_iter):
        con = duckdb.connect()
        try:
            query_0 = f"""
                SELECT id, molecule_smiles, protein_name, binds
                FROM parquet_scan('{parquet_path}')
                WHERE binds = 0
                ORDER BY random()
                LIMIT {limit} 
                OFFSET {offset_0}
            """
            query_1 = f"""
                SELECT id, molecule_smiles, protein_name, binds
                FROM parquet_scan('{parquet_path}')
                WHERE binds = 1
                ORDER BY random()
                LIMIT {limit} 
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

        smiles_list = data['molecule_smiles'].tolist()
        data = preprocess_data(data, smiles_list, ctd_path, save_dir, radius, dim)
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
            'num_leaves': n_leaves,
            'learning_rate': lr,
            'feature_fraction': ff,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        }

        evals_result = {}
        model = lgb.train(
            params,
            train_data,
            num_boost_round=nbr,
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

        offset_0 += limit
        offset_1 += limit
        
        del data, X_train, X_test, y_train, y_test, train_data, test_data
        gc.collect()


if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_config(config_path)

    DEBUG = config['debug']
    PREPARE = config['prepare']

    DATA_DIR = config['data_dir']
    OUTPUT_DIR = config['output_dir']
    SAVE_DIR = config['save_dir']

    TRAIN_PARQUET = config['train_parquet']
    TEST_PARQUET = config['test_parquet']
    UNIPROT_DICT = config['uniprot_dicts']

    N_ITER = config['n_iter']
    LIMIT = config['limit']
    RADIUS = config['radius']
    DIM = config['vec_dim']

    NBR = config['num_boost']
    LR = config['learning_rate']
    FF = config['feature_frac']
    N_LEAVES = config['num_leaves']

    if PREPARE:
        prepare_dataset(TRAIN_PARQUET, TEST_PARQUET, OUTPUT_DIR, UNIPROT_DICT, RADIUS, DIM, DEBUG)

    # parameter_search(TRAIN_PARQUET, f"{OUTPUT_DIR}/ctd.parquet", LIMIT, RADIUS, DIM, SAVE_DIR)
    train(TRAIN_PARQUET, f"{OUTPUT_DIR}/ctd.parquet", LIMIT, RADIUS, DIM, N_ITER, SAVE_DIR, NBR, LR, FF, N_LEAVES)