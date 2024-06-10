import os
import gc
import json
import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb

from time import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from utils.utils import load_config, prepare_dataset, save_normalizers
from data.dataset import process_smiles_list, normalize_ctd, normalize_mol_descriptors


def train(parquet_path, ctd_path, limit, radius, dim, n_iter, save_dir, nbr=100, lr=0.01, ff=1.0, n_leaves=30):
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
        start_time = time()
        con = duckdb.connect()
        try:
            data_0 = con.query(f"""
                SELECT *
                FROM parquet_scan('{parquet_path}')
                WHERE binds = 0
                ORDER BY random()
                LIMIT {limit} OFFSET {offset_0}
            """).df()

            data_1 = con.query(f"""
                SELECT *
                FROM parquet_scan('{parquet_path}')
                WHERE binds = 1
                ORDER BY random()
                LIMIT {limit} OFFSET {offset_1}
            """).df()
        finally:
            con.close()

        if data_1.empty:
            con = duckdb.connect()
            try:
                random_data_1 = con.query(f"""
                    SELECT *
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
        print(f"Iter {i+1} : Dataset shape : {data.shape}, binds=0 count : {binds_0_count}, binds=1 count : {binds_1_count}")

        smiles_list = data['molecule_smiles'].tolist()
        results = process_smiles_list(smiles_list, radius, dim, desc=True)

        fingerprints = [result['fingerprint'] for result in results]
        data['fingerprints'] = fingerprints

        fingerprints = data['fingerprints'].apply(lambda x: np.array([int(char) for char in x]))
        fingerprint_df = pd.DataFrame(fingerprints.tolist(), index=data.index)
        data = pd.concat([data, fingerprint_df], axis=1)
        data.drop(columns=['fingerprints'], inplace=True)
        print("Preprocess 1 : FingerPrints Merged.")
        print(f"Time for Fingerprints Merging: {time() - start_time:.2f} seconds")
        
        start_time = time()
        if i == 0:
            ctd_df = pd.read_parquet(ctd_path, engine='pyarrow')
            ctd_df = normalize_ctd(ctd_df, utils_dir)
            ctd_df.to_parquet(os.path.join(f"{save_dir}/utils", 'normalized_ctd.parquet'), engine='pyarrow')
        else:
            ctd_df = pd.read_parquet(f"{save_dir}/utils/normalized_ctd.parquet", engine='pyarrow')

        data = pd.merge(data, ctd_df, on='protein_name', how='left')
        print("Preprocess 2 : Protein CTD Merged.")
        print(f"Time for Protein CTD Merging: {time() - start_time:.2f} seconds")

        start_time = time()
        protein_one_hot = pd.get_dummies(data['protein_name'], prefix='protein_')
        data = pd.concat([data, protein_one_hot], axis=1)
        data.drop(columns=['protein_name'], inplace=True)
        print("Preprocess 3 : protein_name Onehot encoded.")
        print(f"Time for Protein One-hot Encoding: {time() - start_time:.2f} seconds")

        start_time = time()
        descriptors_list = [result['descriptors'] for result in results]
        descriptor_df = pd.DataFrame(descriptors_list)
        excluded_descriptors = descriptor_df.columns[descriptor_df.isna().any()].tolist()
        descriptor_df.drop(columns=excluded_descriptors, inplace=True)
        descriptor_df, preprocessor = normalize_mol_descriptors(descriptor_df)
        
        normalizer_path = os.path.join(utils_dir, 'descriptor_normalizer.pkl')
        save_normalizers(preprocessor, normalizer_path)
        
        data = pd.concat([data, descriptor_df], axis=1)
        print("Preprocess 4 : Normalized Molecule Descriptor Merged.")
        print(f"Time for Molecule Descriptor Normalization: {time() - start_time:.2f} seconds")

        exclude_columns = ['id', 'buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles']
        data.drop(columns=exclude_columns, inplace=True)

        features = data.drop(columns=['binds'])
        targets = data['binds'].tolist()

        features = features.astype('float32')
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
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

    train(TRAIN_PARQUET, f"{OUTPUT_DIR}/ctd.parquet", LIMIT, RADIUS, DIM, N_ITER, SAVE_DIR, NBR, LR, FF, N_LEAVES)