import os
import duckdb
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq

from tqdm import tqdm
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from data.dataset import process_smiles_list, compute_fingerprint
from utils.utils import load_config, prepare_dataset, normalize_ctd, load_models, predict_with_models


def predict(test_parquet_path, saved_dir, radius, dim, chunk_size=10000):
    offset = 0
    conn = duckdb.connect()
    total_rows_query = f"SELECT COUNT(*) FROM '{test_parquet_path}'"
    total_rows = conn.execute(total_rows_query).fetchone()[0]

    weights_dir = os.path.join(saved_dir, 'weights')
    models = load_models(weights_dir)

    result_file = './my_submission.csv'
    with open(result_file, 'w') as f:
        f.write('id,binds\n')

    while offset < total_rows:
        query = f"""
        SELECT * FROM '{test_parquet_path}'
        LIMIT {chunk_size} 
        OFFSET {offset}
        """
        chunk_df = conn.execute(query).fetchdf()
        print(f"{offset:>09} : {chunk_df.shape}")

        smiles_list = chunk_df['molecule_smiles'].tolist()
        results = process_smiles_list(smiles_list, radius, dim, desc=False)

        fingerprints = [result['fingerprint'] for result in results]
        chunk_df['fingerprints'] = fingerprints
        fingerprints = chunk_df['fingerprints'].apply(lambda x: np.array([int(char) for char in x]))
        fingerprint_df = pd.DataFrame(fingerprints.tolist(), index=chunk_df.index)
        chunk_df = pd.concat([chunk_df, fingerprint_df], axis=1)
        chunk_df.drop(columns=['fingerprints'], inplace=True)

        ctd_df = pd.read_parquet(f"{saved_dir}/normalized_ctd.parquet", engine='pyarrow')
        chunk_df = pd.merge(chunk_df, ctd_df, on='protein_name', how='left')

        protein_one_hot = pd.get_dummies(chunk_df['protein_name'], prefix='protein_')
        chunk_df = pd.concat([chunk_df, protein_one_hot], axis=1)
        chunk_df.drop(columns=['protein_name'], inplace=True)

        exclude_columns = ['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles']
        chunk_df.drop(columns=exclude_columns, inplace=True)

        features = chunk_df.drop(columns=['id']).astype(float)
        predictions = predict_with_models(models, features)
        
        chunk_results = pd.DataFrame({
            'id': chunk_df['id'],
            'binds': predictions
        })

        # 예측 결과를 파일에 append
        chunk_results.to_csv(result_file, mode='a', header=False, index=False)
        
        offset += chunk_size

    conn.close()
    print("Prediction results saved to ./my_submission.csv")


def train(parquet_path, ctd_path, limit, radius, dim, n_iter, save_dir):
    utils_dir = os.path.join(save_dir, 'utils')
    weights_dir = os.path.join(save_dir, 'weights')
    os.makedirs(utils_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    for i in range(n_iter):
        con = duckdb.connect()
        data = con.query(f"""(SELECT *
                            FROM parquet_scan('{parquet_path}')
                            WHERE binds = 0
                            ORDER BY random()
                            LIMIT {limit})
                            UNION ALL
                            (SELECT *
                            FROM parquet_scan('{parquet_path}')
                            WHERE binds = 1
                            ORDER BY random()
                            LIMIT {limit})""").df()
        con.close()
        print(data.shape)

        smiles_list = data['molecule_smiles'].tolist()
        results = process_smiles_list(smiles_list, radius, dim, desc=False)

        # Features
        # Feature1: Fingerprints
        fingerprints = [result['fingerprint'] for result in results]
        data['fingerprints'] = fingerprints

        ## fingerprints 컬럼을 숫자형 벡터로 변환
        fingerprints = data['fingerprints'].apply(lambda x: np.array([int(char) for char in x]))

        ## fingerprints 벡터를 개별 열로 확장
        fingerprint_df = pd.DataFrame(fingerprints.tolist(), index=data.index)

        ## 원래 데이터프레임에 추가하고 fingerprints 컬럼을 제거
        data = pd.concat([data, fingerprint_df], axis=1)
        data.drop(columns=['fingerprints'], inplace=True)

        ## Feature2: Target Protein & Protein Descriptors
        ctd_df = pd.read_parquet(ctd_path, engine='pyarrow')
        ctd_df = normalize_ctd(ctd_df, utils_dir)
        
        # Save normalized ctd_df
        ctd_df.to_parquet(os.path.join(save_dir, 'normalized_ctd.parquet'), engine='pyarrow')

        data = pd.merge(data, ctd_df, on='protein_name', how='left')

        ## Feature3: OneHot Encoding of protein_name
        protein_one_hot = pd.get_dummies(data['protein_name'], prefix='protein_')
        data = pd.concat([data, protein_one_hot], axis=1)

        ## 이미 protein_name 컬럼이 삭제되었으므로 중복 삭제 방지
        data.drop(columns=['protein_name'], inplace=True)

        ## Exclude unwanted columns
        exclude_columns = ['id', 'buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles']
        data.drop(columns=exclude_columns, inplace=True)

        ## Merge Features
        features = data.drop(columns=['binds'])

        ## Targets
        targets = data['binds'].tolist()

        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        print(f"Train data shape : {X_train.shape}, {len(y_train)}")
        print(f"Test data shape : {X_test.shape}, {len(y_test)}")

        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,  # 리프노드의 수
            'learning_rate': 0.05,
            'feature_fraction': 1.0,  # 낮을수록 임의로 선정되는 비중이 커짐.
            'device': 'gpu',  # GPU 사용 설정
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        }

        evals_result = {}
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
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

        print(f"Iteration {i+1}:")
        print(f"Accuracy: {accuracy}")
        print(f"ROC AUC: {roc_auc}")

        # Save the model
        model.save_model(os.path.join(weights_dir, f'model_{i}.txt'))

        print(f"Model and scaler for iteration {i+1} saved.\n")


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

    N_ITER = config['n_iter']
    LIMIT = config['limit']
    RADIUS = config['radius']
    DIM = config['vec_dim']

    TARGET_PROTEIN = config['target_proteins']
    UNIPROT_DICT = config['uniprot_dicts']

    if PREPARE:
        prepare_dataset(TRAIN_PARQUET, TEST_PARQUET, OUTPUT_DIR, UNIPROT_DICT, RADIUS, DIM, DEBUG)

    train(TRAIN_PARQUET, f"{OUTPUT_DIR}/ctd.parquet", LIMIT, RADIUS, DIM, N_ITER, SAVE_DIR)
    predict(TEST_PARQUET, SAVE_DIR, RADIUS, DIM, LIMIT)