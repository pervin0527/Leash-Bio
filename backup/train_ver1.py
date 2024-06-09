import yaml
import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb

from rdkit import Chem
from rdkit.Chem import AllChem
from data.dataset import process_smiles_list
from utils.utils import load_config, prepare_dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalize_ctd(ctd_df):
    # Initialize scalers
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # Apply Min-Max normalization to columns with prefix '_Polarizability' and '_SolventAccessibility'
    polarizability_columns = [col for col in ctd_df.columns if '_Polarizability' in col]
    solvent_accessibility_columns = [col for col in ctd_df.columns if '_SolventAccessibility' in col]
    ctd_df[polarizability_columns] = min_max_scaler.fit_transform(ctd_df[polarizability_columns])
    ctd_df[solvent_accessibility_columns] = min_max_scaler.fit_transform(ctd_df[solvent_accessibility_columns])

    # Apply Z-score normalization to columns with prefix '_SecondaryStr' and '_Hydrophobicity'
    secondary_str_columns = [col for col in ctd_df.columns if '_SecondaryStr' in col]
    hydrophobicity_columns = [col for col in ctd_df.columns if '_Hydrophobicity' in col]
    ctd_df[secondary_str_columns] = standard_scaler.fit_transform(ctd_df[secondary_str_columns])
    ctd_df[hydrophobicity_columns] = standard_scaler.fit_transform(ctd_df[hydrophobicity_columns])

    return ctd_df


def train(parquet_path, ctd_path, limit, radius, dim):
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
    ctd_df = normalize_ctd(ctd_df)
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
    features = data.drop(columns=['binds'])  # Exclude the target column

    ## Targets
    targets = data['binds'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    print(f" Train data shape : {X_train.shape}, {len(y_train)}")
    print(f" Test data shape : {X_test.shape}, {len(y_test)}")


    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31, ## 리프노드의 수
        'learning_rate': 0.05,
        'feature_fraction': 1.0, ## 낮을수록 임의로 선정되는 비중이 커짐.
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

    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC: {roc_auc}")


if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_config(config_path)

    DEBUG = config['debug']
    PREPARE = config['prepare']

    DATA_DIR = config['data_dir']
    OUTPUT_DIR = config['output_dir']
    TRAIN_PARQUET = config['train_parquet']
    TEST_PARQUET = config['test_parquet']

    LIMIT = config['limit']
    RADIUS = config['radius']
    DIM = config['vec_dim']

    TARGET_PROTEIN = config['target_proteins']
    UNIPROT_DICT = config['uniprot_dicts']

    if PREPARE:
        prepare_dataset(TRAIN_PARQUET, TEST_PARQUET, OUTPUT_DIR, UNIPROT_DICT, RADIUS, DIM, DEBUG)

    train(TRAIN_PARQUET, f"{OUTPUT_DIR}/ctd.parquet", LIMIT, RADIUS, DIM)