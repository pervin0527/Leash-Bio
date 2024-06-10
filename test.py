import os
import json
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.dataset import process_smiles_list
from utils.utils import load_config, load_models, load_top_k_models, predict_with_models

def plot_feature_importance(weights_dir, k):
    utils_dir = os.path.join(weights_dir, 'utils')
    performance_file = os.path.join(utils_dir, 'model_performance.json')
    models = load_top_k_models(performance_file, k)

    feature_importance_dict = {}
    for model in models:
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        for name, score in zip(feature_names, importance):
            if name in feature_importance_dict:
                feature_importance_dict[name].append(score)
            else:
                feature_importance_dict[name] = [score]

    avg_feature_importance = {name: np.mean(scores) for name, scores in feature_importance_dict.items()}    
    sorted_features = sorted(avg_feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    feature_names, importance_scores = zip(*sorted_features)
    plt.figure(figsize=(10, 8))
    plt.barh(feature_names, importance_scores)
    plt.xlabel('Average Gain')
    plt.ylabel('Feature')
    plt.title(f'Top {k} Models Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    os.makedirs(f"{weights_dir}/importance_scores", exist_ok=True)
    importance_plot_path = os.path.join(weights_dir, 'importance_scores', 'feature_importance.png')
    plt.savefig(importance_plot_path)
    plt.close()
    print(f"Feature importance plot saved to {importance_plot_path}")


def predict(test_parquet_path, saved_dir, radius, dim, k=5, chunk_size=10000):
    offset = 0
    conn = duckdb.connect()
    total_rows_query = f"SELECT COUNT(*) FROM '{test_parquet_path}'"
    total_rows = conn.execute(total_rows_query).fetchone()[0]

    weights_dir = os.path.join(saved_dir, 'weights')
    performance_file = os.path.join(saved_dir, 'model_performance.json')
    models = load_top_k_models(weights_dir, performance_file, k)

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

        chunk_results.to_csv(result_file, mode='a', header=False, index=False)        
        offset += chunk_size

    conn.close()
    print("Prediction results saved to ./my_submission.csv")


if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_config(config_path)

    CKPT_DIR = config['ckpt_dir']
    TEST_PARQUET = config['test_parquet']

    LIMIT = config['limit']
    RADIUS = config['radius']
    DIM = config['vec_dim']

    TOP_K = config['predict_top_k']

    plot_feature_importance(CKPT_DIR, TOP_K)
    predict(TEST_PARQUET, CKPT_DIR, RADIUS, DIM, TOP_K, LIMIT)