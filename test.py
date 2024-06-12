## kaggle competitions submit -c leash-BELKA -f my_submission.csv -m "My submission"

import os
import json
import joblib
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.dataset import process_smiles_list, preprocess_data
from util.utils import load_config, load_models, load_top_k_models, predict_with_models

def plot_feature_importance(weights_dir, k, n):
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

    top_n_features = sorted_features[:n]
    feature_names, importance_scores = zip(*top_n_features)

    grouped_feature_importance = {}
    for name, score in avg_feature_importance.items():
        if name.isdigit():
            group_name = "fingerprint"
        else:
            group_name = name
        if group_name in grouped_feature_importance:
            grouped_feature_importance[group_name] += score
        else:
            grouped_feature_importance[group_name] = score

    sorted_grouped_features = sorted(grouped_feature_importance.items(), key=lambda x: x[1], reverse=True)
    grouped_feature_names, grouped_importance_scores = zip(*sorted_grouped_features[:n])

    plt.figure(figsize=(10, 8))
    plt.barh(grouped_feature_names, grouped_importance_scores)
    plt.xlabel('Average Gain')
    plt.ylabel('Feature Group')
    plt.title(f'Top {n} Feature Groups Importance from Top {k} Models')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    os.makedirs(f"{weights_dir}/importance_scores", exist_ok=True)
    importance_plot_path = os.path.join(weights_dir, 'importance_scores', 'feature_importance_grouped.png')
    plt.savefig(importance_plot_path)
    plt.close()
    print(f"Grouped feature importance plot saved to {importance_plot_path}")


def predict(test_parquet_path, saved_dir, radius, dim, k=5, chunk_size=10000):
    offset = 0
    conn = duckdb.connect()
    total_rows_query = f"SELECT COUNT(*) FROM '{test_parquet_path}'"
    total_rows = conn.execute(total_rows_query).fetchone()[0]

    performance_file = os.path.join(f"{saved_dir}/utils", 'model_performance.json')
    models = load_top_k_models(performance_file, k)

    # Load the descriptor normalizer
    normalizer_path = os.path.join(saved_dir, 'utils', 'descriptor_normalizer.pkl')
    preprocessor = joblib.load(normalizer_path)

    result_file = './my_submission.csv'
    with open(result_file, 'w') as f:
        f.write('id,binds\n')

    while offset < total_rows:
        query = f"""
        SELECT id, molecule_smiles, protein_name
        FROM '{test_parquet_path}'
        LIMIT {chunk_size} 
        OFFSET {offset}
        """
        chunk_df = conn.execute(query).fetchdf()
        print(f"{offset:>09} : {chunk_df.shape}")

        smiles_list = chunk_df['molecule_smiles'].tolist()
        chunk_df = preprocess_data(chunk_df,
                                   smiles_list, 
                                   f"{saved_dir}/utils/normalized_ctd.parquet", 
                                   saved_dir, 
                                   radius, 
                                   dim, 
                                   is_train=False,
                                   important_features=['SMR_VSA4', 'SlogP_VSA1', 'fr_phenol', 'NumSaturatedCarbocycles', 'fr_Ar_NH'])

        features = chunk_df.drop(columns=['id', 'molecule_smiles'])
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
    TOP_N = config['feature_top_n']

    plot_feature_importance(CKPT_DIR, TOP_K, TOP_N)
    predict(TEST_PARQUET, CKPT_DIR, RADIUS, DIM, TOP_K, LIMIT)