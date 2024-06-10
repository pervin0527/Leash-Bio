import os
import yaml
import json
import pickle
import numpy as np
import lightgbm as lgb

from sklearn.preprocessing import MinMaxScaler, StandardScaler

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


def normalize_ctd(ctd_df, utils_dir):
    # Initialize scalers
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # Apply Min-Max normalization to columns with prefix '_Polarizability' and '_SolventAccessibility'
    polarizability_columns = [col for col in ctd_df.columns if '_Polarizability' in col]
    solvent_accessibility_columns = [col for col in ctd_df.columns if '_SolventAccessibility' in col]
    ctd_df[polarizability_columns] = min_max_scaler.fit_transform(ctd_df[polarizability_columns])
    ctd_df[solvent_accessibility_columns] = min_max_scaler.fit_transform(ctd_df[solvent_accessibility_columns])

    # Save Min-Max Scaler and column names
    with open(os.path.join(utils_dir, 'min_max_scaler.pkl'), 'wb') as f:
        pickle.dump((min_max_scaler, polarizability_columns, solvent_accessibility_columns), f)

    # Apply Z-score normalization to columns with prefix '_SecondaryStr' and '_Hydrophobicity'
    secondary_str_columns = [col for col in ctd_df.columns if '_SecondaryStr' in col]
    hydrophobicity_columns = [col for col in ctd_df.columns if '_Hydrophobicity' in col]
    ctd_df[secondary_str_columns] = standard_scaler.fit_transform(ctd_df[secondary_str_columns])
    ctd_df[hydrophobicity_columns] = standard_scaler.fit_transform(ctd_df[hydrophobicity_columns])

    # Save Standard Scaler and column names
    with open(os.path.join(utils_dir, 'standard_scaler.pkl'), 'wb') as f:
        pickle.dump((standard_scaler, secondary_str_columns, hydrophobicity_columns), f)
    
    return ctd_df

def load_scalers_and_apply(ctd_df, utils_dir):
    # Load Min-Max Scaler and column names
    with open(os.path.join(utils_dir, 'min_max_scaler.pkl'), 'rb') as f:
        min_max_scaler, polarizability_columns, solvent_accessibility_columns = pickle.load(f)

    # Load Standard Scaler and column names
    with open(os.path.join(utils_dir, 'standard_scaler.pkl'), 'rb') as f:
        standard_scaler, secondary_str_columns, hydrophobicity_columns = pickle.load(f)

    # Ensure columns exist in the new dataframe, if not add columns with zeros
    for col in polarizability_columns:
        if col not in ctd_df.columns:
            ctd_df[col] = 0
    for col in solvent_accessibility_columns:
        if col not in ctd_df.columns:
            ctd_df[col] = 0
    for col in secondary_str_columns:
        if col not in ctd_df.columns:
            ctd_df[col] = 0
    for col in hydrophobicity_columns:
        if col not in ctd_df.columns:
            ctd_df[col] = 0

    # Apply Min-Max normalization to columns with prefix '_Polarizability' and '_SolventAccessibility'
    ctd_df[polarizability_columns] = min_max_scaler.transform(ctd_df[polarizability_columns])
    ctd_df[solvent_accessibility_columns] = min_max_scaler.transform(ctd_df[solvent_accessibility_columns])

    # Apply Z-score normalization to columns with prefix '_SecondaryStr' and '_Hydrophobicity'
    ctd_df[secondary_str_columns] = standard_scaler.transform(ctd_df[secondary_str_columns])
    ctd_df[hydrophobicity_columns] = standard_scaler.transform(ctd_df[hydrophobicity_columns])
    
    return ctd_df


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