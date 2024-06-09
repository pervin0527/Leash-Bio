import yaml

from data.dataset import (num_class_data, unique_per_columns, get_protein_sequences, 
                          load_protein_sequences_from_file, save_protein_ctd_to_parquet, preprocess_molecule)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def prepare_dataset(train_parq_path, test_parq_path, output_path, uniprot_dict, radius, dim, debug=False):
    # num_b0, num_b1 = num_class_data(train_parq_path)
    # num_molecules, target_proteins = unique_per_columns(train_parq_path, output_path)
    # target_pt_seqs = get_protein_sequences(uniprot_dict, output_path)
    
    protein_sequences = load_protein_sequences_from_file(f"{output_path}/protein_sequence.json")
    save_protein_ctd_to_parquet(protein_sequences, output_path)
    
    # preprocess_molecule(train_parq_path,
    #                     f"{output_path}/ctd.parquet",
    #                     output_path,
    #                     "train_added",
    #                     radius=radius,
    #                     dim=dim,
    #                     debug=debug)

    # preprocess_molecule(test_parq_path,
    #                     f"{output_path}/ctd.parquet",
    #                     output_path,
    #                     "test_added",
    #                     radius=radius,
    #                     dim=dim,
    #                     debug=debug)