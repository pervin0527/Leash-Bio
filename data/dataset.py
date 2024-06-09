import os
import time
import json

import duckdb
import requests
import pandas as pd
import pyarrow as pa
import multiprocessing as mp
import pyarrow.parquet as pq

from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import OneHotEncoder

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BitVectToText
from rdkit.Chem import Descriptors

from PyBioMed.Pyprotein import PyProtein


def num_class_data(parquet_path):
    """
    binds=0, 1에 대한 갯수를 측정하고 반환.
    """
    con = duckdb.connect()

    ## binds=0인 데이터만 load
    count_binds_0 = con.query(f"""SELECT COUNT(*) 
                                FROM parquet_scan('{parquet_path}') 
                                WHERE binds = 0""").fetchone()[0]
    print(f"Total binds=0 : {count_binds_0}")

    ## binds=1인 데이터만 load
    count_binds_1 = con.query(f"""SELECT COUNT(*) 
                                FROM parquet_scan('{parquet_path}') 
                                WHERE binds = 1""").fetchone()[0]
    print(f"Total binds=1 : {count_binds_1}")
    con.close()

    ## 전체 데이터 수
    total_count = count_binds_0 + count_binds_1
    print(f"Total data : {total_count}\n")

    return count_binds_0, count_binds_1


def unique_per_columns(parquet_path, output_dir):
    """
    각각의 컬럼에서 고유한 값의 갯수를 측정하고 output_dir에 저장.
    """
    columns = [
        'buildingblock1_smiles', 
        'buildingblock2_smiles', 
        'buildingblock3_smiles', 
        'molecule_smiles', 
        'protein_name'
    ]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    con = duckdb.connect()
    molecule_smiles_unique_count = 0
    protein_name_unique_values = []

    for column in columns:
        print(f"Column : {column}")
        query = f"SELECT {column}, COUNT(*) as count FROM parquet_scan('{parquet_path}') GROUP BY {column}"
        df = con.query(query).df()

        if column == 'molecule_smiles':
            molecule_smiles_unique_count = df.shape[0]
        elif column == 'protein_name':
            protein_name_unique_values = df[column].unique().tolist()

        output_path = os.path.join(output_dir, f"{column}_uniques.parquet")
        df.to_parquet(output_path)

    con.close()

    print(f"Molecules : {molecule_smiles_unique_count}")
    print(f"Proteins : {protein_name_unique_values}\n")
    print(f"Parquet File Saved at {output_path}")
    return molecule_smiles_unique_count, protein_name_unique_values


def get_protein_sequences(uniprot_dicts, output_path):
    """
    표적 단백질에 대한 sequence 계산.
    """
    def fetch_sequence(uniprot_id):
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        response = requests.get(url)
        if response.status_code == 200:
            response_text = response.text
            lines = response_text.splitlines()
            seq = "".join(lines[1:])
            return seq
        else:
            return None

    protein_seq_dicts = {}
    for protein_name, uniprot_id in uniprot_dicts.items():
        protein_sequence = fetch_sequence(uniprot_id)
        if protein_sequence:
            protein_seq_dicts[protein_name] = protein_sequence
        else:
            print(f"Failed to retrieve sequence for {protein_name} ({uniprot_id})")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with open(f"{output_path}/protein_sequence.json", 'w') as file:
        json.dump(protein_seq_dicts, file)

    print(f"Protein Sequence \n {protein_seq_dicts}")
    print(f"Protein Sequence Saved at {output_path}/protein_sequence.json \n")


def load_protein_sequences_from_file(file_path):
    with open(file_path, 'r') as file:
        protein_seq_dicts = json.load(file)
        
    return protein_seq_dicts


def save_protein_ctd_to_parquet(protein_seq_dicts, output_path):
    """
    표적 단백질에 대한 CTD를 계산하고 저장.
    """
    ctd_features = []
    for protein_name, sequence in protein_seq_dicts.items():
        protein_class = PyProtein(sequence)
        ctd = protein_class.GetCTD()
        ctd = {'protein_name': protein_name, **ctd}
        ctd_features.append(ctd)

    ctd_df = pd.DataFrame(ctd_features)
    ctd_df = ctd_df[['protein_name'] + [col for col in ctd_df.columns if col != 'protein_name']]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ctd_df.to_parquet(f"{output_path}/ctd.parquet", index=False)
    ctd_df.to_csv(f"{output_path}/ctd.csv", index=False)
    print(f"Target Proteins CTD Saved at {output_path}/ctd.parquet \n")


def compute_fingerprint(mol, radius, dim):
    """
    molecule을 Morgan FingerPrint로 변환.
    """
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=dim)

    return BitVectToText(fp)


def calculate_descriptors(smiles):
    """
    molecule로부터 descriptor 계산.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    descriptors = Descriptors.CalcMolDescriptors(mol)
    return descriptors


def process_row(smiles, radius, vector_dim, descriptor=True):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return {'fingerprint': None, 'descriptors': {}}
    
    fingerprint = compute_fingerprint(mol, radius, vector_dim)

    if descriptor:
        descriptors = calculate_descriptors(smiles)
        return {'fingerprint': fingerprint, 'descriptors': descriptors}
    else:
        return {'fingerprint' : fingerprint}


def process_smiles_list(smiles_list, radius, dim, desc=True):
    with mp.Pool() as pool:
        results = pool.starmap(process_row, [(smiles, radius, dim, desc) for smiles in smiles_list])
    return results


def preprocess_molecule(parquet_path, ctd_file_path, output_path, output_file_name, radius=2, dim=1024, chunk_size=10000, debug=False):
    ctd_df = pd.read_parquet(ctd_file_path, engine="pyarrow")
    num_workers = cpu_count()
    pool = Pool(num_workers)

    offset = 0
    first_chunk = True
    con = duckdb.connect()
    
    excluded_descriptors_file = f"{output_path}/excluded_descriptors.txt"
    used_descriptors_file = f"{output_path}/used_descriptors.txt"
    
    with open(excluded_descriptors_file, 'w') as excl_file, open(used_descriptors_file, 'w') as used_file:
        while True:
            start_time = time.time()
            
            if debug and offset == 20000:
                break
            
            chunk = con.execute(f"""SELECT *
                                    FROM parquet_scan('{parquet_path}')
                                    LIMIT {chunk_size} 
                                    OFFSET {offset}
                                 """).fetch_df()

            if chunk.empty:
                break

            smiles_list = chunk['molecule_smiles'].tolist()
            results = process_smiles_list(smiles_list, radius, dim)

            fingerprints = [result['fingerprint'] for result in results]
            descriptors_list = [result['descriptors'] for result in results]
            
            chunk['fingerprints'] = fingerprints
            descriptor_df = pd.DataFrame(descriptors_list)
            excluded_descriptors = descriptor_df.columns[descriptor_df.isna().any()].tolist()
            descriptor_df.drop(columns=excluded_descriptors, inplace=True)
            used_descriptor = descriptor_df.columns.tolist()

            if first_chunk:
                print(f"제외된 descriptors: {len(excluded_descriptors)}, 사용된 descriptors: {len(used_descriptor)}")
                # excl_file.write("Excluded Descriptors:\n")
                excl_file.write("\n".join(excluded_descriptors) + "\n")
                # used_file.write("Used Descriptors:\n")
                used_file.write("\n".join(used_descriptor) + "\n")

            # CTD 데이터 병합 (protein_name 기준)
            merged_chunk = pd.merge(chunk, ctd_df, on='protein_name', how='left')

            # protein_name 원핫인코딩
            protein_one_hot = pd.get_dummies(merged_chunk['protein_name'], prefix='protein')
            merged_chunk = pd.concat([merged_chunk, protein_one_hot], axis=1)
            merged_chunk.drop(columns=['protein_name'], inplace=True)

            merged_chunk = pd.concat([merged_chunk, descriptor_df], axis=1)
            
            table = pa.Table.from_pandas(merged_chunk)

            if first_chunk:
                writer = pq.ParquetWriter(f"{output_path}/{output_file_name}.parquet", table.schema)
                first_chunk = False

            writer.write_table(table)
            offset += chunk_size

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Processed offset: {offset} saved to {output_path}/{output_file_name}.parquet Time taken: {elapsed_time:.2f} seconds")

        pool.close()
        pool.join()

        writer.close()
        con.close()