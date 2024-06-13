import os
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import gc

from util.utils import load_config
from data.dataset import preprocess_data

def generate_datasets(cfg):
    os.makedirs(f"{cfg['save_dir']}/utils", exist_ok=True)
    print(cfg['save_dir'])
    save_file = os.path.join(cfg['save_dir'], 'train_added.parquet')
    
    try:
        con = duckdb.connect()
        
        total_query = f"""
            SELECT COUNT(*) AS total_count
            FROM parquet_scan('{cfg['train_parquet']}')
        """
        total_data_count = con.execute(total_query).fetchone()[0]
        print(f"Total data count: {total_data_count}")
        
        offset = 0
        processed_data_count = 0

        # Create an empty Parquet file with the appropriate schema if it doesn't exist
        if not os.path.exists(save_file):
            empty_df = pd.DataFrame()
            empty_df.to_parquet(save_file)

        while True:
            query = f"""
                SELECT id, molecule_smiles, protein_name, binds
                FROM parquet_scan('{cfg['train_parquet']}')
                LIMIT {cfg['limit']} 
                OFFSET {offset}
            """
            
            try:
                data = con.execute(query).fetchdf()
            except Exception as e:
                print(f"Query failed: {e}")
                break

            if data.empty:
                break

            binds_0_count = data[data['binds'] == 0].shape[0]
            binds_1_count = data[data['binds'] == 1].shape[0]
            print(f"Dataset shape : {data.shape}, binds=0 count : {binds_0_count}, binds=1 count : {binds_1_count}")
            
            smiles_list = data['molecule_smiles'].tolist()
            processed_data = preprocess_data(data, smiles_list, f"{cfg['data_dir']}/ctd.parquet", cfg['save_dir'], cfg['radius'], cfg['vec_dim'], is_train=True)            
            
            # Ensure all column names are strings
            processed_data.columns = [str(col) for col in processed_data.columns]

            # Append processed data to the Parquet file
            table = pa.Table.from_pandas(processed_data)
            with pq.ParquetWriter(save_file, table.schema, compression='snappy', use_dictionary=True) as writer:
                writer.write_table(table)

            del data
            del processed_data
            gc.collect()
            
            processed_data_count += cfg['limit']
            remaining_data_count = total_data_count - processed_data_count
            print(f"Processed data count: {processed_data_count}, Remaining data count: {remaining_data_count}")
            
            offset += cfg['limit']

    finally:
        con.close()


if __name__ == "__main__":
    config_path = 'config.yaml'
    cfg = load_config(config_path)

    generate_datasets(cfg)
