import os
import json
import random
import glob
from tqdm import tqdm
from termcolor import cprint

class StructuralProbeGenerator:

    def __init__(self, tables_dir, num_samples_per_type=500):
        self.tables_dir = tables_dir
        self.num_samples_per_type = num_samples_per_type
        self.table_files = glob.glob(os.path.join(self.tables_dir, "*.json"))
        cprint(f"Found {len(self.table_files)} tables in '{self.tables_dir}'.", "cyan")

    def _load_table(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def _get_column_name(self, headers, col_idx):
        if headers and len(headers) > col_idx:
            return headers[col_idx]
        return f"Column {col_idx + 1}"
    
    def _is_valid_value(self, value):
        if value is None:
            return False
        str_value = str(value).strip()
        if len(str_value) == 0:
            return False
        if str_value.lower() in ['none', 'null', 'n/a', 'na', '']:
            return False
        return True

    def generate_row_lookups(self):
        samples = []
        pbar = tqdm(total=self.num_samples_per_type, desc="Generating Row Lookups")
        
        while len(samples) < self.num_samples_per_type:
            table_file = random.choice(self.table_files)
            table = self._load_table(table_file)
            
            if not table.get('data') or len(table['data']) < 2:
                continue
            num_cols = len(table['data'][0])
            if num_cols < 2:
                continue

            try:
                row_idx = random.randint(0, len(table['data']) - 1)
                row_data = table['data'][row_idx]
                key_col_idx, target_col_idx = random.sample(range(num_cols), 2)
                
                key_header = self._get_column_name(table.get('columns'), key_col_idx)
                target_header = self._get_column_name(table.get('columns'), target_col_idx)

                key_value = row_data[key_col_idx]
                ground_truth_answer = row_data[target_col_idx]
                
                if not self._is_valid_value(key_value) or not self._is_valid_value(ground_truth_answer):
                    continue
                
                if not self._is_valid_value(key_header) or not self._is_valid_value(target_header):
                    continue

                variation = random.choice(['standard', 'action', 'conditional'])
                question = ""
                if variation == 'standard':
                    question = f"What is the value of '{target_header}' for the entry where '{key_header}' is '{key_value}'?"
                elif variation == 'action':
                    question = f"From the table, find the '{target_header}' that corresponds to the '{key_header}' of '{key_value}'."
                elif variation == 'conditional':
                    question = f"If the '{key_header}' is listed as '{key_value}', what is its corresponding '{target_header}'?"

                samples.append({
                    "table_id": os.path.splitext(os.path.basename(table_file))[0],
                    "question": question,
                    "answer": [str(ground_truth_answer)], 
                    "probe_label": "row_lookup"
                })
                pbar.update(1)
            except (ValueError, IndexError):
                continue
        pbar.close()
        return samples

    def generate_column_lookups(self):
        samples = []
        pbar = tqdm(total=self.num_samples_per_type, desc="Generating Column Lookups")
        
        while len(samples) < self.num_samples_per_type:
            table_file = random.choice(self.table_files)
            table = self._load_table(table_file)

            if not table.get('data') or len(table['data']) < 2:
                continue
            num_cols = len(table['data'][0])
            if num_cols < 1:
                continue

            try:
                variation = random.choice(['list_all', 'reverse_lookup', 'count'])
                question = ""
                ground_truth_answer = None
                
                col_idx = random.randint(0, num_cols - 1)
                header = self._get_column_name(table.get('columns'), col_idx)
                
                if not self._is_valid_value(header):
                    continue
                
                column_data = [str(row[col_idx]) for row in table['data']]

                if variation == 'list_all':
                    valid_column_data = [v for v in column_data if self._is_valid_value(v)]
                    if not valid_column_data:
                        continue
                    question = f"List all the data entries present in the column named '{header}'."
                    ground_truth_answer = valid_column_data

                elif variation == 'reverse_lookup':
                    non_empty_values = [v for v in column_data if self._is_valid_value(v)]
                    if not non_empty_values:
                        continue
                    cell_value = random.choice(non_empty_values)
                    question = f"In which column of the table can the value '{cell_value}' be found?"
                    ground_truth_answer = [header]

                elif variation == 'count':
                    valid_rows = [row[col_idx] for row in table['data'] if self._is_valid_value(row[col_idx])]
                    if not valid_rows:
                        continue
                    question = f"How many data rows are there in the '{header}' column?"
                    ground_truth_answer = [str(len(valid_rows))]

                samples.append({
                    "table_id": os.path.splitext(os.path.basename(table_file))[0],
                    "question": question,
                    "answer": ground_truth_answer,
                    "probe_label": "column_lookup"
                })
                pbar.update(1)
            except (ValueError, IndexError):
                continue
        pbar.close()
        return samples

    def generate(self, output_file):
        cprint("\nStarting Structural Probe Dataset Generation", "magenta", attrs=["bold"])
        row_samples = self.generate_row_lookups()
        col_samples = self.generate_column_lookups()

        full_dataset = row_samples + col_samples
        random.shuffle(full_dataset)

        with open(output_file, 'w') as f:
            json.dump(full_dataset, f, indent=2)
        
        cprint(f"\nSuccessfully generated {len(full_dataset)} samples.", "green")
        cprint(f"Dataset saved to '{output_file}'.", "green")
        cprint(f"Class distribution: {len(row_samples)} row_lookups, {len(col_samples)} column_lookups.", "cyan")

if __name__ == "__main__":
    
    TABLES_DIRECTORY = "./data/processed/tables"
    OUTPUT_FILE_PATH = "./data/processed/probes/structural_probe_dataset.json"
    
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    generator = StructuralProbeGenerator(tables_dir=TABLES_DIRECTORY, num_samples_per_type=1000)
    generator.generate(output_file=OUTPUT_FILE_PATH)