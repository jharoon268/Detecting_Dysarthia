import os
import pandas as pd

def load_dataset(base_dir, excel_name, audio_folder_name, result_name):
    excel_path = os.path.join(base_dir, excel_name)
    audio_root = os.path.join(base_dir, audio_folder_name)
    results_dir = os.path.join(os.path.dirname(__file__), "results") if "__file__" in globals() else "results"
    os.makedirs(results_dir, exist_ok=True)
    results_csv = os.path.join(results_dir, result_name)
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip().str.lower()
    audio_files = []
    for root, _, files in os.walk(audio_root):
        for f in files:
            if f.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, f))
    data_records = []
    for _, row in df.iterrows():
        file_id = str(row['id']).strip()
        matched = [f for f in audio_files if file_id in os.path.basename(f)]
        for fpath in matched:
            data_records.append({
                'file_path': os.path.abspath(fpath),
                'id': file_id,
                'age': row.get('age', None),
                'sex': row.get('sex', None),
                'class': row.get('class', None) if 'class' in df.columns else None
            })
    result_df = pd.DataFrame(data_records)
    result_df.to_csv(results_csv, index=False)
    print(f"{result_name} saved with {len(result_df)} entries at {results_csv}")
    return result_df

if __name__ == "__main__":
    current_dir = os.getcwd()
    train_dir = os.path.join(current_dir, "training_data")
    test_dir = os.path.join(current_dir, "testing_data")
    print("\nLoading training dataset...")
    load_dataset(train_dir, "sand_task_1.xlsx", "training", "train_data.csv")
    print("\nLoading testing dataset...")
    load_dataset(test_dir, "sand_task1_test.xlsx", "test", "test_data.csv")
