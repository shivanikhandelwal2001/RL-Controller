import os
import pandas as pd

actual_dir = 'env_logs/episode_actual_data'  
prediction_dir = 'env_logs/episode_prediction_data'  

output_csv = 'env_logs/episodes_eval.csv' 


actual_files = sorted([f for f in os.listdir(actual_dir) if f.endswith('.csv')])
prediction_files = sorted([f for f in os.listdir(prediction_dir) if f.endswith('.csv')])

if len(actual_files) != len(prediction_files):
    print("Mismatch in the number of actual and prediction files.")
    exit()

results = []

for actual_file, prediction_file in zip(actual_files, prediction_files):
    actual_path = os.path.join(actual_dir, actual_file)
    prediction_path = os.path.join(prediction_dir, prediction_file)

    actual_df = pd.read_csv(actual_path)
    prediction_df = pd.read_csv(prediction_path)
    
    if list(actual_df.columns) != list(prediction_df.columns):
        print(f"Column mismatch in files {actual_file} and {prediction_file}.")
        continue
    
    # Calculate min, max, and average for each column
    for col in actual_df.columns:
        actual_col = actual_df[col]
        prediction_col = prediction_df[col]
        
        result = {
            'File': f"Episode {actual_file.split('.')[0].split('_')[-1]}",
            'Column': col,
            'Actual Min': actual_col.min(),
            'Actual Max': actual_col.max(),
            'Actual Avg': actual_col.mean(),
            'Prediction Min': prediction_col.min(),
            'Prediction Max': prediction_col.max(),
            'Prediction Avg': prediction_col.mean(),
        }
        results.append(result)

results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)

print(f"Results have been saved to {output_csv}")
