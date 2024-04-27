import joblib
import pandas as pd

# Load the saved pipeline from the joblib file
loaded_pipeline = joblib.load('parkinson_predictor_model.joblib')
print("Loaded the best pipeline from 'parkinson_predictor_model.joblib'.")

# Unseen custom data in a DataFrame (you can change this as needed)
# For example, assume `unseen_data.csv` contains the unseen custom data
unseen_data_file_path = r"E:\Hasib's Github\Data Science Intern@CodeClause\Task-3 Parkinson Disease Detection\unseen_data_for_testing.csv"

unseen_data = pd.read_csv(unseen_data_file_path)

# Predict the unseen data using the loaded pipeline
predictions = loaded_pipeline.predict(unseen_data)

# Display the predictions
print("Predictions on unseen custom data:")
print(predictions)