from data_collection import load_data
from train_test_split import split_data
from feature_engineering import create_pipeline
from resampling import apply_smote
from model_training import perform_grid_search
import joblib

# Define file path
# Define file path using an absolute path and a raw string (r prefix):
file_path = r"E:\Hasib's Github\Data Science Intern@CodeClause\Task-3 Parkinson Disease Detection\Parkinsson disease.csv"

# Load data
df = load_data(file_path)

# Split data
x_train, x_test, y_train, y_test = split_data(df)

# Apply SMOTE resampling
x_train_resampled, y_train_resampled = apply_smote(x_train, y_train)

# Create a pipeline with preprocessing and model
pipeline = create_pipeline(x_train_resampled, y_train_resampled)

# Define the hyperparameter grid
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.1, 0.05, 0.01],
    'classifier__max_depth': [3, 4, 5],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV with the pipeline
best_pipeline = perform_grid_search(pipeline, param_grid, x_train_resampled, y_train_resampled)

# Save the best pipeline to a joblib file
joblib.dump(best_pipeline, 'parkinson_predictor_model.joblib')
print("Best pipeline saved to 'parkinson_predictor_model.joblib'.")
